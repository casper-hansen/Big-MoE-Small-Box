# Big MoE, Small Box

Big MoE, Small Box serves Mixture-of-Experts models that are far larger than the GPU from one 24 GB card, 64 GB of RAM and an NVMe drive: [GLM-5.3-Flash](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3) (321B parameters, EXL3 3.05 bpw, 125 GB) and [DeepSeek-V4.1-Flash](https://huggingface.co/turboderp/DeepSeek-V4.1-Flash-1017-exl3) (EXL3 2.25 bpw, 361 GB including 203 GB of Engram tables), each behind an OpenAI-compatible server on [exllamav3](https://github.com/turboderp-org/exllamav3) 1.6.6 (one release for both, `models/runtime.py`). A token only uses a few routed experts per MoE layer (8 of 288 for GLM, 6 of 384 for DeepSeek), so the routed experts, which are most of the weights, live in three tiers and every layer is served from wherever its experts are:

- **Expert store.** The routed experts are packed into one file of fixed-size records per layer and read with `O_DIRECT`; the checkpoint keeps everything else and its expert bytes are punched out after the store is verified. The loader never reads the experts from the checkpoint: every expert tensor starts as a view of one VRAM bounce slot.
- **Three tiers, exllamav3's own math.** A CLOCK cache of uniform slots fills all free VRAM (layers with different expert sizes share it, each in its own compact layout); a pinned, GPU-mapped RAM tier holds the experts that are not in VRAM; the rest stays on NVMe. exllamav3's kernels read every expert through its live pointer tables, which the engine points at the expert's current home, so the bytes and the arithmetic are exllamav3's own.
- **A scheduler per layer.** The GPU publishes each layer's picks to mapped memory without a host sync; a host planner sends every pick to the cheapest lane: a VRAM hit, a copy in from RAM, a zero-copy read, an NVMe read (demand reads ahead of prefetches, prefetches guessed one layer ahead from the next router), or the CPU lane. Admitted experts evict VRAM victims back into RAM.
- **CPU lane.** RAM-resident and freshly landed experts can be computed on the CPU (trellis decoding with AVX-512 VBMI, or AVX2 for GLM's 3-bit experts; int16 dot products; results combined on the GPU), which takes PCIe and NVMe pressure off the GPU. This is mode `nvme`, the default, and it is near-exact: on the same forced tokens it picks the same top-1 token as exact decode 99.3 % of the time for DeepSeek (mean KL 0.0013, without speculation) and 97.1 % for GLM (mean KL 0.008). Mode `nvme-exact` keeps every expert on the GPU path, so decode is exllamav3's own computation on the same bytes.
- **Bit-exact CPU lane (`x16_layers`, off by default).** The CPU lane's experts differ from the GPU kernel's in the last bits (the GPU accumulates in fp16 with its own rounding); for GLM that alone costs its top-1 agreement. `x16_layers=N` computes the CPU's experts in the first `N` MoE layers bit-identically to the GPU (its tensor-core rounding emulated on the CPU), self-tested at every start: GLM with 21 layers agrees with exact decode on 99.3 % of top-1 tokens (mean KL 0.0012) at 12 % lower decode speed, DeepSeek with all 40 layers decodes exactly. Details: [docs/x16.md](docs/x16.md).
- **Prefill** stages each layer's experts into VRAM buffers one layer ahead of the compute, in memory borrowed from the VRAM cache while the prefill runs (8192-token chunks). Staging every expert reads every NVMe-only one, ~15 s per chunk for DeepSeek whatever its length, which is what each agent step pays for its tool result. Warm prefill (`engine/warm_prefill.py`; option `warm_prefill`, mode `nvme`, on by default for DeepSeek, off for GLM, whose answers react more to it): a chunk runs on the experts already in VRAM or RAM, plus, per layer once its routing is known, the 64 NVMe-only experts the chunk gives the most routing weight, read in parallel; the picks of the remaining ones are dropped and each token's weights rescaled. exllamav3 also prefills a prompt's last partial page as a forward of its own (a checkpoint for follow-ups); mode `nvme` keeps that only when the tail is short enough to skip staging (63 tokens for GLM, 85 for DeepSeek; `engine/last_page.py`), otherwise it costs a whole extra forward. Teacher-forced against exact prefill (prompt and answer through it, 128 answer positions each): DeepSeek top-1 100 %, mean KL 0.023; GLM 94.5 %, mean KL 0.042. On a DeepSeek Harness agent turn (DeepSeek, decode exact after the prefill, 330 forced answer positions): top-1 99.7 %, mean KL 0.0021. Mode `nvme-exact` keeps the exact prefill.
- **Serving agents.** The OpenAI-compatible servers take `tools` and return OpenAI `tool_calls` (DeepSeek's DSML calls are parsed in `models/dsv41/toolcalls.py`), refuse a prompt that does not fit the cache with an immediate 400 `context_length_exceeded` (agent clients compact and retry; `bigmoe/serving.py`), report errors as JSON or in-band SSE events instead of dropping the stream, answer `/health` with 503 and exit if the generator loop dies, and report `usage.prompt_tokens_details.cached_tokens` (prefix reuse) plus server-side `usage.timings` (queue / prefill / decode), with one log line per request. Prompt prefixes are reused from the GPU cache; with `-o disk_cache=<dir>` the pages it evicts (one long conversation evicts the others) go through exllamav3's pinned-RAM tier to disk, written only on eviction or when the server is idle, so a prefix also survives other conversations and restarts. A prefix comes back from disk through that RAM tier, all of it at once, so `cpu_cache_gb` defaults to the whole context (`models/prefix_cache.py`): 0.25 GiB for DeepSeek's 64k at 3.2 KB per token, 2.5 GiB for GLM's 128k at 17.6 KB; a 0.25 GiB tier gave GLM back nothing of a 17k-token prompt. Measured: a 17k-token prompt prefilled in 27.0 s (DeepSeek) / 27.9 s (GLM) took 5.7 s / 5.5 s after a restart.
- **Speculative decoding.** DeepSeek-V4.1-Flash uses its own DSpark block drafter (up to 5 tokens per step, cut by its confidence head at 0.5) by default; GLM-5.3-Flash can draft from the context (`speculation="ngram"`).
- **One command.** `AutoBox` pins and builds the Python runtime (the same exllamav3 release for both models) and the CUDA extensions, downloads and verifies the weights, packs and verifies the store, and starts the server in a memory-capped systemd scope with its threads pinned (main, scheduler, CPU lane and NVMe readers on separate cores).

```python
from bigmoe import AutoBox

box = AutoBox.from_pretrained("deepseek-v4.1-flash")   # or "glm-5.3-flash"; first run downloads, packs and builds
box.start()                                            # OpenAI-compatible server on http://127.0.0.1:30000/v1
print(box.chat("Why do MoE models fit on small boxes?"))
box.stop()
```

The offloading and scheduling engine started as 0xSero's [glm-flash-lite](https://github.com/sybil-solutions/glm-flash-lite) (MIT, `LICENSE.glm-flash-lite`); this package is Apache-2.0 (`LICENSE`).

## Build

- RTX 3090 (used)
- 64GB DDR5-5600 (used)
- WD SN850X (used)
- Ryzen 5 7600X
- ASRock B650 Pro RS
- Peerless Assassin 120 SE
- Lian Li V100
- ASUS TUF GAMING 850G EVO 2026

## GLM-5.3-Flash

`bigmoe bench` prompts (Python source files and a question about them), greedy, 128k context, FP16 KV cache. KL and top-1: mean per-token KL divergence from exact decode (`nvme-exact`) and the share of positions with the same most likely token, on the same forced tokens, six chat prompts, 897 positions.

| mode | prompt tokens | prefill | decode | KL | top-1 |
|---|---|---|---|---|---|
| `nvme` | 1,024 | 113 tok/s | 11.4 tok/s | 0.012 | 97.2 % |
| `nvme` | 8,192 | 776 tok/s | 11.4 tok/s | 0.012 | 97.2 % |
| `nvme` | 32,768 | 833 tok/s | 11.8 tok/s | 0.012 | 97.2 % |
| `nvme` | 120,000 | 826 tok/s | 11.4 tok/s | 0.012 | 97.2 % |
| `nvme-exact` | 1,024 | 60 tok/s | 7.8 tok/s | 0 (reference) | 100 % (reference) |
| `nvme-exact` | 8,192 | 427 tok/s | 7.9 tok/s | 0 (reference) | 100 % (reference) |

(Measured on exllamav3 1.5.1 with exact prefill.) On exllamav3 1.6.6 with warm prefill, mode `nvme` measured 172 / 703 / 744 tok/s prefill at 1,024 / 8,192 / 32,768 prompt tokens and 11.1-11.3 tok/s decode, top-1 97.1 % (mean KL 0.008; six chat prompts, 1,152 positions); the 1.5.1 runtime on the same box and code: 168 / 674 / 715 tok/s, 11.4-11.5 tok/s. exllamav3's fused hyper-connection sites (`EXL3_HC_FOLD=1`) replace the custom decode kernels the 1.5.1 runtime needed.

`bigmoe bench`, prefill on a cached 8,192-token prefix, time to first token by new tokens, mode `nvme`: exact prefill on exllamav3 1.5.1, warm prefill and the last-page split on 1.6.6 (800 new tokens leave a 31-token tail, which keeps its own forward):

| new tokens | 8 | 80 | 120 | 400 | 800 | 2,000 | 5,000 | 10,000 |
|---|---|---|---|---|---|---|---|---|
| exact prefill | 3.0 s | 8.9 s | 8.9 s | 9.0 s | 9.0 s | 9.2 s | 9.8 s | 19.5 s |
| warm prefill | 1.2 s | 3.8 s | 5.1 s | 5.5 s | 10.1 s | 6.8 s | 9.2 s | 20.8 s |

## DeepSeek-V4.1-Flash

`bigmoe bench` prompts, greedy, 64k context, FP16 KV cache, DSpark speculative decoding. KL and top-1: mean per-token KL divergence from exact single-token decode (`nvme-exact`, no speculation) and the share of positions with the same most likely token, on the same forced tokens, six chat prompts, 1,110 positions; with DSpark even `nvme-exact` differs slightly, because the verify forwards round in other GEMM shapes.

| mode | prompt tokens | prefill | decode | KL | top-1 |
|---|---|---|---|---|---|
| `nvme` | 1,024 | 33 tok/s | 15.6 tok/s | 0.0008 | 99.6 % |
| `nvme` | 8,192 | 250 tok/s | 18.1 tok/s | 0.0008 | 99.6 % |
| `nvme` | 32,768 | 405 tok/s | 18.5 tok/s | 0.0008 | 99.6 % |
| `nvme-exact` | 1,024 | 35 tok/s | 11.0 tok/s | 0.0007 | 99.7 % |
| `nvme-exact` | 8,192 | 268 tok/s | 12.6 tok/s | 0.0007 | 99.7 % |

(Measured with exact prefill, before warm prefill.) `bigmoe bench`, prefill on a cached 8,192-token prefix, time to first token by new tokens, mode `nvme` with exact prefill (exllamav3's own last-page forward: 400, 2,000 and 5,000 new tokens take two forwards) and with warm prefill and the last-page split:

| new tokens | 8 | 80 | 120 | 400 | 800 | 2,000 | 5,000 | 10,000 |
|---|---|---|---|---|---|---|---|---|
| exact prefill | 0.7 s | 3.3 s | 14.8 s | 29.0 s | 17.2 s | 29.6 s | 29.8 s | 33.1 s |
| warm prefill | 1.2 s | 3.5 s | 3.2 s | 5.3 s | 8.2 s | 6.9 s | 8.8 s | 19.2 s |
