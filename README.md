# Big MoE, Small Box

Big MoE, Small Box serves Mixture-of-Experts models that are far larger than the GPU from one 24 GB card, 64 GB of RAM and an NVMe drive: [GLM-5.3-Flash](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3) (321B parameters, EXL3 3.05 bpw, 125 GB) and [DeepSeek-V4.1-Flash](https://huggingface.co/turboderp/DeepSeek-V4.1-Flash-1017-exl3) (EXL3 2.25 bpw, 361 GB including 203 GB of Engram tables), each behind an OpenAI-compatible server on [exllamav3](https://github.com/turboderp-org/exllamav3). A token only uses a few routed experts per MoE layer (8 of 288 for GLM, 6 of 384 for DeepSeek), so the routed experts, which are most of the weights, live in three tiers and every layer is served from wherever its experts are:

- **Expert store.** The routed experts are packed into one file of fixed-size records per layer and read with `O_DIRECT`; the checkpoint keeps everything else and its expert bytes are punched out after the store is verified. The loader never reads the experts from the checkpoint: every expert tensor starts as a view of one VRAM bounce slot.
- **Three tiers, exllamav3's own math.** A CLOCK cache of uniform slots fills all free VRAM (layers with different expert sizes share it, each in its own compact layout); a pinned, GPU-mapped RAM tier holds the experts that are not in VRAM; the rest stays on NVMe. exllamav3's kernels read every expert through its live pointer tables, which the engine points at the expert's current home, so the bytes and the arithmetic are exllamav3's own.
- **A scheduler per layer.** The GPU publishes each layer's picks to mapped memory without a host sync; a host planner sends every pick to the cheapest lane: a VRAM hit, a copy in from RAM, a zero-copy read, an NVMe read (demand reads ahead of prefetches, prefetches guessed one layer ahead from the next router), or the CPU lane. Admitted experts evict VRAM victims back into RAM.
- **CPU lane.** RAM-resident and freshly landed experts can be computed on the CPU (trellis decoding with AVX-512 VBMI, or AVX2 for GLM's 3-bit experts; int16 dot products; results combined on the GPU), which takes PCIe and NVMe pressure off the GPU. This is mode `nvme`, the default, and it is near-exact: on the same forced tokens it picks the same top-1 token as exact decode 99.3 % of the time for DeepSeek (mean KL 0.0013, without speculation) and 97.2 % for GLM (mean KL 0.012). Mode `nvme-exact` keeps every expert on the GPU path, so decode is exllamav3's own computation on the same bytes.
- **Prefill** stages each layer's experts into VRAM buffers one layer ahead of the compute, in memory borrowed from the VRAM cache while the prefill runs (8192-token chunks). Staging every expert reads every NVMe-only one, ~15 s per chunk for DeepSeek whatever its length, which is what each agent step pays for its tool result. Mode `nvme` therefore uses warm prefill (`engine/warm_prefill.py`): a chunk runs on the experts already in VRAM or RAM, plus, per layer once its routing is known, the 64 NVMe-only experts the chunk gives the most routing weight, read in parallel; the picks of the remaining ones are dropped and each token's weights rescaled. Teacher-forced against exact prefill (prompt and answer through it, 128 answer positions each): DeepSeek top-1 100 %, mean KL 0.023; GLM 96.1 %, mean KL 0.035. On a DeepSeek Harness agent turn (DeepSeek, decode exact after the prefill, 330 forced answer positions): top-1 99.7 %, mean KL 0.0021. Mode `nvme-exact` keeps the exact prefill.
- **Speculative decoding.** DeepSeek-V4.1-Flash uses its own DSpark block drafter (up to 5 tokens per step, cut by its confidence head at 0.5) by default; GLM-5.3-Flash can draft from the context (`speculation="ngram"`).
- **One command.** `AutoBox` pins and builds the Python runtime and the CUDA extensions, downloads and verifies the weights, packs and verifies the store, and starts the server in a memory-capped systemd scope with its threads pinned (main, scheduler, CPU lane and NVMe readers on separate cores).

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

(Measured with exact prefill, before warm prefill.) `bigmoe bench`, prefill on a cached 8,192-token prefix, time to first token by new tokens, mode `nvme` with exact and with warm prefill:

| new tokens | 8 | 80 | 120 | 400 | 800 | 2,000 | 5,000 | 10,000 |
|---|---|---|---|---|---|---|---|---|
| exact prefill | 3.0 s | 8.9 s | 8.9 s | 9.0 s | 9.0 s | 9.2 s | 9.8 s | 19.5 s |
| warm prefill | 1.9 s | 4.8 s | 5.3 s | 5.7 s | 6.0 s | 6.9 s | 9.4 s | 18.3 s |

## DeepSeek-V4.1-Flash

`bigmoe bench` prompts, greedy, 64k context, FP16 KV cache, DSpark speculative decoding. KL and top-1: mean per-token KL divergence from exact single-token decode (`nvme-exact`, no speculation) and the share of positions with the same most likely token, on the same forced tokens, six chat prompts, 1,110 positions; with DSpark even `nvme-exact` differs slightly, because the verify forwards round in other GEMM shapes.

| mode | prompt tokens | prefill | decode | KL | top-1 |
|---|---|---|---|---|---|
| `nvme` | 1,024 | 33 tok/s | 15.6 tok/s | 0.0008 | 99.6 % |
| `nvme` | 8,192 | 250 tok/s | 18.1 tok/s | 0.0008 | 99.6 % |
| `nvme` | 32,768 | 405 tok/s | 18.5 tok/s | 0.0008 | 99.6 % |
| `nvme-exact` | 1,024 | 35 tok/s | 11.0 tok/s | 0.0007 | 99.7 % |
| `nvme-exact` | 8,192 | 268 tok/s | 12.6 tok/s | 0.0007 | 99.7 % |

(Measured with exact prefill, before warm prefill.) `bigmoe bench`, prefill on a cached 8,192-token prefix, time to first token by new tokens, mode `nvme` with exact and with warm prefill (400, 2,000 and 5,000 new tokens take two forwards: exllamav3 prefills a prompt's last partial page separately):

| new tokens | 8 | 80 | 120 | 400 | 800 | 2,000 | 5,000 | 10,000 |
|---|---|---|---|---|---|---|---|---|
| exact prefill | 0.7 s | 3.3 s | 14.8 s | 29.0 s | 17.2 s | 29.6 s | 29.8 s | 33.1 s |
| warm prefill | 1.1 s | 3.3 s | 3.3 s | 9.5 s | 8.2 s | 11.3 s | 13.1 s | 19.5 s |
