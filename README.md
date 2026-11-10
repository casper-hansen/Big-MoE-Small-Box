# Big MoE, Small Box

Big MoE, Small Box serves Mixture-of-Experts models that are far larger than the GPU from one 24 GB card, 64 GB of RAM and an NVMe drive: [GLM-5.3-Flash](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3) (321B parameters, EXL3 3.05 bpw, 125 GB) and [DeepSeek-V4.1-Flash](https://huggingface.co/turboderp/DeepSeek-V4.1-Flash-1017-exl3) (EXL3 2.25 bpw, 361 GB including 203 GB of Engram tables). Both are served by vLLM 0.33.0 (its OpenAI/Anthropic server, tool and reasoning parsers, and its scheduler), with [exllamav3](https://github.com/turboderp-org/exllamav3) 1.6.6 as the model executor it drives (one release for both, `models/runtime.py`). A token only uses a few routed experts per MoE layer (8 of 288 for GLM, 6 of 384 for DeepSeek), so the routed experts, which are most of the weights, live in three tiers and every layer is served from wherever its experts are:

- **Expert store.** The routed experts are packed into one file of fixed-size records per layer and read with `O_DIRECT`; the checkpoint keeps everything else and its expert bytes are punched out after the store is verified. The loader never reads the experts from the checkpoint: every expert tensor starts as a view of one VRAM bounce slot.
- **Three tiers, exllamav3's own math.** A CLOCK cache of uniform slots fills all free VRAM (layers with different expert sizes share it, each in its own compact layout); a pinned, GPU-mapped RAM tier holds the experts that are not in VRAM; the rest stays on NVMe. exllamav3's kernels read every expert through its live pointer tables, which the engine points at the expert's current home, so the bytes and the arithmetic are exllamav3's own.
- **A scheduler per layer.** The GPU publishes each layer's picks to mapped memory without a host sync; a host planner sends every pick to the cheapest lane: a VRAM hit, a copy in from RAM, a zero-copy read, an NVMe read (demand reads ahead of prefetches, prefetches guessed one layer ahead from the next router), or the CPU lane. Admitted experts evict VRAM victims back into RAM.
- **CPU lane.** RAM-resident and freshly landed experts can be computed on the CPU (trellis decoding with AVX-512 VBMI, or AVX2 for GLM's 3-bit experts; int16 dot products; results combined on the GPU), which takes PCIe and NVMe pressure off the GPU. This is mode `nvme`, the default, and it is near-exact: on the same forced tokens it picks the same top-1 token as exact decode 99.3 % of the time for DeepSeek (mean KL 0.0013, without speculation) and 97.3 % for GLM (mean KL 0.0096). Mode `nvme-exact` keeps every expert on the GPU path, so decode is exllamav3's own computation on the same bytes.
- **Bit-exact CPU lane (`x16_layers`, off by default).** The CPU lane's experts differ from the GPU kernel's in the last bits (the GPU accumulates in fp16 with its own rounding); for GLM that alone costs its top-1 agreement. `x16_layers=N` computes the CPU's experts in the first `N` MoE layers bit-identically to the GPU (its tensor-core rounding emulated on the CPU), self-tested at every start: GLM with 21 layers agrees with exact decode on 99.2 % of top-1 tokens (mean KL 0.0011; default 97.3 %) at 13 % lower decode speed, DeepSeek with all 40 layers decodes exactly (with DSpark, whose verify forwards stay on the GPU: 99.7 %, at 40 % lower decode speed). Details: [docs/x16.md](docs/x16.md).
- **Prefill** stages each layer's experts into VRAM buffers one layer ahead of the compute, in memory borrowed from the VRAM cache while the prefill runs (8192-token chunks). Staging every expert reads every NVMe-only one, ~15 s per chunk for DeepSeek whatever its length, which is what each agent step would pay for its tool result. DeepSeek (mode `nvme`, `engine/warm_prefill.py`) reads only what a chunk routes to: it runs on the experts already in VRAM or RAM, plus, per layer once its routing is known, its NVMe-only experts, read in parallel. A chunk of up to 1,024 rows (an agent step) reads every one its routing touches (75-165 of ~250 per layer) and drops none, so it computes what staging every expert does. A longer chunk reads the 64 with the most routing weight, guessing the next layer's with that layer's router while the current one computes, and drops the picks of the rest, rescaling each token's weights. DeepSeek also runs no separate forward for a prompt's last partial page: its recurrent state rewinds by cursor, so the checkpoint at the last page boundary, where a follow-up resumes, is stashed from the end of the prompt's last forward (`engine/last_page.py`). GLM keeps exllamav3's split there for tails of up to 63 tokens; a longer split would cost a whole extra staged forward. On a cached 8k prefix, 100-1,000 new tokens take 5.0-11.9 s instead of 15.1-20.3 s (71.5 s for one step of each size instead of 145.8 s), at top-1 99.76 % and mean KL 0.0005 against exact prefill over 3,344 forced positions. The only differences come from the tail sharing its forward with the tokens before it: exact staging with the same chunking gives bit-identical logits ([docs/vllm.md](docs/vllm.md#tail-checkpoints-deepseek-mode-nvme-2026-11-10)). Measured while every chunk was capped at 64 (short ones were not exact yet), teacher-forced against exact prefill (prompt and answer through it, 128 answer positions each): top-1 99.2 %, mean KL 0.023; on a DeepSeek Harness agent turn (decode exact after the prefill, 330 forced answer positions): top-1 99.7 %, mean KL 0.0021. GLM prefills exactly: its answers react more to approximate prefill (teacher-forced top-1 94.5 %, mean KL 0.042). Below ~86 new tokens prefill is exact anyway. Mode `nvme-exact` keeps exllamav3's prefill.
- **Serving agents.** The servers take `tools` and return OpenAI `tool_calls`, streamed as argument deltas (vLLM's parsers: DeepSeek's DSML calls, GLM's `<tool_call>` blocks), refuse a prompt that does not fit the cache with an immediate 400 `context_length_exceeded` (agent clients compact and retry; `bigmoe/serving.py`), report errors as JSON or in-band SSE events instead of dropping the stream, send an SSE comment on a stream that is silent for 15 s, free what a request that ran out of GPU memory held (and run it again with half-size prefill chunks if no token was streamed yet), and report `usage.prompt_tokens_details.cached_tokens` (prefix reuse, counted by vLLM's scheduler from what exllamav3 reused), with one log line per request. Requests may carry an OpenAI-style `priority` (lower first; vLLM's priority scheduling orders the queue). vLLM's grammars are enforced in exllamav3's sampler, so `response_format` (JSON schemas) works and tool calls follow vLLM's strict tool-call grammar. `bigmoe bench-agent` (and `bigmoe verify`) runs coding tasks through the OpenAI API with tool calling and checks the results (`bench/agent.py`). Prompt prefixes are reused from the GPU cache; with `-o disk_cache=<dir>` the pages it evicts (one long conversation evicts the others) go through exllamav3's pinned-RAM tier to disk, written only on eviction or when the server is idle, so a prefix also survives other conversations and restarts. A prefix comes back from disk through that RAM tier, all of it at once, so `cpu_cache_gb` defaults to the whole context (`models/prefix_cache.py`): 0.25 GiB for DeepSeek's 64k at 3.2 KB per token, 2.5 GiB for GLM's 128k at 17.6 KB; a 0.25 GiB tier gave GLM back nothing of a 17k-token prompt. Measured: a 17k-token prompt prefilled in 27.0 s (DeepSeek) / 27.9 s (GLM) took 5.7 s / 5.5 s after a restart.
- **Image input, on by default, free while unused.** Both checkpoints ship a vision tower (DeepSeek-V4.1: ~0.4 GiB of 6-bit weights, GLM-5.3: ~0.5 GiB) and both servers take OpenAI / Anthropic image blocks (`bigmoe/multimodal.py`, [docs/multimodal.md](docs/multimodal.md)). Nothing of the tower is resident: a request with a new image loads it from the checkpoint on NVMe (0.05-0.26 s), encodes its images (a new image costs 0.26-0.34 s in all) and unloads it before the job is queued, in VRAM that torch has free between forwards or that the expert cache lends for the encode (written back to the RAM tier, reallocated right after). Text-only serving keeps the same VRAM expert cache and RAM tier: measured against `multimodal=False` on the same box, prefill and decode are unchanged and `nvme-exact` answers are byte-identical to the baselines. Images are keyed by content: a resent image skips the tower (RAM, then an SSD cache that survives restarts) and keeps its token ids, so follow-up turns reuse the prefix through the image (DeepSeek, 11.5k-token conversation with a screenshot: 3-4 s per turn instead of 24 s). For DeepSeek-V4.1 the server turns off exllamav3's exact image chunking, which is meant for V4-Flash-Vision-Exp and cost V4.1 two extra prefill forwards per image and every checkpoint after it.
- **Speculative decoding.** DeepSeek-V4.1-Flash uses its own DSpark block drafter (up to 5 tokens per step, cut by its confidence head at 0.5) by default; GLM-5.3-Flash can draft from the context (`speculation="ngram"`).
- **One command.** `AutoBox` pins and builds the Python runtime (the same exllamav3 release for both models) and the CUDA extensions, downloads and verifies the weights, packs and verifies the store, and starts the server in a memory-capped systemd scope with its threads pinned (main, scheduler, CPU lane and NVMe readers on separate cores).

```python
from bigmoe import AutoBox

box = AutoBox.from_pretrained("deepseek-v4.1-flash")   # or "glm-5.3-flash"; first run downloads, packs and builds
box.start()                                            # OpenAI-compatible server on http://127.0.0.1:30000/v1
print(box.chat("Why do MoE models fit on small boxes?"))
print(box.chat("What does this chart show?", images=["chart.png"]))   # paths, bytes, PIL images or URLs
box.stop()
```

The offloading and scheduling engine started as 0xSero's [glm-flash-lite](https://github.com/sybil-solutions/glm-flash-lite) (MIT, `LICENSE.glm-flash-lite`); this package is Apache-2.0 (`LICENSE`).

## Serving engine

vLLM 0.33.0 is the serving engine: the OpenAI/Anthropic API server, chat renderer, tool and reasoning parsers, detokenizer, stop handling and multimodal input pipeline, and its scheduler, which decides admission, preemption, chunked-prefill sizes under its token budget, and which speculative tokens are verified (and accounts the accepted ones). An out-of-tree plugin (`bigmoe.vllm`) gives that scheduler a KV-cache manager over exllamav3's page table (capacity and prefix-cache hits from the GPU, RAM and disk tiers) and a worker that executes each scheduled step on exllamav3: the prefill chunks, one batched sample / verify forward, and the drafter's next drafts. The EXL3 kernels, nv2 expert tiers, prefix caches, DSpark / n-gram drafters and lazy vision towers are the measured ones; exllamav3's own scheduling loop is not used. vLLM 0.33 cannot execute either model's attention on sm86 and has no EXL3 quantization, so the forwards stay exllamav3's. Architecture, compatibility notes and verification results: [docs/vllm.md](docs/vllm.md).

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

`bigmoe bench` prompts (Python source files and a question about them), greedy, 128k context, FP16 KV cache, exllamav3 1.6.6 (`bigmoe verify`, 2026-11-06). KL and top-1: mean per-token KL divergence from exact decode (`nvme-exact`) and the share of positions with the same most likely token, on the same forced tokens, six chat prompts, 1,152 positions.

| mode | prompt tokens | prefill | decode | KL | top-1 |
|---|---|---|---|---|---|
| `nvme` | 1,024 | 114 tok/s | 11.1 tok/s | 0.0096 | 97.3 % |
| `nvme` | 8,192 | 789 tok/s | 11.3 tok/s | 0.0096 | 97.3 % |
| `nvme` | 32,768 | 846 tok/s | 11.5 tok/s | 0.0096 | 97.3 % |
| `nvme` | 120,000 | 817 tok/s | 11.3 tok/s | 0.0096 | 97.3 % |
| `nvme-exact` | 1,024 | 59 tok/s | 7.9 tok/s | 0 (reference) | 100 % (reference) |
| `nvme-exact` | 8,192 | 418 tok/s | 8.0 tok/s | 0 (reference) | 100 % (reference) |

exllamav3's fused hyper-connection sites (`EXL3_HC_FOLD=1`) replace the custom decode kernels the 1.5.1 runtime needed. GLM prefills exactly. Warm prefill was measured for it (an option until 2026-11-10): after 4k / 8k / 16k-token prompts, forced decode agreed with exact decode on 96.2 % of 288 positions (mean KL 0.018) with it, 99.7 % (0.002) without. `bigmoe bench`, prefill on a cached 8,192-token prefix, time to first token by new tokens, mode `nvme` (800 new tokens leave a 31-token tail, which keeps its own forward):

| new tokens | 8 | 80 | 120 | 400 | 800 | 2,000 | 5,000 | 10,000 |
|---|---|---|---|---|---|---|---|---|
| default | 1.4 s | 8.8 s | 8.7 s | 8.9 s | 12.9 s | 9.2 s | 9.7 s | 22.0 s |
| with warm prefill (removed for GLM) | 1.2 s | 3.7 s | 5.1 s | 5.6 s | 10.0 s | 6.9 s | 9.3 s | 20.8 s |

## DeepSeek-V4.1-Flash

`bigmoe bench` prompts, greedy, 128k context, FP16 KV cache, DSpark speculative decoding (`bigmoe verify`, 2026-11-06). KL and top-1: mean per-token KL divergence from exact single-token decode (`nvme-exact`, no speculation) and the share of positions with the same most likely token, on the same forced tokens, six chat prompts, 1,110 positions; with DSpark even `nvme-exact` differs slightly, because the verify forwards round in other GEMM shapes. This is the decode path on short prompts; an agent turn that also prefills new tokens through warm prefill agrees less (98.3-98.8 % top-1 on the same test for the old engine and vLLM, 99.4 % with exact prefill at 76 % more time per turn, measured before short chunks were exact; [docs/vllm.md](docs/vllm.md#agent-turn-quality-deepseek-mode-nvme-2026-11-09)).

| mode | prompt tokens | prefill | decode | KL | top-1 |
|---|---|---|---|---|---|
| `nvme` | 1,024 | 174 tok/s | 17.5 tok/s | 0.0011 | 99.6 % |
| `nvme` | 8,192 | 680 tok/s | 20.2 tok/s | 0.0011 | 99.6 % |
| `nvme` | 32,768 | 725 tok/s | 19.0 tok/s | 0.0011 | 99.6 % |
| `nvme-exact` | 1,024 | 33 tok/s | 10.6 tok/s | 0.0007 | 99.7 % |
| `nvme-exact` | 8,192 | 248 tok/s | 11.8 tok/s | 0.0007 | 99.7 % |

A 120,000-token prompt prefills at 657 tok/s (183 s) and decodes at 18.5 tok/s. Larger chunks prefill long prompts faster (`-o chunk_size=10240`: 815 tok/s) but leave too little VRAM at 128k: a ~125k-token re-prefill peaks 300 MiB below the card's limit (8192: 900 MiB), and a long agent session ran out of GPU memory with them; `/stats` reports GPU memory (`gpu`). Mode `nvme` prefills with warm prefill (teacher-forced against exact prefill: top-1 99.2 %, mean KL 0.023); `nvme-exact` answers are identical to the runs before it. `bigmoe bench`, prefill on a cached 8,192-token prefix, time to first token by new tokens, mode `nvme` with exact prefill (before warm prefill and the last-page split: 400, 2,000 and 5,000 new tokens took two forwards) and now:

| new tokens | 8 | 80 | 120 | 400 | 800 | 2,000 | 5,000 | 10,000 |
|---|---|---|---|---|---|---|---|---|
| exact prefill | 0.7 s | 3.3 s | 14.8 s | 29.0 s | 17.2 s | 29.6 s | 29.8 s | 33.1 s |
| warm prefill | 1.2 s | 3.3 s | 3.3 s | 5.4 s | 8.2 s | 6.9 s | 8.9 s | 18.6 s |
| + next-layer read-ahead | 0.6 s | 4.1 s | 3.4 s | 5.5 s | 8.3 s | 6.3 s | 7.4 s | 16.5 s |

For chunks of at least 1,024 rows, warm prefill now guesses the next layer's cold experts with that layer's own router and reads them while the current layer computes, then reads only what the real routing still needs. The fetched experts and outputs are unchanged (1,200/1,200 teacher-forced positions bit-identical, KL 0); long cached-prefix steps are 10-20 % faster on this box. Measurements and the matched-prompt A/B are in [docs/vllm.md](docs/vllm.md#warm-prefill-read-ahead-2026-11-09).

## Agentic benchmark

`bigmoe verify` also runs `bench/agent.py` (or `bigmoe bench-agent` alone): three coding tasks in fresh git repositories, solved over the OpenAI chat API with `bash` / `read_file` / `write_file` function tools, every turn streamed, each result checked by a command independent of what the model says; then two of them with two agents at once. Mode `nvme`, 2026-11-09:

| model | tasks solved | turns | tool calls (malformed) | wall | prefix reuse after turn 1 | mean time to first token |
|---|---|---|---|---|---|---|
| DeepSeek-V4.1-Flash | 3 / 3 | 14 | 13 (0) | 165 s | 80 % | 5.7 s |
| DeepSeek-V4.1-Flash, two agents | 2 / 2 | 10 | 10 (0) | 84 s | 84 % | 14.1 s |
| GLM-5.3-Flash | 3 / 3 | 21 | 28 (0) | 376 s | 21 % | 9.3 s |
| GLM-5.3-Flash, two agents | 2 / 2 | 13 | 17 (0) | 217 s | 44 % | 11.4 s |

Both models have recurrent state, so a conversation resumes only from a recurrent checkpoint. DeepSeek checkpoints every 256 tokens while decoding (`rc_interval`, mode `nvme`), so the next turn resumes inside the previous answer: 80 % reuse instead of 59 %, and its resumed turns are closer to exact than a re-prefill through warm prefill (KL 0.0044 instead of 0.0142). For GLM that measured as a wash, so these short GLM conversations (under 3k tokens) still reuse only what their earlier prompt lengths happened to checkpoint, which is why GLM's reuse varies so much. Details, including concurrency: [docs/vllm.md](docs/vllm.md#agentic-benchmark-benchagentpy-mode-nvme).
