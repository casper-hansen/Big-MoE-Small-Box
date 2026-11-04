# Big MoE, Small Box

Big MoE, Small Box serves Mixture-of-Experts models that are far larger than the GPU from one 24 GB card, 64 GB of RAM and an NVMe drive: [GLM-5.3-Flash](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3) (321B parameters, EXL3 3.05 bpw, 125 GB) and [DeepSeek-V4.1-Flash](https://huggingface.co/turboderp/DeepSeek-V4.1-Flash-1017-exl3) (EXL3 2.25 bpw, 361 GB including 203 GB of Engram tables), each behind an OpenAI-compatible server on [exllamav3](https://github.com/turboderp-org/exllamav3). A token only uses a few routed experts per MoE layer (8 of 288 for GLM, 6 of 384 for DeepSeek), so the routed experts, which are most of the weights, live in three tiers and every layer is served from wherever its experts are:

- **Expert store.** The routed experts are packed into one file of fixed-size records per layer and read with `O_DIRECT`; the checkpoint keeps everything else and its expert bytes are punched out after the store is verified. The loader never reads the experts from the checkpoint: every expert tensor starts as a view of one VRAM bounce slot.
- **Three tiers, exllamav3's own math.** A CLOCK cache of uniform slots fills all free VRAM (layers with different expert sizes share it, each in its own compact layout); a pinned, GPU-mapped RAM tier holds the experts that are not in VRAM; the rest stays on NVMe. exllamav3's kernels read every expert through its live pointer tables, which the engine points at the expert's current home, so the bytes and the arithmetic are exllamav3's own.
- **A scheduler per layer.** The GPU publishes each layer's picks to mapped memory without a host sync; a host planner sends every pick to the cheapest lane: a VRAM hit, a copy in from RAM, a zero-copy read, an NVMe read (demand reads ahead of prefetches, prefetches guessed one layer ahead from the next router), or the CPU lane. Admitted experts evict VRAM victims back into RAM.
- **CPU lane.** RAM-resident and freshly landed experts can be computed on the CPU (AVX-512 VBMI trellis decoding, int16 dot products, results combined on the GPU), which takes PCIe and NVMe pressure off the GPU. This is mode `nvme`, the default, and it is near-exact: on the same forced tokens it picks the same top-1 token as exact decode 99.3 % of the time for DeepSeek (mean KL 0.0013) and 97.2 % for GLM (mean KL 0.012). Mode `nvme-exact` keeps every expert on the GPU path, so decode is exllamav3's own computation on the same bytes.
- **Prefill** stages each layer's experts into VRAM buffers one layer ahead of the compute, in memory borrowed from the VRAM cache while the prefill runs (8192-token chunks, NVMe-bound).
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

`bigmoe bench` prompts (Python source files and a question about them), greedy, 128k context, FP16 KV cache.

| mode | prompt tokens | prefill | decode |
|---|---|---|---|
| `nvme` | 1,024 | 113 tok/s | 11.4 tok/s |
| `nvme` | 8,192 | 776 tok/s | 11.4 tok/s |
| `nvme` | 32,768 | 833 tok/s | 11.8 tok/s |
| `nvme` | 120,000 | 826 tok/s | 11.4 tok/s |
| `nvme-exact` | 1,024 | 60 tok/s | 7.8 tok/s |
| `nvme-exact` | 8,192 | 427 tok/s | 7.9 tok/s |

## DeepSeek-V4.1-Flash

`bigmoe bench` prompts, greedy, 64k context, FP16 KV cache, DSpark speculative decoding.

| mode | prompt tokens | prefill | decode |
|---|---|---|---|
| `nvme` | 1,024 | 33 tok/s | 15.6 tok/s |
| `nvme` | 8,192 | 250 tok/s | 18.1 tok/s |
| `nvme` | 32,768 | 405 tok/s | 18.5 tok/s |
| `nvme-exact` | 1,024 | 35 tok/s | 11.0 tok/s |
| `nvme-exact` | 8,192 | 268 tok/s | 12.6 tok/s |
