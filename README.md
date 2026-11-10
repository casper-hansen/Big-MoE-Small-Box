# Big MoE, Small Box

Serve Mixture-of-Experts models that are far larger than the GPU from one 24 GB card, 64 GB of RAM and an NVMe drive:

| model | checkpoint | size |
|---|---|---|
| GLM-5.3-Flash | [turboderp/GLM-5.3-Flash-exl3](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3), EXL3 3.05 bpw | 125 GB |
| DeepSeek-V4.1-Flash | [turboderp/DeepSeek-V4.1-Flash-1017-exl3](https://huggingface.co/turboderp/DeepSeek-V4.1-Flash-1017-exl3), EXL3 2.25 bpw | 361 GB, 203 GB of it Engram tables |

A token uses only a few routed experts per MoE layer (8 of 288 for GLM, 6 of 384 for DeepSeek), so the routed experts,
most of the weights, live in three tiers: a cache in all free VRAM, a pinned RAM tier, and a packed store on NVMe read
with `O_DIRECT`. A per-layer planner serves each picked expert from wherever it is, and can compute RAM-resident
experts on the CPU instead of moving them over PCIe. [vLLM](https://github.com/vllm-project/vllm) 0.33.0 is the
server and scheduler (OpenAI and Anthropic APIs, tool calling, reasoning, structured outputs, images);
[exllamav3](https://github.com/turboderp-org/exllamav3) 1.6.6 runs the forwards. How it works:
[docs/architecture.md](docs/architecture.md).

## Quick start

```sh
pip install -e .
bigmoe doctor                          # what this box has
bigmoe start deepseek-v4.1-flash       # or glm-5.3-flash; OpenAI-compatible server on http://127.0.0.1:30000/v1
bigmoe chat "Why do MoE models fit on small boxes?"
bigmoe stop
```

The first start builds a pinned Python runtime and the CUDA extensions, downloads and verifies the checkpoint, packs
and verifies the expert store, and punches the expert bytes out of the checkpoint (the store has them). Later starts
take about a minute. The server runs in a memory-capped systemd user scope with its threads pinned.

From Python:

```python
from bigmoe import AutoBox

box = AutoBox.from_pretrained("deepseek-v4.1-flash")
box.start()
print(box.chat("What does this chart show?", images=["chart.png"]))   # paths, bytes, PIL images or URLs
box.stop()
```

Other commands: `bigmoe prepare` (build and download only), `serve` (foreground), `status`, `logs`, `bench`,
`bench-agent`, `verify` (speed, exact answers and the agentic benchmark against the stored baselines), `models` (every
backend option and its default).

## Options

Pass them as keyword arguments to `AutoBox` or as `-o key=value` on the command line.

| option | default | |
|---|---|---|
| `mode` | `nvme` | `nvme`: the CPU lane computes some experts, near-exact. `nvme-exact`: every expert on the GPU, exllamav3's own arithmetic |
| `context` | 131072 | context length in tokens (an `AutoBox` argument, `--context`) |
| `memory_gb` | 56 on a 64 GB box | memory cap of the server; the RAM tier is sized from it (`--memory-gb`) |
| `speculation` | DeepSeek `dspark`, GLM `none` | DeepSeek's own block drafter, or `ngram` drafts from the context for GLM |
| `x16_layers` | 0 | bit-exact CPU lane in the first N MoE layers ([docs/x16.md](docs/x16.md)) |
| `disk_cache` | off | directory for a prefix cache that survives restarts |
| `cpu_cache_gb` | 0, or the whole context with `disk_cache` | pinned-RAM prefix-cache tier |
| `multimodal`, `image_tokens` | on, model's budget | image input |
| `cache_quant`, `chunk_size` | FP16, 8192 | K/V cache bits, prefill chunk |

## Results

RTX 3090, Ryzen 5 7600X, 64 GB DDR5-5600, WD SN850X. `bigmoe verify`: greedy, 128k context, FP16 K/V cache; the
numbers are the stored baselines in `src/bigmoe/models/*/baselines`.

**GLM-5.3-Flash** (2026-11-06)

| mode | prompt tokens | prefill | decode |
|---|---|---|---|
| `nvme` | 1,024 | 114 tok/s | 11.1 tok/s |
| `nvme` | 8,192 | 789 tok/s | 11.3 tok/s |
| `nvme` | 32,768 | 846 tok/s | 11.5 tok/s |
| `nvme` | 120,000 | 817 tok/s | 11.3 tok/s |
| `nvme-exact` | 1,024 | 59 tok/s | 7.9 tok/s |
| `nvme-exact` | 8,192 | 418 tok/s | 8.0 tok/s |

**DeepSeek-V4.1-Flash**, DSpark speculative decoding (2026-11-10)

| mode | prompt tokens | prefill | decode |
|---|---|---|---|
| `nvme` | 1,024 | 79 tok/s | 16.4 tok/s |
| `nvme` | 8,192 | 767 tok/s | 21.3 tok/s |
| `nvme` | 32,768 | 820 tok/s | 20.3 tok/s |
| `nvme-exact` | 1,024 | 33 tok/s | 10.6 tok/s |
| `nvme-exact` | 8,192 | 250 tok/s | 12.1 tok/s |

Short prompts prefill slowly: a prefill forward reads its NVMe-only experts at the SSD's speed however few tokens it
has. What an agent waits for is the time to the first token after a cached prefix, here
8,192 tokens (`bigmoe bench`):

| new tokens | 8 | 80 | 400 | 800 | 2,000 | 10,000 |
|---|---|---|---|---|---|---|
| GLM-5.3-Flash | 1.4 s | 8.8 s | 8.9 s | 12.9 s | 9.2 s | 22.0 s |
| DeepSeek-V4.1-Flash | 1.2 s | 3.4 s | 8.0 s | 9.7 s | 6.5 s | 16.7 s |

**Agreement with exact decode.** On the same forced tokens (six chat prompts), mode `nvme` picks the same most likely
token as `nvme-exact` 97.3 % of the time for GLM (mean KL 0.0096) and 99.6 % for DeepSeek (mean KL 0.0011). With
`x16_layers=21` GLM reaches 99.2 % at 13 % lower decode speed. `nvme-exact` answers are byte-identical to the baselines.

**Agentic benchmark** (`bigmoe bench-agent`, mode `nvme`): three coding tasks in fresh git repositories, solved over
the OpenAI API with `bash`, `read_file` and `write_file` tools, each checked by a command independent of what the
model says; then two of them with two agents at once.

| | tasks solved | turns | tool calls (malformed) | wall | prefix reuse after turn 1 | mean time to first token |
|---|---|---|---|---|---|---|
| DeepSeek-V4.1-Flash | 3 / 3 | 14 | 13 (0) | 153 s | 89 % | 4.3 s |
| DeepSeek-V4.1-Flash, two agents | 2 / 2 | 10 | 10 (0) | 79 s | 88 % | 12.9 s |
| GLM-5.3-Flash | 3 / 3 | 21 | 28 (0) | 376 s | 21 % | 9.3 s |
| GLM-5.3-Flash, two agents | 2 / 2 | 13 | 17 (0) | 217 s | 44 % | 11.4 s |

DeepSeek resumes each agent request where the previous one stopped; GLM's recurrent state cannot be moved back, so it
resumes only at a stored checkpoint, and short conversations reuse little.

## Credits

The offloading and scheduling engine started as 0xSero's
[glm-flash-lite](https://github.com/sybil-solutions/glm-flash-lite) (MIT, `LICENSE.glm-flash-lite`). This package is
Apache-2.0 (`LICENSE`).
