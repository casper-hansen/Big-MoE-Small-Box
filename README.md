# Big MoE, Small Box

Big MoE, Small Box takes GLM-5.3-Flash from its 328 GB release down to a 125 GB 3-bit quantization, and serves it from one 24 GB GPU, 64 GB of RAM and an NVMe drive with an expert offloading and scheduling system, at 7-10 tokens/s with up to 512k tokens of context. The routed experts are 117 GB of those 125 GB, and each token uses only 8 of the 288 experts in each MoE layer. So the experts live in three tiers: a cache in free VRAM, a pinned RAM tier, and a packed store on NVMe read with `O_DIRECT`. For every layer of every token, a scheduler decides where each picked expert runs: from VRAM, copied in from RAM or NVMe, or on the CPU with AVX2. Big MoE, Small Box was created and improved upon from 0xSero's original work, [glm-flash-lite](https://github.com/sybil-solutions/glm-flash-lite).

```python
from bigmoe import AutoBox

box = AutoBox.from_pretrained("glm-5.3-flash")   # first run: downloads, converts and builds what is missing
box.start()                                      # OpenAI-compatible server in the background
print(box.chat("Why do MoE models fit on small boxes?"))
box.stop()
```

## Install

```bash
cd ~/big-moe-small-box
uv venv && uv pip install -e .          # or: python -m venv .venv && .venv/bin/pip install -e .
```

The package needs only `huggingface_hub` and `uv`, not torch or CUDA. The server runs in its own process and its own pinned Python environment, which bigmoe builds the first time it is needed. Requirements: Linux, an NVIDIA driver, and a local filesystem with `O_DIRECT` (ext4, xfs) for the artifacts. The memory cap uses `systemd-run --user`. No root, no Docker.

## Usage

```python
from bigmoe import AutoBox

box = AutoBox("glm-5.3-flash", context=262144)   # resolves the model, checks options; downloads nothing yet
box.prepare()                                    # downloads / converts / builds what is missing; instant once done
box.start()                                      # reuses a server already running with this configuration

box.chat("Hello!")                                     # -> str
for piece in box.chat("Tell me a story", stream=True, thinking=True):
    print(piece, end="")
box.generate("def fibonacci(n):")                      # raw continuation, no chat template
box.client.chat_completion("Hi", thinking=True)        # full OpenAI response, incl. reasoning_content
box.url                                                # "http://127.0.0.1:30000/v1", for any OpenAI client
box.bench([1024, 32768])                               # long-prompt prefill / decode benchmark
box.status(), box.logs(), box.stop()

with AutoBox.from_pretrained("glm-5.3-flash", mode="nvme-exact") as box:   # stopped at exit if started here
    print(box.chat("Bit-exact decode, please"))
```

```bash
bigmoe start                                   # glm-5.3-flash, mode nvme, 128k context, port 30000
bigmoe start --context 524288 --cache-quant 8 --chunk-size 4096     # 512k context
bigmoe chat "What is an expert store?"         # no prompt: interactive
bigmoe status | logs -f | stop
bigmoe serve --mode nvme-exact                 # foreground
bigmoe prepare --dry-run                       # what would be downloaded / converted / built
bigmoe bench --tokens 1024 32768 120000
bigmoe verify                                  # ~15 min: speed + outputs against the reference results
bigmoe doctor | models | verify-store
```

`examples/` has the same as scripts: `quickstart.py`, `serve.py`, `prepare.py`, `chat.py`, `benchmark.py`.

## Benchmarks

One RTX 3090 24 GB, Ryzen 5 7600X (6 cores / 12 threads), 64 GB DDR5-5600, one 1 TB WD_BLACK SN850X (ext4), Ubuntu 26.04. Mode `nvme`, `memory_gb=52`, `context=131072`, FP16 KV cache, greedy, answers run to their natural end.

| prefill size | prefill speed | decode speed | concurrency | kv cache | gpu count |
|---|---|---|---|---|---|
| 8,192 | 358 tok/s | 9.20 tok/s | 1 | 131,072 | 1 |
| 8,192 | - | 9.48 tok/s aggregate (4.87 per stream) | 2 | 131,072 | 1 |
| 8,192 | - | 8.98 tok/s aggregate (2.51 per stream) | 4 | 131,072 | 1 |
| 32,768 | 613 tok/s | 8.59 tok/s | 1 | 131,072 | 1 |

That table is glm-flash-lite's `bench/sweep.py`: random-token prompts, 3 reps; chat answers of ~1.6k-2.7k tokens. The next one is `bigmoe bench`: Python source files plus a question about them, ~200-250-token answers, thinking off.

| prompt tokens | context | time to first token | prefill | decode |
|---|---|---|---|---|
| 1,024 | 131,072 | 19.6 s | 52 tok/s | 7.97 tok/s |
| 8,192 | 131,072 | 21.7 s | 377 tok/s | 7.89 tok/s |
| 32,768 | 131,072 | 53.4 s | 614 tok/s | 8.06 tok/s |
| 120,000 | 131,072 | 2.8 min | 711 tok/s | 8.07 tok/s |
| 250,000 | 262,144 | 5.7 min | 725 tok/s | 7.76 tok/s |
| 510,000 | 524,288 (8-bit KV, chunk 4096) | 21.9 min | 387 tok/s | 7.40 tok/s |

- **Long chat answers** (1.6k-3k tokens): 9.9-10.1 tok/s. **Mode `nvme-exact`:** 8.4-8.7 tok/s on chat, 6.5 tok/s after the long prompts.
- **Quality:** top-1 agreement 1.0000 and mean KL 0.000000 against exllamav3's reference logits (2,154 positions). Every long-prompt answer names the first three files of the prompt correctly.
- **Short prompts** wait ~19 s for the first token: the recurrent-state tail forward re-streams the experts (an upstream issue).
- **Memory:** the server holds ~47 GB of RAM (cap 52 GiB) and the whole GPU.
- **Tuning:** the scheduler's CPU cost per expert is 0.40 ms instead of glm-flash-lite's 0.165 ms (which is calibrated for 22 EPYC cores): +6 %. Running the CPU lane on all 10 hardware threads (9.13 tok/s, controls 9.45-9.66) or with `-march=znver4` (9.32) did not help.

These numbers were measured before the engine became this package. `bigmoe verify` reproduces them through the package: in mode `nvme-exact` the answers are byte-identical and prefill / decode speeds are within 0.2 %; in mode `nvme`, prefill is within 1 % and decode 0.3-2.2 % lower, with one 4.2 % outlier at 120k that a second run did not repeat (1.1 %). Mode `nvme` answers drift apart from run to run after a few hundred characters: which experts the CPU lane computes in fp32 depends on timing.

## How it works

`AutoBox(model)` resolves the model to a backend: an alias, a Hugging Face repo id, or a local checkpoint directory, pinned to a commit. `prepare()` (and `from_pretrained`, `start`) runs every step that is not done yet. Each step remembers its result, so the check is instant once everything is in place:

| step | what | first run |
|---|---|---|
| `runtime` | Python 3.12 environment with the exact pins the engine was measured with (torch 2.13.0+cu130, Triton 3.7.1, transformers 5.12.1); exllamav3 1.5.1 built from source in a twin environment with the CUDA 13.0 compiler from pip | ~4 min |
| `kernels` | the engine's 4 CUDA extensions (expert cache, fused hyper-connections, NVMe tier, scheduler + CPU lane), cached by a hash of their sources | ~2 min |
| `weights` | `turboderp/GLM-5.3-Flash-exl3` @ `332ab457` (3.05 bpw), 125 GB, every file checked against the Hub's sha256 | download |
| `store` | the 12,384 routed experts packed into one 117.3 GB file of 9.47 MB records; 384 records byte-compared with the checkpoint, every record sha256-checked with `O_DIRECT` reads | ~2-3 min |
| `slim` | the expert bytes punched out of the checkpoint: 125 GB apparent, 8.3 GB on disk | seconds |

`start()` checks the GPU, CPU flags, free RAM and `O_DIRECT`, then launches the server in a systemd scope with `MemoryMax=<memory_gb>G` and no swap; the RAM tier sizes itself from that cap. Threads are pinned to a layout derived from the CPU topology: on the 7600X, main thread 0, scheduler 6, CPU lane 1-5, NVMe readers 7-11. A server already running with the same configuration is reused; one with a different configuration is restarted. Everything lives in `~/.cache/bigmoe` (`BIGMOE_HOME`), about 132 GB: store 117.3 GB, checkpoint 8.3 GB, runtime 5.4 GB, build twin 0.3 GB. `bigmoe adopt --model-dir D --store-dir S` takes over a checkpoint or store that is already on the box.

## Configuration

| `AutoBox(...)` | default | |
|---|---|---|
| `model` | | alias, Hugging Face repo id, or local checkpoint directory |
| `revision` | `3.05bpw` | only the validated quant; the repo's 2.05 / 4.05 bpw and self-calibrated branches are refused |
| `context` | 131072 | tokens; multiple of 256 |
| `memory_gb` | 56 | cgroup cap of the server (60 GiB+ hosts: 56, the measured best with 4.8 GiB of host headroom at peak; smaller hosts: MemTotal - 8) |
| `host`, `port` | 127.0.0.1, 30000 | no auth: keep it on localhost |
| `home` | `~/.cache/bigmoe` | must support `O_DIRECT` |
| `runtime` | `"managed"` | `"current"`, or the path of an interpreter that already has the packages |
| `env`, `extra_args` | | extra server environment / arguments |
| `mode` | `"nvme"` | `"nvme-exact"`: every expert on the GPU, bit-exact with stock exllamav3, 15-20 % slower |
| `cache_quant` | `None` | KV-cache bits; `8` for 512k context |
| `chunk_size` | 8192 | prefill chunk; `4096` for 512k context |
| `vram_expert_slots` | `None` | all free VRAM (1,408 expert slots on a 24 GB card); an integer caps the cache, e.g. 1376 leaves room for a desktop on the GPU |
| `speculation`, `ngram_min`, `draft_tokens` | `"none"`, 2, 5 | `speculation="ngram"` drafts tokens from the context and verifies them with the target model; off by default, see below |
| `planner_cost`, `cpus`, `max_batch_size`, `slim`, `verify_weights` | | see `bigmoe models` |

## Optional: n-gram speculation

`bigmoe start -o speculation=ngram` drafts tokens by matching the last `ngram_min` tokens against the context (prompt lookup, no draft model) and verifies `draft_tokens` drafts per step with the target model. It is off by default, and we have not found a workload where it is faster on this box. Measured with single runs of 384 new tokens:

| prompt | mode | `none` | `ngram` |
|---|---|---|---|
| copy a 178-line source file | `nvme` | 7.31 tok/s | 7.24 tok/s |
| copy a 178-line source file | `nvme-exact` | 5.78 tok/s | 2.54 tok/s |
| MoE explanation (prose) | `nvme-exact` | 7.37 tok/s | 4.58 tok/s |

Verification touches more cold experts per accepted token than single-token steps (77 vs 54 NVMe picks per token on the copy prompt in `nvme` mode), so the extra accepted tokens do not translate into speed here.

Outputs are not bit-identical to `none`. On the copy prompt the output matched byte for byte. On the prose prompt it matched for 370 characters and then differed at a near-tie token: verification runs the same weights in different GEMM shapes, and fp16 rounding can flip a near-tie. The `none` path reproduces its own output exactly across runs.

## Context length

The 11 sparse-attention (MLA/DSA) layers keep ~17 KB of KV cache per token in VRAM; the 34 linear-attention (KDA) layers have constant-size state. Every extra 128k tokens costs ~220 VRAM expert slots, and prefill needs ~590 slots free for its staging buffers. Contexts are capped at 512k on purpose; 256k keeps the FP16 KV cache and costs ~3-4 % of decode speed.

| `context` | KV cache | VRAM expert slots | result |
|---|---|---|---|
| 131,072 | FP16 | 1,376 (1,408 uncapped) | default |
| 262,144 | FP16 | 1,184 | 250k prompt: 725 / 7.76 tok/s |
| 524,288 | 8-bit, chunk 4096 | 960 | 510k prompt: 387 / 7.40 tok/s |
| 524,288 | FP16 | 672 | starts; not run with a full prompt |
| 1,048,576 | 4-bit, chunk 2048 | 608 | a 1.04M prompt ran out of VRAM after ~530k tokens (twice); not supported |

## Decode ceiling at 256k context

**About 15-20 tok/s in theory**, twice today's 7.8-9.6 tok/s at `context=262144`. That needs a perfect scheduler; software tuning alone is worth maybe 20-30 % (10-12 tok/s). Each token uses 336 experts of 9.4 MB: on chat, 165 are already in VRAM, 135 come from RAM and 35 from NVMe; after a 250k-token document, 128 / 160 / 48.

| resource | limit | used today | fastest it allows |
|---|---|---|---|
| NVMe | 6.7-6.9 GB/s | 4.6-5.0 GB/s; 55 reads per token, ~17 of them unused prefetch | 35-48 reads: **15-20 tok/s** |
| PCIe 4.0 x16 + CPU lane | 26.8 / 26.4 GB/s; ~0.6 ms per expert on 5 cores | 8.9 / 8.1 GB/s; lane busy 62 of 104 ms | 170-208 experts outside VRAM: 18-22 tok/s |
| DRAM (2 channels) | ~60 GB/s | - | 2.9-3.6 GB per token: 16-20 tok/s |

The three limits are within ~15 % of each other, so fixing one moves the bottleneck. Today's speed is about half of what they allow, because each of the 42 MoE layers waits for its slowest lane. A second NVMe in RAID0 or 128 GB of RAM would lift the ceiling to ~20 tok/s; more CPU cores help only together with one of those (estimates, not measured).

## Adding a model

A backend is a `ModelBackend` subclass, registered with `register_backend` or through the `bigmoe.backends` entry-point group. It says which repos and aliases it accepts, its options, its prepare steps, the Python runtime its server needs, and how to launch it:

```python
from bigmoe import AutoBox, ModelBackend, register_backend
from bigmoe.launcher import LaunchSpec

@register_backend
class MyMoE(ModelBackend):
    name = "mymoe"
    aliases = {"my-moe": ("me/my-moe-exl3", "main")}

    def steps(self, box):          # [Step(name, title, done, run)]: downloads, stores, builds
        return []

    def launch_spec(self, box):
        return LaunchSpec(argv=[str(box.runtime.python), "-m", "my_server", "--port", str(box.port)], env={},
                          host=box.host, port=box.port, memory_gb=box.memory_gb)

box = AutoBox.from_pretrained("my-moe")
```

The shared components are model-agnostic: `hub` (pinned, verified downloads), `store` (EXL3 expert-store packing, verification, slimming), `runtime` (pinned environments, source builds), `kernels` (cached CUDA extensions), `system` and `cpus` (hardware probes, thread layout), `launcher` (memory-capped server processes), `client`, and `bench`. The offload engine in `models/glm53/engine` still assumes GLM-5.3-Flash's expert shapes; a second model needs those assumptions lifted.

## Credits

- [0xSero](https://github.com/sybil-solutions/glm-flash-lite) came up with the idea and built glm-flash-lite, the offloading and scheduling engine this package runs (MIT, `LICENSE.glm-flash-lite`).
- [turboderp](https://github.com/turboderp-org/exllamav3): exllamav3 and the 3.05 bpw [EXL3 quantization](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3) of GLM-5.3-Flash.
- [Z.ai](https://huggingface.co/zai-org/GLM-5.3-Flash): GLM-5.3-Flash, 321B parameters, released in FP8 (328 GB, MIT).

License: Apache-2.0 (`LICENSE`). The engine code from glm-flash-lite keeps its MIT license (`LICENSE.glm-flash-lite`); the weights carry their own.

Tests: `uv pip install -e ".[dev]" && .venv/bin/python -m pytest` (29 tests, ~6 s, no GPU).
