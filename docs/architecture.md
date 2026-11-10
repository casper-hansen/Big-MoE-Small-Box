# Architecture

## Processes

vLLM 0.33.0 serves; exllamav3 1.6.6 computes. vLLM cannot run either model on an RTX 3090: DeepSeek-V4.1's sparse-MLA
attention and GLM-5.3's sparse indexer need sm90 or newer, and vLLM has no EXL3 quantization or expert offloading. So
the forwards, the K/V cache and the expert tiers are exllamav3's and this package's, and every scheduling decision is
vLLM's.

| process | code | does |
|---|---|---|
| API server | vLLM, `bigmoe/vllm/routes.py`, `mm.py` | OpenAI / Anthropic HTTP APIs, chat templates, tool and reasoning parsers (DeepSeek: `deepseek_v41`, GLM: `glm47`), image preprocessing, detokenization, SSE |
| EngineCore | vLLM's `Scheduler` as `BoxScheduler` (`scheduler.py`) | admission, preemption, priority, chunked prefill under the token budget, speculative-token scheduling and acceptance, stop checks |
| EngineCore | `BoxKVCacheManager` (`scheduler.py`) | vLLM's K/V-manager interface over exllamav3's page table |
| EngineCore | `BoxWorker` + `ExlEngine` (`worker.py`, `engine.py`) | runs exactly the step vLLM scheduled on exllamav3 |
| EngineCore | `bigmoe/engine` | the expert tiers, the planner, the CPU lane, staged prefill |

exllamav3's own scheduling loop (`Generator.iterate`) is never called. The vLLM model classes are attention-free stubs
(`vllm/models.py`), so vLLM allocates no K/V of its own and its block-hash prefix cache is off.

## Expert tiers

- **Store.** `bigmoe prepare` packs the routed experts into one file of fixed-size records per layer (`store/`; v1 for
  GLM, v2 with per-layer record sizes for DeepSeek, whose quant mixes 2.5 and 2.0 bpw layers), verifies every record
  against the checkpoint and by hash, then punches the expert bytes out of the checkpoint. The loader never reads
  experts from the checkpoint: every expert tensor starts as a view of one VRAM bounce slot (`engine/tiers.py`).
- **VRAM.** One CLOCK cache of uniform slots over all routed layers fills the free VRAM. A slot holds one expert in its
  layer's compact layout (gate | up | down), so layers with different expert sizes share it.
- **RAM.** A pinned, GPU-mapped host arena, exclusive of VRAM (an expert admitted to VRAM leaves RAM; a VRAM victim is
  written back), evicting by LRU. DeepSeek in mode `nvme` uses a segmented LRU: an expert that arrives from NVMe is on
  probation until it is used again, and 80 % of the slots are kept for experts that were reused.
- **NVMe.** `O_DIRECT` reads by a reader pool, demand reads ahead of prefetches ahead of prefill reads.

exllamav3's MoE kernels read each expert through pointer tables; the engine points each row at the expert's current
home. The bytes and the arithmetic are exllamav3's.

## Decode

Per MoE layer call (`engine/nv2.py`, `kernels/nv2/`), without a host sync: the GPU publishes the layer's picks to mapped
memory; the host controller plans every pick into a lane (VRAM hit; copy in from RAM; zero-copy read from RAM; NVMe
read then copy or zero-copy; CPU lane) by a cost model, starts the reads and replies; the GPU admits and copies, runs
exllamav3's kernel on the GPU picks, and adds the CPU lane's partial sum. The next layer's picks are guessed with its
router on the current input and prefetched.

The CPU lane computes RAM-resident and freshly read experts on the host (EXL3 trellis decode with AVX-512 VBMI, or
AVX2 for GLM's 3-bit experts; int16 dot products), which takes PCIe and NVMe load off the GPU. That is mode `nvme`. Its
results differ from the GPU's in the last bits, so it is near-exact; mode `nvme-exact` keeps every pick on the GPU.
`x16_layers` makes the CPU lane bit-exact in the leading layers ([x16.md](x16.md)).

## Prefill

A forward with at least `EC_STAGE_MIN` picks per layer is staged: each layer's experts are copied into one of two VRAM
buffers one layer ahead of the compute, in memory the VRAM cache lends for the duration of the prefill. Staging every
expert reads every NVMe-only expert once per forward, however short the forward.

DeepSeek in mode `nvme` stages only what the forward uses (`engine/warm_prefill.py`): the experts already in VRAM or
RAM, plus, once a layer's routing is known, its NVMe-only experts:

- A forward of at most 1,024 rows reads every NVMe-only expert its routing touches and drops nothing: the same experts
  on the same bytes as staging all of them, so the logits are identical.
- A longer forward reads the 64 with the most routing weight per layer, guessing the next layer's with that layer's
  router while the current one computes, and drops the other cold picks, rescaling each token's weights.

Mode `nvme-exact` and GLM stage every expert.

exllamav3 runs a prompt's last partial page as a forward of its own, so that a follow-up can resume at the page
boundary from a recurrent checkpoint (`engine/last_page.py`). GLM keeps that split only for tails of up to 63 tokens,
which take the per-pick path; a longer split would cost a whole extra staged forward. DeepSeek's recurrent state can
be moved back by its cursor, so it runs no split and stashes the page-boundary checkpoint from the end of the prompt's
last forward.

## Scheduling

- **Admission.** When vLLM looks a waiting request up in the prefix cache, `BoxKVCacheManager` reserves it in the
  engine: an exllamav3 job with its pages allocated (cached pages revived from GPU, RAM or disk) and its cached prefix
  skipped. The tokens skipped are vLLM's prefix hit, so `usage.prompt_tokens_details.cached_tokens` is vLLM's count of
  exllamav3's reuse. Without free pages the request waits, or vLLM preempts one.
- **Concurrency.** exllamav3 runs as many sequences as its cache has recurrent-state slots: DeepSeek 1, GLM 4 in mode
  `nvme` without speculation, otherwise 1. vLLM gets that as `max_num_seqs`; other requests wait in its queue, ordered
  by `priority`.
- **Chunked prefill.** A step holds one prefill chunk beside every running decode. `BoxScheduler` shortens a chunk to
  where exllamav3 can end a forward (`engine.plan_chunk`: a page boundary, the maximum chunk, the last-page split), so
  the engine computes exactly what was scheduled. vLLM turns chunked prefill off for a model without K/V cache groups;
  `BoxScheduler` turns it back on.
- **Speculation.** The drafter (DeepSeek's DSpark block drafter, GLM's n-gram policy) proposes drafts after each
  verify forward; vLLM schedules them as spec tokens in the next step and rolls back the rejected ones. When the engine
  drafts, a prompt's last chunk stops one token short, so the first decode step verifies the first drafts as
  exllamav3's own loop would.
- **Grammars.** vLLM's structured outputs and strict tool-call grammars are enforced in exllamav3's sampler: each
  step's bitmask rows from vLLM are applied by an exllamav3 filter (`engine.grammar_filter`).
- **Reconciliation.** After each step `BoxScheduler` compares vLLM's computed-token count with the engine's and
  corrects the rare difference (an out-of-memory retry, a requeue waiting for pages), counted in `/stats` under
  `scheduler.corrections`. A request the engine ends on its own (cache full, a failed job) is finished there.

## Prefix reuse

K/V pages are shared by hash: the GPU page cache, then optionally `cpu_cache_gb` of pinned RAM and a `disk_cache`
directory that survives restarts. A prefix returns from disk through the RAM tier, so with `disk_cache` that tier is
sized for the whole context by default (`models/prefix_cache.py`).

Both models have recurrent state, so a prompt resumes only where a recurrent checkpoint was stored: at a chunk end, at a
prompt's last page boundary (GLM: when the tail after it is short, see Prefill), and while decoding every 2,048 tokens
(DeepSeek in mode `nvme`: every 256). DeepSeek in mode
`nvme` also keeps turn checkpoints (`engine/turn_resume.py`): the state where each request stopped and where its prompt
ended, so the next request of a conversation, which extends the last one, prefills only what is new. The chat template
overlay (`vllm/serve.py`) renders a tool-call turn to exactly the tokens the model generated, so the history matches.
GLM's recurrent state cannot be moved back, so it has no turn checkpoints.

## Image input

Both checkpoints ship a vision tower; neither is resident (`bigmoe/multimodal.py`). vLLM loads and preprocesses the
image with exllamav3's own preprocessing; when the request is admitted the engine loads the tower from the checkpoint,
encodes the request's new images and unloads it, in VRAM that torch has free or that the expert cache lends for the
encode. Embeddings are cached by image content, in RAM and then on disk, and an image keeps its placeholder token ids
for the life of the process, so a resent image skips the tower and the prefix cache matches through it. Text-only
requests run exactly as without image input.

GLM-5.3's template is text-only; the tokenizer overlay renders image blocks as the model's image tokens and adds an
`enable_thinking` switch. For DeepSeek-V4.1 the server clears exllamav3's `mm_exact_chunks`, which is meant for a
model whose image span attends bidirectionally; V4.1 is causal over it.

| variable | default | |
|---|---|---|
| `BIGMOE_MM` | 1 | 0: requests with images get a 400 (`multimodal=False`) |
| `BIGMOE_MM_MAX_TOKENS` | the model's budget | tokens per image (`image_tokens`) |
| `BIGMOE_MM_CACHE_MB` | 256 | embedding cache in RAM |
| `BIGMOE_MM_DISK_DIR`, `BIGMOE_MM_DISK_MB` | `~/.cache/bigmoe/mm-cache/<backend>`, 2048 | embedding cache on disk |
| `BIGMOE_MM_MAX_IMAGES`, `BIGMOE_MM_MAX_BYTES` | 16, 20,000,000 | per request, per image |
| `BIGMOE_MM_FETCH`, `BIGMOE_MM_FILES` | 1, 0 | fetch http(s) URLs; accept `file://` URLs |

## Serving behaviour

- A prompt that leaves less than min(`max_tokens`, 1024) tokens of room (plus the drafter's window) is refused before
  anything is streamed with a 400 `context_length_exceeded`, which agent clients answer by compacting.
- A streaming response that is silent for 15 s gets an SSE comment, so HTTP clients do not drop it during a long
  prefill or a queue wait.
- A job that runs out of GPU memory frees what the failed forward held; before its first token it is run once more with
  half-size prefill chunks.
- A request may name any `model`; it is mapped to the served model.
- Refused with a 400 rather than ignored: logprobs, `allowed_token_ids`, `bad_words`, `n > 1`, beam search.
  Teacher-forced logprobs are available at `/generate` with `return_logprob`.
- Beside vLLM's routes: `/stats` (engine, scheduler, tier and vision counters, GPU memory), `/server_info`, `/generate`
  (token ids in, SGLang-shaped), and the diagnostics `/kl_capture`, `/kl_force`, `/kl_teacher`, `/nv_verify`, `/nv2_rate`,
  `/cpu`.

## Memory and CPUs

`memory_gb` is the systemd scope's cap; the RAM expert tier is what is left after exllamav3's host caches.
`frontend_extra_ram_gb` (DeepSeek 4.5, GLM 1.5) is added to the cap for vLLM's API server process. DeepSeek's RAM tier
has a fixed size (46.0 GiB in mode `nvme`, 47.5 GiB in `nvme-exact`), because sizing it from the cgroup at start varies
by several GiB between identical starts.

The default CPU layout (`cpus.py`) puts the GPU-feeding thread and the planner's controller on the first core, the CPU
lane on one thread of every other core and the NVMe readers on their SMT siblings. vLLM's API server may use every CPU
but the model thread's.

## Verification

- `pytest` runs vLLM's real scheduler over a CPU-only stand-in for the engine (`tests/test_vllm.py`), the agentic
  benchmark against a scripted server, the launch configuration of both backends, the store format, and turn
  checkpoints over fake states and exllamav3's real classes. GPU tests skip without a GPU.
- `bigmoe verify <model>` restarts the server for each mode, runs the long-context benchmark and, in mode `nvme`, the
  agentic benchmark, and compares with `src/bigmoe/models/*/baselines`: `nvme-exact` answers must be byte-identical.
- `src/bigmoe/models/dsv41/tools/` measures agreement with exact decode: forced decode through the server
  (`kl_force.py`), reference logits from stock exllamav3 (`ref_logits.py`), agent turns (`turn_kl.py`), resumed turns
  (`resume_kl.py`).

## Tried and not kept

- Warm prefill for GLM: its answers react more to dropped experts (forced-decode agreement after long prompts fell from
  99.7 % to 96.2 %).
- Dropping cold experts in forwards of up to 1,024 rows: faster, but reading every routed expert is exact, and leaving
  out even 0.2 % of the routing weight cost more agreement than it saved time.
- Decode-time checkpoints for GLM: a staged GLM forward costs about the same whatever its length, so resuming inside an
  answer saves little.
- vLLM's async scheduling: vLLM spends about 0.2 ms between steps.
- Drafting the text a tool-call grammar forces: about two tokens per argument.
- Eviction policies for the RAM tier other than the segmented LRU (LRU-2, 2Q, ARC, decayed counts): none beat it on a
  replayed agent session; knowing the future would be worth far more, which takes a routing predictor, not a policy.
- `chunk_size` 10240 for DeepSeek: faster long prefills, too little VRAM headroom at 128k context.
