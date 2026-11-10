# vLLM serving engine

vLLM 0.33.0 is the only serving engine of both backends. It owns the HTTP APIs, rendering, tool and reasoning parsing,
detokenization and stop handling, and its scheduler makes every scheduling decision. exllamav3 1.6.6 over the shared
expert-offload engine (`bigmoe.engine`) is the model executor it drives: the EXL3 forwards, the paged K/V cache with its
prefix-cache tiers, and the drafters. There is no other server: the former standalone exllamav3 servers
(`models/*/engine/server.py`), their DSML tool parser (`models/dsv41/toolcalls.py`) and the `frontend` option are gone.

```sh
bigmoe start deepseek-v4.1-flash
bigmoe start glm-5.3-flash
bigmoe verify deepseek-v4.1-flash          # long-context speed + answers, then the agentic benchmark
bigmoe bench-agent glm-5.3-flash           # the agentic benchmark alone (bench/agent.py)
```

## Why the forwards stay in exllamav3

vLLM 0.33.0 ships both architecture families but cannot run either on this box: DeepSeek-V4.1's sparse-MLA attention
backends require sm90, sm100 or sm120 (`v1/attention/backends/mla/*sparse*.py`), GLM-5.3's sparse indexer requires
DeepGEMM (also sm90/sm100/sm120), and the RTX 3090 is sm86. vLLM also has no EXL3 quantization method, and nothing that
offloads routed experts to a RAM and NVMe tier, which is how a 125-361 GB checkpoint runs from a 24 GB card. The
forwards, the expert tiers and the K/V memory therefore stay exllamav3's and bigmoe's; everything that decides what runs
when is vLLM's.

## What runs where

| process | code | responsibility |
|---|---|---|
| API server | vLLM + `bigmoe.vllm.routes` / `mm` | OpenAI and Anthropic HTTP APIs, chat rendering, tool/reasoning parsing (DeepSeek DSML: `deepseek_v41`, GLM: `glm47`), media loading and preprocessing, detokenization, stop strings, SSE streaming |
| EngineCore | vLLM `Scheduler` (`BoxScheduler`) | admission and preemption (FCFS / priority, `max_num_seqs`), chunked prefill under the token budget and long-prefill threshold, speculative-token scheduling and acceptance accounting, prefix-cache accounting, stop checks, outputs, metrics |
| EngineCore | `BoxKVCacheManager` | vLLM's KV-manager interface over exllamav3's page table: capacity, prefix hits, frees |
| EngineCore | `BoxWorker` + `ExlEngine` (`engine.py`) | executes each scheduled step: prefill chunks (`Job.prefill`), one batched sampling / verify forward (`Generator.iterate_gen` over exactly the scheduled batch), the drafter's next drafts |
| EngineCore | `bigmoe.engine` | nv2 VRAM CLOCK cache, pinned-RAM tier, O_DIRECT NVMe store, per-layer planner, CPU and x16 lanes, staged / warm prefill |

exllamav3's own scheduling loop is not used: `Generator.iterate()` (its admission in `iterate_start_jobs`, its prefill
rotation, cooldown and shared chunk, its draft-then-verify round) is never called.

### Scheduling

- **Admission.** vLLM's scheduler admits waiting requests in its order. When it looks a request up in the prefix
  cache, `BoxKVCacheManager.get_computed_blocks` reserves the request in the engine: an exllamav3 job whose pages are
  allocated (cached pages revived from the GPU, the pinned-RAM tier or the disk tier, with the recurrent checkpoint
  they need) and whose cached prefix is skipped at once. The number of prompt tokens skipped is vLLM's local prefix hit,
  so `num_computed_tokens`, the prefix-cache metrics and `usage.prompt_tokens_details.cached_tokens` are vLLM's own
  accounting of exllamav3's reuse. Without free pages `allocate_slots` fails: the request keeps waiting, and a running
  request that needs fresh pages (exllamav3's 4096-token page budget, `MAX_RQ_TOKENS`, is used up) makes vLLM preempt
  one, as it would for its own KV blocks. Images are encoded at admission (nonresident tower, content-keyed cache).
- **Concurrency.** exllamav3 runs as many sequences at once as its cache has recurrent-state slots (`-ambs`, 1 by
  default, both models have recurrent layers), whatever `max_batch_size` says. `serve.py` gives vLLM that number as
  `max_num_seqs`; other requests wait in vLLM's queue (the previous servers had the same limit inside exllamav3).
- **Chunked prefill.** A step's budget is one prefill chunk (`chunk_size`, 8192) plus every running request's decode
  token and drafts (`max_num_batched_tokens`), with `long_prefill_token_threshold = chunk_size`. vLLM sizes every chunk;
  `BoxScheduler._reserve_prefill_lookahead` (the last word on a chunk's size in both of vLLM's scheduling loops) shortens
  it to where exllamav3 can end a forward (`engine.plan_chunk`: a 256-token page, the max chunk, the recurrent
  last-page split of `engine/last_page.py`), so the engine computes exactly what was scheduled. vLLM's EngineCore turns
  chunked prefill off for a model without KV cache groups; `BoxScheduler` turns it back on.
- **Speculative decoding.** DSpark (DeepSeek) and n-gram drafts (GLM `speculation="ngram"`) are proposed by
  exllamav3's drafters right after each step's verify forward, for exactly the requests that sampled, and handed to
  vLLM (`take_draft_token_ids`). vLLM schedules them as spec tokens in the next step (`num_speculative_tokens` from the
  engine's window; the config's method is `ngram` only as a name, vLLM runs no proposer), the engine verifies what was
  scheduled, and vLLM rolls back the rejected ones (`num_computed_tokens -= rejected`) and keeps the spec-decode
  metrics. A step that also samples a request's first token verifies no drafts (vLLM counts them as rejected).
- **Reconciliation.** After each step `BoxScheduler.update_from_output` lets vLLM do its bookkeeping, then compares each
  request's computed-token count with the engine's. They differ only when the engine did something the schedule could
  not foresee (an out-of-memory retry with half-size chunks, a requeue waiting for pages, a multimodal chunk boundary);
  the count is then corrected and the cause counted in `/stats` (`scheduler.corrections`). On the verify and agent
  runs below it stayed empty.
- **Finishes.** vLLM ends requests (EOS, stop strings, `max_tokens`, aborts) and `BoxKVCacheManager.free` returns their
  pages to exllamav3's prefix cache. A request the engine ends on its own (the cache is full, a failed job) is finished
  by `BoxScheduler` with `length` or `error` and its message.

### Memory and CPUs

The API server may use every CPU except the model's logical CPU. Only the GPU-feeding EngineCore thread is pinned to
the backend's main CPU; nv2's controller, CPU lane and NVMe readers keep their existing affinities.
`frontend_extra_ram_gb` (4.5 for DeepSeek, 1.5 for GLM) is added to the systemd memory cap for the API process and
vLLM's footprint, which would otherwise come out of nv2's RAM expert tier. DeepSeek's tier is fixed at 46.0 GiB
(`nvme`) / 47.5 GiB (`nvme-exact`): identical cold starts sized it anywhere from 43.66 to 51.60 GiB from the cgroup.
An explicit `DSV41_NV_RAM_GB` wins; `frontend_extra_ram_gb=0` sizes the tier automatically from `memory_gb`.

## Compatibility

- **Chat and completions.** vLLM's DeepSeek-V4.1 and GLM-4.7 renderers/parsers with the checkpoints' own chat
  templates. Requests without a `model`, or with any alias, are mapped to the served model name. GLM thinking defaults
  on; DeepSeek thinking defaults off.
- **Tools and reasoning.** DeepSeek DSML calls and GLM `<tool_call>` blocks are parsed by vLLM, including streaming
  argument deltas. GLM string arguments keep leading/trailing whitespace (file-writing tools need it).
- **Images.** vLLM loads and validates the image and calls exllamav3's own resize/normalize/patchify code; the
  executor encodes it with the nonresident vision tower (VRAM lending, content-addressed RAM/SSD embedding cache) when
  vLLM admits the request. A repeated image keeps its placeholder IDs, so the prefix cache reuses the prompt through it.
- **Prefix caching.** The GPU page cache, pinned-RAM tier, disk tier and recurrent checkpoints are exllamav3's; vLLM's
  block-hash cache stays off (`--disable-prefix-caching`) and vLLM asks `BoxKVCacheManager` instead. Models with
  recurrent state reuse a prefix only up to a checkpoint: a chunk end, a short last page, or a page boundary while
  decoding (every 256 tokens for DeepSeek in mode `nvme`, `rc_interval`; exllamav3's 2,048 otherwise). `/stats`
  exposes `prefix_cache` occupancy and hit counters. Idle disk flushes continue between requests.
- **Operational behavior.** `/health`, `/stats` (with `scheduler`: steps, reservations, page refusals, verify steps,
  corrections, time in the executor and between steps), `/server_info`, `/generate`, `/tokenize`, `/kl_capture`, `/kl_force`, `/kl_teacher`, `/nv_verify`,
  `/nv2_rate` and `/cpu` are available. Streaming responses receive an SSE comment after 15 silent seconds.
  Context-length admission reserves min(`max_tokens`, 1024) output tokens plus the drafter window and answers 400
  `context_length_exceeded`. A CUDA OOM before the first token frees the failed forward and retries once with half-size
  prefill chunks.
- **Token counts.** OpenAI/vLLM `completion_tokens` excludes a stop token.

- **Structured outputs and strict tool calling.** vLLM's grammars are enforced: `response_format` (JSON schemas,
  regex, grammars) and the structural tags vLLM's DSML / GLM tool parsers attach for strict tool calling
  (`VLLM_ENFORCE_STRICT_TOOL_CALLING`, on by default) are vLLM's xgrammar state machines. vLLM computes each step's
  bitmask (one row per sampled position, draft positions included) between `execute_model` and `sample_tokens`; the
  worker runs the step in `sample_tokens`, and a `Filter` on the exllamav3 job (`engine.grammar_filter`) applies the
  rows in exllamav3's sampler (its packed int32 bitmask format is xgrammar's). vLLM advances and checks the state
  machine with the accepted tokens and limits the drafts it verifies. Before this, vLLM ended a request with a 500
  ("grammar rejected tokens") whenever the unconstrained sampler left the tool-call grammar.

The following sampling features are rejected with a 400 rather than accepted and ignored: output/prompt logprobs,
`allowed_token_ids`, `bad_words`, `n > 1`, and beam search. Teacher-forced logprobs remain available through
`/generate` with `return_logprob`, `/kl_teacher`, and the quality tools.

## Multimodal template details

DeepSeek's template already understands image blocks. GLM-5.3's shipped template is text-only and renders a reminder
instead; `serve.py` creates a tokenizer overlay that renders `<|begin_of_image|><|image|><|end_of_image|>` and adds an
`enable_thinking=False` switch. The overlay is keyed by the resolved checkpoint path and template contents, so
same-named or replaced checkpoints cannot inherit stale tokenizer files.

For DeepSeek, one `<｜deepseek_image｜>` expands to `2 + h * (w + 1)` embedding positions. For GLM, the image token expands
to `t * h * w / merge^2` positions between the real start/end tokens. The executor substitutes exllamav3's stable
placeholder IDs and passes the corresponding `MMEmbedding` objects to the job. These positions are prompt tokens and
are included in prefill cost.

## Verification

The unit suite runs vLLM's real scheduler in-process over a `FakeEngine` with the executor's interface
(`tests/test_vllm.py`): chunk sizes under the budget and threshold, page-aligned by the plan, with the engine asserting
that it computes exactly what was scheduled; a second request admitted into the leftover budget; a long prefill that
does not stall a decoding request; admission waiting for pages; vLLM-reported prefix hits; drafts scheduled, verified
and rolled back by vLLM; engine-side errors. `tests/test_agent_bench.py` runs the agentic benchmark against a scripted
OpenAI server.

## Measured results (RTX 3090 box, 2026-11-09)

`bigmoe verify` against the backends' baselines (`models/*/baselines`, 2026-11-06), greedy, 128k context. Numbers
are prefill / decode tok/s. Mode `nvme` answers are near-exact and differ between any two runs; mode `nvme-exact`
answers must be byte-identical.

### DeepSeek-V4.1-Flash (DSpark on)

| prompt tokens | baseline `nvme` | now `nvme` | baseline `nvme-exact` | now `nvme-exact` |
|---|---|---|---|---|
| 1,024 | 174.1 / 17.54 | 171.1 / 16.96 | 32.7 / 10.57 | 33.1 / 10.56, identical |
| 8,192 | 679.6 / 20.19 | 675.1 / 18.63 | 248.5 / 11.83 | 250.5 / 12.06, identical |
| 32,768 | 724.8 / 19.03 | 718.2 / 19.02 | — | — |

The first run of this scheduler had `nvme-exact` differ at 1k: it sampled a request's first token in a forward of its
own, while exllamav3 verifies DSpark's first drafts in that forward, so the verify windows (and their fp16 GEMM shapes)
shifted. The last prefill chunk now stops one token short when the engine drafts (`plan_chunk`, `hold_last`) and the
first decode step is one of vLLM's spec-decode steps: the same forwards as exllamav3's own loop, answers identical.
The `nvme` decode gap (3 % at 1k; 8k 18.6-19.3 tok/s over two runs) is the one the earlier vLLM measurement showed.

Forced decode through the server (`models/dsv41/tools/kl_force.py`, reference `q/gen_v0.json`, 1,110 positions):
**99.73 % top-1, mean KL 0.000659, max KL 0.083** (baseline with DSpark: 99.64 %, 0.000775). Prefill on a cached
8,192-token prefix (`bigmoe bench --prefill-tokens`), time to first token for 8 / 80 / 120 / 400 / 800 / 2,000 / 5,000 /
10,000 new tokens: 1.35 / 4.87 / 3.52 / 5.46 / 8.22 / 6.97 / 9.27 / 20.13 s (README: 1.2 / 3.3 / 3.3 / 5.4 / 8.2 / 6.9 /
8.9 / 18.6 s); two repeats of the short sizes gave 0.54-1.87 s for 8 and 2.36-5.20 s for 80 new tokens, so those
differences are run-to-run noise. Images: a synthetic two-color image was described correctly; tower load 0.26 s,
encode 0.30 s, the follow-up turn hit the embedding cache.

### GLM-5.3-Flash

| prompt tokens | baseline `nvme` | now `nvme` | baseline `nvme-exact` | now `nvme-exact` |
|---|---|---|---|---|
| 1,024 | 113.7 / 11.08 | 110.1 / 10.81 | 58.9 / 7.86 | 58.0 / 7.69, identical |
| 8,192 | 788.6 / 11.33 | 761.7 / 10.87 | 418.0 / 8.03 | 413.1 / 7.87, identical |
| 32,768 | 845.6 / 11.45 | 811.6 / 11.19 | — | — |
| 120,000 | 816.9 / 11.31 | 809.2 / 11.07 | — | — |

The 2-4 % gap matches the earlier vLLM run's (same-day A/B then: 765 / 11.16 vLLM against 759 / 11.28 for the old
server at 8k). Forced decode against `q/glm_ref_exact.json` (897 positions): **97.21 % top-1, mean KL 0.0102**
(README: 97.3 %, 0.0096; earlier vLLM run: 97.10 %, 0.0131). With `speculation="ngram"` a code-copy request was
reproduced exactly with 309 of 309 drafted tokens accepted over 64 verify steps, all scheduled by vLLM; a JSON-schema
`response_format` request returned a valid object. Images: described correctly, tower load 0.19 s, encode 0.59 s.

### Agentic benchmark (`bench/agent.py`, mode `nvme`)

Three coding tasks in fresh git repositories (fix a failing test suite, write a CLI to a spec, refactor duplicated code
into a module), solved over the OpenAI chat API with three function tools (`bash`, `read_file`, `write_file`), every
turn streamed with tool-call argument deltas, each task checked by a command independent of the model's claims. Then
two of the tasks with two agents at once. Baselines: `models/*/baselines/agent_nvme*.json`.

| | tasks solved | turns | tool calls (malformed) | wall | prefix reuse after turn 1 | mean time to first token |
|---|---|---|---|---|---|---|
| DeepSeek, one agent | 3 / 3 | 14 | 13 (0) | 165 s | 80 % | 5.7 s |
| DeepSeek, two agents | 2 / 2 | 10 | 10 (0) | 84 s | 84 % | 14.1 s |
| GLM, one agent | 3 / 3 | 21 | 28 (0) | 376 s | 21 % | 9.3 s |
| GLM, two agents | 2 / 2 | 13 | 17 (0) | 217 s | 44 % | 11.4 s |
| GLM `speculation="ngram"`, one agent | 3 / 3 | 23 | 27 (0) | 443 s | 62 % | 10.2 s |

DeepSeek's rows are with decode-time checkpoints (its default in mode `nvme`, see below); before them: 175 s, 59 %,
7.0 s and 87 s, 68 %, 14.6 s. `scheduler.corrections` stayed empty in every run (vLLM's counts and the engine's never differed), with no page
refusals and no dropped drafts. What the benchmark shows about these models on this box:

- **Prefix reuse is bounded by recurrent checkpoints.** Both models have recurrent state, so a conversation resumes
  only from a checkpoint: a chunk end, a page boundary at least `rc_interval` tokens (exllamav3: 2,048) past the last
  checkpoint, or a prompt
  whose tail past its last page boundary is short enough to split off cheaply (`engine/last_page.py`: 85 tokens for
  DeepSeek, 63 for GLM). Short agent conversations (all of these stay under 3k tokens) therefore reuse only what their
  earlier prompts' lengths happened to checkpoint: GLM's runs range from 0 % to 70 % per task for that reason alone.
  Splitting every tail would not pay on this box: a staged GLM forward costs ~8.8 s whatever its length, about what
  the re-prefill it would save costs. Long agent sessions get a checkpoint at every 8,192-token chunk end. For
  DeepSeek, decode-time checkpoints now also put one every 256 tokens inside each answer.
- **Concurrency.** DeepSeek has one recurrent-state slot, so vLLM runs one sequence and queues the other; two agents
  still finish in 87 s instead of 113 s because one's prefill and decode overlap the other's tool execution and client
  time. GLM has four slots (`-ambs 4`), so vLLM batches both agents; steps that carry the other agent's 8.8 s prefill
  chunk stall the decode beside it, and two GLM agents took 217 s against 191 s for the same tasks one after the other.

## What running on vLLM made possible (tried and measured, 2026-11-09)

Each idea was measured on this box before it was kept. Quality is measured against exact references (mode
`nvme-exact`, full prefill): answer identity, top-1 agreement and mean KL per position (`models/dsv41/tools/kl_force.py`,
`turn_kl.py`, `resume_kl.py`).

### Kept

- **Decode-time recurrent checkpoints (DeepSeek, mode `nvme`: `rc_interval=256`, the default).** An agent's next turn
  extends its previous answer, but both models have recurrent state, so a prompt resumes only from a checkpoint, and
  exllamav3 checkpoints a decode only every 2,048 tokens: most turns re-prefilled the previous answer. With a checkpoint
  every 256 decoded tokens the next turn resumes inside the answer.
  - Quality, six ~600-token answers each followed by a ~630-token second turn that resumes at token 512, against the
    exact full prefill: in mode `nvme`, re-prefilling (warm prefill) gives 98.40 % top-1, mean KL 0.0142 in 82.8 s;
    resuming gives **99.01 %, 0.0044 in 61.6 s**. Faster and closer to exact: the resumed state comes from the near-exact
    decode path, a re-prefill goes through warm prefill.
  - Agent benchmark, mode `nvme`: 175.2 s -> **165.3 s**, prefix reuse 59 % -> **80 %**, mean time to first token per
    task 6.8-7.1 s -> 4.6-6.4 s; two agents 86.8 s -> 84.0 s (reuse 68 % -> 84 %). Mode `nvme-exact` (deterministic,
    same trajectories), where it was tested before being restricted to `nvme`: 423 s -> 359 s, reuse 71 % -> 79 %,
    turns that resumed inside the previous answer 31.6 s -> 16.5 s and 31.7 s -> 2.7 s.
  - Cost: none measured. Decode at 32k in back-to-back pairs: 16.54 / 17.42 tok/s with it, 16.43 / 17.42 without; the
    verify run: 170.7 / 17.21, 667.8 / 20.03, 717.4 / 17.63 tok/s at 1k / 8k / 32k (baseline 174.1 / 17.54,
    679.6 / 20.19, 724.8 / 19.03). Forced decode 99.55 % top-1, mean KL 0.00093 (earlier runs 99.55-99.73 %). A
    checkpoint is 36 MB; `prefix_cache_gb` (0.5 GiB) holds 14 and evicts the oldest.
  - Mode `nvme-exact` keeps exllamav3's schedule, because there a decode-time checkpoint is not bit-identical to what
    a prefill computes (KL 0.0039 against 0 for a prefill-time checkpoint at the same position), and the stop at each
    checkpoint boundary moves DSpark's verify windows (a 1k answer changed). `nvme-exact` answers stay identical.
- **Priority scheduling** (`--scheduling-policy priority`). A request's `priority` (lower first) orders vLLM's waiting
  queue; equal priorities stay first-come first-served, so default traffic is scheduled exactly as before, and a
  non-zero `priority` is no longer refused. Preemption still happens only when pages run out (with one sequence at a
  time, DeepSeek, a priority request waits for the running one, then goes first). `tests/test_vllm.py` checks the order
  on vLLM's real scheduler.
- **A failed sampling forward no longer kills the engine.** exllamav3's loop contained prefill failures only; an
  exception in the batched sample / verify forward (found while testing grammar drafts: GLM's KDA layers cannot rewind a
  speculative pass without draft history) took the EngineCore down and every request with it. It now ends the requests
  in that forward with an error and the server goes on.

### Measured, not kept

- **Async scheduling.** vLLM's time between two engine steps (update_from_output, schedule, outputs) is 0.2 ms per
  step, 0.2 % of a DeepSeek decode (463 tokens, 303 steps, 28.2 s in the executor, 0.06 s between). Nothing to hide.
- **GLM one sequence at a time (`-ambs 1` instead of 4).** The extra recurrent slots do not come out of the expert cache
  (1,472 VRAM slots either way) and single-sequence speed is unchanged (111.0 / 10.86 tok/s at 1k against 110.1 / 10.81).
  Two agents at once took 224 s against 227 s for the same tasks run one after the other: GLM's tool time is too short
  to overlap, and batching (217 s against 191 s with `-ambs 4`) is within the run-to-run spread of agent trajectories.
  No change.
- **Grammar-forced draft tokens.** With vLLM's tool-call grammar in the sampler, the text the grammar forces could be
  drafted for free. It is short: xgrammar constrains text, not token splits, so no position allows a single token, and
  the forced text (`find_jump_forward_string`) is about two tokens per argument (`</arg_key><arg_value>`) for GLM;
  DeepSeek's parser attaches no tool grammar unless a tool is `strict`. GLM without a drafter keeps no draft history and
  cannot verify drafts at all. With `speculation="ngram"` it drafted 21 tokens in four replayed agent turns, the turns
  took 28.61 / 21.98 / 44.74 / 12.62 s against 28.27 / 22.78 / 44.24 / 12.71 s, and one `nvme-exact` answer changed (a
  verify window flips a near-tie, as DSpark's do). Removed.
- **Decode-time checkpoints for GLM** (opt-in: `-o rc_interval=256`). The same six turns as for DeepSeek (below), GLM
  mode `nvme`, against the exact full prefill: re-prefilling 98.47 % top-1, mean KL 0.0029, 126.9 s; resuming from the
  decode-time checkpoint 99.04 %, 0.0034, 119.5 s (-6 %). GLM's prefill is near-exact already (no warm prefill by
  default) and a staged GLM forward costs about the same whatever its length, so the remainder after the checkpoint
  costs nearly what the whole turn did. Quality is a wash and the saving small, so GLM keeps exllamav3's schedule.
  (In `nvme-exact`: resuming from a decode-time checkpoint 98.64 %, 0.0031 against 98.97 %, 0.0021 for resuming from a
  prefill-time one at the same position.) Single-request answers stay byte-identical with it (no drafter), decode
  speed unchanged.
- **Suffix decoding** (vLLM's `suffix` method, Arctic Inference; `arctic-inference` 0.3.0 installs). Not tried: its
  advantage is drafting from earlier *requests*, while what agents here repeat is within the context (files read, then
  written), which DSpark and GLM's n-gram policy already draft; and vLLM's proposer is tied to its own model runner.

## Agent-turn quality (DeepSeek, mode `nvme`, 2026-11-09)

The README's DeepSeek top-1 (99.6 %) is `kl_force` on six short chat prompts: the decode path on a reference answer,
with almost no prefill. An agent turn also prefills its new tokens, and DeepSeek's mode `nvme` prefills through warm
prefill (on by default since before the vLLM migration), so it agrees less with exact. Measured with
`resume_kl.py`: six ~600-token answers, then a ~630-token second turn whose 128-token answer is teacher-forced along
the exact (`nvme-exact`, full prefill) answer, so every position is compared on the same token path, 719 positions.
The old engine is commit f9f7285 with `frontend=native`, run on the same day.

| configuration | second-turn top-1 | mean KL | 6 turns | README test (`kl_force`, 1,110 positions) |
|---|---|---|---|---|
| old engine | 98.33 % | 0.0150 | 82.9 s | 99.46 %, KL 0.00156 |
| vLLM, re-prefill | 98.33 % | 0.0149 | 82.5 s | 99.55-99.73 %, KL 0.00066-0.00093 |
| vLLM, resumed from decode-time checkpoint (default) | 98.75 % | 0.0059 | 61.3 s | |
| vLLM, re-prefill, `warm_fetch=128` | 98.75 % | 0.0057 | 101.5 s | |
| vLLM, re-prefill, `warm_fetch=192` | 99.30 % | 0.0041 | 103.4 s | |
| vLLM, resumed, `warm_fetch=192` | 98.89 % | 0.0048 | 61.5 s | |
| vLLM, re-prefill, `warm_prefill=False` | 99.44 % | 0.0030 | 145.0 s | |

Free-running instead of forced (both answers generated, compared up to their first difference): the old engine
96.65 %, vLLM re-prefill 98.40 %, vLLM resumed 99.01 %.

So vLLM matches the old engine on both tests, and its default (resuming from decode-time checkpoints) is better and
faster than either. The remaining distance to the decode-only figure is warm prefill: re-prefilling exactly restores
99.44 % at 76 % more time per turn, and reading more NVMe-only experts per layer (`warm_fetch`, default 64 of 384) buys
most of it back at 25 % more time on fresh prompts. On resumed turns, whose short chunks route few cold experts,
`warm_fetch=192` costs nothing measurable but gains little; long chunks (file reads, first turns) pay the most. Both
knobs stay at their defaults; set `-o warm_fetch=192` or `-o warm_prefill=false` to trade prefill time for agreement.

## Warm-prefill read-ahead (2026-11-09)

Warm prefill used to wait for each layer's NVMe-only experts immediately before that layer's MoE. Now, for staged chunks
of at least 1,024 rows, layer i + 1's router runs on layer i's router input while layer i computes. The predicted cold
experts are read on a background thread into the next layer's staging buffer. Once the real routing is known, unread
guesses are discarded and any missed experts are queued ahead of the remaining guesses. CUDA events order the copies
before the MoE; pending reads are drained before staging memory returns to the decode pool. The actual fetched set is
still the top `warm_fetch` experts by real routing weight, so read-ahead changes I/O timing, not the model's computation.
It is on by default for DeepSeek warm prefill (`warm_ahead`), with 64 guesses per layer; `warm_ahead_n` can widen that
window. Short chunks keep the old synchronous path (`EC_WARM_AHEAD_MIN`, default 1,024).

Four matched-prompt repetitions per configuration (same token ids and lengths within each A/B cycle, unique prefix to
prevent cache reuse) on DeepSeek, mode `nvme`:

| prompt tokens | read-ahead off | 64 guesses | 96 guesses | 128 guesses |
|---|---|---|---|---|
| 460 | 5.52 s | 5.70 s (+3 %) | 5.76 s | 5.77 s |
| 1,600 | 10.48 s | **8.70 s (-17 %)** | 10.90 s | 9.98 s |
| 3,000 | 8.48 s | **6.37 s (-25 %)** | 6.42 s | 6.47 s |
| 6,000 | 10.07 s | **8.19 s (-19 %)** | 8.82 s | 9.44 s |

The default 64-expert window was best overall; wider windows added I/O without improving coverage enough. Below 1,024
rows the default now avoids the +0.18 s seen at 460 tokens. On a fixed teacher-forced prompt, read-ahead off versus on
produced **1,200/1,200 bit-identical positions, top-1 100 %, KL 0**, and took 6.67 s -> 5.43 s.

Cached-prefix time to first token (8,192-token cached prefix, one run per size):

| new tokens | 8 | 80 | 120 | 400 | 800 | 2,000 | 5,000 | 10,000 |
|---|---|---|---|---|---|---|---|---|
| read-ahead off | 0.87 s | 5.08 s | 3.50 s | 5.60 s | 8.28 s | 7.00 s | 9.24 s | 20.25 s |
| read-ahead on | 0.62 s | 4.11 s | 3.38 s | 5.48 s | 8.32 s | 6.32 s | 7.42 s | 16.49 s |

The 2,000-10,000-token steps are 10-20 % faster. Short-step differences are within the run-to-run spread documented
above and below 1,024 rows the code path is unchanged. Cold-server forced second turns produced the same six answers
with read-ahead off and on. The decode-only reference remained in its usual range: **99.64 % top-1, mean KL 0.000796**
over 1,111 positions. The no-read-ahead agent baseline solved 3/3 tasks in 167.7 s.

The full `bigmoe verify deepseek-v4.1-flash` run passed. In mode `nvme`, 1k / 8k / 32k prefill measured
170.0 / **780.2** / **839.2 tok/s** against the 174.1 / 679.6 / 724.8 baselines: long-chunk prefill is 15-16 % faster,
and decode is in the same range as the previous vLLM runs. Both `nvme-exact` answers are **byte-identical** (1k and 8k).
The agent benchmark solved 3/3 tasks in 164.5 s with read-ahead versus 167.7 s without (mean TTFT 5.75 s versus 5.68 s,
within noise), and the two-agent run solved 2/2 in 83.7 s. There were no malformed tool calls and no scheduler
corrections. GLM warm prefill remains off by default; with `-o warm_prefill=true`, its off/on teacher-forced check was
also **1,200/1,200 bit-identical, KL 0**, and took 9.11 s -> 7.33 s.


## Warm prefill on small agent steps (DeepSeek, mode `nvme`, 2026-11-09)

An agent step adds a tool result of N tokens to a cached conversation. `models/dsv41/tools/warm_kl.py` measures what
warm-prefilling only those N tokens costs. On both servers, the same 8,192-token prefix was computed exactly and is
cached. Each sample then adds N new tokens of its own (a file excerpt plus the assistant turn's opening) and a greedy
128-token answer. The answer is generated with warm prefill off and forced along the same tokens with it on. Decode runs
on the bit-exact GPU path on both (`/cpu?on=0`), so the N-token prefill is the only difference. Three samples per size,
~360 answer positions each. Read-ahead does not apply below 1,024 rows. Prefill times are the engine's per-request log
lines.

| new tokens | prefill path | exact prefill | warm prefill | top-1 | mean KL |
|---|---|---|---|---|---|
| 8 | per pick (no staging) | 0.90 s | 0.90 s | 100 % | 0 (bit-identical) |
| 20 | per pick | 1.87 s | 1.90 s | 100 % | 0 (bit-identical) |
| 40 | per pick | 2.90 s | 2.90 s | 100 % | 0 (bit-identical) |
| 60 | per pick | 4.23 s | 4.27 s | 100 % | 0 (bit-identical) |
| 80 | per pick | 4.83 s | 4.83 s | 100 % | 0 (bit-identical) |
| 100 | staged | 15.13 s | **4.47 s** | 98.5 % | 0.0035 |
| 120 | staged | 15.17 s | **4.77 s** | 96.5 % | 0.0057 |
| 160 | staged | 15.20 s | **4.93 s** | 98.1 % | 0.0051 |
| 200 | staged | 15.17 s | **5.00 s** | 97.7 % | 0.0066 |
| 300 | staged + tail forward | 20.27 s | **9.70 s** | 97.4 % | 0.0053 |
| 400 | staged | 15.20 s | **5.20 s** | 96.4 % | 0.0082 |
| 600 | staged | 15.20 s | **5.40 s** | 96.1 % | 0.0097 |
| 800 | staged + tail forward | 19.13 s | **9.07 s** | 98.2 % | 0.0073 |
| 1,000 | staged | 15.30 s | **5.70 s** | 97.8 % | 0.0070 |

- Below 512 expert picks (N <= 86: the prefill forward holds N - 1 tokens, 6 experts each), neither exact nor warm
  staging applies and every pick is served individually. Those sizes are the control: all 15 samples are bit-identical,
  so the measurement has no noise floor.
- From 100 to 1,000 new tokens, warm prefill saves about 10 s per step (15.2 s -> 4.5-5.7 s). Over the 3,344 staged
  positions it agrees with exact prefill on 97.4 % of top-1 tokens, mean KL 0.0065, rising from 0.0035 at 100 tokens to
  0.007-0.010 at 400-1,000. 84 of the 87 disagreements are near-ties: the exact model's own probability for its top-1
  is within 0.2 of the token warm prefill picked. 3 are clear flips (largest: 0.905 vs 0.095, at 120 tokens).
- 300 and 800 new tokens end 43 and 31 tokens past a page boundary. exllamav3 splits such a short tail into a forward of
  its own, which checkpoints the recurrent state at the boundary (`engine/last_page.py`). That forward costs ~5 s in both
  modes, so about a third of step sizes (tails of 1-85 tokens) pay twice.
- A request whose prompt was cached up to its last token (a resent prompt ending one token past a checkpoint) failed
  with a 500: exllamav3 restores it complete at allocation, so `Job.prefill`, which stamps the prefill start time, never
  ran. The executor stamps it now (`tests/test_vllm.py`).

## Exact short prefills (DeepSeek, mode `nvme`, 2026-11-09)

Exact staging reads every NVMe-only expert of every layer, about 250 of 384 per layer (~95 GB), however few tokens the
chunk has. A short chunk routes far fewer of them. Recorded on the samples above (`warm_trace` RPC): 75 per layer at
100 new tokens and 153 at 1,000. Warm prefill reads the 64 heaviest and drops the rest, and that drop is where the
3 % top-1 loss came from. It does not have to read the other ~100-175: no token routes to them.

Now a staged forward of at most `warm_exact_rows` rows (default 1,024) reads **every** routed NVMe-only expert and drops
nothing. The MoE then runs the same experts on the same bytes as exact staging, so the logits are bit-identical. Longer
chunks are unchanged (the 64-expert cap, read-ahead). Same test, same reference (`warm_kl.py`, `ref.json` from the
`warm_prefill=False` server), 3 samples per size, prefill times from the engine's request log:

| new tokens | exact prefill | warm prefill (fetch 64) | **exact short prefill** (new default) | top-1 / mean KL |
|---|---|---|---|---|
| 100 | 15.1 s | 4.5 s | **5.0 s** | 100 % / 0, 3/3 bit-identical |
| 120 | 15.2 s | 4.8 s | **5.2 s** | 100 % / 0, 3/3 |
| 160 | 15.2 s | 4.9 s | **6.0 s** | 100 % / 0, 3/3 |
| 200 | 15.2 s | 5.0 s | **6.6 s** | 100 % / 0, 3/3 |
| 300 | 20.3 s | 9.7 s | **11.9 s** | 100 % / 0, 3/3 |
| 400 | 15.2 s | 5.2 s | **8.4 s** | 100 % / 0, 3/3 |
| 600 | 15.2 s | 5.4 s | **10.0 s** | 100 % / 0, 3/3 |
| 800 | 19.1 s | 9.1 s | **14.0 s** | 100 % / 0, 3/3 |
| 1,000 | 15.3 s | 5.7 s | **11.9 s** | 100 % / 0, 3/3 |
| all 3,344 positions | | 97.4 % / 0.0065 | **100 % / 0** | |

Sum over one step of each size: exact 145.8 s, warm prefill 54.2 s, exact short prefill 79.0 s. That is exact prefill's
answers in 54 % of its time; against warm prefill it costs +0.4-1.6 s per step up to 200 new tokens and +3.2-6.2 s at
400-1,000. The forward is NVMe-bound: 1,078 GB in 27 forwards in 182 s of reads (5.9 GB/s; raw O_DIRECT reads of
expert-sized records on this SSD reach 5.9-6.2 GB/s). The 300 and 800 rows still pay a second forward for their short
tail (`engine/last_page.py`). A 4,000-token prompt (one 4k-row forward) and a 4,600-token one take no exact layer calls,
so long prefills are unaffected. The run went through `bigmoe start deepseek-v4.1-flash` with no options set.
`-o warm_exact_rows=0` restores warm prefill's speed for short steps, and `-o warm_exact_rows=256` keeps exact
prefill only for steps up to 256 rows (the +0.4-1.6 s ones).

GLM: the engine default is off (`EC_WARM_EXACT_ROWS=0`); GLM's warm prefill is off by default and this was not measured
there.

Measured, not kept (same samples, sizes 100-1,000, sum of prefill times / top-1 / mean KL):

| variant | prefill | top-1 | mean KL |
|---|---|---|---|
| warm prefill, fetch 64 (before) | 54.2 s | 97.40 % | 0.0065 |
| fetch 64; each dropped pick goes to the token's next-best warm expert, weights recomputed as the router does | 54.5 s | 97.07 % | 0.0058 |
| `warm_fetch=384 warm_drop=0.005`: read until <= 0.5 % of the call's routing weight is left out | 69.9 s | 98.09 % | 0.0034 |
| `warm_fetch=384 warm_drop=0.002` | 74.5 s | 98.06 % | 0.0027 |
| every routed expert (= the new default) | 79.0 s | 100 % | 0 |
| every routed expert + read-ahead (64 / 384 guesses, `EC_WARM_AHEAD_MIN=0`) | 78.5 / 78.7 s | 100 % | 0 |

- Leaving out even 0.2 % of the routing weight keeps KL at 0.0027, and it is only 6 % faster than reading every routed
  expert. Approximation does not pay at these sizes. `warm_drop` is kept as an option for longer chunks (with `warm_fetch=384`), not measured there yet.
- Substituting dropped picks with warm experts moves KL little and top-1 the wrong way.
- Read-ahead guesses are right (93-95 % of them are used), but it covers only 19-20 % of the reads. A layer's compute is
  much shorter than its reads, so there is little time to run ahead of the real routing, and the reads already saturate
  the SSD.
- Offline (recorded routings): fetching to minimise each token's squared dropped weight, instead of the summed weight,
  cuts the worst token's dropped weight by a third at the same budget. That comes at a slightly higher mean, so it was
  not tried on the server. Keeping the experts a step read in the RAM tier would cut the next step's cold reads (on
  these samples the cold routing weight would fall from ~0.5 to 0.06-0.15 with an unbounded tier). But a bounded LRU
  evicts decode-hot experts, and it changes decode, so it was not tried either.

Tools: `warm_kl.py force --min-n 100` skips the per-pick control sizes. The debug RPCs `warm_fetch` (`n`, `drop`,
`exact_rows`, at runtime) and `warm_trace` (records each staged layer call's routing and cold set to a file) need
`BIGMOE_DEBUG_RPC=1`.
