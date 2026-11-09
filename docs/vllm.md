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
  recurrent state reuse a prefix only up to a checkpoint: a chunk end, a short last page, or exllamav3's checkpoint
  interval while decoding. `/stats` exposes `prefix_cache` occupancy and hit counters. Idle disk flushes continue
  between requests.
- **Operational behavior.** `/health`, `/stats` (with `scheduler`: steps, reservations, page refusals, verify steps,
  corrections), `/server_info`, `/generate`, `/tokenize`, `/kl_capture`, `/kl_force`, `/kl_teacher`, `/nv_verify`,
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
| DeepSeek, one agent | 3 / 3 | 14 | 13 (0) | 175 s | 59 % | 7.0 s |
| DeepSeek, two agents | 2 / 2 | 10 | 10 (0) | 87 s | 68 % | 14.6 s |
| GLM, one agent | 3 / 3 | 21 | 28 (0) | 376 s | 21 % | 9.3 s |
| GLM, two agents | 2 / 2 | 13 | 17 (0) | 217 s | 44 % | 11.4 s |
| GLM `speculation="ngram"`, one agent | 3 / 3 | 23 | 27 (0) | 443 s | 62 % | 10.2 s |

`scheduler.corrections` stayed empty in every run (vLLM's counts and the engine's never differed), with no page
refusals and no dropped drafts. What the benchmark shows about these models on this box:

- **Prefix reuse is bounded by recurrent checkpoints.** Both models have recurrent state, so a conversation resumes
  only from a checkpoint: a chunk end, a page boundary at least 2,048 tokens past the last checkpoint, or a prompt
  whose tail past its last page boundary is short enough to split off cheaply (`engine/last_page.py`: 85 tokens for
  DeepSeek, 63 for GLM). Short agent conversations (all of these stay under 3k tokens) therefore reuse only what their
  earlier prompts' lengths happened to checkpoint: GLM's runs range from 0 % to 70 % per task for that reason alone.
  Splitting every tail would not pay on this box: a staged GLM forward costs ~8.8 s whatever its length, about what
  the re-prefill it would save costs. Long agent sessions get a checkpoint at every 8,192-token chunk end.
- **Concurrency.** DeepSeek has one recurrent-state slot, so vLLM runs one sequence and queues the other; two agents
  still finish in 87 s instead of 113 s because one's prefill and decode overlap the other's tool execution and client
  time. GLM has four slots (`-ambs 4`), so vLLM batches both agents; steps that carry the other agent's 8.8 s prefill
  chunk stall the decode beside it, and two GLM agents took 217 s against 191 s for the same tasks one after the other.
