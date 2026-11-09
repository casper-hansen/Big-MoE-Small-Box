# vLLM serving frontend

Both backends now default to `frontend="vllm"`: vLLM 0.33.0 is the serving engine in front of the same exllamav3 1.6.6
and shared expert-offload engine that the native servers used. `frontend="native"` keeps the old server for A/B runs and
as a fallback.

```sh
bigmoe start deepseek-v4.1-flash                         # vLLM frontend
bigmoe start glm-5.3-flash                               # vLLM frontend
bigmoe start deepseek-v4.1-flash -o frontend=native      # previous server
bigmoe start deepseek-v4.1-flash -o frontend_extra_ram_gb=0   # keep the native 56 GiB cgroup cap
bigmoe verify glm-5.3-flash -o frontend=native           # native A/B
```

This is deliberately a **frontend migration, not a kernel migration**. vLLM 0.33.0 ships both architecture families,
but on this box it cannot execute either one: DeepSeek-V4.1's sparse-MLA backends require sm90, sm100 or sm120, and
GLM-5.3's sparse indexer requires DeepGEMM (also sm90/sm100/sm120). The RTX 3090 is sm86. vLLM 0.33.0 also has no EXL3
quantization method. The supported route therefore keeps the verified exllamav3/EXL3 math and nv2 expert tiers instead
of silently requantizing the checkpoints or requiring a different GPU.

## What runs where

`bigmoe.vllm` is an out-of-tree vLLM plugin (`vllm.general_plugins` and `vllm.endpoint_plugins`). `bigmoe.vllm.serve`
translates the backend's existing exllamav3 command line and starts vLLM's OpenAI server.

| process | code | responsibility |
|---|---|---|
| API server | vLLM + `bigmoe.vllm.routes` / `mm` | OpenAI and Anthropic HTTP APIs, chat rendering, tool/reasoning parsing, media loading and preprocessing, detokenization, stop strings, SSE streaming |
| EngineCore | vLLM `BoxScheduler` + `BoxWorker` | admission, per-request state, stop tokens, outputs, metrics; one in-process worker |
| same EngineCore | `bigmoe.vllm.engine.ExlEngine` | one exllamav3 `Generator.iterate()` per vLLM step; EXL3 forwards, DSpark / n-gram speculation, recurrent checkpoints, staged / warm prefill |
| same EngineCore | `bigmoe.engine` | nv2 VRAM CLOCK cache, pinned-RAM tier, O_DIRECT NVMe store, per-layer planner, CPU and x16 lanes |

vLLM allocates no KV cache for these models. Its model classes are attention-free registration stubs (`models.py`), not
alternate implementations of the networks. exllamav3 owns the real paged KV cache and all three prefix-cache tiers.
After every step, `BoxScheduler` takes each request's prefill position, sampled tokens and reused-prefix count from the
engine. This is why chunked prefill, speculative multi-token steps and recurrent checkpoints work without teaching
vLLM the models' attention layouts.

The API server may use every CPU except the model's logical CPU. Only the GPU-feeding EngineCore thread is pinned to
the backend's original main CPU; nv2's controller, CPU lane and NVMe readers keep their existing affinities. Pinning
vLLM's additional I/O threads onto the model CPU is avoided. `frontend_extra_ram_gb` defaults to 4.5 for DeepSeek and 1.5 for GLM and is added to
the systemd memory cap: the separate API/detokenizer process and vLLM's larger loader/runtime footprint otherwise come
out of nv2's RAM expert tier, reducing decode speed. GLM keeps automatic sizing with its larger ceiling. DeepSeek's
loader explicitly runs `gc.collect()` and `malloc_trim()` before sizing, but identical cold starts still produced RAM
tiers from 43.66 to 51.60 GiB. Its vLLM configuration therefore sets `DSV41_NV_RAM_GB` explicitly to 46.0 for `nvme`
and 47.5 for `nvme-exact`, with a 60.5 GiB ceiling to accommodate the worst-case startup heap. This is both faster and
safer than letting allocator luck choose the tier; an explicit user `DSV41_NV_RAM_GB` still wins. Set
`frontend_extra_ram_gb=0` to retain the native cgroup limit with automatic sizing.

## Compatibility

- **Chat and completions.** vLLM's DeepSeek-V4.1 and GLM-4.7 renderers/parsers are used with the checkpoints' own chat
  templates. Requests without a `model`, or with an alias the native single-model server would have ignored, are mapped
  to the served model name. GLM thinking defaults on; DeepSeek thinking defaults off, matching the native servers.
- **Tools and reasoning.** DeepSeek DSML calls and GLM `<tool_call>` blocks are parsed by vLLM, including streaming
  argument deltas. GLM string arguments preserve leading/trailing whitespace under vLLM; the native parser stripped it.
  Preserving it is intentional for file-writing tools.
- **Images.** vLLM loads and validates the image and calls exllamav3's own resize/normalize/patchify code. The EngineCore
  worker then uses the existing nonresident vision tower, VRAM lending and content-addressed RAM/SSD embedding cache.
  A repeated image keeps stable placeholder IDs, so exllamav3 can reuse the prefix through the image. `/v1/models`
  advertises `input_modalities: ["text", "image"]` and `/stats` retains the vision counters.
- **Prefix caching.** The GPU page cache, pinned-RAM tier, disk tier and recurrent checkpoints are exllamav3's, not
  vLLM's prefix cache. `usage.prompt_tokens_details.cached_tokens` reports the engine's actual reused prompt tokens;
  `/stats` also exposes `prefix_cache` occupancy and hit counters. Idle disk flushes continue between requests. A real
  16,384-token prefill flushed to disk, followed by a full server stop/start, restored 32 pages (8,192 tokens) from
  disk; both `alloc_disk_pages` and `usage.prompt_tokens_details.cached_tokens` reported 8,192.
- **Speculation.** DSpark and n-gram drafting stay in exllamav3. A vLLM step can therefore emit several accepted tokens.
  Draft acceptance counters remain under `/stats.draft`.
- **Operational behavior.** `/health`, `/stats`, `/server_info`, `/generate`, `/tokenize`, `/kl_capture`, `/kl_force`,
  `/kl_teacher`, `/nv_verify`, `/nv2_rate` and `/cpu` are available. Streaming responses receive an SSE comment after
  15 silent seconds. Context-length admission reserves min(`max_tokens`, 1024) output tokens plus the drafter window,
  as before; engine-side failures retain their 400/503 distinction where vLLM's error protocol allows it. A CUDA OOM
  before the first token frees the failed forward and retries once with half-size prefill chunks.
- **Token counts.** OpenAI/vLLM `completion_tokens` excludes a stop token. The native servers included it, so totals
  can differ by one even when the visible answer is byte-identical.

The following vLLM-only sampling features are rejected with a 400 rather than accepted and ignored: output/prompt
logprobs, structured outputs (`response_format` / JSON schemas), `allowed_token_ids`, `bad_words`, `n > 1`, and beam
search. Structural tags that vLLM adds internally for native DSML/GLM tool parsing are not treated as user
`response_format` constraints and remain supported. Teacher-forced logprobs remain available through `/generate` with
`return_logprob`, `/kl_teacher`, and the quality tools.

## Multimodal template details

DeepSeek's template already understands image blocks. GLM-5.3's shipped template is text-only and renders a reminder
instead; `serve.py` creates a tokenizer overlay that renders `<|begin_of_image|><|image|><|end_of_image|>` and adds the
native server's `enable_thinking=False` handling. The overlay is keyed by the resolved checkpoint path and template
contents, so same-named or replaced checkpoints cannot inherit stale tokenizer files.

For DeepSeek, one `<｜deepseek_image｜>` expands to `2 + h * (w + 1)` embedding positions. For GLM, the image token expands
to `t * h * w / merge^2` positions between the real start/end tokens. The worker substitutes exllamav3's stable
placeholder IDs and passes the corresponding `MMEmbedding` objects to the job. These positions are prompt tokens and
are included in prefill cost as before.

## Verification

The unit suite covers the scheduler/worker loop with a fake engine, multi-token speculative outputs, engine-side
errors, template overlays, middleware, image-ID splicing, context budgets and launch/runtime wiring. On the GPU, both
models were run through `bigmoe verify` and native A/B checks. Prompt rendering was checked for plain conversations,
system messages, thinking on/off, tools, tool results and multiple text blocks. DSML and GLM tool output was checked
against the native parsers, streaming and nonstreaming.

Exact-mode (`nvme-exact`, no speculative drafting) answers to the same synthetic image were byte-identical between the
native and vLLM servers for both models. The follow-up turns were also identical; GLM reported the same 256 reused
prefix tokens in both servers.

## Measured results (RTX 3090 box, 2026-11-09)

`bigmoe verify`, greedy, 128k context, one run of each frontend on the same day. Native is `frontend="native"`; vLLM is
the default with `frontend_extra_ram_gb=1.5`. Numbers are prefill / decode tok/s. Long answers differ in mode `nvme`
for the same near-exact reason as two native runs; mode `nvme-exact` was byte-identical in every row.

### GLM-5.3-Flash

| prompt tokens | native `nvme` | vLLM `nvme` | native `nvme-exact` | vLLM `nvme-exact` |
|---|---|---|---|---|
| 1,024 | 110.9 / 10.79 | 110.9 / 10.71 | 58.5 / 7.81 | 58.4 / 7.74 |
| 8,192 | 759.5 / 11.28 | 765.3 / 11.16 | 411.0 / 7.99 | 414.4 / 7.91 |
| 32,768 | 821.3 / 11.42 | 821.0 / 11.11 | — | — |
| 120,000 | 824.2 / 11.32 | 817.5 / 11.25 | — | — |

The first migration run was 4–7% slower in `nvme`. Two avoidable causes were found and fixed: the API process and its
I/O threads were allowed to compete with the model thread, and the second process's roughly 1.3 GiB footprint reduced
the RAM expert tier. The final numbers above use the CPU mask and explicit frontend RAM budget described earlier.

### DeepSeek-V4.1-Flash

The 1k/8k native `nvme` column is a fresh back-to-back run on 2026-11-09; the other native numbers are the same boxes'
2026-11-08 verification runs. The vLLM tier is fixed at 46.0 / 47.5 GiB as described above.

| prompt tokens | native `nvme` | vLLM `nvme` | native `nvme-exact` | vLLM `nvme-exact` |
|---|---|---|---|---|
| 1,024 | 163.4 / 17.15 | 171.1 / 16.67 | 31.8 / 10.62 | 33.1 / 10.57 |
| 8,192 | 692.4 / 19.56 | 664.8 / 18.61 | 240.1 / 11.96 | 250.2 / 12.06 |
| 32,768 | 707.2 / 19.76 | 716.7 / 19.07 | — | — |

Every exact answer above is byte-identical. There is a residual near-exact decode gap of about 3% at 1k and 5% at 8k;
32k differs by about 3% as well. Larger RAM tiers were also tested: they can exceed native decode speed but make host
memory pressure depend on the startup heap, so the deterministic native-sized tier is the default rather than a claim
of universally faster serving.

### Output quality

DeepSeek forced decode (`models/dsv41/tools/kl_force.py`, six prompts, 1,110 positions, reference `q/gen_v0.json`)
through vLLM: **99.55% top-1 agreement, mean KL 0.001023, max KL 0.172478**. This matches the native baseline of
about 99.6% top-1 and 0.0011 mean KL. A real streamed write-tool request reassembled to valid JSON with all 60
requested lines and incremental argument deltas. Overlong prompts return 400 with `context_length_exceeded`; disconnect
cleanup, unsupported-option errors, and the OOM retry logic are also covered.

GLM forced decode against `q/glm_ref_exact.json` covered 897 positions (two shorter answers stop at EOS). Native:
96.99% top-1, mean KL 0.008679. vLLM: **97.10% top-1, mean KL 0.013140**. Top-1 is therefore comparable;
KL differs most on the Danish-story prompt and is retained here rather than averaged away. With `speculation="ngram"`,
a repeated-code request produced 73 accepted and 1 rejected draft token over 17 rounds. A real 60-line GLM write-tool
call then streamed 125 argument deltas and reassembled as valid JSON.
