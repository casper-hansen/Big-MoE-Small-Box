# Image input

Both servers take images in chat messages, on by default (backend option `multimodal=True`). The code is
`bigmoe/multimodal.py`; the servers call it from `/v1/chat/completions` only when a message carries an image block.

```python
box = AutoBox.from_pretrained("deepseek-v4.1-flash")
box.start()
print(box.chat("What does the chart show?", images=["chart.png"]))   # paths, bytes, PIL images or URLs
```

```sh
bigmoe chat --image screenshot.png "What is wrong in this dialog?"
```

Accepted blocks, in any message (user, tool, system) and nested in Anthropic `tool_result` blocks:

| format | block |
|---|---|
| OpenAI chat | `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..." \| "https://..."}}` |
| OpenAI Responses | `{"type": "input_image", "image_url": "data:..."}` |
| Anthropic | `{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}}` or `{"type": "url", "url": ...}` |

`file://` URLs are refused unless the server runs with `BIGMOE_MM_FILES=1` (the servers can listen on 0.0.0.0).
`/v1/models` reports `"input_modalities": ["text", "image"]`, and `/stats` has a `vision` section (encodes, tower loads,
load and encode seconds, VRAM lent, cache hits).

## No cost while no image comes in

The vision towers are small next to the text models but not free: DeepSeek-V4.1's is 400-440 MiB of 6-bit weights
(32 ViT blocks + aligner), GLM-5.3's ~506 MiB (24 blocks + merger). Kept resident they would take ~40 (DeepSeek) or
~54 (GLM) VRAM expert slots, or
the same RAM from the RAM tier, which is decode speed. So nothing of the tower is resident:

- **Weights stay on the SSD**, in the checkpoint they ship in (the slim step only punches out routed experts). A
  request with an image that is not cached loads the tower (exllamav3's batched safetensors reader), encodes every new
  image of the request and unloads it before the request's job is queued. Measured: a load takes 0.26 s from cold NVMe
  (DeepSeek, in isolation) and 0.05 s from the page cache; a 1280x960 image encodes in ~0.2 s (738 tokens on
  DeepSeek, 1,612 on GLM). In the servers a new image costs 0.26-0.34 s of tower load + encode in all (DeepSeek, a 1920x1080 screenshot: 0.34 s;
  0.9 s before the load skipped exllamav3's garbage pass, below). The first
  encode in a process also compiles the tower's kernels; the startup self-test pays that when torch's free VRAM covers
  it (DeepSeek, GLM mode nvme), otherwise the first image does (GLM mode nvme-exact: 1.3 s once). exllamav3's `Model.load`
  starts with a full `gc.collect()` when a model is already resident; in the server's heap that alone took ~0.4 s, so
  the vision load skips it (there is no garbage VRAM to free) and freezes the heap instead.
- **VRAM is borrowed, not reserved.** The encode needs the tower plus its activations (~620 MiB peak for a 1280x960
  image on DeepSeek, ~765 MiB on GLM; estimated beforehand from the checkpoint headers and the patch count). It first uses what torch has free between forward passes (the `EC_RESERVE_GB` headroom the
  prefill activations use: 3 GB for DeepSeek, 1.5 GB for GLM mode nvme). When that is short (large images, GLM's
  smaller headroom) the expert pool lends whole chunks of its VRAM cache (`tiers.Pool.lend`): their experts are written
  back to the RAM tier, the slots leave the CLOCK, the memory goes to torch; after the encode the chunks are
  reallocated and their slots come back empty (`Pool.give_back`). The encode runs when vLLM's scheduler admits the
  request (`vllm/engine.py`: `reserve`), on the EngineCore thread between two steps, so no forward sees the cache shrink.
  `/stats` counts the lends (DeepSeek: `tiers[0].vram.lent`, GLM: `expert_cache.lent`). GLM mode nvme-exact (1 GB
  of headroom), a 1920x1080 screenshot: 4 chunks (1,152 MiB) lent, 123 experts written back, all reallocated, 1,536
  slots before and after.
- **CPU work runs in a thread pool**: base64 decode, URL fetches, PIL decode (EXIF orientation applied), and the model's
  own resize / normalize / patchify (exllamav3's `preprocess`, the reference preprocessing). Other requests keep
  decoding meanwhile; only the GPU encode itself holds the event loop.

The text-only path is the same code as before plus one scan of the messages' content lists: same tokens, same jobs,
same VRAM cache (864 slots for DeepSeek, 1,408 for GLM at 128k, as without image input), same RAM tier. Measured
(2026-11-08, `bigmoe bench`, mode nvme, default options, alternating restarts, two runs each):

| | DeepSeek `multimodal=False` | DeepSeek default | GLM `multimodal=False` | GLM default |
|---|---|---|---|---|
| prefill 1,024 tokens | 165.1 / 163.6 tok/s | 167.2 / 165.7 tok/s | 109.7 / 111.1 tok/s | 110.5 / 110.5 tok/s |
| prefill 8,192 tokens | 658.0 / 657.9 tok/s | 658.5 / 660.3 tok/s | 754.4 / 758.1 tok/s | 755.3 / 759.1 tok/s |
| decode after 1,024 | 16.84 / 17.08 tok/s | 17.30 / 17.03 tok/s | 10.65 / 10.84 tok/s | 10.67 / 10.81 tok/s |
| decode after 8,192 | 19.52 / 19.69 tok/s | 20.18 / 20.85 tok/s | 11.06 / 11.17 tok/s | 11.17 / 11.16 tok/s |

Mode nvme-exact, whose decode is deterministic, gives byte-identical answers to the committed baselines for both
models (DeepSeek `bigmoe verify`: 1,024 and 8,192 tokens; GLM: the same two prompts), at the baselines' speed.

## Content-addressed embeddings: resent images are free

Agents send the whole conversation every turn, images included. Each image is keyed by a hash of its bytes (plus the
checkpoint and the token budget):

- **Stable token ids.** exllamav3 gives an image's embedding rows placeholder token ids (from 10^9 up), allocated per
  embedding. The prefix cache hashes pages by token ids, so a re-encoded image with fresh ids would make every page
  from the image on a miss. The server fixes an image's ids for the process lifetime (`EmbeddingCache.ids`, a few bytes
  per image ever seen; `rebase()` moves a re-encoded embedding onto them), so a follow-up turn finds its prefix, image
  included.
- **Embedding cache**: a RAM LRU (`BIGMOE_MM_CACHE_MB`, 256 MiB: ~34 DeepSeek images at 7.5 MB, ~10 GLM images at
  26 MB) over a directory on the SSD (`mm_cache_mb` option, 2048 MiB, under `~/.cache/bigmoe/mm-cache/<backend>`,
  oldest records trimmed). A hit skips the tower entirely; the disk tier also survives restarts.
- exllamav3 keeps pages after an image out of its *disk* prefix tier (placeholder ids are per process), so across a
  restart the image's prefix is prefilled again; the embedding itself comes from the disk tier.

Measured on DeepSeek, a 10.8k-token system prompt, then a screenshot and four turns about it:

| turn | prompt | cached | prefill |
|---|---|---|---|
| 1 | 11,558 | 0 | 24.3 s |
| 2 | 11,621 | 11,520 | 2.8 s |
| 3 | 11,659 | 11,520 | 3.2 s |
| 4 | 11,677 | 11,520 | 3.3 s |

## DeepSeek-V4.1: exact chunking turned off

exllamav3 1.6.6 sets `mm_exact_chunks` on every DeepSeek-V4-family model with a vision tower. It is meant for
V4-Flash-Vision-Exp, whose image span attends bidirectionally and has to be prefilled as one chunk. V4.1 stays causal
over an image span (`DeepseekV41Model.prepare_inputs` does no chunk check; its Engram layers hash image rows as the pad
token, carried in their state), so any chunking is exact. With the cap on, prefill chunks were cut at both edges of
every image (two extra staged forwards per image), no page-aligned recurrent checkpoint survived past an image, so the
turns above re-prefilled ~3.4k tokens each (15 s instead of 3), and a cached prefix ending inside an image span failed
the request. The dsv41 server clears the cap.

## Token budget

`image_tokens` (backend option, `BIGMOE_MM_MAX_TOKENS`) caps the tokens per image; the default is the model's own:
1024 for DeepSeek-V4.1 (`vision_config.max_image_tokens`; small images are upscaled to `min_pixels`), 8000 for
GLM-5.3 (`processor_config.json`). Image tokens are prompt tokens: they cost prefill like text (a 1280x960 image is 738
tokens on DeepSeek, 1,610 on GLM). Lower budgets cost detail (small text); an agent that sends many screenshots per
conversation may want `-o image_tokens=1024` on GLM.

## Options and environment

| backend option | server environment | default | |
|---|---|---|---|
| `multimodal` | `BIGMOE_MM` | `True` / `1` | `False`: requests with images get a 400 `image_input_not_supported` |
| `image_tokens` | `BIGMOE_MM_MAX_TOKENS` | model's own | tokens per image |
| `mm_cache_mb` | `BIGMOE_MM_DISK_DIR`, `BIGMOE_MM_DISK_MB` | 2048 | embedding cache on the SSD (0: none) |
| | `BIGMOE_MM_CACHE_MB` | 256 | embedding cache in RAM |
| | `BIGMOE_MM_MAX_IMAGES` | 16 | images per request |
| | `BIGMOE_MM_MAX_BYTES` | 20,000,000 | bytes per image |
| | `BIGMOE_MM_FETCH` | 1 | fetch http(s) image URLs |
| | `BIGMOE_MM_FILES` | 0 | accept `file://` URLs |

A tower that fails its startup self-test (a synthetic 640x480 image through the whole load / encode / unload path)
leaves the server text-only with a logged error instead of failing the start.

## Agents (DeepSeek Harness)

A client only sends images to a model it declares as taking them; in DSH that is the model's `input` list in the
profile's `cordis.patch.yml` (`input: [text, image]` under the `local-bigmoe` provider). Checked on 2026-11-08 with both
models, each in a new Web GUI session at effort max: asked to open news.ycombinator.com with the browser plugin
(browse-anything), take a screenshot, look at it with `read_image` and report what only the pixels show, each called
`browser_navigate` -> `browser_take_screenshot` -> `read_image` and answered correctly (orange top bar, the "Y" badge,
the first story's title as rendered). Server side, the screenshot was encoded once (DeepSeek: 930 tokens, 0.30 s of
tower load + encode; GLM: 1,198 tokens, 0.70 s) and `read_image` of the same file was a cache hit (same bytes, same
ids): 4 steps, 1 m 26 s (DeepSeek) and 2 m 18 s (GLM) in all.
