"""Image input for both servers (glm53, dsv41; bigmoe/vllm): OpenAI / Anthropic image blocks -> exllamav3 MMEmbeddings.

The text-only path has to stay exactly as fast as before, so the vision tower costs nothing while no image comes in:

- Not resident. The tower (DeepSeek-V4.1: ~0.4 GiB of 6-bit weights; GLM-5.3: ~0.5 GiB) stays in the checkpoint on
  NVMe. A request with images that are not cached loads it (0.05 s from the page cache, 0.26 s cold from NVMe),
  encodes all of that request's images (~0.2 s for 1280x960), and unloads it before the request is queued. The VRAM expert cache, the RAM
  tier and the decode path are the ones measured without images.
- VRAM for the encode comes from what torch has free between forward passes (the EC_RESERVE_GB headroom the prefill
  activations use); when that is short (GLM's 1.5 GB headroom, large images) the expert pool lends chunks of its cache
  (tiers.Pool.lend: written back to the RAM tier, freed, reallocated and re-enabled right after the encode). The
  executor encodes when vLLM's scheduler admits the request (vllm/engine.py: reserve), on the EngineCore thread between
  two steps, so no forward ever sees the cache shrink.
- CPU work (download, base64 decode, PIL decode, resize / normalize) runs in a thread pool: the decode of other
  requests is not held up by it, only by the GPU encode itself (~0.2 s per 1280x960 image).
- Content-addressed embeddings. Each image is keyed by a hash of its bytes (plus the model and the token budget); its
  placeholder token ids are fixed for the process lifetime, so an agent conversation that sends the same screenshot
  again every turn produces the same token ids and its prefix (image included) is reused from exllamav3's prefix cache
  instead of prefilled again. Embeddings are kept in a RAM LRU (BIGMOE_MM_CACHE_MB, 256) and written to a size-capped
  directory on the SSD (BIGMOE_MM_DISK_DIR / BIGMOE_MM_DISK_MB, 2048), which also survives restarts: a hit there is a
  few ms of NVMe read instead of a tower load + encode.

Environment (set by the backends' launch_spec): BIGMOE_MM (1 = on), BIGMOE_MM_MAX_TOKENS (per-image token budget,
default the model's own), BIGMOE_MM_CACHE_MB, BIGMOE_MM_DISK_DIR, BIGMOE_MM_DISK_MB, BIGMOE_MM_MAX_IMAGES (16 per
request), BIGMOE_MM_MAX_BYTES (20 MB per image), BIGMOE_MM_FETCH (1: http(s) image URLs are fetched), BIGMOE_MM_FILES
(0: file:// URLs are refused).

Stdlib only at import time (torch, PIL and exllamav3 are imported where they are used), so the parsing and caching
logic is testable without the GPU stack.
"""
import asyncio
import base64
import binascii
import collections
import hashlib
import io
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from .serving import RequestError

IMAGE_TYPES = ("image_url", "image", "input_image")
FIRST_MM_ID = 1_000_000_000          # exllamav3 tokenizer.mm_embedding.FIRST_MM_EMBEDDING_INDEX
MiB = 1024 ** 2


def env_int(name, default):
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else default


def enabled() -> bool:
    return os.environ.get("BIGMOE_MM", "1") not in ("0", "", "false", "False")


# --------------------------------------------------------------------------------------------------------------------
# messages: find the image blocks, put the embeddings' text aliases in their place
# --------------------------------------------------------------------------------------------------------------------
def image_source(block):
    """The image a content block refers to, as a string (data: / http(s): / file: URL) or None if it is no image.
    OpenAI chat: {"type": "image_url", "image_url": {"url": ...}} (or the URL as a plain string); Responses API:
    {"type": "input_image", "image_url": ...}; Anthropic: {"type": "image", "source": {"type": "base64", "media_type",
    "data"} | {"type": "url", "url"}}."""
    if not isinstance(block, dict) or block.get("type") not in IMAGE_TYPES:
        return None
    t = block["type"]
    if t in ("image_url", "input_image"):
        u = block.get("image_url")
        if isinstance(u, dict):
            u = u.get("url")
        if t == "input_image" and u is None and block.get("file_id"):
            raise RequestError(400, "input_image file_id references are not supported; send the image as a data URL",
                               param="messages")
        if not isinstance(u, str) or not u:
            raise RequestError(400, f"{t} block without a URL", param="messages")
        return u
    src = block.get("source")
    if isinstance(src, dict):
        if src.get("type") == "base64" and isinstance(src.get("data"), str):
            return f"data:{src.get('media_type') or 'image/png'};base64,{src['data']}"
        if src.get("type") == "url" and isinstance(src.get("url"), str):
            return src["url"]
    if isinstance(block.get("image_url"), (str, dict)):    # {"type": "image", "image_url": ...} (some clients)
        return image_source(dict(block, type="image_url"))
    if isinstance(block.get("image"), str):                  # {"type": "image", "image": "<url or data URL>"}
        return block["image"]
    raise RequestError(400, "image block without a usable source (base64 data or a URL)", param="messages")


def _walk(content, fn):
    """content with every image block replaced by fn(block) (a new list; nested `content` lists of blocks such as
    Anthropic tool_result blocks included). Strings and other blocks are returned as they are."""
    if not isinstance(content, list):
        return content
    out = []
    for b in content:
        if image_source(b) is not None:
            out.append(fn(b))
        elif isinstance(b, dict) and isinstance(b.get("content"), list):
            out.append(dict(b, content=_walk(b["content"], fn)))
        else:
            out.append(b)
    return out


def image_sources(messages) -> list:
    """Every image of a chat request in message / block order."""
    found = []

    def take(b):
        found.append(image_source(b))
        return b
    for m in messages or ():
        if isinstance(m, dict):
            _walk(m.get("content"), take)
    return found


def has_images(messages) -> bool:
    """Cheap pre-check that runs on every chat request: a text-only request never gets further than this."""
    for m in messages or ():
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, list):
            for b in c:
                if isinstance(b, dict) and (b.get("type") in IMAGE_TYPES or
                                            (isinstance(b.get("content"), list) and has_images([b]))):
                    return True
    return False


def with_aliases(messages, aliases) -> list:
    """The messages with the i-th image block replaced by a text block holding aliases[i]. Both chat templates render
    text blocks where they rendered image blocks (DeepSeek: parts joined by blank lines, the image placeholder token
    becoming the alias; GLM: concatenated), so the alias lands exactly where the template puts the image."""
    it = iter(aliases)
    out = []
    for m in messages:
        if isinstance(m, dict) and isinstance(m.get("content"), list):
            m = dict(m, content=_walk(m["content"], lambda b: {"type": "text", "text": next(it)}))
        out.append(m)
    return out


# --------------------------------------------------------------------------------------------------------------------
# image bytes
# --------------------------------------------------------------------------------------------------------------------
def decode_data_url(url: str, max_bytes: int) -> bytes:
    head, sep, payload = url.partition(",")
    if not sep:
        raise RequestError(400, "malformed data URL (no ',')", param="messages")
    if len(payload) > max_bytes * 4 // 3 + 4:
        raise RequestError(400, f"image larger than {max_bytes} bytes", param="messages")
    try:
        if head.endswith(";base64"):
            data = base64.b64decode(payload, validate=False)
        else:
            from urllib.parse import unquote_to_bytes
            data = unquote_to_bytes(payload)
    except (binascii.Error, ValueError) as e:
        raise RequestError(400, f"data URL is not valid base64 ({e})", param="messages")
    return data


def fetch_bytes(src: str, max_bytes: int, timeout: float = 20.0) -> bytes:
    """The image's bytes (runs in the worker pool)."""
    if src.startswith("data:"):
        data = decode_data_url(src, max_bytes)
    elif src.startswith(("http://", "https://")):
        if os.environ.get("BIGMOE_MM_FETCH", "1") == "0":
            raise RequestError(400, "this server does not fetch image URLs; send the image as a data URL",
                               param="messages")
        import httpx
        try:
            with httpx.stream("GET", src, timeout=timeout, follow_redirects=True) as r:
                if r.status_code >= 400:
                    raise RequestError(400, f"fetching the image URL failed: HTTP {r.status_code}", param="messages")
                buf = bytearray()
                for chunk in r.iter_bytes():
                    buf += chunk
                    if len(buf) > max_bytes:
                        raise RequestError(400, f"image larger than {max_bytes} bytes", param="messages")
                data = bytes(buf)
        except httpx.HTTPError as e:
            raise RequestError(400, f"fetching the image URL failed ({type(e).__name__}: {e})", param="messages")
    elif src.startswith("file://"):
        if os.environ.get("BIGMOE_MM_FILES", "0") != "1":
            raise RequestError(400, "file:// image URLs are disabled on this server (BIGMOE_MM_FILES=1 enables them)",
                               param="messages")
        path = src[len("file://"):]
        if os.path.getsize(path) > max_bytes:
            raise RequestError(400, f"image larger than {max_bytes} bytes", param="messages")
        with open(path, "rb") as f:
            data = f.read()
    else:
        raise RequestError(400, f"unsupported image URL scheme: {src[:32]!r} (use a data: or http(s): URL)",
                           param="messages")
    if not data:
        raise RequestError(400, "empty image", param="messages")
    if len(data) > max_bytes:
        raise RequestError(400, f"image larger than {max_bytes} bytes", param="messages")
    return data


def open_image(data: bytes):
    """PIL image, decoded, EXIF orientation applied, first frame of an animation (runs in the worker pool)."""
    from PIL import Image, ImageOps
    Image.MAX_IMAGE_PIXELS = 64 * 1024 * 1024
    try:
        im = Image.open(io.BytesIO(data))
        im.seek(0)
        im.load()
        im = ImageOps.exif_transpose(im)
    except Image.DecompressionBombError as e:
        raise RequestError(400, f"image too large ({e})", param="messages")
    except Exception as e:
        raise RequestError(400, f"cannot decode the image ({type(e).__name__}: {e})", param="messages")
    if im.width < 1 or im.height < 1:
        raise RequestError(400, "image has no pixels", param="messages")
    return im


# --------------------------------------------------------------------------------------------------------------------
# embedding cache: content key -> MMEmbedding, ids fixed per key for the process lifetime
# --------------------------------------------------------------------------------------------------------------------
def rebase(mme, first_index: int):
    """Move an MMEmbedding's placeholder ids to start at first_index (the ids an earlier embedding of the same image
    had), so the prompt's token ids, and with them the prefix cache's page hashes, are the same as before."""
    old = mme.first_index
    if old == first_index:
        return mme
    ts = mme.token_string.clone()
    m = (ts >= old) & (ts < old + mme.mm_length)
    ts[m] += first_index - old
    if mme.text_alias == f"<$EMB_{old}$>":
        mme.text_alias = f"<$EMB_{first_index}$>"
    mme.first_index, mme.last_index = first_index, first_index + mme.mm_length
    mme.token_string, mme.token_list = ts, ts[0].tolist()
    return mme


def mme_nbytes(mme) -> int:
    n = mme.embeddings.numel() * mme.embeddings.element_size()
    for d in (mme.deepstack_embeddings or ()):
        n += d.numel() * d.element_size()
    return n


def _to_record(mme) -> dict:
    """What it takes to rebuild an MMEmbedding: the token string with -1 at the embedding rows."""
    ts = mme.token_string.clone()
    ts[(ts >= mme.first_index) & (ts < mme.last_index)] = -1
    return {"embeddings": mme.embeddings.contiguous(), "token_string": ts,
            "deepstack": list(mme.deepstack_embeddings) if mme.deepstack_embeddings is not None else None,
            "meta": json.dumps({"grid_thw": list(mme.grid_thw) if mme.grid_thw is not None else None,
                                "mrope_merge_size": mme.mrope_merge_size, "align": mme.align,
                                "align_phase": mme.align_phase, "align_lead": mme.align_lead,
                                "metadata": {k: v for k, v in mme.metadata.items()
                                             if isinstance(v, (int, float, str, list, tuple, type(None)))}})}


def _from_record(rec, first_index=None):
    from exllamav3.tokenizer import MMEmbedding
    meta = json.loads(rec["meta"])
    mme = MMEmbedding(embeddings=rec["embeddings"], token_string=rec["token_string"].clone(),
                      deepstack_embeddings=rec.get("deepstack"),
                      grid_thw=tuple(meta["grid_thw"]) if meta["grid_thw"] is not None else None,
                      mrope_merge_size=meta["mrope_merge_size"], align=meta["align"], align_phase=meta["align_phase"],
                      align_lead=meta["align_lead"])
    mme.metadata.update(meta["metadata"])
    return rebase(mme, first_index) if first_index is not None else mme


class EmbeddingCache:
    """RAM LRU (bytes budget) over a size-capped directory of records on the SSD, keyed by content key. ids[key] (the
    first placeholder id of the key's embedding) lives as long as the process, a few bytes per image ever seen."""

    def __init__(self, ram_bytes: int, disk_dir: str = None, disk_bytes: int = 0):
        self.ram_bytes, self.disk_dir, self.disk_bytes = int(ram_bytes), disk_dir, int(disk_bytes)
        self.lru = collections.OrderedDict()      # key -> MMEmbedding
        self.size = 0
        self.ids = {}
        self.stats = {"ram_hits": 0, "disk_hits": 0, "misses": 0, "disk_writes": 0, "disk_errors": 0}
        self._disk_lock = threading.Lock()
        if disk_dir and disk_bytes > 0:
            os.makedirs(disk_dir, exist_ok=True)
        else:
            self.disk_dir = None

    def _path(self, key):
        return os.path.join(self.disk_dir, key + ".pt")

    def get(self, key):
        """The cached MMEmbedding (RAM, else disk) or None. Returned embeddings carry the key's fixed ids."""
        mme = self.lru.get(key)
        if mme is not None:
            self.lru.move_to_end(key)
            self.stats["ram_hits"] += 1
            return mme
        if self.disk_dir:
            p = self._path(key)
            if os.path.exists(p):
                try:
                    import torch
                    rec = torch.load(p, map_location="cpu", weights_only=True)
                    mme = _from_record(rec, self.ids.get(key))
                    self.ids.setdefault(key, mme.first_index)
                    os.utime(p)                    # LRU order on disk: mtime
                    self._put_ram(key, mme)
                    self.stats["disk_hits"] += 1
                    return mme
                except Exception as e:             # a damaged file is a miss
                    print(f" !! mm cache: dropping unreadable {p} ({type(e).__name__}: {e})", flush=True)
                    self.stats["disk_errors"] += 1
                    try:
                        os.unlink(p)
                    except OSError:
                        pass
        self.stats["misses"] += 1
        return None

    def put(self, key, mme, write_disk=True):
        """Store a freshly encoded embedding; returns it rebased onto the key's ids if the key was seen before."""
        if key in self.ids:
            rebase(mme, self.ids[key])
        else:
            self.ids[key] = mme.first_index
        self._put_ram(key, mme)
        return mme

    def _put_ram(self, key, mme):
        if key in self.lru:
            self.size -= mme_nbytes(self.lru.pop(key))
        n = mme_nbytes(mme)
        if n > self.ram_bytes:
            return
        self.lru[key] = mme
        self.size += n
        while self.size > self.ram_bytes and self.lru:
            _, old = self.lru.popitem(last=False)
            self.size -= mme_nbytes(old)

    def write_disk(self, key, mme):
        """Persist one record (worker pool) and trim the directory to disk_bytes, oldest mtime first."""
        if not self.disk_dir:
            return
        import torch
        p = self._path(key)
        if os.path.exists(p):
            return
        tmp = f"{p}.{os.getpid()}.{threading.get_ident()}.tmp"
        try:
            torch.save(_to_record(mme), tmp)
            os.replace(tmp, p)
            self.stats["disk_writes"] += 1
        except OSError as e:
            self.stats["disk_errors"] += 1
            print(f" !! mm cache: cannot write {p} ({e})", flush=True)
            try:
                os.unlink(tmp)
            except OSError:
                pass
            return
        with self._disk_lock:
            files = []
            for f in os.listdir(self.disk_dir):
                if f.endswith(".pt"):
                    try:
                        st = os.stat(os.path.join(self.disk_dir, f))
                        files.append((st.st_mtime, st.st_size, f))
                    except OSError:
                        pass
            total = sum(s for _, s, _ in files)
            for _, s, f in sorted(files):
                if total <= self.disk_bytes:
                    break
                try:
                    os.unlink(os.path.join(self.disk_dir, f))
                    total -= s
                except OSError:
                    pass

    def summary(self):
        return dict(self.stats, ram_entries=len(self.lru), ram_mib=round(self.size / MiB, 1),
                    ram_budget_mib=round(self.ram_bytes / MiB), keys=len(self.ids), disk_dir=self.disk_dir)


# --------------------------------------------------------------------------------------------------------------------
# the vision tower, loaded per encode batch
# --------------------------------------------------------------------------------------------------------------------
class _Sized:
    """Stands in for the PIL image where the tower only reads its size (embedding metadata)."""

    def __init__(self, size):
        self.size = tuple(size)


class Vision:
    """On-demand vision tower + embedding cache for one server."""

    def __init__(self, config, tokenizer, device="cuda:0", log=print):
        import torch
        from exllamav3 import Model
        self.torch = torch
        self.config, self.tok, self.device, self.log = config, tokenizer, torch.device(device), log
        self.arch = config.architecture
        self.max_tokens = self._apply_token_budget(env_int("BIGMOE_MM_MAX_TOKENS", 0))
        # the tower's modules, constructed but not loaded (no tensors): preprocess() only reads the config
        self.tower = Model.from_config(config, component="vision")
        self.cache = EmbeddingCache(env_int("BIGMOE_MM_CACHE_MB", 256) * MiB, os.environ.get("BIGMOE_MM_DISK_DIR"),
                                    env_int("BIGMOE_MM_DISK_MB", 2048) * MiB)
        self.max_images = env_int("BIGMOE_MM_MAX_IMAGES", 16)
        self.max_bytes = env_int("BIGMOE_MM_MAX_BYTES", 20 * 1000 * 1000)
        self.pool = ThreadPoolExecutor(max_workers=max(1, env_int("BIGMOE_MM_THREADS", 2)),
                                       thread_name_prefix="bigmoe-mm")
        # VRAM model of an encode: weights (from the checkpoint headers, then measured) + activations per vision patch
        # (conservative until an encode measures it)
        self.weight_bytes = self._weights_on_disk() or 1024 * MiB
        self.patch_bytes = 64 * 1024
        self.patch_measured = False
        self.margin = 256 * MiB
        self.stats = {"requests": 0, "images": 0, "encoded": 0, "tower_loads": 0, "load_s": 0.0, "encode_s": 0.0,
                      "lent_mib": 0, "oom_retries": 0, "image_tokens": 0}
        # key part that changes the embedding: model, quant revision, token budget
        self.fingerprint = hashlib.blake2b(json.dumps([self.arch, str(config.directory), self.max_tokens]).encode(),
                                           digest_size=8).hexdigest()

    def _weights_on_disk(self) -> int:
        """Bytes of the tower's tensors in the checkpoint (+10 %), before anything is loaded."""
        try:
            names, total = set(), 0
            for m in self.tower.modules:
                for k, meta in self.config.stc.list_tensors(m.key).items():
                    if k not in names:
                        names.add(k)
                        total += int(meta["n_bytes"])
            return int(total * 1.1)
        except Exception:
            return 0

    def _apply_token_budget(self, n):
        """Per-image token budget: BIGMOE_MM_MAX_TOKENS overrides the model's own (0 keeps it)."""
        c = self.config
        pp = getattr(c, "vision_pp", None)
        if pp is not None and hasattr(pp, "max_image_tokens"):        # GLM-5.3 (Glm4V preprocessing)
            if n:
                pp.max_image_tokens = max(int(getattr(pp, "min_image_tokens", 16) or 16), n)
            return int(pp.max_image_tokens)
        v = getattr(c, "vision", None)
        if v is not None and hasattr(v, "max_n_token"):                # DeepSeek-V4.1
            if n:
                v.max_n_token = max(4, n)
            return int(v.max_n_token)
        return n

    def key(self, data: bytes) -> str:
        return hashlib.blake2b(data, digest_size=16, person=b"bigmoe-mm-v1").hexdigest() + "-" + self.fingerprint

    # ---- CPU side (worker pool) ----
    def _prepare(self, src, force=False):
        """(key, PIL image or None if cached, preprocessed patches or None)"""
        data = fetch_bytes(src, self.max_bytes)
        key = self.key(data)
        if not force and (key in self.cache.lru or (self.cache.disk_dir and os.path.exists(self.cache._path(key)))):
            return key, None, None
        im = open_image(data)
        # the class's method: _encode_now shadows the instance attribute while it encodes (on the event loop, while
        # this runs in a worker thread for another request)
        return key, im, type(self.tower).preprocess(self.tower, im)

    # ---- GPU side (event loop, between generator steps) ----
    def _load_tower(self):
        """Model.load without its garbage pass: with the text model resident exllamav3 runs free_mem() (a full
        gc.collect() + empty_cache) first, to return VRAM held by an earlier model's garbage. There is none here, and in
        the server's heap that collection alone took ~0.4 s, longer than reading the tower from NVMe. The heap is frozen
        for the load, as exllamav3 itself does on a first load."""
        import gc
        import exllamav3.model.model as xm
        orig, xm.free_mem = xm.free_mem, (lambda: None)
        froze = gc.get_freeze_count() == 0
        if froze:
            gc.freeze()
        try:
            self.tower.load(str(self.device))
        finally:
            xm.free_mem = orig
            if froze:
                gc.unfreeze()

    def _n_patches(self, prepped):
        x = prepped[0]
        return int(x.shape[0]) if hasattr(x, "shape") else 0

    def _encode_now(self, todo, measure=False):
        """Load the tower, encode every (key, image, prepped) of a request, unload it. Lends VRAM from the expert pool
        when torch's free memory is short. Runs synchronously: no generator step can interleave. measure (the startup
        self-test only: it resets torch's peak counter, which /stats reports) sizes the activation estimate."""
        torch = self.torch
        tiers = None
        if os.environ.get("BIGMOE_ENGINE_PREFIX"):
            from .engine import tiers
        need = self.weight_bytes + max(self._n_patches(p) for _, _, p in todo) * self.patch_bytes + self.margin
        lent, out = [], []
        t0 = time.perf_counter()
        try:
            for attempt in (0, 1):
                torch.cuda.synchronize(self.device)
                free = torch.cuda.mem_get_info(self.device)[0] + torch.cuda.memory_reserved(self.device) \
                    - torch.cuda.memory_allocated(self.device)
                short = need - free if attempt == 0 else need     # the retry lends the whole plan on top
                if short > 0 and tiers is not None:
                    for p in tiers.POOLS:
                        took = p.lend(short)
                        lent.append((p, took))
                        self.stats["lent_mib"] += round(len(took) * p.spc * p.rec / MiB)
                base = torch.cuda.memory_allocated(self.device)
                if measure:
                    torch.cuda.reset_peak_memory_stats(self.device)
                peak0 = torch.cuda.max_memory_allocated(self.device)
                try:
                    tl = time.perf_counter()
                    self._load_tower()
                    torch.cuda.synchronize(self.device)
                    self.stats["tower_loads"] += 1
                    self.stats["load_s"] += time.perf_counter() - tl
                    wbytes = torch.cuda.memory_allocated(self.device) - base
                    out = []
                    for key, im, prepped in todo:
                        self.tower.preprocess = lambda *_a, _p=prepped, **_k: _p   # done in the worker pool
                        try:
                            mme = self.tower.get_image_embeddings(self.tok, im)
                        finally:
                            del self.tower.preprocess
                        out.append((key, mme))
                    torch.cuda.synchronize(self.device)
                    self.weight_bytes = max(wbytes, 64 * MiB)
                    peak1 = torch.cuda.max_memory_allocated(self.device)
                    # the process-wide peak counter (reported by /stats) is reset only by the startup self-test; an
                    # encode that raised it has its own peak there
                    if measure or (not self.patch_measured and peak1 > peak0):
                        n = max(1, max(self._n_patches(p) for _, _, p in todo))
                        self.patch_bytes = max(int((peak1 - base - wbytes) / n * 1.5), 4096)
                        self.patch_measured = True
                    break
                except torch.OutOfMemoryError as e:
                    if attempt:
                        raise RequestError(503, f"not enough GPU memory to encode the image(s) ({e})",
                                           type="server_error", code="server_overloaded")
                    self.stats["oom_retries"] += 1
                    self.patch_bytes *= 2
                    self.log(f" !! vision encode ran out of GPU memory ({need / MiB:.0f} MiB planned); lending more "
                             f"VRAM from the expert cache and retrying")
                finally:
                    self.tower.unload()
        finally:
            if any(took for _, took in lent):
                for p, took in lent:
                    p.give_back(took)      # empties torch's cache first: the chunks are big contiguous blocks
        self.stats["encode_s"] += time.perf_counter() - t0
        return out

    async def prepare(self, messages):
        """(messages with every image block replaced by its embedding's alias, [MMEmbedding] in prompt order, info)"""
        srcs = image_sources(messages)
        if not srcs:
            return messages, [], None
        if len(srcs) > self.max_images:
            raise RequestError(400, f"too many images in one request ({len(srcs)}; this server takes at most "
                                    f"{self.max_images})", param="messages")
        t0 = time.perf_counter()
        loop = asyncio.get_running_loop()
        prepped = await asyncio.gather(*(loop.run_in_executor(self.pool, self._prepare, s) for s in srcs))
        t_cpu = time.perf_counter() - t0
        embs, todo = {}, []
        for i, (key, im, pp) in enumerate(prepped):
            if key in embs or any(k == key for k, _, _ in todo):
                continue
            mme = None if im is not None else self.cache.get(key)
            if mme is not None:
                embs[key] = mme
                continue
            if im is None:      # evicted from RAM / gone from disk since _prepare looked: decode it after all
                key, im, pp = await loop.run_in_executor(self.pool, self._prepare, srcs[i], True)
            todo.append((key, im, pp))
        t1 = time.perf_counter()
        if todo:
            for key, mme in self._encode_now(todo):
                embs[key] = self.cache.put(key, mme)
                self.stats["encoded"] += 1
                if self.cache.disk_dir:
                    self.pool.submit(self.cache.write_disk, key, mme)
        t_gpu = time.perf_counter() - t1
        order = [embs[k] for k, _, _ in prepped]
        uniq = list({id(e): e for e in order}.values())
        n_tok = sum(len(e.token_list) for e in order)
        self.stats["requests"] += 1
        self.stats["images"] += len(order)
        self.stats["image_tokens"] += n_tok
        info = {"images": len(order), "cached": len(order) - len(todo), "encoded": len(todo), "image_tokens": n_tok,
                "prepare_ms": round(t_cpu * 1e3, 1), "encode_ms": round(t_gpu * 1e3, 1)}
        self.log(f" -- images: {len(order)} ({len(order) - len(todo)} cached), {n_tok} tokens, decode/resize "
                 f"{t_cpu:.2f} s, tower load + encode {t_gpu:.2f} s [{time.strftime('%H:%M:%S')}]")
        return with_aliases(messages, [e.text_alias for e in order]), uniq, info

    def embed_prepped(self, items):
        """[MMEmbedding] for [(content key, (width, height) of the original image, preprocessed tensors)] that the
        caller decoded and preprocessed already (the vLLM processor, bigmoe/vllm/mm.py, runs preprocess() in the API
        server). Cached embeddings (RAM, then disk) skip the tower; the rest are encoded in one tower load. Each key keeps
        its placeholder ids for the process lifetime, as in prepare()."""
        t0 = time.perf_counter()
        embs, todo = {}, []
        for key, size, prepped in items:
            if key in embs or any(k == key for k, _, _ in todo):
                continue
            mme = self.cache.get(key)
            if mme is not None:
                embs[key] = mme
            else:
                todo.append((key, _Sized(size), prepped))
        if todo:
            for key, mme in self._encode_now(todo):
                embs[key] = self.cache.put(key, mme)
                self.stats["encoded"] += 1
                if self.cache.disk_dir:
                    self.pool.submit(self.cache.write_disk, key, mme)
        order = [embs[k] for k, _, _ in items]
        n_tok = sum(len(e.token_list) for e in order)
        self.stats["requests"] += 1
        self.stats["images"] += len(order)
        self.stats["image_tokens"] += n_tok
        self.log(f" -- images: {len(order)} ({len(order) - len(todo)} cached), {n_tok} tokens, tower load + encode "
                 f"{time.perf_counter() - t0:.2f} s [{time.strftime('%H:%M:%S')}]")
        return order

    def self_test(self):
        """Startup check of the whole path on a synthetic image, while torch's free VRAM covers it (no lending): loads
        the tower once (compiles its kernels and measures its memory), encodes, unloads. Returns a log line."""
        from PIL import Image, ImageDraw
        im = Image.new("RGB", (640, 480), (235, 235, 235))
        d = ImageDraw.Draw(im)
        d.rectangle([40, 40, 300, 240], fill=(200, 40, 40))
        d.ellipse([340, 160, 600, 440], fill=(40, 70, 200))
        prepped = type(self.tower).preprocess(self.tower, im)
        torch = self.torch
        free = torch.cuda.mem_get_info(self.device)[0] + torch.cuda.memory_reserved(self.device) \
            - torch.cuda.memory_allocated(self.device)
        need = self.weight_bytes + self._n_patches(prepped) * self.patch_bytes + self.margin
        if free < need:
            return f"vision tower: self-test skipped ({free / MiB:.0f} MiB free < {need / MiB:.0f} MiB)"
        t = time.perf_counter()
        (_, mme), = self._encode_now([("selftest", im, prepped)], measure=True)
        torch.cuda.empty_cache()
        assert mme.embeddings.shape[0] > 0 and bool(torch.isfinite(mme.embeddings.float()).all()), "bad embeddings"
        for k in self.stats:      # the self-test is not traffic
            self.stats[k] = 0 if isinstance(self.stats[k], int) else 0.0
        return (f"vision tower: {self.weight_bytes / MiB:.0f} MiB of weights, loaded per request from "
                f"{self.config.directory}; self-test 640x480 -> {len(mme.token_list)} tokens in "
                f"{time.perf_counter() - t:.2f} s; per-image budget {self.max_tokens} tokens")

    def summary(self):
        s = dict(self.stats)
        s["load_s"], s["encode_s"] = round(s["load_s"], 3), round(s["encode_s"], 3)
        return dict(s, max_image_tokens=self.max_tokens, resident=False, weight_mib=round(self.weight_bytes / MiB),
                    cache=self.cache.summary())


def create(config, tokenizer, log=print):
    """The server's Vision, or None when multimodal input is off (BIGMOE_MM=0) or the checkpoint has no vision tower.
    A tower that fails its startup self-test leaves the server text-only (logged) instead of failing the start."""
    if not enabled():
        log(" -- multimodal: off (BIGMOE_MM=0); requests with images get a 400")
        return None
    if "vision" not in getattr(config, "model_classes", {}) or getattr(config, "vision", None) is None:
        log(" -- multimodal: the checkpoint has no vision tower; text only")
        return None
    try:
        v = Vision(config, tokenizer, log=log)
        log(" -- multimodal: " + v.self_test())
        return v
    except Exception as e:
        import traceback
        traceback.print_exc()
        log(f" !! multimodal: the vision tower failed its self-test ({type(e).__name__}: {e}); serving text only")
        return None


def not_supported() -> RequestError:
    return RequestError(400, "this server was started without image input (multimodal off or no vision tower); "
                             "send text only", param="messages", code="image_input_not_supported")


# --------------------------------------------------------------------------------------------------------------------
# what the servers call
# --------------------------------------------------------------------------------------------------------------------
async def prepare_chat(vision, messages):
    """(messages for the chat template, [MMEmbedding]). Text-only requests come back unchanged after one scan of the
    content lists; images need the server's Vision (else a 400)."""
    if not has_images(messages):
        return messages, []
    if vision is None:
        raise not_supported()
    messages, embs, _info = await vision.prepare(messages)
    return messages, embs


def encode_prompt(tok, prompt: str, embeddings):
    """(token ids, the embeddings whose alias is in the prompt). Without embeddings this is the text-only encode."""
    if not embeddings:
        return tok.encode(prompt, encode_special_tokens=True), []
    used = [e for e in embeddings if e.text_alias in prompt]
    return tok.encode(prompt, encode_special_tokens=True, embeddings=used), used


def text_of(content) -> str:
    """Plain text of a message content (string or block list) for the servers' built-in fallback templates."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text") or "" if isinstance(b, dict) else str(b) for b in content
                       if isinstance(b, str) or (isinstance(b, dict) and b.get("type") in ("text", "input_text")))
    return ""


def model_card_fields(vision) -> dict:
    """/v1/models extras: the input modalities this server takes (OpenRouter-style), so clients can tell."""
    mods = ["text", "image"] if vision is not None else ["text"]
    return {"input_modalities": mods, "capabilities": {"vision": vision is not None}}
