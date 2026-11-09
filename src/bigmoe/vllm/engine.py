"""The compute behind vLLM: exllamav3's Generator over the shared expert engine, one Generator.iterate() per vLLM step.

ExlEngine loads exactly what the native servers loaded (models/<backend>/engine/server.py: model_init.init with the
same exllamav3 arguments, the shared engine's tiers, the lazy vision tower, DSpark / n-gram drafting) and builds the
same Generator, so the prefill chunking, prefix reuse (GPU pages, RAM and disk tiers, recurrent checkpoints), staged /
warm prefill and speculative decoding are the measured ones. vLLM's requests become exllamav3 Jobs (identifier = the
vLLM request id); step() runs one iterate() and reports, per request, the tokens sampled since the last report and the
engine's progress (Progress), which BoxScheduler hands to vLLM's request state.

What the native servers did around a job is here too: the EOS set, an out-of-memory failure before the first token
re-run once with half-size prefill chunks (serving.RetryingJob), a per-request log line, draft statistics, idle disk
flushes. Stdlib + torch only at import time; exllamav3 is imported in ExlEngine.load().
"""
import argparse
import dataclasses
import json
import os
import time

from .. import serving

PAGE_SIZE = 256


@dataclasses.dataclass
class Progress:
    """One request's state in the engine after a step. computed: tokens of the request's sequence whose K/V is in the
    cache (vLLM's num_computed_tokens), i.e. the prefill position, or all but the last sampled token once decoding.
    cached: prompt tokens reused from the prefix cache (set once the prefill is done). finished: the engine ended the
    request on its own ("length": the cache is full, "error": a failed job), with error the message."""
    computed: int
    cached: int | None = None
    finished: str | None = None
    error: str | None = None
    status: int = 500            # HTTP status of the error (serving.RequestError): 400 = the request's own fault


@dataclasses.dataclass
class Request:
    req_id: str
    prompt: list
    params: object                       # vllm SamplingParams
    mm: list = dataclasses.field(default_factory=list)
    force: list | None = None
    job: object = None
    started: bool = False                # first token sampled
    retried: bool = False
    restore_chunk: int | None = None
    new_tokens: list = dataclasses.field(default_factory=list)
    n_out: int = 0
    finished: str | None = None
    error: str | None = None
    status: int = 500
    t_add: float = dataclasses.field(default_factory=time.time)
    cached: int | None = None


def _cpus(spec: str) -> list:
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def pin_unpinned_threads(cpus: set) -> int:
    """Pin the calling (GPU-feeding) thread to `cpus`. vLLM's I/O threads stay with the OS: pinning all of them onto
    the model's single logical CPU adds scheduler work to every decode step. The shared engine's own threads keep the
    affinities they selected during load; the native servers likewise pinned only their main thread."""
    import threading
    os.sched_setaffinity(threading.get_native_id(), cpus)
    return 1


def validate_sampling_params(sp):
    """Refuse options that the exllamav3 runner cannot enforce. vLLM would otherwise accept them and either ignore
    the constraint or fail while serializing the answer. The native servers did not implement these either; a clear
    400 is preferable to a plausible but unconstrained answer."""
    unsupported = []
    for attr in ("logprobs", "prompt_logprobs", "prompt_token_id_logprobs"):
        if getattr(sp, attr, None) is not None:
            unsupported.append(attr)
    structured = getattr(sp, "structured_outputs", None)
    # vLLM's own tool parsers add structural_tag constraints for native DSML / GLM tool calls. Those are input to its
    # streaming parser, not an OpenAI response_format request, and must remain allowed. JSON/regex/grammar constraints
    # are the ones this runner cannot enforce during sampling.
    if structured is not None and not structured.all_non_structural_tag_constraints_none():
        unsupported.append("structured_outputs / response_format")
    if getattr(sp, "allowed_token_ids", None):
        unsupported.append("allowed_token_ids")
    if getattr(sp, "bad_words", None):
        unsupported.append("bad_words")
    if int(getattr(sp, "n", 1) or 1) != 1:
        unsupported.append("n > 1")
    if getattr(sp, "use_beam_search", False):
        unsupported.append("beam search")
    if unsupported:
        raise serving.RequestError(400, "the exllamav3 runner does not support " + ", ".join(unsupported)
                                   + "; omit these parameters (teacher-forced logprobs remain available at /generate "
                                     "and /kl_teacher)", param="logprobs" if "logprobs" in unsupported else None)


def parse_exl_args(argv: list):
    """The native servers' command line (exllamav3's model_init arguments + the server's own)."""
    from exllamav3 import model_init
    ap = argparse.ArgumentParser(prog="bigmoe-exl")
    model_init.add_args(ap, cache=True, add_draft_model_args=True, default_chunk_size=2048)
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--max-batch-size", type=int, default=8)
    ap.add_argument("--served-name", default=None)
    ap.add_argument("--max-q-size", type=int, default=8)
    return ap.parse_args(argv)


class ExlEngine:
    """exllamav3 + the shared engine, driven step by step."""

    def __init__(self, backend: str, argv: list, log=print):
        self.backend, self.argv, self.log = backend, list(argv), log
        self.reqs: dict = {}
        self.by_job: dict = {}
        self.stats = {"requests": 0, "completion_tokens": 0}
        self.kl = {"on": os.environ.get(f"{self.prefix}KL_CAPTURE") == "1", "prompt": None, "gen": [], "steps": []}
        self.vision = None

    @property
    def prefix(self) -> str:
        return "DSV41_" if self.backend == "dsv41" else "GLM53_"

    # ---- loading -----------------------------------------------------------------------------------------------
    def load(self):
        import torch
        from exllamav3 import model_init
        from ..kernels import install_prebuilt_loader
        install_prebuilt_loader()
        self.torch = torch
        args = self.args = parse_exl_args(self.argv)
        engine = None
        if self.backend == "dsv41":
            if os.environ.get("DSV41_NV_STORE"):
                from ..engine import tiers as engine
                engine.install()
        else:
            from ..models.glm53.engine import server as glm_server
            for var, mod in glm_server._NOT_INCLUDED.items():
                if os.environ.get(var) not in (None, "", "0"):
                    raise SystemExit(f"{var}={os.environ[var]} needs upstream glm-flash-lite's {mod}, which bigmoe "
                                     f"does not include")
            if os.environ.get("GLM53_NV") == "2":
                assert os.environ.get("GLM53_EC"), "GLM53_NV=2 needs GLM53_EC=1 (the VRAM expert cache)"
                from ..engine import tiers as engine
                engine.install()
            elif os.environ.get("GLM53_NV") not in (None, "", "0") or os.environ.get("GLM53_EC"):
                raise SystemExit("GLM53_NV must be 2 (the shared engine) when the expert cache is on")
        self.tiers = engine
        if engine is not None:
            engine.ARGV = list(self.argv)     # the RAM tier leaves -rcs / -ccs to exllamav3 (tiers._rcs_bytes)
        t0 = time.time()
        (self.model, self.config, self.cache, self.tok, self.draft_model, _dc,
         self.draft_cache) = model_init.init(args)
        if engine is not None:
            # vLLM keeps more loader frames and module objects alive than the native server. Reclaim temporary
            # checkpoint/drafter allocations before nv2 sizes the RAM tier, or they silently become fewer experts.
            import ctypes
            import gc
            gc.collect()
            ctypes.CDLL(None).malloc_trim(0)
            engine.attach(self.model)
        from .. import multimodal
        self.vision = multimodal.create(self.config, self.tok, log=self.log)
        if self.backend == "dsv41" and self.vision is not None and self.model.caps.get("mm_exact_chunks"):
            # see models/dsv41/engine/server.py: V4.1's trunk stays causal over an image span, any chunking is exact
            self.model.caps["mm_exact_chunks"] = False
        self.t_load = time.time() - t0
        self.log(f" -- loaded in {self.t_load:.1f} s")
        self.eos = self._eos_ids()
        self.ctx = args.cache_size
        self._make_generator()
        self._install_forward_hooks()
        cpus = os.environ.get(f"{self.prefix}MAIN_CPUS")
        if cpus:   # the GPU-feeding thread off the CPU-lane / reader cores
            pin_unpinned_threads(set(_cpus(cpus)))
            torch.set_num_threads(int(os.environ.get(f"{self.prefix}TORCH_THREADS", "1")))
            self.log(f" -- engine thread on cpus {sorted(os.sched_getaffinity(0))}; vLLM I/O threads unpinned")

    def _eos_ids(self) -> list:
        eos = set()
        for e in (getattr(self.config, "eos_token_id_list", None) or []):
            eos.add(int(e))
        if self.tok.eos_token_id is not None:
            eos.add(int(self.tok.eos_token_id))
        try:
            gc = json.load(open(os.path.join(self.args.model_dir, "generation_config.json")))
            v = gc.get("eos_token_id")
            for e in (v if isinstance(v, list) else [v]):
                if e is not None:
                    eos.add(int(e))
        except Exception:
            pass
        return sorted(eos)

    def _make_generator(self):
        from exllamav3.generator import Generator
        a = self.args
        self.gen = Generator(model=self.model, cache=self.cache, tokenizer=self.tok, max_batch_size=a.max_batch_size,
                             max_chunk_size=a.chunk_size, max_q_size=a.max_q_size, draft_model=self.draft_model,
                             draft_cache=self.draft_cache, num_draft_tokens=a.num_draft_tokens,
                             dynamic_draft_tokens=a.dynamic_draft, draft_confidence=a.draft_confidence,
                             ngram_match_min=a.ngram_match_min,
                             record_draft_stats=bool(self.draft_model is not None or a.ngram_match_min),
                             recurrent_cache_size=int(a.recurrent_cache_size * 1024 ** 3),
                             **serving.prefix_cache_tiers(a))
        if self.backend == "glm53" and a.ngram_match_min and os.environ.get("GLM53_NG_POLICY", "1") == "1":
            from exllamav3.generator.generator import Generator as G
            from exllamav3.generator.job import Job
            from ..models.glm53.engine import ngram_policy
            ngram_policy.install(G, Job)

    def _install_forward_hooks(self):
        """KL capture (quality tools): top-256 logprobs of every forward step of a single sequence. GLM's verify flag
        for staged forwards (BCP: nv2.VERIFY around multi-token forwards) is the native server's _bcp_install."""
        torch = self.torch
        orig = self.model.forward
        kl, backend = self.kl, self.backend
        nv2 = None
        if backend == "glm53" and self.tiers is not None:
            from ..engine import nv2
        lossy_dec = os.environ.get("GLM53_NV_LOSSY_DEC") == "1"

        def fwd(*a, **kw):
            ids = kw["input_ids"] if "input_ids" in kw else (a[0] if a else None)
            params = kw["params"] if "params" in kw else (a[1] if len(a) > 1 else None)
            staged = isinstance(params, dict) and params.get("pinned_staging")
            if nv2 is not None and staged:
                nv2.VERIFY[0] = int(ids.shape[-1]) > 1 or lossy_dec
            try:
                out = orig(*a, **kw)
            finally:
                if nv2 is not None and staged:
                    nv2.VERIFY[0] = False
            if kl["on"] and ids is not None and ids.shape[0] == 1 and isinstance(params, dict) and out is not None \
                    and (backend == "dsv41" or staged):
                cs = params.get("cache_seqlens")
                p0 = int(cs[0]) if cs is not None else -1
                with torch.inference_mode():
                    v, ix = out[0, -min(out.shape[1], 64):].float().log_softmax(-1).topk(256, dim=-1)
                kl["steps"].append({"p0": p0, "n": int(ids.shape[-1]), "ids": [int(x) for x in ids[0, -64:].tolist()],
                                    "lp": [[round(x, 4) for x in row] for row in v.tolist()], "ix": ix.tolist()})
            return out
        self.model.forward = fwd

    # ---- requests ----------------------------------------------------------------------------------------------
    def _sampler(self, sp):
        from exllamav3.generator.sampler import ArgmaxSampler, ComboSampler
        t = float(sp.temperature if sp.temperature is not None else 1.0)
        rep = float(getattr(sp, "repetition_penalty", 1.0) or 1.0)
        freq = float(getattr(sp, "frequency_penalty", 0.0) or 0.0)
        pres = float(getattr(sp, "presence_penalty", 0.0) or 0.0)
        bias = getattr(sp, "logit_bias", None)
        if t < 1e-5 and rep == 1.0 and freq == 0.0 and pres == 0.0 and not bias:
            return ArgmaxSampler()
        top_k = int(sp.top_k or 0)
        return ComboSampler(temperature=max(t, 1e-5), top_p=float(sp.top_p or 1.0), top_k=max(top_k, 0),
                            min_p=float(sp.min_p or 0.0), rep_p=rep, freq_p=freq, pres_p=pres,
                            logit_bias={int(k): float(v) for k, v in (bias or {}).items()} or None)

    def _make_job(self, r: Request):
        from exllamav3.generator import Job
        torch = self.torch
        sp = r.params
        ids = torch.tensor([r.prompt], dtype=torch.long)
        max_new = sp.max_tokens if sp.max_tokens is not None else None
        rq = int(os.environ.get(f"{self.prefix}MAX_RQ_TOKENS", "0")) or None
        extra = getattr(sp, "extra_args", None) or {}
        job = Job(input_ids=ids, max_new_tokens=max_new, min_new_tokens=int(sp.min_tokens or 0),
                  sampler=self._sampler(sp), seed=sp.seed, stop_conditions=list(self.eos) if not sp.ignore_eos else [],
                  decode_special_tokens=False, identifier=r.req_id, embeddings=r.mm or None,
                  **({"max_rq_tokens": rq} if rq else {}))
        r.force = extra.get("bigmoe_force")     # /kl_force: constrain only after the job is enqueued
        # With recurrent checkpoints the final job's cached_pages can be zero even when allocation restored pages
        # through the RAM/disk tiers. Measure this job's own allocation-counter delta before any requeue can reshape
        # its accounting; taking the generator-wide delta at a later "started" event would mix concurrent jobs.
        metrics = self.gen.pagetable.metrics
        allocate = job.allocate_pages

        def tracked_allocate():
            before = metrics.get("alloc_cached_pages", 0)
            out = allocate()
            hit = metrics.get("alloc_cached_pages", 0) - before
            r.cached = hit * PAGE_SIZE + int(getattr(job, "cached_tokens", 0))
            return out
        job.allocate_pages = tracked_allocate
        return job

    def _enqueue(self, r: Request):
        self.gen.enqueue(r.job)
        if r.force is not None:
            r.job.constrain_output_now(self.torch.tensor([[int(x) for x in r.force]], dtype=self.torch.long))

    def _images(self, r: Request, feats: list):
        """Embeddings of the request's images (mm.features_of), the prompt's placeholder rows given their ids."""
        from . import mm
        if self.vision is None:
            from .. import multimodal
            raise multimodal.not_supported()
        items = []
        for ident, _off, _n, pixels, meta in feats:
            prepped, size = mm.prepped_of(self.backend, pixels, meta)
            items.append((f"{ident}-{self.vision.fingerprint}", size, prepped))
        embs = self.vision.embed_prepped(items)
        r.prompt = mm.splice(r.prompt, feats, embs)
        r.mm = list({id(e): e for e in embs}.values())

    def add(self, req_id: str, prompt: list, params, mm=None):
        r = Request(req_id, list(prompt), params)
        try:
            validate_sampling_params(params)
            if mm:
                self._images(r, mm)
            r.job = self._make_job(r)
            self._enqueue(r)
        except Exception as e:      # does not fit the cache (exllamav3 asserts) or a bad parameter
            err = serving.as_request_error(e, len(prompt), self.ctx)
            r.finished, r.error, r.status = "error", err.message, err.status
            self.log(f" !! request {req_id}: {err.message}")
        self.reqs[req_id] = r
        if r.job is not None:
            self.by_job[id(r.job)] = r
        self.stats["requests"] += 1
        if self.kl["on"]:
            self.kl["prompt"], self.kl["gen"], self.kl["steps"] = list(prompt), [], []

    def cancel(self, req_id: str):
        r = self.reqs.pop(req_id, None)
        if r is None:
            return
        if r.job is not None:
            self.by_job.pop(id(r.job), None)
            if not r.finished:
                try:
                    self.gen.cancel(r.job)
                except Exception as e:
                    self.log(f" !! cancel {req_id}: {type(e).__name__}: {e}")
        if r.restore_chunk is not None:
            self.gen.max_chunk_size, r.restore_chunk = r.restore_chunk, None

    def has_work(self) -> bool:
        return bool(self.gen.num_remaining_jobs())

    @property
    def idle_pending(self) -> bool:
        return not self.gen.num_remaining_jobs() and self.gen.has_idle_work()

    def idle(self) -> bool:
        """One slice of exllamav3's idle work (the disk tier's flush). Returns whether more is left."""
        if self.gen.has_idle_work():
            self.gen.idle_work()
        return self.gen.has_idle_work()

    # ---- one step ----------------------------------------------------------------------------------------------
    def step(self) -> dict:
        """One Generator.iterate(). Returns {req_id: (new tokens, Progress)} for every live request."""
        results = self.gen.iterate() if self.gen.num_remaining_jobs() else []
        for res in results:
            r = self.by_job.get(id(res["job"]))
            if r is None:
                continue
            stage = res.get("stage")
            if stage == "error":
                self._failed(r, res["error"])
            elif stage == "started":
                # allocate_pages() has just restored the prefix (possibly through the RAM/disk tiers)
                if r.cached is None:
                    r.cached = self._cached(r.job)
            elif stage == "streaming":
                tids = res.get("token_ids")
                if tids is not None and tids.numel():
                    toks = [int(x) for x in tids.flatten().tolist()]
                    r.new_tokens += toks
                    r.n_out += len(toks)
                    if self.kl["on"]:
                        self.kl["gen"] += toks
                if not r.started and (r.new_tokens or res.get("eos")):
                    r.started = True
                    if r.cached is None:
                        r.cached = self._cached(r.job)
                    if r.restore_chunk is not None:     # the retried prefill is done: full chunks again
                        self.gen.max_chunk_size, r.restore_chunk = r.restore_chunk, None
                if res.get("eos"):
                    self._eos(r, res)
        out = {}
        for rid, r in self.reqs.items():
            toks, r.new_tokens = r.new_tokens, []
            out[rid] = (toks, self._progress(r))
        return out

    def _cached(self, job) -> int:
        return int(getattr(job, "cached_pages", 0)) * PAGE_SIZE + int(getattr(job, "cached_tokens", 0))

    def _progress(self, r: Request) -> Progress:
        if r.finished:
            return Progress(computed=0, cached=r.cached, finished=r.finished, error=r.error, status=r.status)
        job = r.job
        if r.started:
            # decoding: everything but the last sampled token is in the cache
            return Progress(computed=len(r.prompt) + r.n_out - 1, cached=r.cached)
        seq = job.sequences[0] if job is not None and job.sequences else None
        pos = int(seq.kv_position) if seq is not None else 0
        return Progress(computed=min(pos, len(r.prompt) - 1))

    def _eos(self, r: Request, res: dict):
        reason = res.get("eos_reason")
        if r.cached is None and res.get("cached_tokens") is not None:
            r.cached = int(res["cached_tokens"])
        self.stats["completion_tokens"] += r.n_out
        if res.get("accepted_draft_tokens") is not None:   # speculative decoding (DSpark / n-gram)
            sd = self.stats.setdefault("draft", {"accepted": 0, "rejected": 0, "rounds": 0, "windows": {},
                                                 "accepted_hist": {}})
            sd["accepted"] += int(res.get("accepted_draft_tokens") or 0)
            sd["rejected"] += int(res.get("rejected_draft_tokens") or 0)
            for _pos, win, acc in (getattr(r.job, "draft_stats", None) or []):
                sd["rounds"] += 1
                sd["windows"][str(win)] = sd["windows"].get(str(win), 0) + 1
                sd["accepted_hist"][str(acc)] = sd["accepted_hist"].get(str(acc), 0) + 1
        fin = {"reason": str(reason), "timings": serving.job_timings(res)}
        serving.log_request(int(res.get("prompt_tokens") or len(r.prompt)), r.n_out, fin)
        # vLLM normally ends the request on the same token (EOS, max_tokens); a job that ends first (the cache is
        # full, a stop token vLLM does not know) is ended here
        r.finished = "length" if reason in ("max_new_tokens", None) else "stop"
        self.by_job.pop(id(r.job), None)

    def _failed(self, r: Request, exc: BaseException):
        """A job the generator contained (Generator.reap_failed_job). Out of GPU memory before the first token: free
        what the failed forward held and run the job once more with half-size prefill chunks (serving.RetryingJob)."""
        self.by_job.pop(id(r.job), None)
        if serving.is_cuda_oom(exc):
            serving.release_failed(exc)
            if not r.started and not r.retried:
                r.retried = True
                if r.restore_chunk is None:
                    r.restore_chunk = self.gen.max_chunk_size
                self.gen.max_chunk_size = max(256, self.gen.max_chunk_size // 2 // 256 * 256)
                self.log(f" !! out of GPU memory before the first token; retrying the request with "
                         f"{self.gen.max_chunk_size}-token prefill chunks")
                try:
                    r.job = self._make_job(r)
                    self._enqueue(r)
                    self.by_job[id(r.job)] = r
                    return
                except Exception as e:
                    exc = e
        if r.restore_chunk is not None:
            self.gen.max_chunk_size, r.restore_chunk = r.restore_chunk, None
        err = serving.as_request_error(exc, len(r.prompt), self.ctx)
        r.finished, r.error, r.status = "error", err.message, err.status
        self.log(f" !! request {r.req_id} failed: {err.message}")

    # ---- diagnostics (RPC from the API server) -----------------------------------------------------------------
    def summary(self) -> dict:
        st = json.loads(json.dumps(self.stats))
        st["recurrent_cache"] = serving.recurrent_cache_stats(self)
        st["prefix_cache"] = self.gen.get_cache_stats()
        st["gpu"] = serving.gpu_memory_stats()
        st["vision"] = self.vision.summary() if self.vision is not None else None
        if self.tiers is not None:
            st["tiers"] = self.tiers.summary()
        return st

    @property
    def generator(self):          # serving.recurrent_cache_stats reads .generator.recurrent_cache
        return self.gen

    def model_card(self) -> dict:
        from .. import multimodal
        return {"max_model_len": self.ctx, **multimodal.model_card_fields(self.vision)}

    def nv_verify(self, n: int = 32):
        if self.tiers is None:
            return {"error": "the shared engine is off"}
        return self.tiers.verify(n)

    def nv2_rate(self, mbps: float = 0.0) -> dict:
        """NVMe emulation (diagnostics): reads complete no faster than a single device of `mbps` MB/s (0 = off)."""
        from ..engine import nv2
        nv2.ext().set_read_rate(float(mbps))
        return {"read_rate_mbps": mbps}

    def cpu_switch(self, on: int = 1) -> dict:
        """Runtime CPU-lane switch (mode nvme): on=0 puts every pick on the GPU path (bit-exact), for A/B runs."""
        from ..engine import nv2
        if nv2.NV is None:
            return {"error": "nv2 off"}
        nv2.NV.set_cpu(bool(on))
        return {"cpu": bool(on) and nv2.NV.cpu_on}

    def server_info(self) -> dict:
        a = self.args
        sa = {k: v for k, v in vars(a).items() if isinstance(v, (int, float, str, bool, type(None)))}
        sa.update({"context_length": self.ctx, "max_running_requests": a.max_batch_size,
                   "max_total_num_tokens": self.ctx, "chunked_prefill_size": a.chunk_size,
                   "kv_cache_dtype": a.cache_quant or "fp16", "engine": "exllamav3 behind vLLM",
                   "env": {k: v for k, v in os.environ.items() if k.startswith((self.prefix, "EXL3_", "BIGMOE_"))}})
        return {"context_length": self.ctx, "server_args": sa, "load_seconds": round(self.t_load, 1)}

    def teacher_forced(self, ids: list, start: int, topn: int, want_ids: list) -> dict:
        torch = self.torch
        x = torch.tensor([ids], dtype=torch.long)
        with torch.inference_mode():
            logits = self.model.forward(x, {"attn_mode": "flash_attn_nc"}).float()[0]
        lp = torch.log_softmax(logits, -1)
        L = x.shape[-1]

        def dist(row):
            v, ix = lp[row].topk(topn)
            top = [[float(a), int(b), None] for a, b in zip(v.tolist(), ix.tolist())]
            sel = [[float(lp[row, t]), int(t), None] for t in want_ids] if want_ids else []
            return top, sel
        in_top, in_ids = [], []
        for t in range(start, L):
            if t == 0:
                in_top.append(None)
                in_ids.append(None)
                continue
            a, b = dist(t - 1)
            in_top.append(a)
            in_ids.append(b)
        o_top, o_ids = dist(L - 1)
        out = {"input_top_logprobs": in_top, "input_token_ids_logprobs": in_ids, "output_top_logprobs": [o_top],
               "output_token_ids_logprobs": [o_ids], "next_token": int(lp[L - 1].argmax())}
        del logits, lp
        torch.cuda.empty_cache()
        return out

    def kl_teacher(self, ids: list, start: int, count: int) -> dict:
        torch = self.torch
        x = torch.tensor([ids], dtype=torch.long)
        with torch.inference_mode():
            logits = self.model.forward(x, {"attn_mode": "flash_attn_nc"})
            v, ix = logits[0, start:start + count].float().log_softmax(-1).topk(256, dim=-1)
        del logits
        torch.cuda.empty_cache()
        return {"start": start, "count": count, "lp": [[round(a, 4) for a in r] for r in v.tolist()],
                "ix": ix.tolist()}

    def kl_capture(self, on: int = -1, keep: bool = False) -> dict:
        if on >= 0:
            self.kl["on"] = bool(on)
        d = {"prompt": self.kl["prompt"], "gen": self.kl["gen"], "steps": self.kl["steps"], "eos": list(self.eos),
             "on": self.kl["on"]}
        if not keep:
            self.kl["steps"] = []
        return d


class FakeEngine:
    """CPU-only stand-in for tests of the vLLM plumbing: a prompt takes ceil(P / chunk) steps to prefill (the first
    `cached` tokens are free), then one token per step (two with `spec`): token i of the answer is
    (prompt[-1] + i + 1) % vocab, eos after `answer` tokens."""

    def __init__(self, chunk=64, answer=8, cached=0, spec=False, vocab=1000, eos=None):
        self.chunk, self.answer, self.cached0, self.spec, self.vocab, self.eos = chunk, answer, cached, spec, vocab, eos
        self.reqs = {}
        self.stats = {"requests": 0, "completion_tokens": 0}
        self.ctx = 1 << 20
        self.log = print
        self.fail = set()
        self.script = None

    def set_script(self, ids):
        """The next requests answer exactly these token ids (then stop), until set_script(None)."""
        self.script = list(ids) if ids is not None else None
        return True

    def add(self, req_id, prompt, params, mm=None):
        n_ans = self.answer if self.script is None else len(self.script)
        n_ans = n_ans if params.max_tokens is None else min(n_ans, params.max_tokens)
        self.reqs[req_id] = {"prompt": list(prompt), "pos": min(self.cached0, len(prompt) - 1), "out": [],
                             "n_ans": n_ans, "fin": None, "err": None, "started": False, "script": self.script}
        self.stats["requests"] += 1

    def cancel(self, req_id):
        self.reqs.pop(req_id, None)

    def has_work(self):
        return any(r["fin"] is None for r in self.reqs.values())

    def idle(self):
        return False

    idle_pending = False

    def step(self):
        out = {}
        for rid, r in self.reqs.items():
            new = []
            P = len(r["prompt"])
            if rid in self.fail and r["fin"] is None:
                r["fin"], r["err"] = "error", "injected failure"
            if r["fin"] is None and r["prompt"] and r["prompt"][0] == self.vocab - 1:   # tests: a request it rejects
                r["fin"], r["err"], r["status"] = "error", "rejected by the engine (test)", 400
            elif r["fin"] is None and r["pos"] < P - 1:
                r["pos"] = min(P - 1, r["pos"] + self.chunk)
            elif r["fin"] is None:
                for _ in range(2 if (self.spec and r["started"]) else 1):
                    i = len(r["out"])
                    t = (r["prompt"][-1] + i + 1) % self.vocab
                    if self.eos is not None and i + 1 == r["n_ans"]:
                        t = self.eos
                    if r["script"] is not None:
                        t = r["script"][i]
                    r["out"].append(t)
                    new.append(t)
                    r["started"] = True
                    if len(r["out"]) >= r["n_ans"]:
                        r["fin"] = "stop" if r["script"] is not None else "length"
                        break
            if r["started"]:
                pr = Progress(computed=P + len(r["out"]) - 1, cached=min(self.cached0, P - 1), finished=r["fin"],
                              error=r["err"])
            else:
                pr = Progress(computed=r["pos"], finished=r["fin"], error=r["err"], status=r.get("status", 500))
            out[rid] = (new, pr)
        return out

    def summary(self):
        return dict(self.stats, fake=True)

    def server_info(self):
        return {"context_length": self.ctx, "server_args": {"engine": "fake"}, "load_seconds": 0.0}

    def model_card(self):
        return {"max_model_len": self.ctx, "input_modalities": ["text"], "capabilities": {"vision": False}}

    def kl_capture(self, on=-1, keep=False):
        return {"prompt": None, "gen": [], "steps": [], "eos": [], "on": False}
