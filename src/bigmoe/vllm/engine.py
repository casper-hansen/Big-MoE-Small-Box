"""The model executor behind vLLM's scheduler: exllamav3's model, paged K/V cache and prefix-cache tiers over the
shared expert engine, driven one vLLM step at a time.

vLLM decides; this module executes. exllamav3's own scheduling loop (Generator.iterate: its admission in
iterate_start_jobs, its prefill rotation / cooldown / shared chunk, its draft-then-verify rounds) is never called.
Instead:

  admission     BoxKVCacheManager (scheduler.py) asks reserve() when vLLM's scheduler looks a waiting request up in its
                prefix cache: the request becomes an exllamav3 Job whose pages are allocated (reviving cached prefix pages
                from the GPU, pinned-RAM and disk tiers, and the recurrent checkpoint they need), and the cached prefix
                is skipped right away. The number of prompt tokens skipped is vLLM's prefix-cache hit (its
                num_computed_tokens, its usage.prompt_tokens_details.cached_tokens). No free pages: None, vLLM keeps the
                request waiting (or preempts, for a running one that needs pages to continue, see ensure()).
  chunking      vLLM's token budget and long-prefill threshold size every prefill chunk; plan() aligns a chunk's end to
                where exllamav3 can end a forward (a 256-token page, the generator's max chunk, the recurrent last-page
                split), so the engine computes exactly what vLLM scheduled.
  a step        execute() runs, for the requests vLLM scheduled and only those: their prefill chunk (Job.prefill), then
                one batched target forward for every request whose scheduled range reaches its last token
                (Generator.iterate_gen over exactly that batch, verifying the draft tokens vLLM scheduled), then the
                drafter (DSpark / MTP, a draft model, n-gram) for the next step's drafts, which go back to vLLM
                (take_drafts -> Scheduler.update_draft_token_ids). vLLM schedules them as spec tokens and does the
                acceptance accounting (num_computed_tokens -= rejected) and spec-decode metrics itself.

  grammars      vLLM's structured outputs (response_format / JSON schemas, and the structural tags its tool parsers add
                for strict tool calling) are vLLM's xgrammar state machines; each step's bitmask rows (one per
                sampled position, drafts included) reach exllamav3's sampler through a Filter on the job
                (grammar_filter), and vLLM advances and checks the state machine with the accepted tokens.

The engine reports each scheduled request's position after the step (Progress); BoxScheduler compares it with vLLM's
own count and only corrects (and counts, /stats "scheduler.corrections") where they differ: an out-of-memory retry, an
exllamav3 requeue that is waiting for pages, a multimodal chunk boundary.

Around a job: the EOS set, an out-of-memory failure before the first token
re-run once with half-size prefill chunks, a per-request log line, draft statistics, idle disk flushes. Stdlib only at
import time; torch and exllamav3 are imported in ExlEngine.load().
"""
import argparse
import dataclasses
import json
import os
import time

from .. import serving

PAGE_SIZE = 256


def inference(fn):
    """Run an engine entry point the way Generator.iterate runs (torch.inference_mode): the cache, page and job
    tensors exllamav3 updates in place are inference tensors."""
    import functools

    @functools.wraps(fn)
    def wrapped(self, *a, **kw):
        torch = getattr(self, "torch", None)
        if torch is None:
            return fn(self, *a, **kw)
        with torch.inference_mode():
            return fn(self, *a, **kw)
    return wrapped


@dataclasses.dataclass
class Progress:
    """One request's state in the engine after a step. computed: tokens of the request's sequence whose K/V is in the
    cache (vLLM's num_computed_tokens), i.e. the prefill position, or all but the last sampled token once decoding.
    cached: prompt tokens reused from the prefix cache (known once the request is reserved). finished: the engine ended
    the request on its own ("length": the cache is full, "stop": a stop token vLLM did not end it on, "error": a failed
    or refused job), with error the message and status its HTTP status (400 = the request's own fault)."""
    computed: int
    cached: int | None = None
    finished: str | None = None
    error: str | None = None
    status: int = 500


@dataclasses.dataclass
class Request:
    req_id: str
    prompt: list                         # the sequence at reservation (a preempted request's includes its output)
    params: object                       # vllm SamplingParams
    max_new: int | None = None           # tokens left to generate (max_tokens minus the output so far)
    min_new: int = 0
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
    stalled: int = 0
    grammar: object = None               # grammar_filter of a structured-output request


STALL_STEPS = 200


def _cpus(spec: str) -> list:
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def pin_unpinned_threads(cpus: set) -> int:
    """Pin the calling (GPU-feeding) thread to `cpus`. vLLM's I/O threads stay with the OS: pinning all of them onto
    the model's single logical CPU adds scheduler work to every decode step. The shared engine's own threads keep the
    affinities they selected during load."""
    import threading
    os.sched_setaffinity(threading.get_native_id(), cpus)
    return 1


def validate_sampling_params(sp):
    """Refuse options that the exllamav3 executor cannot honour. vLLM would otherwise accept them and either ignore
    them or fail while serializing the answer; a clear 400 is preferable to a plausible but wrong answer. (Structured
    outputs are honoured: grammar_filter.)"""
    unsupported = []
    for attr in ("logprobs", "prompt_logprobs", "prompt_token_id_logprobs"):
        if getattr(sp, attr, None) is not None:
            unsupported.append(attr)
    if getattr(sp, "allowed_token_ids", None):
        unsupported.append("allowed_token_ids")
    if getattr(sp, "bad_words", None):
        unsupported.append("bad_words")
    if int(getattr(sp, "n", 1) or 1) != 1:
        unsupported.append("n > 1")
    if getattr(sp, "use_beam_search", False):
        unsupported.append("beam search")
    if unsupported:
        raise serving.RequestError(400, "the exllamav3 executor does not support " + ", ".join(unsupported)
                                   + "; omit these parameters (teacher-forced logprobs remain available at /generate "
                                     "and /kl_teacher)", param="logprobs" if "logprobs" in unsupported else None)


def exl_arg_parser():
    """The engine's command line: exllamav3's model_init arguments plus the server's own."""
    from exllamav3 import model_init
    ap = argparse.ArgumentParser(prog="bigmoe-exl")
    model_init.add_args(ap, cache=True, add_draft_model_args=True, default_chunk_size=2048)
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--max-batch-size", type=int, default=8)
    ap.add_argument("--served-name", default=None)
    ap.add_argument("--max-q-size", type=int, default=8)
    return ap


def parse_exl_args(argv: list):
    return exl_arg_parser().parse_args(argv)


_GRAMMAR_FILTER = []


def grammar_filter(tokenizer):
    """An exllamav3 Filter that masks each sampled position with the row of vLLM's grammar bitmask vLLM computed for
    it (packed int32, xgrammar's layout, which exllamav3's sampler takes as is). The state machine is vLLM's: it
    is advanced and checked by vLLM's structured-output manager with the tokens the step accepted, so the filter
    keeps no state of its own. Rows run out only on a step vLLM sent no bitmask for: nothing is masked then."""
    if not _GRAMMAR_FILTER:
        from exllamav3.generator.filter.filter import Filter

        class VllmGrammarFilter(Filter):
            def __init__(self, tok):
                super().__init__(tok, None, None, False)
                self.rows = []

            def reset(self):
                pass

            def accept_token(self, token):
                pass

            def is_completed(self):
                return False

            def use_background_worker(self):
                return False

            def get_next_logit_mask(self):
                import torch
                if self.rows:
                    return self.rows.pop(0)
                return torch.full((1, (self.vocab_size + 31) // 32), -1, dtype=torch.int32)
        _GRAMMAR_FILTER.append(VllmGrammarFilter)
    return _GRAMMAR_FILTER[0](tokenizer)


def plan_chunk(num_computed: int, num_new: int, num_tokens: int, max_chunk: int, recurrent: bool,
               last_page_max: int | None, page: int = PAGE_SIZE, hold_last: bool = False) -> int:
    """How many of the `num_new` tokens vLLM offers a request starting at `num_computed` (of `num_tokens`) the engine
    can compute in one step, as exllamav3's Job.prefill ends a forward: the chunk ends on a page boundary and inside
    the generator's max chunk, unless it is the last one (it reaches the sequence's last token, which the step's
    sampling forward takes); with recurrent state the last partial page is a forward of its own when it is short
    (engine/last_page.py, last_page_max; None = always, exllamav3's stock rule). Decode steps (one token plus drafts)
    pass unchanged. With hold_last (the engine drafts) the last chunk stops one token short: the drafter proposes the
    first output tokens right after the prefill and the last prompt token goes into the next step's verify forward
    with them, as one of vLLM's spec-decode steps (exllamav3's own order: prefill, draft, verify). 0: nothing fits."""
    if num_new <= 0:
        return 0
    c, last = num_computed, num_tokens - 1
    if c >= last:                                    # decoding: the last token (+ spec tokens)
        return num_new
    cap = (c + max_chunk) // page * page             # Job.prefill's window
    final = c + num_new >= num_tokens and cap >= last
    end = last if final else min(c + num_new, cap) // page * page
    if recurrent:
        lpb = last // page * page
        if c < lpb <= end and (last_page_max is None or last - lpb <= last_page_max):
            return lpb - c
    if final:
        return last - c if hold_last else num_new
    return max(0, end - c)


class ExlEngine:
    """exllamav3 + the shared engine, stepped by vLLM's scheduler (see the module docstring)."""

    def __init__(self, backend: str, argv: list, log=print):
        self.backend, self.argv, self.log = backend, list(argv), log
        self.reqs: dict = {}
        self.by_job: dict = {}
        self.drafts: dict = {}               # req_id -> draft tokens proposed after the last step
        self.draft_rows: list = []           # job ids of the drafted rows, in order (calibration / sparse head)
        self.draft_limit: dict = {}
        self.t_draft = 0.0
        self.stats = {"requests": 0, "completion_tokens": 0,
                      "scheduler": {"steps": 0, "reserved": 0, "no_pages": 0, "requeue_waits": 0,
                                    "prefill_tokens": 0, "verify_steps": 0, "drafts_dropped": 0}}
        self.kl = {"on": os.environ.get(f"{self.prefix}KL_CAPTURE") == "1", "prompt": None, "gen": [], "steps": []}
        self.vision = None
        self.tiers = None

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
            from ..models.glm53.engine import NOT_INCLUDED
            for var, mod in NOT_INCLUDED.items():
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
            # Reclaim temporary checkpoint/drafter allocations before nv2 sizes the RAM tier, or they silently
            # become fewer experts.
            import ctypes
            import gc
            gc.collect()
            ctypes.CDLL(None).malloc_trim(0)
            engine.attach(self.model)
        from .. import multimodal
        self.vision = multimodal.create(self.config, self.tok, log=self.log)
        if self.backend == "dsv41" and self.vision is not None and self.model.caps.get("mm_exact_chunks"):
            # V4.1's trunk stays causal over an image span, so any chunking is exact; exllamav3's exact image
            # chunking (meant for V4-Flash-Vision-Exp) would cost two extra prefill forwards per image
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
                             ngram_match_min=a.ngram_match_min, hybrid_draft_tokens=a.hybrid_draft_tokens,
                             record_draft_stats=bool(self.draft_model is not None or a.ngram_match_min),
                             recurrent_cache_size=int(a.recurrent_cache_size * 1024 ** 3),
                             **serving.prefix_cache_tiers(a))
        if self.backend == "glm53" and a.ngram_match_min and os.environ.get("GLM53_NG_POLICY", "1") == "1":
            from exllamav3.generator.generator import Generator as G
            from exllamav3.generator.job import Job
            from ..models.glm53.engine import ngram_policy
            ngram_policy.install(G, Job)

    @property
    def drafting(self) -> bool:
        g = self.gen
        return bool(g.draft_model is not None or g.ngram_match_min or g.hybrid_draft_tokens)

    def _install_forward_hooks(self):
        """KL capture (quality tools): top-256 logprobs of every forward step of a single sequence. GLM's verify flag
        for staged forwards (nv2.VERIFY around multi-token forwards)."""
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

    # ---- jobs --------------------------------------------------------------------------------------------------
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
        sp = r.params
        ids = self.torch.tensor([r.prompt], dtype=self.torch.long)
        rq = int(os.environ.get(f"{self.prefix}MAX_RQ_TOKENS", "0")) or None
        return Job(input_ids=ids, max_new_tokens=r.max_new, min_new_tokens=r.min_new,
                   sampler=self._sampler(sp), seed=sp.seed, stop_conditions=list(self.eos) if not sp.ignore_eos else [],
                   decode_special_tokens=False, identifier=r.req_id, embeddings=r.mm or None,
                   filters=[r.grammar] if r.grammar is not None else None,
                   **({"max_rq_tokens": rq} if rq else {}))

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

    def _fits(self, job) -> bool:
        g = self.gen
        rows = sum(len(j.sequences) for j in g.active_jobs)
        need, free = job.current_new_pages_required(), g.pagetable.num_unreferenced_pages()
        ok = rows + len(job.sequences) <= g.max_batch_size and need <= free
        if not ok:
            self.stats["scheduler"]["last_refusal"] = {"rows": rows, "max_rows": g.max_batch_size, "pages_needed": need,
                                                       "pages_free": free, "pages": g.pagetable.max_pages}
        return ok

    def _activate(self, r: Request) -> bool:
        """Move the request's pending job into the active set with its pages allocated (exllamav3's
        iterate_start_jobs for one job, on vLLM's decision) and skip its cached prefix. False: not enough free pages."""
        g, job = self.gen, r.job
        if not self._fits(job):
            return False
        g.pending_jobs.remove(job)
        g.active_jobs.append(job)
        job.activate()
        try:
            job.allocate_pages()
        except Exception as e:
            g.reap_failed_job(job, e, [])
            self._failed(r, e)
            return True
        self._skip_cached(job)
        if not r.started:
            r.cached = self._position(r)
        return True

    def _skip_cached(self, job):
        """Walk the job past every cached prefix page now (Job.prefill with a zero-token chunk skips cached pages within
        its window and runs no forward), so the position vLLM learns at admission is where computing starts."""
        seq = job.sequences[0]
        while not job.is_prefill_done():
            pos = seq.kv_position
            job.prefill([], 0)
            if seq.kv_position == pos:
                break

    def _position(self, r: Request) -> int:
        seq = r.job.sequences[0] if r.job is not None and r.job.sequences else None
        return int(seq.kv_position) if seq is not None else 0

    # ---- the scheduler's side (BoxKVCacheManager, BoxScheduler) -------------------------------------------------
    @inference
    def reserve(self, request) -> int | None:
        """vLLM looks a waiting request up: an exllamav3 job for its sequence, pages allocated, cached prefix skipped.
        Returns the prompt tokens reused from the prefix cache, or None when there are no free pages (the request stays
        waiting; the job stays prepared, so its page hashes and images are not redone). A request the engine refuses
        (too long for the cache, an option it cannot honour, an image without a vision tower) is admitted with 0 and
        reported failed after its first step."""
        rid = request.request_id
        r = self.reqs.get(rid)
        if r is None:
            r = self._new_request(request)
            if r.finished:
                return 0
        if r.finished:
            return 0
        if r.job in self.gen.pending_jobs:
            if not self._activate(r):
                self.stats["scheduler"]["no_pages"] += 1
                return None
            self.stats["scheduler"]["reserved"] += 1
        if r.finished:
            return 0
        return r.cached

    def _new_request(self, request) -> Request:
        sp = request.sampling_params
        out = request.num_output_tokens
        max_new = None if sp.max_tokens is None else max(1, int(sp.max_tokens) - out)
        r = Request(request.request_id, list(request.all_token_ids), sp, max_new=max_new,
                    min_new=max(0, int(sp.min_tokens or 0) - out))
        self.reqs[r.req_id] = r
        self.stats["requests"] += 1
        try:
            validate_sampling_params(sp)
            if request.mm_features:
                from . import mm
                self._images(r, mm.features_of(request))
            if getattr(request, "use_structured_output", False):
                r.grammar = grammar_filter(self.tok)
            r.job = self._make_job(r)
            self.gen.enqueue(r.job)                    # hashes the pages, starts disk-tier prefetches
            force = (getattr(sp, "extra_args", None) or {}).get("bigmoe_force")
            if force is not None:                      # /kl_force: constrain only after the job is enqueued
                r.job.constrain_output_now(self.torch.tensor([[int(x) for x in force]], dtype=self.torch.long))
            self.by_job[id(r.job)] = r
        except Exception as e:      # does not fit the cache (exllamav3 asserts) or a bad parameter
            err = serving.as_request_error(e, len(r.prompt), self.ctx)
            r.finished, r.error, r.status = "error", err.message, err.status
            self.log(f" !! request {r.req_id}: {err.message}")
            if r.job is not None and r.job in self.gen.pending_jobs:
                self.gen.pending_jobs.remove(r.job)
        if self.kl["on"]:
            self.kl["prompt"], self.kl["gen"], self.kl["steps"] = list(r.prompt), [], []
        return r

    @inference
    def ensure(self, request) -> bool:
        """vLLM allocates slots for a scheduled request: True if the engine can run it this step. A running request
        whose job exllamav3 requeued (its page budget, max_rq_tokens, is used up) needs fresh pages; without them vLLM
        preempts a request to make room, as it does when its own KV blocks run out."""
        r = self.reqs.get(request.request_id)
        if r is None:
            return self.reserve(request) is not None
        if r.finished or r.job is None:
            return True
        if r.job in self.gen.pending_jobs:
            if not self._activate(r):
                self.stats["scheduler"]["requeue_waits"] += 1
                return False
        return True

    @inference
    def release(self, req_id: str):
        """vLLM finished or preempted the request: its pages go back (to the prefix cache)."""
        r = self.reqs.pop(req_id, None)
        self.drafts.pop(req_id, None)
        if r is None:
            return
        if r.job is not None:
            self.by_job.pop(id(r.job), None)
            try:
                self.gen.cancel(r.job)
            except Exception as e:
                self.log(f" !! cancel {req_id}: {type(e).__name__}: {e}")
        if r.restore_chunk is not None:
            self.gen.max_chunk_size, r.restore_chunk = r.restore_chunk, None

    cancel = release

    def cached_of(self, req_id: str) -> int:
        r = self.reqs.get(req_id)
        return int(r.cached or 0) if r is not None else 0

    def page_usage(self) -> float:
        pt = self.gen.pagetable
        return 1.0 - pt.num_unreferenced_pages() / max(1, pt.max_pages)

    def plan(self, req_id: str, num_computed: int, num_new: int, num_tokens: int) -> int:
        """The part of vLLM's offer of num_new tokens that one step computes (plan_chunk)."""
        recurrent = self.gen.recurrent_cache is not None
        limit = None
        try:
            from exllamav3.generator import job as J
            if getattr(J.Job.prefill, "_bigmoe_last_page", False):
                limit = J._BIGMOE_LAST_PAGE_MAX
        except ImportError:
            pass
        return plan_chunk(num_computed, num_new, num_tokens, self.gen.max_chunk_size, recurrent, limit,
                          hold_last=self.drafting)

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
    def _only(self, jobs: list):
        """Generator methods that walk active_jobs (iterate_gen, the drafters) see exactly `jobs`, in this order; the
        page table is not defragmented while other active jobs hold block tables."""
        engine = self

        class Scope:
            def __enter__(self_):
                g = engine.gen
                self_.full = g.active_jobs
                self_.ids = {id(j) for j in jobs}
                g.active_jobs = list(jobs)
                g.on_queue_drained = lambda: None
                return self_

            def __exit__(self_, *exc):
                g = engine.gen
                kept = {id(j) for j in g.active_jobs}
                g.active_jobs = [j for j in self_.full if id(j) not in self_.ids or id(j) in kept]
                del g.on_queue_drained
                return False
        return Scope()

    @inference
    def execute(self, plan: dict, spec: dict, grammar=None) -> dict:
        """One vLLM step. plan: {req_id: (num_computed, num_scheduled, num_tokens)} in vLLM's order (vLLM's counts
        before the step); spec: {req_id: [draft tokens]} vLLM scheduled for verification; grammar: vLLM's
        GrammarOutput for the step's structured-output requests (or None). Returns {req_id: (new tokens, Progress)}
        for every scheduled request."""
        g = self.gen
        results = []
        self.draft_limit = self._grammar_rows(grammar, spec)
        busy_before = g.num_remaining_jobs()
        self.stats["scheduler"]["steps"] += 1
        sample = []
        for rid, (c, n, total) in plan.items():
            r = self.reqs.get(rid)
            if r is None or r.finished or r.job is None or r.job in g.pending_jobs:
                continue
            job = r.job
            reaches_end = c + n >= total
            pos = self._position(r)
            if not job.is_prefill_done():
                try:
                    if c + n >= total - 1:      # the last chunk (Job.prefill stops a token short of the end)
                        job.prefill(results, None)
                    elif c + n > pos:           # a chunk_size bounds the forward to whole pages
                        job.prefill(results, c + n - pos)
                except Exception as e:
                    g.reap_failed_job(job, e, results)
                self.stats["scheduler"]["prefill_tokens"] += max(0, self._position(r) - pos)
            if reaches_end and job in g.active_jobs and job.is_prefill_done():
                sample.append(r)
            elif not r.finished and job in g.active_jobs and self._position(r) == pos:
                r.stalled += 1
                if r.stalled >= STALL_STEPS:   # a schedule the engine cannot follow must not spin forever
                    self._failed(r, RuntimeError(f"no progress in {STALL_STEPS} steps at position {pos} "
                                                 f"(scheduled {c}+{n} of {total})"))
                continue
            r.stalled = 0
        if g.recurrent_cache is not None:
            g.recurrent_checkpoint()
        if sample:
            self._sample(sample, spec, results)
        self._results(results)
        for r in list(self.reqs.values()):          # requeued jobs (a new page budget) start again if pages allow
            if not r.finished and r.job is not None and r.job in g.pending_jobs and r.started:
                self._activate(r)
        self.drafts = {}
        if self.drafting:   # every scheduled request that can decode next: it just sampled, or its prefill is done
            self._draft([r for rid in plan if (r := self.reqs.get(rid)) is not None and not r.finished
                         and r.job in g.active_jobs and r.job.is_prefill_done()])
        if g.disk_page_cache is not None and g.active_jobs and g.disk_cache_write_interval > 0:
            g.write_behind()
        if busy_before and not g.num_remaining_jobs():
            g.on_queue_drained()
        out = {}
        for rid in plan:
            r = self.reqs.get(rid)
            if r is None:
                continue
            toks, r.new_tokens = r.new_tokens, []
            out[rid] = (toks, self._progress(r))
        return out

    def _grammar_rows(self, grammar, spec: dict) -> dict:
        """Hand each structured-output request its bitmask rows for this step (vLLM's layout: a fixed stride of rows
        per request, the first 1 + its spec tokens used). Returns {req_id: drafts the bitmask constrained}: drafts past
        that may not be accepted."""
        limit = {}
        for r in self.reqs.values():
            if r.grammar is not None:
                r.grammar.rows = []
        if grammar is None or not grammar.structured_output_request_ids:
            return limit
        torch = self.torch
        ids = grammar.structured_output_request_ids
        bm = grammar.grammar_bitmask
        stride = bm.shape[0] // len(ids)
        for i, rid in enumerate(ids):
            r = self.reqs.get(rid)
            if r is None or r.grammar is None:
                continue
            n = 1 + len(spec.get(rid) or ())
            r.grammar.rows = [torch.from_numpy(bm[i * stride + j:i * stride + j + 1].copy()) for j in range(n)]
            if grammar.num_acceptable_drafts is not None:
                limit[rid] = int(grammar.num_acceptable_drafts[i])
            elif spec.get(rid):
                limit[rid] = len(spec[rid])
        return limit

    def _sample(self, rows: list, spec: dict, results: list):
        """One batched target forward over `rows`: each request's next token, verifying the drafts vLLM scheduled
        when every row has some (a row that just finished its prefill has none yet: then nobody's are verified this
        step, which vLLM counts as rejected)."""
        torch, g = self.torch, self.gen
        drafts = [[int(t) for t in (spec.get(r.req_id) or [])] for r in rows]
        k = min(len(d) for d in drafts) if drafts else 0
        for r in rows:
            if r.req_id in self.draft_limit:
                k = min(k, self.draft_limit[r.req_id])
        if k and any(t < 0 for d in drafts for t in d[:k]):
            k = 0
        if not k and any(drafts):
            self.stats["scheduler"]["drafts_dropped"] += 1
        draft_tokens = torch.tensor([d[:k] for d in drafts], dtype=torch.long) if k else None
        jobs = [r.job for r in rows]
        if draft_tokens is None or [id(j) for j in jobs] != self.draft_rows:
            g._draft_conf_round = None           # the calibrator labels only the round it drafted
        t0 = time.perf_counter()
        with self._only(jobs):
            g.iterate_gen(results, draft_tokens)
        if draft_tokens is not None:
            self.stats["scheduler"]["verify_steps"] += 1
            if g.sparse_head is not None:
                g.sparse_head.note_round(self.t_draft + time.perf_counter() - t0, k, k if g.mtp_draft else 1)

    def _draft(self, rows: list):
        """The drafter's tokens for the next step (exllamav3's draft round, for exactly these rows)."""
        g = self.gen
        self.draft_rows = []
        if not rows:
            return
        jobs = [r.job for r in rows]
        t0 = time.perf_counter()
        with self._only(jobs):
            window = g.row_limited(g.num_draft_tokens) if g.draft_model else 0
            if g.sparse_head is not None and window > 0:
                g._sparse_flag = g.sparse_head.round_flag()
            d, results = None, []
            if g.draft_model is not None and window > 0:
                if g.dflash_draft:
                    d = g.iterate_draftmodel_dflash_gen(results, window)
                elif g.mtp_draft:
                    d = g.iterate_draftmodel_mtp_gen(results, window)
                else:
                    d = g.iterate_draftmodel_gen(results, window)
            elif g.ngram_match_min:
                d = g.iterate_ngram_gen(results)
            if g.hybrid_draft_tokens:
                d = g.iterate_hybrid_gen(d, g.row_limited(g.hybrid_draft_tokens))
        self.t_draft = time.perf_counter() - t0
        if d is None or d.shape[-1] == 0:
            g._draft_conf_round = None
            return
        rows_d = d[:len(rows)].tolist()
        for r, ids in zip(rows, rows_d):
            self.drafts[r.req_id] = [int(x) for x in ids]
        self.draft_rows = [id(j) for j in jobs]

    def take_drafts(self) -> dict:
        d, self.drafts = self.drafts, {}
        return d

    def _results(self, results: list):
        for res in results:
            r = self.by_job.get(id(res["job"]))
            if r is None:
                continue
            stage = res.get("stage")
            if stage == "error":
                self._failed(r, res["error"])
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
                    if r.restore_chunk is not None:     # the retried prefill is done: full chunks again
                        self.gen.max_chunk_size, r.restore_chunk = r.restore_chunk, None
                if res.get("eos"):
                    self._eos(r, res)

    def _progress(self, r: Request) -> Progress:
        if r.finished:
            return Progress(computed=0, cached=r.cached, finished=r.finished, error=r.error, status=r.status)
        if r.started:
            # decoding: everything but the last sampled token is in the cache
            return Progress(computed=len(r.prompt) + r.n_out - 1, cached=r.cached)
        return Progress(computed=min(self._position(r), len(r.prompt) - 1), cached=r.cached)

    def _eos(self, r: Request, res: dict):
        reason = res.get("eos_reason")
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
        fin = {"reason": str(reason), "timings": dict(serving.job_timings(res), cached_tokens=r.cached)}
        serving.log_request(int(res.get("prompt_tokens") or len(r.prompt)), r.n_out, fin)
        # vLLM normally ends the request on the same token (EOS, max_tokens); a job that ends first (the cache is
        # full, a stop token vLLM does not know) is ended by BoxScheduler
        r.finished = "length" if reason in ("max_new_tokens", None) else "stop"
        self.by_job.pop(id(r.job), None)

    def _failed(self, r: Request, exc: BaseException):
        """A job the generator contained (Generator.reap_failed_job). Out of GPU memory before the first token: free
        what the failed forward held and run the job once more with half-size prefill chunks."""
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
                    self.gen.enqueue(r.job)
                    self.by_job[id(r.job)] = r
                    self._activate(r)        # no pages now: ensure() retries before the next step
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
        from . import state
        st["scheduler"]["corrections"] = dict(state.CORRECTIONS)
        st["recurrent_cache"] = serving.recurrent_cache_stats(self)
        st["prefix_cache"] = self.gen.get_cache_stats()
        st["gpu"] = serving.gpu_memory_stats()
        st["vision"] = self.vision.summary() if self.vision is not None else None
        if self.tiers is not None:
            st["tiers"] = self.tiers.summary()
        if self.backend == "glm53" and self.args.ngram_match_min:
            from ..models.glm53.engine import ngram_policy
            st["ngram"] = ngram_policy.summary()
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
        sa.update({"context_length": self.ctx, "max_running_requests": self.gen.max_batch_size,
                   "max_total_num_tokens": self.ctx, "chunked_prefill_size": a.chunk_size,
                   "kv_cache_dtype": a.cache_quant or "fp16", "engine": "vLLM scheduler, exllamav3 executor",
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
    """CPU-only stand-in with ExlEngine's scheduler-facing interface, for tests of the vLLM plumbing. A cache of
    `pages` pages of `page` tokens; a request reserves ceil((sequence + answer) / page) of them, and a prompt whose
    first `cached` tokens were seen before (any earlier prompt with the same first `cached` tokens) skips them.
    Prefill computes what plan_chunk allows (max chunk `chunk`), token i of an answer is (prompt[-1] + i + 1) % vocab
    (eos after `answer` tokens if `eos` is set). With `spec`, every decode step proposes 2 drafts, the first right,
    the second wrong (unless `spec_all`), so vLLM's acceptance accounting is exercised."""

    def __init__(self, chunk=64, answer=8, cached=0, spec=False, vocab=1000, eos=None, page=16, pages=1 << 16,
                 spec_all=False, max_batch=8):
        self.chunk, self.answer, self.cached0, self.spec, self.vocab, self.eos = chunk, answer, cached, spec, vocab, eos
        self.page, self.pages, self.spec_all, self.max_batch = page, pages, spec_all, max_batch
        self.reqs = {}
        self.seen = set()
        self.stats = {"requests": 0, "completion_tokens": 0, "reserve_calls": 0, "no_pages": 0, "steps": [],
                      "verified": 0, "accepted": 0}
        self.ctx = 1 << 20
        self.log = print
        self.fail = set()
        self.script = None
        self.drafts = {}

    drafting = property(lambda self: self.spec)

    def set_script(self, ids):
        """The next requests answer exactly these token ids (then stop), until set_script(None)."""
        self.script = list(ids) if ids is not None else None
        return True

    def _need(self, r):
        return -(-(len(r["seq"]) + r["n_ans"]) // self.page)

    def used_pages(self):
        return sum(self._need(r) for r in self.reqs.values() if r["active"])

    def reserve(self, request):
        self.stats["reserve_calls"] += 1
        rid = request.request_id
        r = self.reqs.get(rid)
        if r is None:
            sp = request.sampling_params
            n_ans = self.answer if self.script is None else len(self.script)
            if sp.max_tokens is not None:
                n_ans = min(n_ans, sp.max_tokens - request.num_output_tokens)
            seq = list(request.all_token_ids)
            r = {"seq": seq, "base": len(seq), "pos": 0, "out": [], "n_ans": n_ans, "fin": None, "err": None,
                 "status": 500, "started": False, "script": self.script, "active": False,
                 "prev_out": request.num_output_tokens}
            self.reqs[rid] = r
            self.stats["requests"] += 1
            if seq and seq[0] == self.vocab - 1:            # tests: a request it refuses
                r["fin"], r["err"], r["status"] = "error", "rejected by the engine (test)", 400
                return 0
        if r["fin"]:
            return 0
        if not r["active"]:
            if self.used_pages() + self._need(r) > self.pages or \
                    sum(x["active"] for x in self.reqs.values()) >= self.max_batch:
                self.stats["no_pages"] += 1
                return None
            r["active"] = True
            key = tuple(r["seq"][:self.cached0])
            hit = self.cached0 if self.cached0 and key in self.seen else 0
            self.seen.add(key)
            r["pos"] = min(hit, len(r["seq"]) - 1)
            r["cached"] = r["pos"]
        return r["pos"]

    def ensure(self, request):
        return self.reserve(request) is not None

    def release(self, req_id):
        self.reqs.pop(req_id, None)
        self.drafts.pop(req_id, None)

    cancel = release

    def cached_of(self, req_id):
        r = self.reqs.get(req_id)
        return int(r.get("cached", 0)) if r else 0

    def page_usage(self):
        return self.used_pages() / self.pages

    def plan(self, req_id, num_computed, num_new, num_tokens):
        return plan_chunk(num_computed, num_new, num_tokens, self.chunk, False, None, self.page, hold_last=self.spec)

    def has_work(self):
        return any(r["fin"] is None for r in self.reqs.values())

    def idle(self):
        return False

    idle_pending = False

    def _token(self, r, i):
        if r["script"] is not None:
            return r["script"][i]
        t = (r["seq"][r["base"] - 1] + r["prev_out"] + i + 1) % self.vocab
        if self.eos is not None and i + 1 == r["n_ans"]:
            t = self.eos
        return t

    def execute(self, plan, spec, grammar=None):
        self.stats["steps"].append({rid: (c, n, total, list(spec.get(rid) or [])) for rid, (c, n, total) in plan.items()})
        out = {}
        new_drafts = {}
        for rid, (c, n, total) in plan.items():
            r = self.reqs.get(rid)
            if r is None:
                continue
            new = []
            L = len(r["seq"])
            if rid in self.fail and r["fin"] is None:
                r["fin"], r["err"] = "error", "injected failure"
            if r["fin"] is None and not r["started"]:
                assert r["pos"] == c, f"{rid}: engine at {r['pos']}, vLLM at {c}"
                r["pos"] = min(L - 1, c + n)
            if r["fin"] is None and c + n >= total and (r["started"] or r["pos"] >= L - 1):
                drafts = list(spec.get(rid) or [])
                accept = True
                for k in range(1 + len(drafts)):
                    i = len(r["out"])
                    t = self._token(r, i)
                    r["out"].append(t)
                    r["seq"].append(t)
                    new.append(t)
                    r["started"] = True
                    if len(r["out"]) >= r["n_ans"]:
                        r["fin"] = "stop" if r["script"] is not None else "length"
                        break
                    if k < len(drafts):
                        self.stats["verified"] += 1
                        accept = drafts[k] == t
                        if not accept:
                            break
                        self.stats["accepted"] += 1

            if self.spec and r["fin"] is None and (r["started"] or r["pos"] >= len(r["seq"]) - 1):
                good = self._token(r, len(r["out"]))
                new_drafts[rid] = [good, good if self.spec_all else (good + 7) % self.vocab]
            if r["started"]:
                pr = Progress(computed=len(r["seq"]) - 1, cached=r.get("cached"), finished=r["fin"], error=r["err"])
            else:
                pr = Progress(computed=r["pos"], cached=r.get("cached"), finished=r["fin"], error=r["err"],
                              status=r.get("status", 500))
            out[rid] = (new, pr)
        self.drafts = new_drafts
        return out

    def take_drafts(self):
        d, self.drafts = self.drafts, {}
        return d

    def summary(self):
        return dict(self.stats, fake=True, steps=len(self.stats["steps"]))

    def trace(self):
        return self.stats["steps"]

    def server_info(self):
        return {"context_length": self.ctx, "server_args": {"engine": "fake"}, "load_seconds": 0.0}

    def model_card(self):
        return {"max_model_len": self.ctx, "input_modalities": ["text"], "capabilities": {"vision": False}}

    def kl_capture(self, on=-1, keep=False):
        return {"prompt": None, "gen": [], "steps": [], "eos": [], "on": False}
