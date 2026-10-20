"""Cost-aware n-gram drafting for the offload engine (GLM53_NG_POLICY=1, default on with -ngram).

On this box a draft token is not cheap: every token of a verify window brings its own routed experts, and the cold ones
come from RAM or NVMe. Stock prompt lookup drafts whenever the last ngram_match_min tokens occur earlier in the context,
which on prose and loosely structured output (JSON with varying values) mostly proposes tokens that get rejected:
measured 9.6 -> 6.3 tok/s on prose and 9.2 -> 4.2 tok/s on CSV->JSON. This policy drafts only when the evidence says the
model is copying:
  - the suffix match must be at least GLM53_NG_MIN tokens long (default 4), and
  - the job's recent per-token draft acceptance (EWMA over drafted tokens, GLM53_NG_ALPHA, starts at 0.5) must be
    >= GLM53_NG_ACC (default 0.7): the draft is the stock num_draft_tokens cut to the length whose last token is still
    accepted with probability >= GLM53_NG_KEEP (default 0.5); or
  - the match is at least GLM53_NG_LONG tokens (default 12) and the acceptance is >= GLM53_NG_ACC_LONG (0.35): a probe of
    GLM53_NG_PROBE tokens (default 2), which is how a job (re-)enters drafting;
  - matched steps that do not draft let a low acceptance recover slowly toward 0.5 (GLM53_NG_RECOVER per step).
wants_draft() tells k_lookahead whether the next step will draft, so single-token steps keep the lookahead otherwise.
Exact: drafts only decide what gets verified, the target model's tokens are unchanged (verification shapes can still
flip an fp16 near-tie, as with stock speculation). Measured (README): copy +27 %, file edit +13 %, others -6..-1 %.
"""
import os
import torch

MIN = int(os.environ.get("GLM53_NG_MIN", "4"))
ACC = float(os.environ.get("GLM53_NG_ACC", "0.7"))
LONG = int(os.environ.get("GLM53_NG_LONG", "12"))
ALPHA = float(os.environ.get("GLM53_NG_ALPHA", "0.3"))
KEEP = float(os.environ.get("GLM53_NG_KEEP", "0.5"))
ACC_LONG = float(os.environ.get("GLM53_NG_ACC_LONG", "0.35"))   # a long match probes only above this acceptance
PROBE = int(os.environ.get("GLM53_NG_PROBE", "2"))               # draft length of a probe
RECOVER = float(os.environ.get("GLM53_NG_RECOVER", "0.01"))      # acceptance recovery per matched step without a draft
STATS = {"calls": 0, "matched": 0, "drafted": 0, "draft_tokens": 0, "steps": 0, "accepted": 0, "rejected": 0,
         "skip_short": 0, "skip_acc": 0}


def _draft_len(p, kmax):
    """largest k <= kmax whose k-th draft token is still accepted with probability p**k >= KEEP"""
    k, q = 0, 1.0
    while k < kmax:
        q *= p
        if q < KEEP:
            break
        k += 1
    return max(1, k)


def wants_draft(job):
    """Would the policy draft for this job right now (lookahead hybrid: no lookahead launch when it would)?"""
    sam = getattr(job, "sam", None)
    if sam is None:
        return False
    seq = job.sequences[0].sequence_ids.torch()
    last = getattr(job, "_ng_last", None)
    if last is not None and last[0] == seq.shape[-1]:
        L = last[1]      # same history as this step's get_ngram_draft (a second accept_tensor sees no new tokens)
    else:
        beg, end = sam.accept_tensor(seq)
        L = int(end - beg)
        job._ng_last = (seq.shape[-1], L, int(end))
    if L < max(MIN, job.generator.ngram_match_min):
        return False
    p = getattr(job, "_ng_acc", 0.5)
    return p >= ACC or (L >= LONG and p >= ACC_LONG)


def install(generator_cls, job_cls):
    if getattr(job_cls, "_ng_policy", False):
        return
    orig_iter = generator_cls.iterate_gen

    def get_ngram_draft(self, draft_length):
        assert self.sam
        STATS["calls"] += 1
        seq = self.sequences[0].sequence_ids.torch()
        last = getattr(self, "_ng_last", None)
        if last is not None and last[0] == seq.shape[-1]:
            L, end = last[1], last[2] if len(last) > 2 else None
        else:
            beg, end = self.sam.accept_tensor(seq)
            L = int(end - beg)
            self._ng_last = (seq.shape[-1], L, int(end))
        if end is None:
            return torch.empty((1, 0), dtype=torch.long)
        if not hasattr(self, "_ng_acc"):
            self._ng_acc = 0.5
        empty = torch.empty((1, 0), dtype=torch.long)
        if L < max(MIN, self.generator.ngram_match_min):
            STATS["skip_short"] += 1
            return empty
        STATS["matched"] += 1
        p = self._ng_acc
        if p >= ACC:
            k = _draft_len(p, draft_length)    # copying recently: as many as the acceptance rate makes worth verifying
        elif L >= LONG and p >= ACC_LONG:
            k = min(PROBE, draft_length)       # a long verbatim match re-arms drafting with a short probe
        else:
            STATS["skip_acc"] += 1
            if p < 0.5:
                self._ng_acc = min(0.5, p + RECOVER)   # slow recovery: a rejected streak does not disarm the job for good
            return empty
        draft = seq[:, end:end + k]
        if draft.shape[-1]:
            STATS["drafted"] += 1
            STATS["draft_tokens"] += int(draft.shape[-1])
        return draft

    def iterate_gen(self, results, draft_tokens=None):
        if draft_tokens is None:
            return orig_iter(self, results, draft_tokens)
        jobs = [j for j in self.active_jobs if j.is_prefill_done()]
        before = [(j, j.accepted_draft_tokens, j.rejected_draft_tokens) for j in jobs]
        out = orig_iter(self, results, draft_tokens)
        STATS["steps"] += 1
        for j, a0, r0 in before:
            da, dr = j.accepted_draft_tokens - a0, j.rejected_draft_tokens - r0
            STATS["accepted"] += da; STATS["rejected"] += dr
            if da + dr > 0:
                j._ng_acc = (1 - ALPHA) * getattr(j, "_ng_acc", 0.5) + ALPHA * (da / (da + dr))
        return out

    job_cls.get_ngram_draft = get_ngram_draft
    generator_cls.iterate_gen = iterate_gen
    job_cls._ng_policy = True
    if os.environ.get("GLM53_NG_LA", "1") == "1":
        try:
            from . import k_lookahead
            k_lookahead.NG_HOOK = wants_draft
        except Exception:
            pass
    print(f" -- ngram policy: min match {MIN}, acceptance >= {ACC} (EWMA {ALPHA}) or match >= {LONG}, keep {KEEP}",
          flush=True)


def summary():
    s = dict(STATS)
    s["accept_rate"] = round(s["accepted"] / max(1, s["accepted"] + s["rejected"]), 3)
    return s
