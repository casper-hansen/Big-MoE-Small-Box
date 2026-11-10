"""Cost-aware n-gram drafting for GLM-5.3-Flash (speculation="ngram").

A draft token is not free here: every token of a verify window brings its own routed experts, and the cold ones come
from RAM or NVMe. exllamav3's prompt lookup drafts whenever the last ngram_match_min tokens occur earlier in the
context, which on prose mostly proposes tokens that are rejected. This policy drafts only when the model is copying:
  - the suffix match is at least MIN tokens long, and
  - the job's recent per-token draft acceptance (an EWMA with weight ALPHA, starting at 0.5) is at least ACC: the draft
    is exllamav3's, cut to the length whose last token is still accepted with probability >= KEEP; or
  - the match is at least LONG tokens and the acceptance at least ACC_LONG: a probe of PROBE tokens, which is how a job
    enters drafting again;
  - a matched step that does not draft lets a low acceptance recover toward 0.5 by RECOVER.
Drafts only decide what is verified; the tokens are the target model's.
"""
import torch

MIN = 4
ACC = 0.7
LONG = 12
ALPHA = 0.3
KEEP = 0.5
ACC_LONG = 0.35
PROBE = 2
RECOVER = 0.01
MM_FIRST = 1_000_000_000   # exllamav3 FIRST_MM_EMBEDDING_INDEX: image placeholder ids
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


def install(generator_cls, job_cls):
    if getattr(job_cls, "_ng_policy", False):
        return
    orig_iter = generator_cls.iterate_gen

    def get_ngram_draft(self, draft_length):
        assert self.sam
        STATS["calls"] += 1
        seq = self.sequences[0].sequence_ids.torch()
        beg, end = self.suffix_match(seq)      # exllamav3's per-round cache: the automaton sees each token once
        L = int(end - beg)
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
        mm = (draft[0] >= MM_FIRST).nonzero()    # an image's placeholder ids are never sampled: draft up to them
        if mm.numel():
            draft = draft[:, :int(mm[0])]
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
    print(f" -- ngram policy: min match {MIN}, acceptance >= {ACC} (EWMA {ALPHA}) or match >= {LONG}, keep {KEEP}",
          flush=True)


def summary():
    s = dict(STATS)
    s["accept_rate"] = round(s["accepted"] / max(1, s["accepted"] + s["rejected"]), 3)
    return s
