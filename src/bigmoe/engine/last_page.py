"""Last-page prefill split sized for staged prefill (exllamav3 generator/job.py, both backends).

For models with recurrent state (GLM-5.3's GDN / KDA layers, DeepSeek-V4's SWA ring and compressor buffers), exllamav3
ends a prompt's prefill on its last 256-token page boundary and runs the rest of the prompt (up to 255 tokens) as a
separate forward, so the state can be checkpointed at that page for a follow-up that shares the prompt. Here a prefill
forward costs nearly the same whatever its length once a layer call has EC_STAGE_MIN picks (it stages the experts), so
a tail of more than EC_STAGE_MIN / experts_per_token tokens (63 for GLM, 85 for DeepSeek, at 512 picks) is a whole
extra forward, while a shorter one runs on the per-pick path for a fraction of that.

The split is kept only for such short tails. A longer tail stays in the prompt's last forward: no extra forward now,
and a follow-up resumes from the last chunk-boundary checkpoint instead. Applied by tiers.attach: Job.prefill is
recompiled from its source with one condition added (the installed file is not modified); a source without the expected
stock snippet is left alone with a warning.

Tail checkpoints (models whose recurrent state rewinds by cursor: DeepSeek-V4, rewindable()): no split at all. The prompt's last forward runs to
its end, and the checkpoint at the last page boundary is stashed afterwards from the end state with its position moved
back to the boundary. DeepSeek-V4's recurrent state is position-derived and keeps at least a page of history (the SWA
ring, the compressor rings and the CSA overlap snapshots: exllamav3 rewinds them by moving the cursor, cache/dsa.py), and
V4.1's Engram layers carry only the last token ids (no conv state), which are rebuilt for the boundary. The checkpoint
is the one the split would have stashed (same position, same page), without the tail's forward of its own, which re-read
from NVMe most of the experts the forward before it had just read (~3-4 s per agent step whose tail is 1-85 tokens, and
a checkpoint that longer tails did not get at all).
"""
import ast
import inspect
import sys
import textwrap

STOCK = """\
            if self.generator.recurrent_cache is not None:
                seqlen = len(seq.sequence_ids) - 1
                last_page_b = seqlen // PAGE_SIZE * PAGE_SIZE
                if prefill_start < last_page_b <= prefill_end:
                    prefill_end = last_page_b
                    recurrent_last_page = True
"""
PATCHED = """\
            if self.generator.recurrent_cache is not None:
                seqlen = len(seq.sequence_ids) - 1
                last_page_b = seqlen // PAGE_SIZE * PAGE_SIZE
                # bigmoe (engine/last_page.py): split off the last partial page only when it costs no staged forward
                if prefill_start < last_page_b <= prefill_end and seqlen - last_page_b <= _BIGMOE_LAST_PAGE_MAX:
                    prefill_end = last_page_b
                    recurrent_last_page = True
"""
STOCK_STASH = """\
                if recurrent_last_page:
                    self.maybe_stash_recurrent(self.generator.recurrent_cache, PAGE_SIZE)
"""
PATCHED_STASH = """\
                if recurrent_last_page:
                    self.maybe_stash_recurrent(self.generator.recurrent_cache, PAGE_SIZE)
                elif _BIGMOE_TAIL_CKPT is not None and self.generator.recurrent_cache is not None:
                    _BIGMOE_TAIL_CKPT(self, seq, prefill_start, prefill_end)   # bigmoe (engine/last_page.py)
"""

# tail checkpoints: stashed / skipped (the state could not be moved back to the boundary) / page not complete
STATS = {"tail_ckpt": 0, "tail_ckpt_skipped": 0, "tail_ckpt_page": 0}


def tail_limit(stage_min: int, experts_per_token: int) -> int:
    """Longest tail whose layer calls stay under the staging threshold (picks < stage_min)."""
    return max(0, (stage_min - 1) // max(1, experts_per_token))


def rewindable(model) -> tuple:
    """(ok, what): every module with recurrent state is a DeepSeek-V4 attention (position-derived state, cursor rewind
    of up to a page) or an Engram / PLE layer without conv state (its state is the last token ids)."""
    seen, bad = set(), []

    def walk(m):
        caps = getattr(m, "caps", None) or {}
        if caps.get("recurrent_cache"):
            name = type(m).__name__
            if name == "PLELayer" and int(getattr(m, "conv_state_len", 1)) == 0:
                seen.add(name)
            elif name == "DSV4Attention":
                seen.add(name)
            else:
                bad.append(name)
        for c in getattr(m, "modules", []) or []:
            walk(c)
    for top in getattr(model, "modules", []) or []:
        walk(top)
    return (not bad and "DSV4Attention" in seen), (sorted(seen) + [f"unsupported {b}" for b in sorted(set(bad))])


def _skip(why: str):
    STATS["tail_ckpt_skipped"] += 1
    STATS.setdefault("skipped_why", {})
    STATS["skipped_why"][why] = STATS["skipped_why"].get(why, 0) + 1


def stash_tail(job, seq, prefill_start: int, prefill_end: int):
    """After a prompt's last forward (no last-page split): stash the recurrent checkpoint at the last page boundary
    from the end state, as Job.prefill's split would have. Every side of the paired state (the target and DSpark's
    drafter blocks) is a DSV4State and is moved back by its cursor."""
    from exllamav3.constants import PAGE_SIZE
    seqlen = len(seq.sequence_ids) - 1
    lpb = seqlen // PAGE_SIZE * PAGE_SIZE
    if not (prefill_start < lpb < prefill_end) or seq.kv_position != prefill_end:
        return
    rs = job.recurrent_state
    comps = list(getattr(rs, "components", None) or [])
    if not comps:
        return _skip("no state")
    for st in comps:
        if type(st).__name__ != "DSV4State":
            return _skip(type(st).__name__)
        if not 0 < st.position - lpb <= st.rollback_capacity():
            return _skip("rollback")
    page = seq.allocated_pages[lpb // PAGE_SIZE - 1]
    if page.kv_position != PAGE_SIZE:
        STATS["tail_ckpt_page"] += 1
        return
    layers = list({id(l): l for st in comps for l in st.cache.get_all_recurrent_layers().values()}.values())
    for l in layers:
        name = type(l).__name__
        if name not in ("DSV4LayerState", "PLELayerState") or (name == "PLELayerState" and l.win):
            return _skip(name)
    from exllamav3.cache.recurrent import host_copy
    import torch
    patched = []
    for l in layers:
        if type(l).__name__ != "PLELayerState":
            continue
        ctx = l.ctx
        lo = max(0, lpb - ctx)
        ids = l.module._prepare_ids(seq.sequence_ids.torch_slice(lo, lpb).reshape(1, -1)).reshape(-1)
        if ids.numel() < ctx:
            pad = torch.full((ctx - ids.numel(),), l.module.ple_embedding.eos_token_id, dtype=ids.dtype)
            ids = torch.cat([pad, ids])
        ids = ids.to(l.id_state.dtype)

        def stash(slot, position=0, _l=l, _ids=ids):
            return host_copy(_l.conv_state[slot, :, :_l.win]), host_copy(_ids)
        l.stash = stash                # instance attribute: this stash only
        patched.append(l)
    pos = [st.position for st in comps]
    for st in comps:
        st.position = lpb
    try:
        job.generator.recurrent_cache.put(page.phash, rs)
    finally:
        for st, p in zip(comps, pos):
            st.position = p
        for l in patched:
            del l.stash
    job.last_recurrent_checkpoint_pos = lpb
    STATS["tail_ckpt"] += 1


def apply(stage_min: int, experts_per_token: int, tail_ckpt: bool = False) -> str:
    """Recompile exllamav3's Job.prefill with the tail limit, or with no split and the tail-checkpoint hook
    (tail_ckpt: the model's state rewinds, rewindable()). Returns a status line."""
    from exllamav3.generator import job as J
    limit = tail_limit(stage_min, experts_per_token)
    J._BIGMOE_LAST_PAGE_MAX = limit
    J._BIGMOE_TAIL_CKPT = None
    if not getattr(J.Job.prefill, "_bigmoe_last_page", False):
        src = textwrap.dedent(inspect.getsource(J.Job.prefill))
        # dedent(getsource) leaves the method body at 4 spaces; the snippets sit two levels deeper
        stock, patched = (textwrap.indent(textwrap.dedent(t), " " * 8) for t in (STOCK, PATCHED))
        if src.count(stock) != 1:
            print(" !! last_page: exllamav3's Job.prefill lacks the stock last-page split; prefill left as is",
                  file=sys.stderr, flush=True)
            return "last-page split: stock (unexpected exllamav3 source)"
        src = src.replace(stock, patched)
        stock_s, patched_s = (textwrap.indent(textwrap.dedent(t), " " * 12) for t in (STOCK_STASH, PATCHED_STASH))
        has_hook = src.count(stock_s) == 1
        if has_hook:
            src = src.replace(stock_s, patched_s)
        tree = ast.parse(src)
        ast.increment_lineno(tree, inspect.getsourcelines(J.Job.prefill)[1] - 1)   # tracebacks point into job.py
        ns = {}
        exec(compile(tree, J.__file__, "exec"), J.__dict__, ns)
        fn = ns["prefill"]
        fn._bigmoe_last_page = True
        fn._bigmoe_tail_ckpt = has_hook
        fn.__qualname__ = J.Job.prefill.__qualname__
        J.Job.prefill = fn
    if tail_ckpt:
        if not getattr(J.Job.prefill, "_bigmoe_tail_ckpt", False):
            print(" !! last_page: exllamav3's Job.prefill lacks the stock last-page stash; tail checkpoints off",
                  file=sys.stderr, flush=True)
        else:
            J._BIGMOE_LAST_PAGE_MAX = -1
            J._BIGMOE_TAIL_CKPT = stash_tail
            return "no last-page split: tail checkpoints stashed after the prompt's last forward"
    return f"last-page split for tails <= {limit} tokens"
