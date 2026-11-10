"""Turn checkpoints (models whose recurrent state rewinds by cursor: DeepSeek-V4, last_page.rewindable()).

exllamav3 resumes a cached prompt from a recurrent checkpoint, and a checkpoint sits on a 256-token page boundary
(the K/V pages are shared by hash, a page is hashed when it is full). An agent's next request extends the last one
(prompt + answer + tool result), so every step prefilled again the end of the last prompt and most of the answer the
model had just decoded: 100-500 tokens, a staged forward of 4-8 s, where the tokens that are really new are often a
few dozen (a short tool result), which the per-pick path takes in 1-2 s.

Here the state is also stashed where a request stops:

  turn end     when a job gives its pages back (Job.deallocate_pages: finished, stopped by vLLM, requeued), at its K/V
               position: the prompt and every answer token but the last sampled one
  prompt end   after the prompt's last forward, at its last token (for a follow-up that keeps the prompt but not the
               answer as it was generated: a client that drops the reasoning, a retried request, a template that
               renders the answer differently)

and a starting job (Job.allocate_pages) whose prompt begins with such a sequence resumes there: at the stashed
position, or below it where the two diverge when the state can be moved back exactly (the cursor rewind exllamav3
uses for rejected drafts: up to a page, and not below the rows the SWA ring still holds for a whole window). The
K/V rows of the page the resume point lies in are copied from the page that holds them (exllamav3's partial-page
reuse, which it skips for recurrent models because their checkpoints are page-aligned); the full pages below are the
prefix cache's. Whatever cannot be verified (the entry was evicted, a page of the prefix is gone, the state cannot be
moved that far) falls back to exllamav3's page-aligned resume.

The store is a small LRU beside exllamav3's RecurrentCache (whose keys are page hashes), of `entries` checkpoints
(DSV41_TURN_CKPTS, default 4, ~36 MB each for DeepSeek-V4.1), taken out of the recurrent cache's budget. An entry
that a newer one covers (same sequence, and the newer one reaches back at least as far) is dropped.

Installed by tiers.attach in mode nvme (mode nvme-exact keeps exllamav3's schedule, so its answers stay those of
exllamav3). Job.prefill must accept a start inside a page: last_page.apply relaxes its assert (_BIGMOE_MID_PAGE).
"""
import sys
import time
from collections import OrderedDict

from . import last_page

STATS = {"stashed": 0, "stash_skipped": 0, "kept": 0, "covered": 0, "evicted": 0,
         "resumed": 0, "resumed_tokens": 0, "resumed_exact": 0, "resumed_moved": 0, "not_better": 0, "missed": 0,
         "errors": 0, "stash_ms": 0.0, "resume_ms": 0.0}
STORE = OrderedDict()      # serial -> Entry, least recently used first
CONFIG = {"entries": 0, "budget_taken": False, "owner": None}
_serial = [0]


class Entry:
    """One stashed state: `stash` (PairedState.stash at `pos`), the sequence up to `pos`, and `floor`, the lowest
    position the state can be moved back to exactly."""
    __slots__ = ("stash", "pos", "floor", "ids", "size", "kind")

    def __init__(self, stash, pos, floor, ids, size, kind):
        self.stash, self.pos, self.floor, self.ids, self.size, self.kind = stash, pos, floor, ids, size, kind


def _why(key: str, why: str):
    STATS[key] += 1
    d = STATS.setdefault(key + "_why", {})
    d[why] = d.get(why, 0) + 1


def _release(e: Entry):
    from exllamav3.cache.recurrent import host_pool, note_freed
    host_pool.give(e.stash)
    note_freed(e.size)


def clear():
    while STORE:
        _release(STORE.popitem(last=False)[1])


def _own(gen):
    """The store belongs to one generator (its page table): another one starts empty."""
    if CONFIG["owner"] != id(gen):
        clear()
        CONFIG["owner"], CONFIG["budget_taken"] = id(gen), False


def _take_budget(rc, size: int):
    """The entries' host memory comes out of the recurrent cache's budget (at most half of it)."""
    if CONFIG["budget_taken"] or size <= 0:
        return
    CONFIG["budget_taken"] = True
    take = min(CONFIG["entries"] * size, int(rc.max_size) // 2)
    CONFIG["entries"] = take // size
    rc.max_size = int(rc.max_size) - CONFIG["entries"] * size
    STATS["budget_bytes"] = CONFIG["entries"] * size


def floor_of(comps: list) -> int:
    """The lowest position every side of the state can be moved back to exactly: a page below where it stands
    (compressor buffers, overlap snapshots), and a whole attention window above the first row its ring holds."""
    from exllamav3.constants import PAGE_SIZE
    window = max([int(getattr(l, "window", 0) or 0) for l in last_page.state_layers(comps)] or [0])
    floor = 0
    for st in comps:
        floor = max(floor, st.position - PAGE_SIZE, st.window_beg + window if st.window_beg > 0 else 0)
    return floor


def capture(job, kind: str):
    """Stash the job's recurrent state at its K/V position."""
    import torch
    if not CONFIG["entries"] or getattr(job, "_bigmoe_no_turn", False):
        return
    gen, rs = job.generator, job.recurrent_state
    rc = getattr(gen, "recurrent_cache", None)
    if rc is None or rs is None or not job.sequences:
        return
    seq = job.sequences[0]
    if not seq.allocated_pages:
        return
    if len(job.sequences) != 1 or job.embeddings:
        return _why("stash_skipped", "multimodal or multi-sequence")
    pos = int(seq.kv_position)
    if pos <= 0:
        return
    _own(gen)
    comps = list(rs.components)
    why = last_page.cannot_move(comps, pos)
    if why is not None:
        return _why("stash_skipped", why)
    ids = seq.sequence_ids.torch_slice(0, pos).reshape(-1).clone()
    for key, e in STORE.items():       # this very state is there (a request repeated, a job resumed and ended at once)
        if e.pos == pos and torch.equal(e.ids, ids):
            STORE.move_to_end(key)
            STATS["kept"] += 1
            return
    _take_budget(rc, int(rs.checkpoint_size))
    if not CONFIG["entries"]:
        return
    floor = min(floor_of(comps), pos)
    for key in [k for k, e in STORE.items() if e.pos <= pos and e.floor >= floor and torch.equal(e.ids, ids[:e.pos])]:
        _release(STORE.pop(key))       # the new entry reaches every resume point of this one
        STATS["covered"] += 1
    while len(STORE) >= CONFIG["entries"]:      # before the stash, so its buffers are the ones just returned
        _release(STORE.popitem(last=False)[1])
        STATS["evicted"] += 1
    t0 = time.perf_counter()
    stash = last_page.stash_at(rs, seq.sequence_ids.torch_slice, pos, rs.stash)
    _serial[0] += 1
    STORE[_serial[0]] = Entry(stash, pos, floor, ids, int(stash["checkpoint_size"]), kind)
    STATS["stash_ms"] = round(STATS["stash_ms"] + (time.perf_counter() - t0) * 1e3, 1)
    STATS["stashed"] += 1
    STATS[f"stashed_{kind}"] = STATS.get(f"stashed_{kind}", 0) + 1


def find_page(pagetable, prev_hash, want, exclude=None):
    """A page holding the K/V rows of the tokens `want` right after the page chain `prev_hash` (exllamav3's own
    rule for partial-page reuse: prev_hash and the page's token ids describe its rows)."""
    import torch
    m = int(want.shape[-1])
    best = None
    for page in pagetable.all_pages:
        if page.kv_position < m or page.prev_hash != prev_hash or page is exclude:
            continue
        if torch.equal(page.sequence[0, :m], want) and (best is None or page.access_serial > best.access_serial):
            best = page
    return best


def plan(job):
    """Before the job's pages are allocated: the entry its prompt can resume from, where, and the page whose rows
    the resume point's page needs (protected from this allocation's evictions). None: no entry applies."""
    from exllamav3.constants import PAGE_SIZE
    if not CONFIG["entries"] or not STORE:
        return None
    gen = job.generator
    if getattr(gen, "recurrent_cache", None) is None or CONFIG["owner"] != id(gen):
        return None
    if len(job.sequences) != 1 or job.embeddings or job.sequences[0].page_hashes is None:
        return None
    seq = job.sequences[0]
    ids = seq.sequence_ids.torch().reshape(-1)
    n = int(ids.shape[0])
    best = None
    for key in reversed(STORE):
        e = STORE[key]
        k = min(e.pos, n - 1)
        if k <= 0 or (best is not None and k <= best[0]):
            continue
        neq = (ids[:k] != e.ids[:k]).nonzero()
        c = int(neq[0]) if neq.numel() else k
        if c <= 0 or (c != e.pos and c < e.floor):
            continue
        if best is None or c > best[0]:
            best = (c, key)
    if best is None:
        return None
    d, key = best
    e = STORE[key]
    i, m = divmod(d, PAGE_SIZE)
    src = None
    if m:
        src = find_page(gen.pagetable, seq.page_hashes[i - 1] if i else None, ids[i * PAGE_SIZE:d])
        if src is None:        # nobody holds the page's rows: its start, if the state moves back that far
            d = i * PAGE_SIZE
            if d <= 0 or d < e.floor:
                return _why("missed", "page rows")
    return key, d, src


def moved(gen, e: Entry, ids, d: int) -> dict:
    """The entry's stash with the state standing at `d` < e.pos: positions, and the Engram layers' id context."""
    out = dict(e.stash, position=d)
    for side, cache in (("target", gen.cache), ("draft", getattr(gen, "draft_cache", None))):
        s = e.stash.get(side)
        if s is None or cache is None:
            continue
        s = dict(s, position=d)
        for k, l in cache.get_all_recurrent_layers().items():
            if type(l).__name__ == "PLELayerState" and k in s:
                s[k] = (s[k][0], last_page.engram_ids(l, ids.reshape(1, -1), d))
        out[side] = s
    return out


def resume(job, planned):
    """After the job's pages are allocated (cached prefix C, a page boundary with a checkpoint): move the job to the
    planned resume point if that is further."""
    import torch
    from exllamav3.constants import PAGE_SIZE
    key, d, src = planned
    e = STORE.get(key)
    gen, seq = job.generator, job.sequences[0]
    if e is None or job.recurrent_state is None:
        return
    t0 = time.perf_counter()
    C = int(seq.kv_position)
    pages = seq.allocated_pages
    ids = seq.sequence_ids.torch().reshape(-1)
    for j in range(d // PAGE_SIZE):            # every page below the resume point holds its rows
        if pages[j].kv_position != PAGE_SIZE:
            d, src = j * PAGE_SIZE, None
            if d < e.floor:
                return _why("missed", "prefix pages")
            break
    if d <= C:
        STATS["not_better"] += 1
        return
    i, m = divmod(d, PAGE_SIZE)
    dst = pages[i] if m else None
    if m and dst.kv_position != PAGE_SIZE:     # (a cached page with this very content already has the rows)
        prev = pages[i - 1].phash if i else None
        want = ids[i * PAGE_SIZE:d]
        if src is None or src is dst or src.kv_position < m or src.prev_hash != prev \
                or not torch.equal(src.sequence[0, :m], want):
            src = find_page(gen.pagetable, prev, want, exclude=dst)     # this allocation took it
        if src is None:
            d, m = i * PAGE_SIZE, 0
            if d <= C or d < e.floor:
                return _why("missed", "page rows")
        else:
            for c in ([gen.cache] if not gen.draft_model else [gen.cache, gen.draft_cache]):
                c.copy_page(c, src.page_index, dst.page_index, m)
            dst.prev_hash = prev
            dst.sequence[:, :m].copy_(want)
            dst.kv_position = m
            dst.can_revert = False
    stash = e.stash if d == e.pos else moved(gen, e, ids, d)
    job.free_recurrent_state()
    try:
        job.recurrent_state = gen.recurrent_state_from_stashed(stash, position=d)
    except BaseException:
        # what the allocation had restored, or a prefill from the start (the pages are written again in place)
        at = job.find_recurrent_stash(C) if C else None
        if at is not None and at["position"] == C:
            job.recurrent_state = gen.recurrent_state_from_stashed(at, position=C)
        else:
            job.recurrent_state = gen.new_recurrent_state()
            seq.kv_position = 0
            job.last_recurrent_checkpoint_pos = None
        raise
    seq.kv_position = d
    job.cached_tokens += d - C
    STORE.move_to_end(key)
    STATS["resume_ms"] = round(STATS["resume_ms"] + (time.perf_counter() - t0) * 1e3, 1)
    STATS["resumed"] += 1
    STATS["resumed_tokens"] += d - C
    STATS["resumed_exact" if d == e.pos else "resumed_moved"] += 1
    STATS[f"resumed_{e.kind}"] = STATS.get(f"resumed_{e.kind}", 0) + 1


def _error(where: str, exc: BaseException):
    STATS["errors"] += 1
    STATS["last_error"] = f"{where}: {type(exc).__name__}: {exc}"
    if STATS["errors"] <= 3:
        import traceback
        print(f" !! turn_resume {where}: {type(exc).__name__}: {exc}\n{traceback.format_exc()}", file=sys.stderr,
              flush=True)


def after_prefill(job, seq, prefill_start: int, prefill_end: int):
    """Job.prefill's hook after a forward: the tail checkpoint (last_page.py), and the prompt-end entry."""
    last_page.stash_tail(job, seq, prefill_start, prefill_end)
    if seq.kv_position == len(seq.sequence_ids) - 1:
        try:
            capture(job, "prompt")
        except Exception as e:
            _error("prompt stash", e)


def install(entries: int) -> str:
    """Wrap exllamav3's Job.allocate_pages / deallocate_pages and Generator.reap_failed_job (class-wide, once)."""
    from exllamav3.generator import generator as G
    from exllamav3.generator import job as J
    CONFIG["entries"] = max(0, int(entries))
    if not getattr(J.Job.prefill, "_bigmoe_mid_page", False) or not getattr(J.Job.prefill, "_bigmoe_tail_ckpt", False):
        CONFIG["entries"] = 0
        return "turn checkpoints: off (exllamav3's Job.prefill is not the expected source)"
    if not CONFIG["entries"]:
        return "turn checkpoints: off"
    J._BIGMOE_MID_PAGE = True
    J._BIGMOE_TAIL_CKPT = after_prefill
    if not getattr(J.Job, "_bigmoe_turn_resume", False):
        alloc, dealloc, reap = J.Job.allocate_pages, J.Job.deallocate_pages, G.Generator.reap_failed_job

        def allocate_pages(self):
            planned, guard = None, None
            try:
                planned = plan(self)
                if planned is not None and planned[2] is not None:
                    guard = planned[2].phash           # not among the pages this allocation evicts first
                    self.all_unique_hashes.append(guard)
            except Exception as e:
                planned = None
                _error("plan", e)
            try:
                alloc(self)
            finally:
                if guard is not None and guard in self.all_unique_hashes:
                    self.all_unique_hashes.remove(guard)
            if planned is not None:
                try:
                    resume(self, planned)
                except Exception as e:
                    _error("resume", e)

        def deallocate_pages(self):
            try:
                capture(self, "turn")
            except Exception as e:
                _error("turn stash", e)
            dealloc(self)

        def reap_failed_job(self, job, error, results):
            job._bigmoe_no_turn = True      # a failed forward: the state is not a position of the sequence
            return reap(self, job, error, results)

        for fn, orig in ((allocate_pages, alloc), (deallocate_pages, dealloc), (reap_failed_job, reap)):
            fn.__qualname__, fn.__doc__, fn._bigmoe_orig = orig.__qualname__, orig.__doc__, orig
        J.Job.allocate_pages, J.Job.deallocate_pages = allocate_pages, deallocate_pages
        G.Generator.reap_failed_job = reap_failed_job
        J.Job._bigmoe_turn_resume = True
    return f"turn checkpoints: {CONFIG['entries']} entries (turn end, prompt end)"


def installed() -> bool:
    from exllamav3.generator import job as J
    return bool(getattr(J.Job, "_bigmoe_turn_resume", False))


def uninstall():
    """Tests: put exllamav3's methods back."""
    from exllamav3.generator import generator as G
    from exllamav3.generator import job as J
    if getattr(J.Job, "_bigmoe_turn_resume", False):
        J.Job.allocate_pages = J.Job.allocate_pages._bigmoe_orig
        J.Job.deallocate_pages = J.Job.deallocate_pages._bigmoe_orig
        G.Generator.reap_failed_job = G.Generator.reap_failed_job._bigmoe_orig
        J.Job._bigmoe_turn_resume = False
    J._BIGMOE_MID_PAGE = False
    CONFIG["entries"] = 0
    STORE.clear()


def summary() -> dict:
    return dict(STATS, limit=CONFIG["entries"],
                entries=[{"kind": e.kind, "pos": e.pos, "floor": e.floor} for e in STORE.values()])
