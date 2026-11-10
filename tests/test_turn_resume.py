"""engine/turn_resume.py: turn checkpoints (stash where a request stops, resume there). Fakes of exllamav3's job, page
table and DeepSeek-V4 state; needs torch and exllamav3 (skips without them), no GPU."""
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("exllamav3.cache.recurrent")

from bigmoe.engine import last_page, turn_resume  # noqa: E402

PAGE = 256


class DSV4LayerState:
    window = 128


class PLELayerState:
    win, ctx = 0, 3

    def __init__(self):
        self.conv_state = torch.zeros((2, 4, 8), dtype=torch.half)
        self.id_state = torch.full((2, 5), 99, dtype=torch.long)
        self.module = types.SimpleNamespace(_prepare_ids=lambda ids: ids * 10,
                                            ple_embedding=types.SimpleNamespace(eos_token_id=1))

    def stash(self, slot, position=0):
        return (self.conv_state[slot, :, :self.win].clone(), self.id_state[slot, :self.ctx].clone())


class Cache:
    def __init__(self):
        self.layers = {"attn.0": DSV4LayerState(), "ple.0": PLELayerState()}
        self.copies = []

    def get_all_recurrent_layers(self):
        return self.layers

    def copy_page(self, target, src, dst, n):
        self.copies.append((src, dst, n))


class DSV4State:
    def __init__(self, cache, position, window_beg):
        self.cache, self.position, self.window_beg, self.slot = cache, position, window_beg, 0

    def rollback_capacity(self):
        return self.position - self.window_beg

    def stash(self):
        return {"position": self.position, "window_beg": self.window_beg, "checkpoint_size": 100,
                **{k: (("ring", self.position) if k.startswith("attn") else l.stash(self.slot, self.position))
                   for k, l in self.cache.get_all_recurrent_layers().items()}}


class Pair:
    checkpoint_size = 100

    def __init__(self, target):
        self.target, self.draft, self.components = target, None, [target]

    def stash(self):
        return {"position": self.target.position, "checkpoint_size": 100, "target": self.target.stash()}

    def free(self):
        pass


class Ids:
    def __init__(self, ids):
        self.t = torch.tensor([ids], dtype=torch.long)

    def __len__(self):
        return self.t.shape[-1]

    def torch(self):
        return self.t

    def torch_slice(self, a, b):
        return self.t[:, a:b]


def page(index, kv=0, prev=None, phash=None, tokens=()):
    seq = torch.zeros((1, PAGE), dtype=torch.long)
    seq[0, :len(tokens)] = torch.tensor(list(tokens), dtype=torch.long)
    return types.SimpleNamespace(page_index=index, kv_position=kv, prev_hash=prev, phash=phash or f"r{index}",
                                 sequence=seq, access_serial=index, can_revert=True)


class Gen:
    def __init__(self, pages):
        self.cache, self.draft_cache, self.draft_model = Cache(), None, None
        self.recurrent_cache = types.SimpleNamespace(max_size=1000)
        self.pagetable = types.SimpleNamespace(all_pages=pages)
        self.restored = []

    def recurrent_state_from_stashed(self, stashed, position):
        assert stashed["position"] == position == stashed["target"]["position"]
        self.restored.append((stashed, position))
        return Pair(DSV4State(self.cache, position, stashed["target"]["window_beg"]))


def job_of(gen, ids, kv, pages, state=None, hashes=None):
    seq = types.SimpleNamespace(sequence_ids=Ids(ids), kv_position=kv, allocated_pages=pages, page_hashes=hashes)
    j = types.SimpleNamespace(generator=gen, recurrent_state=state, sequences=[seq], embeddings=[], cached_tokens=0,
                              all_unique_hashes=[], last_recurrent_checkpoint_pos=None)
    j.free_recurrent_state = lambda: setattr(j, "recurrent_state", None)
    j.find_recurrent_stash = lambda pos: None
    return j


@pytest.fixture(autouse=True)
def store():
    turn_resume.STORE.clear()
    turn_resume.CONFIG.update(entries=4, budget_taken=False, owner=None)
    yield
    turn_resume.STORE.clear()
    turn_resume.CONFIG.update(entries=0, budget_taken=False, owner=None)


def finished_turn(n=700, position=None, window_beg=256, start=1000):
    """A job that stops at K/V position n: pages 0 and 1 full, page 2 holds the rest."""
    ids = list(range(start, start + n + 1))                  # the last sampled token is not in the cache
    pages = [page(0, PAGE, None, "h0", ids[:PAGE]), page(1, PAGE, "h0", "h1", ids[PAGE:2 * PAGE]),
             page(2, n - 2 * PAGE, "h1", None, ids[2 * PAGE:n])]
    gen = Gen(pages + [page(i) for i in range(3, 8)])
    st = DSV4State(gen.cache, position or n, window_beg)
    return gen, job_of(gen, ids, n, pages, Pair(st)), ids


def test_capture_stashes_the_state_at_the_kv_position():
    gen, job, ids = finished_turn(position=703)               # 3 unresolved draft positions at the end
    turn_resume.capture(job, "turn")
    (e,) = turn_resume.STORE.values()
    assert (e.pos, e.kind, e.ids.tolist()) == (700, "turn", ids[:700])
    assert e.stash["position"] == 700 and e.stash["target"]["attn.0"] == ("ring", 700)
    assert e.stash["target"]["ple.0"][1].tolist() == [16970, 16980, 16990]    # the ids before 700, not the end's
    assert e.floor == max(703 - PAGE, 256 + 128)              # a page below the state, a window above the ring start
    assert job.recurrent_state.target.position == 703         # the live state is untouched
    assert gen.recurrent_cache.max_size == 1000 - 4 * 100     # the entries' memory comes out of the checkpoint budget
    turn_resume.capture(job, "turn")                          # the same state again: kept, not stashed twice
    assert len(turn_resume.STORE) == 1 and turn_resume.STATS["kept"] >= 1


def test_capture_skips_states_it_cannot_move():
    gen, job, _ = finished_turn(position=703, window_beg=640)     # the ring starts less than a window below 700
    before = turn_resume.STATS["stash_skipped"]
    turn_resume.capture(job, "turn")
    assert not turn_resume.STORE and turn_resume.STATS["stash_skipped"] == before + 1
    gen, job, _ = finished_turn(position=700, window_beg=640)     # standing at 700, the state is stashed as it is
    turn_resume.capture(job, "prompt")
    (e,) = turn_resume.STORE.values()
    assert e.floor == 700                                         # ... and resumes only there
    job._bigmoe_no_turn = True                                    # a failed forward
    turn_resume.STORE.clear()
    turn_resume.capture(job, "turn")
    assert not turn_resume.STORE


def test_a_newer_entry_drops_the_ones_it_covers_and_the_store_is_lru():
    gen, job, ids = finished_turn(n=600)
    turn_resume.capture(job, "prompt")                            # floor 384
    seq = job.sequences[0]
    seq.sequence_ids = Ids(ids + list(range(5000, 5101)))
    seq.kv_position = 700
    seq.allocated_pages[2].kv_position = 700 - 2 * PAGE
    job.recurrent_state.target.position = 700
    turn_resume.capture(job, "turn")                              # same sequence, floor max(444, 384) > 384: both stay
    assert [(e.kind, e.pos, e.floor) for e in turn_resume.STORE.values()] == [("prompt", 600, 384), ("turn", 700, 444)]
    job.recurrent_state.target.position = 640                     # (a state further back on the same sequence)
    seq.kv_position = 640
    turn_resume.capture(job, "turn")                              # floor 384 <= both floors, but covers only pos <= 640
    assert [(e.kind, e.pos) for e in turn_resume.STORE.values()] == [("turn", 700), ("turn", 640)]
    for k in range(4):                                            # other conversations: least recently used goes first
        g2, j2, _ = finished_turn(n=600 + k, start=20000 * (k + 1))
        j2.generator = gen
        turn_resume.capture(j2, "turn")
    assert len(turn_resume.STORE) == 4 and all(e.ids[0] >= 20000 for e in turn_resume.STORE.values())


def next_request(gen, ids, extra, cached=256):
    """The next request of the conversation: `ids` + extra tokens, allocated by exllamav3 with `cached` tokens
    resumable (a checkpoint at that page boundary)."""
    new = ids + list(extra)
    n_pages = len(new) // PAGE + 1
    old = {p.phash: p for p in gen.pagetable.all_pages}
    hashes = [f"h{i}" if new[:(i + 1) * PAGE] == ids[:(i + 1) * PAGE] else f"n{i}" for i in range((len(new) - 1) // PAGE)]
    free = [p for p in gen.pagetable.all_pages if p.kv_position == 0]
    pages = []
    for i in range(n_pages):
        h = hashes[i] if i < len(hashes) else None
        if h in old:
            pages.append(old[h])
        else:
            p = free.pop()
            p.phash, p.prev_hash, p.kv_position = h or f"u{i}", None, 0
            pages.append(p)
    return job_of(gen, new, cached, pages, Pair(DSV4State(gen.cache, cached, 0)), hashes)


def test_resume_at_the_turn_end():
    gen, job, ids = finished_turn()
    turn_resume.capture(job, "turn")
    nxt = next_request(gen, ids, range(7000, 7100))               # the sampled token, then 99 new ones
    planned = turn_resume.plan(nxt)
    key, d, src = planned
    assert d == 700 and src is job.sequences[0].allocated_pages[2]
    turn_resume.resume(nxt, planned)
    seq = nxt.sequences[0]
    dst = seq.allocated_pages[2]
    assert seq.kv_position == 700 and nxt.cached_tokens == 700 - 256
    assert gen.cache.copies == [(src.page_index, dst.page_index, 700 - 512)]      # the partial page's rows
    assert (dst.kv_position, dst.prev_hash, dst.can_revert) == (188, "h1", False)
    assert dst.sequence[0, :188].tolist() == ids[512:700]
    (stashed, pos), = gen.restored
    assert pos == 700 and stashed is turn_resume.STORE[key].stash
    assert nxt.recurrent_state.target.position == 700


def test_resume_below_the_turn_end_moves_the_state():
    gen, job, ids = finished_turn()
    turn_resume.capture(job, "turn")                              # floor 444
    nxt = next_request(gen, ids[:650], range(7000, 7100))         # the request diverges at 650
    planned = turn_resume.plan(nxt)
    assert planned[1] == 650
    turn_resume.resume(nxt, planned)
    (stashed, pos), = gen.restored
    assert pos == 650 and stashed["target"]["position"] == 650
    assert stashed["target"]["ple.0"][1].tolist() == [16470, 16480, 16490]        # the id context of 650
    assert turn_resume.STORE[planned[0]].stash["position"] == 700                 # the entry itself is unchanged
    assert nxt.sequences[0].kv_position == 650 and gen.cache.copies[0][2] == 650 - 512
    turn_resume.STORE.clear()
    turn_resume.capture(job, "turn")
    assert turn_resume.plan(next_request(gen, ids[:400], range(7000, 7100))) is None   # below the floor: no resume


def test_resume_keeps_what_exllamav3_restored_when_it_is_further_or_pages_are_gone():
    gen, job, ids = finished_turn()
    turn_resume.capture(job, "turn")
    nxt = next_request(gen, ids[:650], range(7000, 7100), cached=768)     # (cannot happen; the rule is "further only")
    turn_resume.resume(nxt, turn_resume.plan(nxt))
    assert nxt.sequences[0].kv_position == 768 and not gen.restored
    gen, job, ids = finished_turn()
    turn_resume.capture(job, "turn")
    nxt = next_request(gen, ids, range(7000, 7100), cached=0)
    planned = turn_resume.plan(nxt)
    nxt.sequences[0].allocated_pages[1].kv_position = 0                   # a page of the prefix was evicted
    turn_resume.resume(nxt, planned)
    assert nxt.sequences[0].kv_position == 0 and not gen.restored         # 256 is below the entry's floor
    gen, job, ids = finished_turn()
    turn_resume.capture(job, "turn")
    job.sequences[0].allocated_pages[2].kv_position = 0                   # the partial page was taken by another job
    nxt = next_request(gen, ids, range(7000, 7100))
    planned = turn_resume.plan(nxt)
    assert planned[1] == 512 and planned[2] is None                       # its start instead: a page boundary
    turn_resume.resume(nxt, planned)
    assert nxt.sequences[0].kv_position == 512 and gen.restored[0][1] == 512 and not gen.cache.copies


def test_install_wraps_exllamav3_and_needs_the_patched_prefill():
    J = pytest.importorskip("exllamav3.generator.job")
    G = pytest.importorskip("exllamav3.generator.generator")
    stock = J.Job.prefill, J.Job.allocate_pages, J.Job.deallocate_pages, G.Generator.reap_failed_job
    try:
        assert turn_resume.install(4).endswith("off (exllamav3's Job.prefill is not the expected source)")
        assert J.Job.allocate_pages is stock[1]
        last_page.apply(512, 6, tail_ckpt=True)
        assert turn_resume.install(4) == "turn checkpoints: 4 entries (turn end, prompt end)"
        assert J._BIGMOE_MID_PAGE and J._BIGMOE_TAIL_CKPT is turn_resume.after_prefill
        assert J.Job.allocate_pages._bigmoe_orig is stock[1] and J.Job.deallocate_pages._bigmoe_orig is stock[2]
        assert "_BIGMOE_MID_PAGE" in J.Job.prefill.__code__.co_names
        turn_resume.install(4)                                            # idempotent
        assert J.Job.allocate_pages._bigmoe_orig is stock[1]
    finally:
        turn_resume.uninstall()
        J.Job.prefill = stock[0]
        J._BIGMOE_TAIL_CKPT = None
    assert (J.Job.allocate_pages, J.Job.deallocate_pages, G.Generator.reap_failed_job) == stock[1:]
