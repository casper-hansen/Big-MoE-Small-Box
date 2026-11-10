"""engine/last_page.py: the last-page prefill split limit. The patch test needs exllamav3 (skips without it), no GPU."""
import pytest

from bigmoe.engine import last_page


def test_tail_limit():
    assert last_page.tail_limit(512, 6) == 85      # DeepSeek: 85 * 6 = 510 picks, still the per-pick path
    assert last_page.tail_limit(512, 8) == 63      # GLM
    assert last_page.tail_limit(0, 6) == 0


def test_patch_applies_to_the_pinned_exllamav3():
    pytest.importorskip("torch")
    J = pytest.importorskip("exllamav3.generator.job")
    stock = J.Job.prefill
    try:
        assert last_page.apply(512, 8) == "last-page split for tails <= 63 tokens"
        fn = J.Job.prefill
        assert fn is not stock and fn._bigmoe_last_page and J._BIGMOE_LAST_PAGE_MAX == 63
        assert "_BIGMOE_LAST_PAGE_MAX" in fn.__code__.co_names
        assert fn.__code__.co_filename == stock.__code__.co_filename   # tracebacks still point at job.py
        assert last_page.apply(512, 6) == "last-page split for tails <= 85 tokens" and J.Job.prefill is fn
    finally:
        J.Job.prefill = stock


def test_tail_checkpoints_drop_the_split_and_hook_the_stash():
    pytest.importorskip("torch")
    J = pytest.importorskip("exllamav3.generator.job")
    stock = J.Job.prefill
    try:
        assert last_page.apply(512, 6, tail_ckpt=True).startswith("no last-page split")
        assert J._BIGMOE_LAST_PAGE_MAX == -1 and J._BIGMOE_TAIL_CKPT is last_page.stash_tail
        assert "_BIGMOE_TAIL_CKPT" in J.Job.prefill.__code__.co_names
        assert last_page.apply(512, 8) == "last-page split for tails <= 63 tokens"    # a model that cannot rewind
        assert J._BIGMOE_LAST_PAGE_MAX == 63 and J._BIGMOE_TAIL_CKPT is None
    finally:
        J.Job.prefill = stock
        J._BIGMOE_TAIL_CKPT = None


def _fake_tail_job(position=8491, window_beg=8192, win=0, draft=None):
    import types
    torch = pytest.importorskip("torch")
    pytest.importorskip("exllamav3.cache.recurrent")

    class DSV4LayerState:
        def stash(self, slot, position):
            return ("ring", position)

    class PLELayerState:
        def __init__(self):
            self.win, self.ctx = win, 3
            self.conv_state = torch.zeros((2, 4, 8), dtype=torch.half)
            self.id_state = torch.full((2, 5), 99, dtype=torch.long)      # the end state's ids: must not be stashed
            self.module = types.SimpleNamespace(_prepare_ids=lambda ids: ids * 10,
                                                ple_embedding=types.SimpleNamespace(eos_token_id=1))

        def stash(self, slot, position=0):
            return (self.conv_state[slot, :, :self.win].clone(), self.id_state[slot, :self.ctx].clone())

    layers = {"attn.0": DSV4LayerState(), "ple.0": PLELayerState()}

    class DSV4State:
        def __init__(self):
            self.position, self.window_beg, self.slot = position, window_beg, 1
            self.cache = types.SimpleNamespace(get_all_recurrent_layers=lambda: layers)

        def rollback_capacity(self):
            return self.position - self.window_beg

    st = DSV4State()
    if draft == "dsv4":                                # DSpark's drafter blocks: a DSV4State of their own
        draft = DSV4State()
        draft.position = position + 1
    put = []

    def put_fn(key, rs):
        put.append((key, [c.position for c in rs.components],
                {k: l.stash(rs.target.slot, rs.target.position) for k, l in layers.items()}))
    ids = torch.arange(9000).reshape(1, -1)

    class Ids:                                         # exllamav3's SequenceIds: 8,492 prompt tokens
        def __len__(self):
            return 8492

        def torch_slice(self, a, b):
            return ids[:, a:b]
    seq = types.SimpleNamespace(sequence_ids=Ids(), kv_position=8491,
                                allocated_pages=[types.SimpleNamespace(kv_position=256, phash=f"p{i}") for i in range(34)])
    comps = [st] + ([draft] if draft is not None else [])
    job = types.SimpleNamespace(recurrent_state=types.SimpleNamespace(target=st, draft=draft, components=comps),
                                generator=types.SimpleNamespace(recurrent_cache=types.SimpleNamespace(put=put_fn)),
                                last_recurrent_checkpoint_pos=8192)
    return job, seq, st, layers, put


def test_stash_tail_checkpoints_the_last_page_boundary_from_the_end_state():
    job, seq, st, layers, put = _fake_tail_job()
    before = dict(last_page.STATS)
    last_page.stash_tail(job, seq, 8192, 8491)       # prompt of 8,492 tokens: last page boundary 8,448, tail 43
    (key, pos, stashed), = put
    assert key == "p32" and pos == [8448]              # the page ending at 8,448, stashed at that position
    assert stashed["attn.0"] == ("ring", 8448)
    assert stashed["ple.0"][1].tolist() == [84450, 84460, 84470]   # prepared ids of tokens 8,445..8,447, not the end's
    assert st.position == 8491 and job.last_recurrent_checkpoint_pos == 8448
    assert "stash" not in vars(layers["ple.0"])        # the per-stash override is gone
    assert last_page.STATS["tail_ckpt"] == before["tail_ckpt"] + 1


def test_stash_tail_moves_every_side_of_the_paired_state():
    job, seq, st, layers, put = _fake_tail_job(draft="dsv4")
    d = job.recurrent_state.draft
    last_page.stash_tail(job, seq, 8192, 8491)
    (key, pos, _), = put
    assert pos == [8448, 8448] and (st.position, d.position) == (8491, 8492)


def test_stash_tail_skips_what_it_cannot_rewind():
    job, seq, st, layers, put = _fake_tail_job(window_beg=8704)          # the ring moved past the boundary
    last_page.stash_tail(job, seq, 8192, 8491)
    job2, seq2, _, _, put2 = _fake_tail_job(win=2)                        # a PLE layer with conv state
    last_page.stash_tail(job2, seq2, 8192, 8491)
    job3, seq3, _, _, put3 = _fake_tail_job(draft=object())               # a drafter state of another kind
    last_page.stash_tail(job3, seq3, 8192, 8491)
    job4, seq4, _, _, put4 = _fake_tail_job()
    last_page.stash_tail(job4, seq4, 8448, 8491)                          # the boundary was not inside this forward
    assert put == put2 == put3 == put4 == []
    assert st.position == 8491


def test_rewindable_only_for_deepseek_v4_state():
    import types

    def mod(name, **kw):
        return type(name, (), dict(caps={"recurrent_cache": True}, modules=[], **kw))()
    ds = types.SimpleNamespace(modules=[types.SimpleNamespace(modules=[mod("DSV4Attention"), mod("PLELayer", conv_state_len=0)])])
    assert last_page.rewindable(ds)[0]
    glm = types.SimpleNamespace(modules=[mod("GatedDeltaNet"), mod("DSV4Attention")])
    ok, what = last_page.rewindable(glm)
    assert not ok and "unsupported GatedDeltaNet" in what
    assert not last_page.rewindable(types.SimpleNamespace(modules=[mod("PLELayer", conv_state_len=3), mod("DSV4Attention")]))[0]
