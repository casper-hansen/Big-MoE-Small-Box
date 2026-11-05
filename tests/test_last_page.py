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
