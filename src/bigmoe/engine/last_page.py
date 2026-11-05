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


def tail_limit(stage_min: int, experts_per_token: int) -> int:
    """Longest tail whose layer calls stay under the staging threshold (picks < stage_min)."""
    return max(0, (stage_min - 1) // max(1, experts_per_token))


def apply(stage_min: int, experts_per_token: int) -> str:
    """Recompile exllamav3's Job.prefill with the tail limit for this engine. Returns a status line."""
    from exllamav3.generator import job as J
    limit = tail_limit(stage_min, experts_per_token)
    J._BIGMOE_LAST_PAGE_MAX = limit
    if getattr(J.Job.prefill, "_bigmoe_last_page", False):
        return f"last-page split for tails <= {limit} tokens"
    src = textwrap.dedent(inspect.getsource(J.Job.prefill))
    # dedent(getsource) leaves the method body at 4 spaces; the snippet sits two levels deeper
    stock, patched = (textwrap.indent(textwrap.dedent(t), " " * 8) for t in (STOCK, PATCHED))
    if src.count(stock) != 1:
        print(" !! last_page: exllamav3's Job.prefill lacks the stock last-page split; prefill left as is",
              file=sys.stderr, flush=True)
        return "last-page split: stock (unexpected exllamav3 source)"
    tree = ast.parse(src.replace(stock, patched))
    ast.increment_lineno(tree, inspect.getsourcelines(J.Job.prefill)[1] - 1)   # tracebacks point into job.py
    ns = {}
    exec(compile(tree, J.__file__, "exec"), J.__dict__, ns)
    fn = ns["prefill"]
    fn._bigmoe_last_page = True
    fn.__qualname__ = J.Job.prefill.__qualname__
    J.Job.prefill = fn
    return f"last-page split for tails <= {limit} tokens"
