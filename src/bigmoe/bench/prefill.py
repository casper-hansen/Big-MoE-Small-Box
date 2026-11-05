"""Prefill on top of a cached prefix: the time to the first token of a request whose first PREFIX tokens are already in
the cache and N tokens are new, for N from a few tokens (a short tool result) to 10k (a file read). This is the latency
of every step of an agent turn.

Each run puts a fresh nonce in front of the prefix, so nothing from an earlier run is reused, and warms the prefix once
(PREFIX is a multiple of the 256-token page and of the default chunk, so the recurrent state is checkpointed exactly
there). Every size then gets new tokens of its own: no request can resume past the prefix. The text is the long-context
benchmark's corpus (bench/longctx.py).
"""
import secrets
import time

from ..client import Client
from .longctx import corpus

PREFIX = 8192
SIZES = (8, 20, 40, 80, 120, 200, 400, 800, 2000, 5000, 10000)


def prompts(client: Client, site_packages, sizes, prefix: int = PREFIX, nonce: str = None):
    """(warm-up ids, {size: ids}): every prompt is the same prefix (opening with a nonce header) plus new tokens that no
    other prompt has; the warm-up's single new token too."""
    head = client.tokenize(f"[prefill benchmark {nonce or secrets.token_hex(4)}]\n")
    need = prefix - len(head) + sum(sizes) + 1
    body = client.tokenize(corpus(int(need * 4.5) + 20000, site_packages))
    assert len(body) >= need, f"corpus too short: {len(body)} < {need} tokens"
    pre = head + body[:prefix - len(head)]
    out, off = {}, prefix - len(head)
    for n in sizes:
        out[n] = pre + body[off:off + n]
        off += n
    return pre + body[off:off + 1], out


def ttft(client: Client, ids: list) -> float:
    t0 = time.perf_counter()
    for _ev in client.generate(input_ids=ids, sampling_params={"temperature": 0, "max_new_tokens": 1}, stream=True):
        return time.perf_counter() - t0
    raise RuntimeError("no token streamed")


def run(client: Client, site_packages, sizes=SIZES, *, prefix: int = PREFIX, log=print) -> dict:
    ctx = int(client.server_info()["context_length"])
    sizes = [n for n in sizes if prefix + n + 256 <= ctx]
    warmup, reqs = prompts(client, site_packages, sizes, prefix)
    warm = ttft(client, warmup)
    res = {"prefix_tokens": prefix, "prefix_s": round(warm, 2), "runs": {}}
    log(f"prefill on a cached {prefix}-token prefix (first prefill {warm:.1f} s): new tokens -> time to first token")
    for n in sizes:
        s = ttft(client, reqs[n])
        res["runs"][str(n)] = {"new_tokens": n, "ttft_s": round(s, 2)}
        log(f"{n:>7} new tok: {s:6.2f} s")
    return res


def compare_runs(base: dict, now: dict) -> list:
    rows = []
    for k in sorted(set(base.get("runs", {})) & set(now.get("runs", {})), key=int):
        a, b = base["runs"][k]["ttft_s"], now["runs"][k]["ttft_s"]
        rows.append(f"  {k:>7} new tok | {a:6.2f} s -> {b:6.2f} s ({(b / a - 1) * 100:+.0f}%)")
    return rows
