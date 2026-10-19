"""Long-context benchmark: prompts of exactly N tokens of real text (the sorted *.py files of exllamav3 + transformers
in the server's environment, so identical on every run of the same runtime) plus a question, greedy, no output cap.
prefill tok/s = N / time to first token; decode tok/s = (completion tokens - 1) / (t_last - t_first). The answer's
sha256 makes two runs comparable.
"""
import argparse
import glob
import hashlib
import json
import os
import time

from ..client import Client

QUESTION = ("\n\nThe text above is a concatenation of Python source files, each starting with a '### file:' header. "
            "Name the first three files in the order they appear, then explain in a few sentences what the very "
            "first file is for.")
CORPUS_PACKAGES = ("exllamav3", "transformers")


def glm_frame(n_tokens: int):
    """GLM-5.x chat framing, thinking off. The reference results were measured with exactly this text."""
    return f"[gMASK]<sop><|user|>[document set {n_tokens}]\n", QUESTION + "<|assistant|><think></think>"


def corpus(min_chars: int, site_packages) -> str:
    parts, n = [], 0
    for root in (os.path.join(str(site_packages), p) for p in CORPUS_PACKAGES):
        for f in sorted(glob.glob(os.path.join(root, "**", "*.py"), recursive=True)):
            t = open(f, encoding="utf-8", errors="replace").read()
            parts.append(f"### file: {os.path.relpath(f, str(site_packages))}\n{t}\n")
            n += len(parts[-1])
            if n >= min_chars:
                return "".join(parts)
    return "".join(parts)


class PromptBuilder:
    def __init__(self, client: Client, site_packages, frame=glm_frame):
        self.client, self.site_packages, self.frame = client, site_packages, frame
        self._body = {}

    def ids(self, n_tokens: int) -> list:
        """The per-length header keeps prompts of different lengths from sharing prefix pages: exllamav3 reuses cached
        pages, which would turn the prefill timing into an incremental one."""
        h, t = self.frame(n_tokens)
        head, tail = self.client.tokenize(h), self.client.tokenize(t)
        need = n_tokens - len(head) - len(tail)
        if need not in self._body:
            ids = self.client.tokenize(corpus(int(need * 4.5) + 20000, self.site_packages))
            assert len(ids) >= need, f"corpus too short: {len(ids)} < {need} tokens"
            self._body[need] = ids[:need]
        return head + self._body[need] + tail


def run_one(client: Client, ids: list, ctx: int) -> dict:
    sp = {"temperature": 0, "max_new_tokens": max(64, ctx - len(ids) - 64)}
    t0, times, last = time.perf_counter(), [], None
    for ev in client.generate(input_ids=ids, sampling_params=sp, stream=True):
        last = ev
        times.append(time.perf_counter())
    if not times:
        raise RuntimeError("no tokens streamed")
    mi = (last or {}).get("meta_info", {})
    n = mi.get("completion_tokens", len(times))
    text = (last or {}).get("text", "")
    ttft = times[0] - t0
    dec = (n - 1) / (times[-1] - times[0]) if n > 1 and times[-1] > times[0] else None
    return {"prompt_tokens": len(ids), "prompt_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            "ttft_s": round(ttft, 2), "prefill_tok_s": round(len(ids) / ttft, 1), "completion_tokens": n,
            "decode_s": round(times[-1] - times[0], 2), "decode_tok_s": round(dec, 2) if dec else None,
            "finish": (mi.get("finish_reason") or {}).get("type"),
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "text": text}


def _page_reserve(env: dict) -> int:
    """Each job reserves pages for max_rq_tokens (GLM53_MAX_RQ_TOKENS) beyond its prompt."""
    return max([int(v) for k, v in (env or {}).items() if k.endswith("_MAX_RQ_TOKENS") and str(v).isdigit()] or [0])


def run(client: Client, tokens, site_packages, *, frame=glm_frame, out=None, log=print) -> dict:
    info = client.server_info()
    ctx = int(info["context_length"])
    env = (info.get("server_args") or {}).get("env")
    res = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "context_length": ctx, "question": QUESTION, "env": env,
           "runs": {}}
    pb = PromptBuilder(client, site_packages, frame)
    client.generate(input_ids=pb.ids(512), sampling_params={"temperature": 0, "max_new_tokens": 8})   # warm-up

    def save():
        if out:
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
            with open(out, "w") as f:
                json.dump(res, f, indent=1)

    for n in tokens:
        if n + _page_reserve(env) + 256 > ctx:
            log(f"{n}: skipped (> context {ctx})")
            continue
        try:
            r = run_one(client, pb.ids(n), ctx)
        except Exception as e:   # e.g. CUDA OOM in the server: the stream ends without a token
            log(f"{n:>7} prompt tok: FAILED ({type(e).__name__}: {e}); see the server log")
            res["runs"][str(n)] = {"prompt_tokens": n, "error": f"{type(e).__name__}: {e}"}
            break
        res["runs"][str(n)] = r
        log(f"{n:>7} prompt tok: prefill {r['ttft_s']:8.1f} s = {r['prefill_tok_s']:7.1f} tok/s | decode "
            f"{r['completion_tokens']} tok at {r['decode_tok_s']} tok/s ({r['finish']}) | sha {r['text_sha256'][:12]}")
        save()
    try:
        res["stats_end"] = client.stats()
    except Exception:
        pass
    res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    save()
    return res


def _pct(a, b):
    return f"{(b / a - 1) * 100:+.1f}%" if a and b else "n/a"


def compare_runs(base: dict, now: dict) -> list:
    rows = []
    for k in sorted(set(base["runs"]) & set(now["runs"]), key=int):
        x, y = base["runs"][k], now["runs"][k]
        if "error" in x or "error" in y:
            rows.append(f"  {k:>7}: error ({x.get('error') or y.get('error')})")
            continue
        n = 0
        while n < min(len(x["text"]), len(y["text"])) and x["text"][n] == y["text"][n]:
            n += 1
        same = "IDENTICAL" if x["text"] == y["text"] else f"differs after {n} of {len(x['text'])} chars"
        rows.append(f"  {k:>7} | {x['prefill_tok_s']:7.1f} -> {y['prefill_tok_s']:7.1f} "
                    f"({_pct(x['prefill_tok_s'], y['prefill_tok_s'])}) | {x['decode_tok_s']:5.2f} -> "
                    f"{y['decode_tok_s']:5.2f} ({_pct(x['decode_tok_s'], y['decode_tok_s'])}) | {same}")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="long-context prefill / decode benchmark of a running box")
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--corpus", help="site-packages directory of the server's environment (prompt text source)")
    ap.add_argument("--tokens", type=int, nargs="*", default=[1024, 32768, 131072])
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2, metavar=("BASE", "NOW"))
    a = ap.parse_args(argv)
    if a.compare:
        base, now = (json.load(open(p)) for p in a.compare)
        print("prompt tokens | prefill tok/s base -> now | decode tok/s base -> now | answer")
        print("\n".join(compare_runs(base, now)))
        return 0
    if not a.corpus:
        ap.error("--corpus is required (bigmoe bench finds it for you)")
    run(Client(a.url), a.tokens, a.corpus, out=a.out, log=lambda m: print(m, flush=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
