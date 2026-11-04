"""Speculative-decoding A/B on a running server: greedy chat answers to a fixed prompt set (quality.py's six + a long
essay), per-prompt and total tok/s (prefill included, the same for every configuration), and the server's draft
statistics (acceptance, window sizes) over the run.

  python -m bigmoe.models.dsv41.tools.spec_bench [--tokens 256] [--label X] [-o results.json]
"""
import argparse
import json
import time
import urllib.request

from .quality import PROMPTS

EXTRA = {"essay": "Write a detailed essay about the history of the printing press, its inventors and its impact on Europe."}


def post(url, body, timeout=3600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get(url, timeout=600):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--label", default="")
    ap.add_argument("-o", "--out", default="")
    a = ap.parse_args()
    d0 = (get(a.url + "/stats").get("draft") or {})
    rows, tn, tt = {}, 0, 0.0
    for name, prompt in {**PROMPTS, **EXTRA}.items():
        t0 = time.time()
        r = post(a.url + "/v1/chat/completions", {"messages": [{"role": "user", "content": prompt}], "temperature": 0,
                                                 "max_tokens": a.tokens, "chat_template_kwargs": {"enable_thinking": False}})
        dt = time.time() - t0
        n = r["usage"]["completion_tokens"]
        rows[name] = {"tokens": n, "secs": round(dt, 2), "tok_s": round(n / dt, 2),
                      "text": r["choices"][0]["message"]["content"]}
        tn += n; tt += dt
        print(f"{name:14s} {n:4d} tokens {dt:6.1f} s {n / dt:6.2f} tok/s", flush=True)
    d1 = (get(a.url + "/stats").get("draft") or {})
    dd = {k: d1.get(k, 0) - d0.get(k, 0) for k in ("accepted", "rejected", "rounds")}
    acc = dd["accepted"] / max(1, dd["accepted"] + dd["rejected"])
    print(f"TOTAL {a.label}: {tn} tokens in {tt:.1f} s = {tn / tt:.2f} tok/s | draft acceptance {acc:.3f}, "
          f"{(dd['accepted'] + dd['rounds']) / max(1, dd['rounds']):.2f} tokens per verify round ({dd['rounds']} rounds)", flush=True)
    if a.out:
        json.dump({"label": a.label, "rows": rows, "tokens": tn, "secs": tt, "draft": dd, "draft_total": d1}, open(a.out, "w"))


if __name__ == "__main__":
    main()
