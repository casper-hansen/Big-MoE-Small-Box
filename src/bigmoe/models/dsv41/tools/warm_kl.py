"""Accuracy of warm prefill on small prefills: an agent step that adds N new tokens (a tool result) to a cached
conversation. The same 8,192-token prefix is computed exactly on both servers (cached, one recurrent checkpoint at its
end); every sample then adds N new tokens of its own (the tool result, ending in the assistant turn's opening) and a
128-token answer. Only the N-token prefill differs between the two runs:

  ref    on a server with warm_prefill=False: each sample's greedy answer and the distribution of every decode step
  force  on a server with warm prefill on: the prefix recomputed exactly when needed (runtime switch off), then each
         sample's answer forced along the reference with warm prefill on (/kl_force)
  compare  per N: top-1 agreement and KL(ref || warm) at every answer position

Decode runs on the bit-exact GPU path on both servers (/cpu?on=0), so the decode lane is not a difference. Below
EC_STAGE_MIN picks (512: the prefill forward of N - 1 tokens times 6 experts, N <= 86) warm prefill does not apply:
those sizes are the control (expected KL 0).

  python -m bigmoe.models.dsv41.tools.warm_kl ref -o ref.json
  python -m bigmoe.models.dsv41.tools.warm_kl force --ref ref.json -o warm.json
  python -m bigmoe.models.dsv41.tools.warm_kl compare ref.json warm.json
"""
import argparse
import json
import math
import statistics
import time
import urllib.request

from bigmoe.bench.longctx import corpus
from bigmoe.models.dsv41.tools.turn_kl import positions

SIZES = (8, 20, 40, 60, 80, 100, 120, 160, 200, 300, 400, 600, 800, 1000)
PREFIX = 8192
ANSWER = 128
SITE = "/home/casper/.cache/bigmoe/envs/dsv41-vllm-py3.12-e9fb4edc78/lib/python3.12/site-packages"
HEAD = "<｜begin▁of▁sentence｜><｜User｜>[warm prefill accuracy test] Files from a repository, for context:\n"
QUESTION = ("\n\nBelow is the output of a tool call that read part of another file of the same repository. "
            "Explain in two or three sentences what it shows.\n\n")
TAIL = "<｜Assistant｜></think>"


def post(url, body, timeout=3600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get(url):
    with urllib.request.urlopen(url, timeout=600) as r:
        return json.loads(r.read())


def samples(url, sizes, reps):
    """(prefix ids, [(name, N, prompt ids)]): every prompt is the prefix plus N new tokens no other prompt has."""
    tok = lambda text: post(url + "/tokenize", {"prompt": text, "add_special_tokens": False})["tokens"]
    head, q, tail = tok(HEAD), tok(QUESTION), tok(TAIL)
    need = PREFIX + reps * sum(sizes) + 1000
    body = tok(corpus(int(need * 4.5) + 20000, SITE))
    assert len(body) >= need, f"corpus too short: {len(body)} < {need}"
    fill = PREFIX - len(head) - len(q)
    prefix = head + body[:fill] + q
    assert len(prefix) == PREFIX
    off, out = fill, []
    for rep in range(reps):
        for n in sizes:
            k = n - len(tail)
            assert k >= 1, f"N={n} is shorter than the assistant opening ({len(tail)} tokens)"
            out.append((f"n{n}_r{rep}", n, prefix + body[off:off + k] + tail))
            off += k
    return prefix, out


def touch(url, prefix):
    """The prefix's recurrent checkpoint in the cache (recomputed if the LRU dropped it); returns the cached count."""
    r = post(url + "/generate", {"input_ids": prefix + [prefix[-1]], "sampling_params": {
        "temperature": 0, "max_new_tokens": 1}})
    return ((r.get("meta_info") or {}).get("finish_reason") or {}).get("timings", {}).get("cached_tokens")


def run(a, mode):
    url = a.url
    print(get(url + "/cpu?on=0"), flush=True)                  # decode on the bit-exact GPU path
    ref = json.load(open(a.ref)) if mode == "force" else None
    reps = ref["reps"] if ref else a.reps
    sizes = ref["sizes"] if ref else a.sizes
    prefix, todo = samples(url, sizes, reps)
    if ref:
        assert ref["prefix"] == prefix, "the reference was measured on a different prefix"
    switch = lambda on: post(url + "/bigmoe/rpc", {"method": "warm_prefill_switch", "args": [on]})
    if mode == "force":
        print("warm prefill switch:", switch(0), flush=True)
    print("prefix warm-up: cached", touch(url, prefix), flush=True)
    res = {"prefix": prefix, "sizes": list(sizes), "reps": reps, "samples": {}}
    for name, n, ids in todo:
        if n < getattr(a, "min_n", 0):
            continue
        if mode == "force":
            switch(0)                                           # the prefix (if it has to be recomputed): exact
        cached_touch = touch(url, prefix)
        if mode == "force":
            switch(1)
        t0 = time.perf_counter()
        if mode == "ref":
            get(url + "/kl_capture?on=1")
            get(url + "/kl_capture")
            r = post(url + "/generate", {"input_ids": ids, "sampling_params": {
                "temperature": 0, "max_new_tokens": ANSWER}})
            cap = get(url + "/kl_capture?on=0")
            cached = ((r.get("meta_info") or {}).get("finish_reason") or {}).get("timings", {}).get("cached_tokens")
        else:
            gen = ref["samples"][name]["gen"]
            cap = post(url + "/kl_force", {"prompt": ids, "ref": gen})
            cap["prompt"], cap["gen"] = ids, gen
            cached = (cap.get("timings") or {}).get("cached_tokens")
        secs = time.perf_counter() - t0
        assert cap["prompt"] == ids, f"{name}: captured prompt differs"
        res["samples"][name] = {"n": n, "prompt_len": len(ids), "gen": cap["gen"], "cached": cached,
                                "cached_touch": cached_touch, "secs": round(secs, 2),
                                "steps": {str(k): v for k, v in positions(cap).items()}}
        print(f"{name}: {len(cap['gen'])} answer tokens, cached {cached} (touch {cached_touch}), {secs:.1f} s",
              flush=True)
        json.dump(res, open(a.out, "w"))
    if mode == "force":
        switch(1)
    print(get(url + "/cpu?on=1"), flush=True)


def kl(ref_ix, ref_lp, ix, lp):
    m = {i: x for i, x in zip(ix, lp)}
    floor = lp[-1]
    return max(0.0, sum(math.exp(x) * (x - m.get(i, floor)) for i, x in zip(ref_ix, ref_lp)))


def compare(a):
    ref, new = json.load(open(a.ref)), json.load(open(a.new))
    rows = {}
    for name, x in ref["samples"].items():
        y = new["samples"].get(name)
        if y is None:
            continue
        assert x["gen"] == y["gen"]
        P = x["prompt_len"]
        r = rows.setdefault(x["n"], {"pos": 0, "top1": 0, "kls": [], "bad": 0, "same": 0, "samples": 0})
        r["samples"] += 1
        r["bad"] += int(x["cached"] != PREFIX or y["cached"] != PREFIX)
        identical = True
        for i in range(len(x["gen"])):
            p = str(P - 1 + i)
            s, t = x["steps"].get(p), y["steps"].get(p)
            if s is None or t is None:
                continue
            r["pos"] += 1
            r["top1"] += int(s["ix"][0] == t["ix"][0])
            r["kls"].append(kl(s["ix"], s["lp"], t["ix"], t["lp"]))
            identical &= s["lp"] == t["lp"] and s["ix"] == t["ix"]
        r["same"] += int(identical)
    print("new tokens | warm prefill used | samples | positions | top-1 | mean KL | max KL | bit-identical samples")
    for n in sorted(rows):
        r = rows[n]
        staged = (n - 1) * 6 >= 512
        print(f"{n:>10} | {'yes' if staged else 'no (per-pick)':>17} | {r['samples']:>7} | {r['pos']:>9} | "
              f"{100 * r['top1'] / max(1, r['pos']):5.1f} % | {statistics.mean(r['kls']) if r['kls'] else 0:.6f} | "
              f"{max(r['kls'] or [0]):.4f} | {r['same']}/{r['samples']}"
              + (f"  ({r['bad']} not resumed from the prefix)" if r["bad"] else ""))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("ref", "force"):
        p = sub.add_parser(cmd)
        p.add_argument("--url", default="http://127.0.0.1:30000")
        p.add_argument("-o", "--out", required=True)
        if cmd == "ref":
            p.add_argument("--reps", type=int, default=3)
            p.add_argument("--sizes", type=lambda s: [int(x) for x in s.split(",")], default=list(SIZES))
        else:
            p.add_argument("--ref", required=True)
            p.add_argument("--min-n", type=int, default=0, help="skip smaller samples (the per-pick sizes are controls)")
    c = sub.add_parser("compare")
    c.add_argument("ref")
    c.add_argument("new")
    a = ap.parse_args()
    if a.cmd == "compare":
        compare(a)
    else:
        run(a, a.cmd)


if __name__ == "__main__":
    main()
