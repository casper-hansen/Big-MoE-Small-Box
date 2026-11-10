"""Follow-ups that resume from a prompt's last-page checkpoint (engine/last_page.py): the last-page split against tail
checkpoints. Each sample is warm_kl's cached 8,192-token prefix plus N new tokens (P, prefilled, one token generated),
then a follow-up P2 = P + 32 filler tokens (as the previous answer) + 100 new tokens + the assistant opening, which
resumes from the checkpoint at P's last page boundary when there is one (the filler stays short of the next page, so no
decode-time checkpoint comes in between). P2's 128-token greedy answer is compared:

  ref    server with -o tail_ckpt=false (the split): P2's greedy answer and distributions
  force  server with tail checkpoints: P2 forced along the reference (/kl_force)
  compare  per N: P2's cached tokens on both, top-1 agreement and KL

  python -m bigmoe.models.dsv41.tools.tail_kl ref -o tail_ref.json
  python -m bigmoe.models.dsv41.tools.tail_kl force --ref tail_ref.json -o tail_new.json
  python -m bigmoe.models.dsv41.tools.tail_kl compare tail_ref.json tail_new.json
"""
import argparse
import json
import statistics
import time

from bigmoe.models.dsv41.tools import warm_kl as W
from bigmoe.models.dsv41.tools.turn_kl import positions

# tails past the last page boundary: 43 / 31 (split before), 93 / 193 (no split and no checkpoint before)
SIZES = (300, 800, 350, 450)
FILLER, NEW = 32, 100


def build(url, reps):
    """(prefix, [(name, N, P, P2)]): per rep, the N-token samples, then one extra body span per sample for its P2."""
    tok = lambda text: W.post(url + "/tokenize", {"prompt": text, "add_special_tokens": False})["tokens"]
    tail = tok(W.TAIL)
    extra_n = FILLER + NEW + len(tail)
    k = len(SIZES)
    prefix, todo = W.samples(url, tuple(SIZES) + (extra_n,) * k, reps)
    out = []
    for rep in range(reps):
        group = todo[rep * 2 * k:(rep + 1) * 2 * k]
        for (name, n, ids), (_, _, ex) in zip(group[:k], group[k:]):
            body = ex[W.PREFIX:W.PREFIX + FILLER + NEW]
            out.append((name, n, ids, ids + body + tail))
    return prefix, out


def run(a, mode):
    url = a.url
    print(W.get(url + "/cpu?on=0"), flush=True)                  # decode on the bit-exact GPU path
    ref = json.load(open(a.ref)) if mode == "force" else None
    reps = ref["reps"] if ref else a.reps
    prefix, todo = build(url, reps)
    switch = lambda on: W.post(url + "/bigmoe/rpc", {"method": "warm_prefill_switch", "args": [on]})
    print("warm prefill switch:", switch(0), flush=True)
    print("prefix warm-up: cached", W.touch(url, prefix), flush=True)
    res = {"prefix": prefix, "reps": reps, "samples": {}}
    for name, n, p, p2 in todo:
        switch(0)
        W.touch(url, prefix)
        switch(1)
        t0 = time.perf_counter()
        r = W.post(url + "/generate", {"input_ids": p, "sampling_params": {"temperature": 0, "max_new_tokens": 1}})
        p_secs = time.perf_counter() - t0
        p_cached = ((r.get("meta_info") or {}).get("finish_reason") or {}).get("timings", {}).get("cached_tokens")
        t0 = time.perf_counter()
        if mode == "ref":
            W.get(url + "/kl_capture?on=1")
            W.get(url + "/kl_capture")
            r = W.post(url + "/generate", {"input_ids": p2, "sampling_params": {"temperature": 0, "max_new_tokens": W.ANSWER}})
            cap = W.get(url + "/kl_capture?on=0")
            cached = ((r.get("meta_info") or {}).get("finish_reason") or {}).get("timings", {}).get("cached_tokens")
        else:
            gen = ref["samples"][name]["gen"]
            cap = W.post(url + "/kl_force", {"prompt": p2, "ref": gen})
            cap["prompt"], cap["gen"] = p2, gen
            cached = (cap.get("timings") or {}).get("cached_tokens")
        secs = time.perf_counter() - t0
        assert cap["prompt"] == p2, f"{name}: captured prompt differs"
        res["samples"][name] = {"n": n, "prompt_len": len(p2), "p_len": len(p), "gen": cap["gen"], "cached": cached,
                                "p_cached": p_cached, "p_secs": round(p_secs, 2), "secs": round(secs, 2),
                                "steps": {str(k): v for k, v in positions(cap).items()}}
        print(f"{name}: P {len(p)} (cached {p_cached}, {p_secs:.1f} s), P2 {len(p2)} cached {cached}, {secs:.1f} s",
              flush=True)
        json.dump(res, open(a.out, "w"))
    switch(1)
    print(W.get(url + "/cpu?on=1"), flush=True)


def compare(a):
    ref, new = json.load(open(a.ref)), json.load(open(a.new))
    rows = {}
    for name, x in ref["samples"].items():
        y = new["samples"].get(name)
        if y is None:
            continue
        P = x["prompt_len"]
        r = rows.setdefault(x["n"], {"pos": 0, "top1": 0, "kls": [], "same": 0, "samples": 0, "cached": []})
        r["samples"] += 1
        r["cached"].append((x["cached"], y["cached"]))
        identical = True
        for i in range(len(x["gen"])):
            s, t = x["steps"].get(str(P - 1 + i)), y["steps"].get(str(P - 1 + i))
            if s is None or t is None:
                continue
            r["pos"] += 1
            r["top1"] += int(s["ix"][0] == t["ix"][0])
            r["kls"].append(W.kl(s["ix"], s["lp"], t["ix"], t["lp"]))
            identical &= s["lp"] == t["lp"] and s["ix"] == t["ix"]
        r["same"] += int(identical)
    print("N | tail | P2 cached (ref -> new) | positions | top-1 | mean KL | max KL | bit-identical")
    allk, allp, allt = [], 0, 0
    for n in sorted(rows):
        r = rows[n]
        tail = (W.PREFIX + n - 1) % 256
        allk += r["kls"]; allp += r["pos"]; allt += r["top1"]
        print(f"{n} | {tail} | {', '.join(f'{c0}->{c1}' for c0, c1 in r['cached'])} | {r['pos']} | "
              f"{100 * r['top1'] / max(1, r['pos']):.1f} % | {statistics.mean(r['kls']) if r['kls'] else 0:.6f} | "
              f"{max(r['kls'] or [0]):.4f} | {r['same']}/{r['samples']}")
    if allk:
        print(f"all | | | {allp} | {100 * allt / allp:.2f} % | {statistics.mean(allk):.6f} | {max(allk):.4f} |")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("ref", "force"):
        p = sub.add_parser(cmd)
        p.add_argument("--url", default="http://127.0.0.1:30000")
        p.add_argument("-o", "--out", required=True)
        if cmd == "ref":
            p.add_argument("--reps", type=int, default=2)
        else:
            p.add_argument("--ref", required=True)
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
