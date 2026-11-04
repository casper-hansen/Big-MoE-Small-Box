"""Quality check of the offload engine against stock exllamav3 (reference logits from ref_logits.py).

  gen      greedy chat answers from a running server with per-step logprob capture (/kl_capture) -> gen.json
  teacher  the server's own teacher-forced logprobs over prompt + answer (/kl_teacher, prefill path) -> teacher.json
  seqs     gen.json -> seqs.json + starts.json for ref_logits.py
  compare  ref.pt vs gen.json (decode path: top-1 agreement, KL) and teacher.json (prefill path: should be identical)

Decode-path KL uses the engine's top-256 logprobs: KL(ref || eng) over the reference's top-256 tokens, a token missing
from the engine's top-256 gets the engine's 256th logprob (an upper bound on its probability, so KL is underestimated
only for mass outside both top-256 sets, which is negligible).
"""
import argparse
import json
import math
import time
import urllib.request

PROMPTS = {
    "explain_moe": "Explain in a few paragraphs how a mixture-of-experts language model decides which experts to use "
                   "for each token, and why that makes them cheap to run.",
    "code_lru": "Write a Python class implementing an LRU cache with O(1) get and put, with a short explanation.",
    "math_sum": "What is the sum of all integers from 1 to 1000 that are divisible by 3 or 5? Show your reasoning.",
    "story_dk": "Skriv en kort historie på dansk om en fyrpasser på Skagen, der en stormfuld nat får besøg.",
    "facts_bridges": "List five famous suspension bridges, with the country, the main span length and the year each "
                     "opened, as a markdown table.",
    "translate": "Translate into German and French: 'The early bird catches the worm, but the second mouse gets the "
                 "cheese.' Then explain both proverbs.",
}


def post(url, body, timeout=3600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get(url, timeout=600):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def cmd_gen(a):
    out = {}
    get(a.url + "/kl_capture?on=1")
    for name, prompt in PROMPTS.items():
        if a.only and name not in a.only.split(","):
            continue
        t0 = time.time()
        r = post(a.url + "/v1/chat/completions", {"messages": [{"role": "user", "content": prompt}], "temperature": 0,
                                                 "max_tokens": a.tokens,
                                                 "chat_template_kwargs": {"enable_thinking": False}})
        cap = get(a.url + "/kl_capture")
        n = r["usage"]["completion_tokens"]
        dt = time.time() - t0
        # decode steps: single-token forwards after the prompt; step i (p0 = P + i - 1) predicts gen[i]
        P = len(cap["prompt"])
        dec = {}
        for s in cap["steps"]:
            if s["n"] == 1 and s["p0"] >= P - 1:
                dec[s["p0"]] = {"lp": s["lp"][-1], "ix": s["ix"][-1]}
            elif s["p0"] + s["n"] == P:      # last prefill chunk: its last row predicts gen[0]
                dec[P - 1] = {"lp": s["lp"][-1], "ix": s["ix"][-1]}
        out[name] = {"prompt": cap["prompt"], "gen": cap["gen"], "steps": dec, "text": r["choices"][0]["message"]["content"],
                     "secs": round(dt, 1), "completion_tokens": n}
        print(f"{name}: {n} tokens in {dt:.1f} s, {len(dec)} captured steps: {out[name]['text'][:100]!r}", flush=True)
    get(a.url + "/kl_capture?on=0")
    json.dump(out, open(a.out, "w"))


def cmd_teacher(a):
    gen = json.load(open(a.gen))
    out = {}
    for name, g in gen.items():
        ids = g["prompt"] + g["gen"]
        P = len(g["prompt"])
        r = post(a.url + "/kl_teacher", {"ids": ids, "start": P - 1, "count": len(g["gen"])})
        out[name] = r
        print(f"{name}: teacher-forced {r['count']} rows", flush=True)
    json.dump(out, open(a.out, "w"))


def cmd_seqs(a):
    gen = json.load(open(a.gen))
    seqs = {n: g["prompt"] + g["gen"] for n, g in gen.items()}
    starts = {n: len(g["prompt"]) - 1 for n, g in gen.items()}
    json.dump(seqs, open(a.out + ".seqs.json", "w"))
    json.dump(starts, open(a.out + ".starts.json", "w"))
    print(f"{len(seqs)} sequences, {sum(len(v) for v in seqs.values())} tokens -> {a.out}.seqs.json / .starts.json")


def kl_row(ref_lp, eng_ix, eng_lp, k=256):
    """KL(ref || eng) from the full reference logprob row (torch) and the engine's top-k (lists)."""
    import torch
    v, ix = ref_lp.topk(k)
    floor = eng_lp[-1]
    m = {int(i): float(l) for i, l in zip(eng_ix, eng_lp)}
    kl = 0.0
    for lr, i in zip(v.tolist(), ix.tolist()):
        kl += math.exp(lr) * (lr - m.get(i, floor))
    return max(kl, 0.0)


def cmd_compare(a):
    import torch
    ref = torch.load(a.ref)
    gen = json.load(open(a.gen))
    teach = json.load(open(a.teacher)) if a.teacher else {}
    tot = {"pos": 0, "top1": 0, "kl": 0.0, "kl_max": 0.0, "tf_pos": 0, "tf_exact_rows": 0, "tf_maxdiff": 0.0,
           "tf_top1": 0}
    for name, g in gen.items():
        R = ref[name]
        lp = R["logits"].float().log_softmax(-1)   # row i -> position start + i
        s0 = R["start"]
        P = len(g["prompt"])
        n_ok = n = 0
        kls = []
        for i, tokid in enumerate(g["gen"]):
            row = lp[P - 1 + i - s0]
            n += 1
            n_ok += int(int(row.argmax()) == tokid)
            st = g["steps"].get(str(P - 1 + i))
            if st is not None:
                kls.append(kl_row(row, st["ix"], st["lp"]))
        tot["pos"] += n; tot["top1"] += n_ok
        tot["kl"] += sum(kls); tot["kl_max"] = max(tot["kl_max"], max(kls) if kls else 0.0)
        line = f"{name}: decode top-1 agreement {n_ok}/{n}, mean KL {sum(kls) / max(1, len(kls)):.6f} " \
               f"({len(kls)} steps), max KL {max(kls) if kls else 0:.6f}"
        if name in teach:
            T = teach[name]
            ex = 0; md = 0.0; t1 = 0
            for j, (tl, ti) in enumerate(zip(T["lp"], T["ix"])):
                row = lp[T["start"] + j - s0]
                v, ix = row.topk(len(ti))
                d = max(abs(x - y) for x, y in zip(v.tolist(), tl))
                same = ix.tolist() == ti
                ex += int(same and d < 1e-3)
                md = max(md, d if same else float("inf"))
                t1 += int(ix[0].item() == ti[0])
            tot["tf_pos"] += len(T["lp"]); tot["tf_exact_rows"] += ex; tot["tf_maxdiff"] = max(tot["tf_maxdiff"], md)
            tot["tf_top1"] += t1
            line += f" | teacher-forced: top-1 {t1}/{len(T['lp'])}, rows equal to the reference {ex}/{len(T['lp'])}, " \
                    f"max |dlogprob| {md:.5f}"
        print(line)
    print(f"TOTAL decode positions {tot['pos']}: top-1 agreement {tot['top1'] / max(1, tot['pos']):.4f}, mean KL "
          f"{tot['kl'] / max(1, tot['pos']):.6f}, max KL {tot['kl_max']:.6f}"
          + (f" | teacher-forced {tot['tf_pos']} rows: top-1 {tot['tf_top1'] / max(1, tot['tf_pos']):.4f}, equal "
             f"{tot['tf_exact_rows']}/{tot['tf_pos']}, max |dlogprob| {tot['tf_maxdiff']:.5f}" if teach else ""))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen"); g.add_argument("--url", default="http://127.0.0.1:30000")
    g.add_argument("--tokens", type=int, default=192); g.add_argument("--only", default=""); g.add_argument("-o", "--out", required=True)
    t = sub.add_parser("teacher"); t.add_argument("--url", default="http://127.0.0.1:30000")
    t.add_argument("--gen", required=True); t.add_argument("-o", "--out", required=True)
    s = sub.add_parser("seqs"); s.add_argument("--gen", required=True); s.add_argument("-o", "--out", required=True)
    c = sub.add_parser("compare"); c.add_argument("--ref", required=True); c.add_argument("--gen", required=True)
    c.add_argument("--teacher", default="")
    a = ap.parse_args()
    {"gen": cmd_gen, "teacher": cmd_teacher, "seqs": cmd_seqs, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
