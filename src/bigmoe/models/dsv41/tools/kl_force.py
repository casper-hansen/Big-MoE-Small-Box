"""Decode-path quality of an engine configuration against a reference decode (e.g. mode nvme vs mode nvme-exact): the
reference's prompts + answers (quality.py gen output) are forced through the running server's decode path (/kl_force),
and every step's top-256 logprobs are compared with the reference step's: top-1 agreement and KL(ref || this).

  python -m bigmoe.models.dsv41.tools.kl_force --ref gen_exact.json [-o forced.json]
"""
import argparse
import json
import math
import time
import urllib.request


def post(url, body, timeout=3600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def kl(ref_ix, ref_lp, ix, lp):
    m = {i: l for i, l in zip(ix, lp)}
    floor = lp[-1]
    return max(0.0, sum(math.exp(a) * (a - m.get(i, floor)) for i, a in zip(ref_ix, ref_lp)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--ref", required=True)
    ap.add_argument("-o", "--out", default="")
    a = ap.parse_args()
    ref = json.load(open(a.ref))
    out, tot_n, tot_top1, tot_kl, kmax = {}, 0, 0, 0.0, 0.0
    for name, g in ref.items():
        t0 = time.time()
        r = post(a.url + "/kl_force", {"prompt": g["prompt"], "ref": g["gen"]})
        P = len(g["prompt"])
        steps = {}
        for s in r["steps"]:
            if s["n"] == 1 and s["p0"] >= P - 1:
                steps[str(s["p0"])] = {"lp": s["lp"][-1], "ix": s["ix"][-1]}
        n = t1 = 0
        ks = []
        for p, rs in g["steps"].items():
            st = steps.get(p)
            if st is None:
                continue
            n += 1
            t1 += int(st["ix"][0] == rs["ix"][0])
            ks.append(kl(rs["ix"], rs["lp"], st["ix"], st["lp"]))
        out[name] = steps
        tot_n += n; tot_top1 += t1; tot_kl += sum(ks); kmax = max([kmax] + ks)
        print(f"{name}: {n} forced decode steps in {time.time() - t0:.1f} s, top-1 agreement {t1}/{n}, mean KL "
              f"{sum(ks) / max(1, len(ks)):.6f}, max KL {max(ks) if ks else 0:.6f}", flush=True)
    print(f"TOTAL {tot_n} positions: top-1 agreement {tot_top1 / max(1, tot_n):.4f}, mean KL {tot_kl / max(1, tot_n):.6f}, "
          f"max KL {kmax:.6f}")
    if a.out:
        json.dump(out, open(a.out, "w"))


if __name__ == "__main__":
    main()
