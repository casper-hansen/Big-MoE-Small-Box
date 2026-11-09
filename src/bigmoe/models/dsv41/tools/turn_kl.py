"""Decode quality on agent turns: replay the turns of an agentic-benchmark run (bench/agent.py results, with tools, so
vLLM's tool-call grammar is on), greedy, with KL capture on (top-256 logprobs of every decode / verify forward), and
compare two server configurations on the same turns: answer identity, and per position up to the first divergence,
top-1 agreement and KL(ref || this). Used for changes that alter the forwards' shapes but not the tokens' targets
(grammar drafts, other speculation).

  python -m bigmoe.models.dsv41.tools.turn_kl capture --agent agent.json -o turns_ref.json [--max-turns 12]
  python -m bigmoe.models.dsv41.tools.turn_kl compare turns_ref.json turns_new.json
"""
import argparse
import json
import time
import urllib.request

from bigmoe.bench.agent import TOOLS
from bigmoe.models.dsv41.tools.kl_force import kl


def post(url, body, timeout=3600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get(url):
    with urllib.request.urlopen(url, timeout=600) as r:
        return json.loads(r.read())


def turns_of(agent: dict, max_turns: int) -> list:
    """(name, messages before an assistant turn) from the benchmark's transcripts, in task order."""
    out = []
    for name, t in agent["tasks"].items():
        msgs = t.get("transcript") or []
        for i, m in enumerate(msgs):
            if m["role"] == "assistant":
                out.append((f"{name}#{sum(1 for x in msgs[:i] if x['role'] == 'assistant')}", msgs[:i]))
    return out[:max_turns]


def positions(cap: dict) -> dict:
    """Prediction made at each position: row j of a forward starting at p0 predicts position p0 + j + 1; the last
    forward covering a position is the one the decode used (rows after a rejected draft are recomputed)."""
    P = len(cap["prompt"] or [])
    steps = {}
    for s in cap["steps"]:
        if s["p0"] >= P - 1 and s["n"] <= 16 and len(s["lp"]) == s["n"]:
            for j in range(s["n"]):
                steps[s["p0"] + j] = {"lp": s["lp"][j], "ix": s["ix"][j]}
    return steps


def capture(url: str, agent_file: str, out: str, max_turns: int, max_tokens: int):
    agent = json.load(open(agent_file))
    model = get(url + "/v1/models")["data"][0]["id"]
    get(url + "/kl_capture?on=1")
    res = {}
    try:
        for name, msgs in turns_of(agent, max_turns):
            get(url + "/kl_capture")                         # drop what an earlier request left
            t0 = time.time()
            r = post(url + "/v1/chat/completions", {"model": model, "messages": msgs, "tools": TOOLS,
                                                     "temperature": 0, "max_tokens": max_tokens,
                                                     "chat_template_kwargs": {"enable_thinking": False}})
            cap = get(url + "/kl_capture")
            m = r["choices"][0]["message"]
            res[name] = {"gen": cap["gen"], "prompt_len": len(cap["prompt"] or []), "secs": round(time.time() - t0, 2),
                         "completion_tokens": r["usage"]["completion_tokens"],
                         "tool_calls": m.get("tool_calls"), "content": m.get("content"),
                         "steps": {str(k): v for k, v in positions(cap).items()}}
            print(f"{name}: {r['usage']['completion_tokens']} tokens in {res[name]['secs']} s", flush=True)
    finally:
        get(url + "/kl_capture?on=0")
    json.dump(res, open(out, "w"))


def compare(ref_file: str, new_file: str):
    ref, new = json.load(open(ref_file)), json.load(open(new_file))
    tot_n = tot_t1 = 0
    tot_kl, kmax, same, t_ref, t_new = 0.0, 0.0, 0, 0.0, 0.0
    for name in ref:
        if name not in new:
            continue
        a, b = ref[name], new[name]
        n = 0
        while n < min(len(a["gen"]), len(b["gen"])) and a["gen"][n] == b["gen"][n]:
            n += 1
        identical = a["gen"] == b["gen"]
        same += identical
        t_ref, t_new = t_ref + a["secs"], t_new + b["secs"]
        P = a["prompt_len"]
        cnt = t1 = 0
        ks = []
        for i in range(n + 1 if not identical else n):      # positions up to and including the first divergence
            p = str(P - 1 + i)
            x, y = a["steps"].get(p), b["steps"].get(p)
            if x is None or y is None:
                continue
            cnt += 1
            t1 += int(x["ix"][0] == y["ix"][0])
            ks.append(kl(x["ix"], x["lp"], y["ix"], y["lp"]))
        tot_n += cnt
        tot_t1 += t1
        tot_kl += sum(ks)
        kmax = max([kmax] + ks)
        print(f"{name}: {'identical' if identical else f'differs after {n} of {len(a['gen'])} tokens'}, "
              f"{cnt} positions, top-1 {t1}/{cnt}, mean KL {sum(ks) / max(1, len(ks)):.6f}, "
              f"{a['secs']} s -> {b['secs']} s")
    print(f"TOTAL {same}/{len(ref)} answers identical, {tot_n} positions: top-1 {tot_t1 / max(1, tot_n):.4f}, "
          f"mean KL {tot_kl / max(1, tot_n):.6f}, max KL {kmax:.6f}; {t_ref:.1f} s -> {t_new:.1f} s "
          f"({(t_new / t_ref - 1) * 100 if t_ref else 0:+.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--url", default="http://127.0.0.1:30000")
    c.add_argument("--agent", required=True)
    c.add_argument("-o", "--out", required=True)
    c.add_argument("--max-turns", type=int, default=12)
    c.add_argument("--max-tokens", type=int, default=512)
    d = sub.add_parser("compare")
    d.add_argument("ref")
    d.add_argument("new")
    a = ap.parse_args()
    if a.cmd == "capture":
        capture(a.url, a.agent, a.out, a.max_turns, a.max_tokens)
    else:
        compare(a.ref, a.new)


if __name__ == "__main__":
    main()
