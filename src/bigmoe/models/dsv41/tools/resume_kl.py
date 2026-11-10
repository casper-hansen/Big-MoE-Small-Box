"""Quality of resuming from a recurrent checkpoint taken while decoding: an agent's next turn extends its previous
answer, and with checkpoints every 256 decoded tokens (DeepSeek, mode nvme) its prefill resumes from a page boundary
inside that answer instead of recomputing it. The resumed state comes from single-token decode forwards, a
full prefill computes it in chunks, so the two differ in fp16 rounding.

On token ids (/generate, no chat template): turn 1 = a prompt, greedy for --answer tokens (the server stashes
checkpoints while decoding); turn 2 = turn 1's prompt + answer + a follow-up, greedy for --decode tokens with KL capture.
Run "turn2" first on a server where turn 1 ran (resumed), then "turn2" on a fresh server or one without decode-time
checkpoints (full prefill), then compare: identical answers, top-1 and KL per position. --prime gives the reference: the
same resume from a checkpoint the prefill stashed (what prefix reuse did already).

  python -m bigmoe.models.dsv41.tools.resume_kl run --tag resumed -o r.json     # runs turn 1, then turn 2
  python -m bigmoe.models.dsv41.tools.resume_kl run --tag full -o f.json --skip-turn1 --from r.json
  python -m bigmoe.models.dsv41.tools.turn_kl compare f.json r.json
"""
import argparse
import json
import time

from bigmoe.models.dsv41.tools.turn_kl import get, positions, post

TOPICS = ["the history of the printing press", "how a refrigerator works", "the rules of chess openings",
          "why the sky is blue", "how compilers optimize loops", "the life cycle of a star"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--tag", required=True)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--answer", type=int, default=600)
    ap.add_argument("--decode", type=int, default=128)
    ap.add_argument("--skip-turn1", action="store_true", help="reuse the turn-1 sequences of --from")
    ap.add_argument("--from", dest="src")
    ap.add_argument("--force-turn1", action="store_true",
                    help="with --from: decode turn 1's answer from --from forced (/kl_force), so this configuration "
                         "stashes its own decode-time checkpoints along the same tokens")
    ap.add_argument("--forced-ref", help="decode turn 2 forced along this run's turn-2 tokens (/kl_force): every position "
                                         "on the same token path, comparable with kl_force's numbers")
    ap.add_argument("--prime", type=int, default=0,
                    help="before turn 2, prefill the first N tokens of the sequence (one output token): with N just past "
                         "a page boundary the last-page split stashes a prefill-time checkpoint there (the reference "
                         "for decode-time checkpoints)")
    a = ap.parse_args()
    tok = lambda text: post(a.url + "/tokenize", {"prompt": text, "add_special_tokens": False})["tokens"]
    src = json.load(open(a.src)) if a.src else {}
    forced = json.load(open(a.forced_ref)) if a.forced_ref else None
    res = {}
    for i, topic in enumerate(TOPICS):
        name = f"t{i}"
        if a.force_turn1:
            seq = src[name]["seq"]
            ids = tok(f"[resume test {i}] Write a long, detailed essay about {topic}.\n\n")
            post(a.url + "/kl_force", {"prompt": ids, "ref": seq[len(ids):]})
        elif a.skip_turn1:
            seq = src[name]["seq"]
        else:
            ids = tok(f"[resume test {i}] Write a long, detailed essay about {topic}.\n\n")
            get(a.url + "/kl_capture?on=1")
            get(a.url + "/kl_capture")
            post(a.url + "/generate", {"input_ids": ids, "sampling_params": {
                "temperature": 0, "max_new_tokens": a.answer}})
            seq = ids + get(a.url + "/kl_capture?on=0")["gen"]     # the answer's token ids, as generated
        follow = tok("\n\nNow summarize the essay above in three sentences.\n\n")
        if a.prime:
            post(a.url + "/generate", {"input_ids": seq[:a.prime], "sampling_params": {
                "temperature": 0, "max_new_tokens": 1}})
        t0 = time.time()
        if forced is not None:
            cap = post(a.url + "/kl_force", {"prompt": seq + follow, "ref": forced[name]["gen"]})
            cap["prompt"], cap["gen"], r = seq + follow, forced[name]["gen"], {}
        else:
            get(a.url + "/kl_capture?on=1")
            get(a.url + "/kl_capture")
            r = post(a.url + "/generate", {"input_ids": seq + follow, "sampling_params": {
                "temperature": 0, "max_new_tokens": a.decode}})
            cap = get(a.url + "/kl_capture?on=0")
        res[name] = {"seq": seq, "gen": cap["gen"], "prompt_len": len(seq + follow), "secs": round(time.time() - t0, 2),
                     "cached": ((r.get("meta_info") or {}).get("finish_reason") or {}).get("timings", {}).get(
                         "cached_tokens"),
                     "steps": {str(k): v for k, v in positions(cap).items()}}
        print(f"{name}: turn 2 of {len(seq + follow)} tokens, cached {res[name]['cached']}, {res[name]['secs']} s",
              flush=True)
    json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
