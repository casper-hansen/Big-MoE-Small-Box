"""Reference logits from STOCK exllamav3 for DeepSeek-V4.1-Flash on a box that cannot hold the model.

Streams the model one module at a time (load -> forward every sequence -> unload), exactly like exllamav3's own
eval/model_diff.py, so the arithmetic is stock exllamav3's prefill path: no expert cache, no NVMe tier, no CPU lane.
One pass reads every weight once (~160 GB at 2.25 bpw), so it costs a few minutes however many sequences it gets.

  python -m bigmoe.models.dsv41.tools.ref_logits -m MODEL_DIR -i seqs.json -o ref.pt [--positions from:to]

seqs.json: {"name": [token ids...], ...}. ref.pt: {"name": {"ids": LongTensor, "start": s, "logits": HalfTensor[n, V]}}
with logits for positions start .. len-1 (logits[i] predicts token start + i + 1). --start-key: per-sequence start in
a second JSON ({"name": start}); default 0 (all positions).
"""
import argparse
import json
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--starts", default=None, help="JSON {name: first position whose logits are kept}")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-seq", type=int, default=0, help="override_dynamic_seq_len (default: longest sequence)")
    args = ap.parse_args()

    from exllamav3 import Config, Model

    seqs = json.load(open(args.input))
    starts = json.load(open(args.starts)) if args.starts else {}
    names = list(seqs)
    device = torch.device(args.device)
    longest = max(len(v) for v in seqs.values())

    config = Config.from_directory(args.model)
    config.override_dynamic_seq_len(max(args.max_seq, longest))
    model = Model.from_config(config)

    states, params = [], []
    for n in names:
        ids = torch.tensor(seqs[n], dtype=torch.long).unsqueeze(0)
        p = {}
        states.append(model.prepare_inputs(ids, p))
        params.append(p)

    out = {}
    t0 = time.time()
    nmod = len(model.modules)
    for idx, module in enumerate(model.modules):
        last = idx == nmod - 1
        t1 = time.time()
        config.stc.begin_deferred_load()
        module.load(device if not module.caps.get("prefer_cpu") else "cpu")
        config.stc.end_deferred_load()
        t2 = time.time()
        for b, n in enumerate(names):
            params[b]["dev_cache"] = None
            x = module.prepare_for_device(states[b], params[b])
            x = module.forward(x, params[b]).clone()
            if last:
                s = int(starts.get(n, 0))
                out[n] = {"ids": torch.tensor(seqs[n], dtype=torch.long), "start": s,
                          "logits": x[0, s:].to(torch.half).cpu()}
                states[b] = None
            else:
                states[b] = x
        module.unload()
        torch.cuda.empty_cache()
        print(f" -- [{idx + 1}/{nmod}] {module.key}: load {t2 - t1:.1f} s, forward {time.time() - t2:.1f} s, "
              f"total {time.time() - t0:.0f} s", flush=True)

    torch.save(out, args.output)
    print(f" -- saved {args.output}: " + ", ".join(f"{n} {tuple(v['logits'].shape)}" for n, v in out.items()), flush=True)


if __name__ == "__main__":
    main()
