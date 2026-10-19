import argparse

from bigmoe import AutoBox
from bigmoe.bench import verify

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="glm-5.3-flash")
ap.add_argument("--context", type=int, default=131072)
ap.add_argument("--mode", default="nvme")
ap.add_argument("--tokens", type=int, nargs="+", default=[1024, 8192, 32768])
ap.add_argument("--out")
ap.add_argument("--verify", action="store_true")
a = ap.parse_args()

if a.verify:
    verify(a.model, out_dir=a.out)
else:
    box = AutoBox(a.model, context=a.context, mode=a.mode)
    res = box.bench(a.tokens, out=a.out)
    for n, r in res["runs"].items():
        print(f"{n:>7} tokens: prefill {r.get('prefill_tok_s')} tok/s, decode {r.get('decode_tok_s')} tok/s")
