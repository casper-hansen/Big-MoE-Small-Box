"""For 512k context: --context 524288 --cache-quant 8 --chunk-size 4096."""
import argparse

from bigmoe import AutoBox

ap = argparse.ArgumentParser()
ap.add_argument("model", nargs="?", default="glm-5.3-flash")
ap.add_argument("--mode", default="nvme", choices=["nvme", "nvme-exact"])
ap.add_argument("--context", type=int, default=131072)
ap.add_argument("--memory-gb", type=float, default=52)
ap.add_argument("--port", type=int, default=30000)
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--cache-quant", type=int)
ap.add_argument("--chunk-size", type=int, default=8192)
a = ap.parse_args()

box = AutoBox(a.model, mode=a.mode, context=a.context, memory_gb=a.memory_gb, port=a.port, host=a.host,
              cache_quant=a.cache_quant, chunk_size=a.chunk_size)
raise SystemExit(box.serve())
