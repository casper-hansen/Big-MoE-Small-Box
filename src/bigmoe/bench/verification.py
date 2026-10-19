import json
import os
import time
from pathlib import Path

from .longctx import compare_runs


def verify(model="glm-5.3-flash", *, out_dir=None, baseline_dir=None, log=print, **box_kwargs) -> str:
    """Restarts the server for each of the backend's verify runs and leaves it stopped."""
    from ..autobox import AutoBox
    probe = AutoBox(model, **box_kwargs)
    runs = probe.backend.verify_runs()
    if not runs:
        raise ValueError(f"backend {probe.backend.name!r} defines no verification runs")
    out = Path(out_dir or probe.home.results / f"verify-{time.strftime('%Y%m%d-%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    base = Path(baseline_dir) if baseline_dir else probe.backend.baseline_dir()
    log(f"[verify] results -> {out}")
    try:
        for label, options, tokens, fname in runs:
            log(f"[verify {time.strftime('%H:%M:%S')}] {label}: {options}")
            box = AutoBox(model, **{**box_kwargs, **options})
            box.start(restart=True)
            box.bench(tokens, out=out / fname, log=log)
    finally:
        probe.stop()
    log(f"[verify {time.strftime('%H:%M:%S')}] done")
    lines = [f"=== baseline {base}", f"=== this run {out}"]
    for label, _options, _tokens, fname in runs:
        pa, pb = (base / fname) if base else None, out / fname
        if not (pa and pa.is_file() and pb.is_file()):
            lines.append(f"\n{label}: no baseline ({fname})")
            continue
        lines.append(f"\n{label}: prompt tokens | prefill tok/s base -> now | decode tok/s base -> now | answer")
        lines += compare_runs(json.load(open(pa)), json.load(open(pb)))
    report = "\n".join(lines)
    (out / "report.txt").write_text(report + "\n")
    log(report)
    return report
