import json
import os
import time
from pathlib import Path

from . import agent as agent_bench
from .longctx import compare_runs

AGENT_FILE = "agent_nvme.json"


def verify(model="glm-5.3-flash", *, out_dir=None, baseline_dir=None, agent=True, log=print, **box_kwargs) -> str:
    """Restarts the server for each of the backend's verify runs and leaves it stopped. With agent, the first run's
    server (mode nvme) also runs the agentic benchmark (bench/agent.py), sequentially and then with two agents at
    once."""
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
            box.bench(tokens, prefill_tokens=(), out=out / fname, log=log)
            if agent and options.get("mode") == "nvme":
                log(f"[verify {time.strftime('%H:%M:%S')}] agentic benchmark")
                agent_bench.run(box.client, out=out / AGENT_FILE, log=log)
                log(f"[verify {time.strftime('%H:%M:%S')}] agentic benchmark, two agents at once")
                agent_bench.run(box.client, ["fix_stats_bug", "refactor_config"], parallel=2,
                                out=out / AGENT_FILE.replace(".json", "_parallel.json"), log=log)
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
    for fname in (AGENT_FILE, AGENT_FILE.replace(".json", "_parallel.json")) if agent else ():
        pa, pb = (base / fname) if base else None, out / fname
        if not pb.is_file():
            continue
        now = json.load(open(pb))
        s = now["summary"]
        lines.append(f"\nagentic benchmark ({fname}): {s['passed']}/{s['tasks']} tasks solved, {s['turns']} turns, "
                     f"{s['invalid_tool_calls']} malformed tool calls, cache hit {s['cache_hit']:.0%}, {s['wall_s']} s")
        if pa and pa.is_file():
            lines += agent_bench.compare_runs(json.load(open(pa)), now)
        else:
            lines.append(f"  no baseline ({fname})")
    report = "\n".join(lines)
    (out / "report.txt").write_text(report + "\n")
    log(report)
    return report
