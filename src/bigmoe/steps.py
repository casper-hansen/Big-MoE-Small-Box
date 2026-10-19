import time
from dataclasses import dataclass, field
from typing import Callable

from . import log
from .errors import PrepareError


@dataclass
class Step:
    name: str
    title: str
    done: Callable[[], bool]  # runs on every prepare: must be cheap (no hashing of big files)
    run: Callable[[], None]
    detail: Callable[[], str] = field(default=lambda: "")


def run_steps(steps, *, dry_run=False, force=()) -> list:
    report = []
    for st in steps:
        done = st.done()
        if dry_run:
            status = "done" if done and st.name not in force else "pending"
            report.append({"step": st.name, "status": status, "title": st.title, "detail": st.detail()})
            continue
        if done and st.name not in force:
            report.append({"step": st.name, "status": "done", "seconds": 0.0})
            continue
        log.info("%s: %s", st.name, st.title)
        t0 = time.time()
        st.run()
        dt = time.time() - t0
        if not st.done():
            raise PrepareError(f"step {st.name!r} ran but is still not complete")
        log.info("%s: finished in %.0f s", st.name, dt)
        report.append({"step": st.name, "status": "ran", "seconds": round(dt, 1)})
    return report
