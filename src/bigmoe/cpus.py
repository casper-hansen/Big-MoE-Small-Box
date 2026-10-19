from dataclasses import dataclass, field


def parse_cpus(spec) -> list:
    """'1-5,7' -> [1, 2, 3, 4, 5, 7]; also accepts an int or an iterable of ints."""
    if isinstance(spec, int):
        return [spec]
    if not isinstance(spec, str):
        return sorted({int(x) for x in spec})
    out = []
    for part in spec.split(","):
        part = part.strip()
        if part:
            a, _, b = part.partition("-")
            out += range(int(a), int(b or a) + 1)
    return out


def format_cpus(cpus) -> str:
    cs, runs = sorted(set(cpus)), []
    for c in cs:
        if runs and c == runs[-1][1] + 1:
            runs[-1][1] = c
        else:
            runs.append([c, c])
    return ",".join(f"{a}-{b}" if b > a else f"{a}" for a, b in runs)


@dataclass(frozen=True)
class CpuLayout:
    main: tuple
    controller: int
    lane: tuple = field(default=())
    readers: tuple = field(default=())

    @classmethod
    def from_dict(cls, d) -> "CpuLayout":
        return cls(tuple(parse_cpus(d["main"])), int(d["controller"]), tuple(parse_cpus(d.get("lane", ()))),
                   tuple(parse_cpus(d.get("readers", ()))))

    def all_cpus(self) -> set:
        return set(self.main) | {self.controller} | set(self.lane) | set(self.readers)

    def describe(self) -> str:
        lane = format_cpus(self.lane) if self.lane else "off"
        return (f"main {format_cpus(self.main)}, controller {self.controller}, CPU lane {lane}, "
                f"NVMe readers {format_cpus(self.readers)}")


def plan_layout(cores, cpu_lane: bool = True) -> CpuLayout:
    """The main (GPU-feeding) thread and the planner's controller are latency-critical; the CPU lane's AVX2 GEMVs are
    FPU-bound, so SMT siblings do not help them, while the NVMe readers mostly sit in pread. With SMT: main and
    controller share the first core, the lane takes one thread of every other core, the readers their siblings
    (on a 7600X: 0 / 6 / 1-5 / 7-11, the measured layout). Without SMT the readers share the last lane CPUs."""
    if not cores:
        raise ValueError("no CPUs")
    if len(cores) >= 2 and all(len(c.threads) >= 2 for c in cores):
        rest = cores[1:]
        return CpuLayout(main=(cores[0].threads[0],), controller=cores[0].threads[1],
                         lane=tuple(c.threads[0] for c in rest) if cpu_lane else (),
                         readers=tuple(t for c in rest for t in c.threads[1:]))
    flat = [t for c in cores for t in c.threads]
    rest = flat[2:] or flat
    return CpuLayout(main=(flat[0],), controller=flat[1 % len(flat)], lane=tuple(rest) if cpu_lane else (),
                     readers=tuple(rest[-min(14, len(rest)):]))
