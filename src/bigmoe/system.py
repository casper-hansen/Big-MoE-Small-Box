import errno
import functools
import mmap
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass

from .errors import BoxError

GiB = 1024 ** 3


@dataclass(frozen=True)
class GPU:
    index: int
    name: str
    memory_total_mib: int
    memory_used_mib: int
    compute_cap: str


def _nvidia_smi(*query, timeout=20):
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        r = subprocess.run([exe, *query], capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout if r.returncode == 0 else None


def gpus() -> list:
    out = _nvidia_smi("--query-gpu=index,name,memory.total,memory.used,compute_cap", "--format=csv,noheader,nounits")
    res = []
    for line in (out or "").strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 5:
            try:
                res.append(GPU(int(p[0]), p[1], int(float(p[2])), int(float(p[3])), p[4]))
            except ValueError:
                pass
    return res


def gpu_compute_apps() -> list:
    out = _nvidia_smi("--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader")
    return [l.strip() for l in (out or "").splitlines() if l.strip()]


@dataclass(frozen=True)
class Core:
    threads: tuple


def cpu_cores(allowed=None) -> list:
    allowed = sorted(set(allowed) if allowed is not None else os.sched_getaffinity(0))
    groups = {}
    for c in allowed:
        base = f"/sys/devices/system/cpu/cpu{c}/topology"
        try:
            key = (int(open(f"{base}/physical_package_id").read()), int(open(f"{base}/core_id").read()))
        except (OSError, ValueError):
            key = (0, 100000 + c)
        groups.setdefault(key, []).append(c)
    return sorted((Core(tuple(sorted(v))) for v in groups.values()), key=lambda k: k.threads[0])


def cpu_flags() -> set:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("flags"):
                return set(line.split(":", 1)[1].split())
    except OSError:
        pass
    return set()


def cpu_model() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "?"


def meminfo() -> dict:
    out = {}
    try:
        for line in open("/proc/meminfo"):
            k, v = line.split(":", 1)
            if k in ("MemTotal", "MemAvailable", "SwapTotal"):
                out[k] = int(v.split()[0]) / 1048576
    except OSError:
        pass
    return out


def cgroup_memory() -> dict:
    out = {}
    cg = "/sys/fs/cgroup"
    if not os.path.exists(f"{cg}/memory.max"):   # not a container: use our own cgroup (e.g. a systemd scope)
        try:
            rel = [l.split(":", 2)[2].strip() for l in open("/proc/self/cgroup") if l.startswith("0::")][0]
            cg = "/sys/fs/cgroup" + rel.rstrip("/")
        except (OSError, IndexError):
            return out
    try:
        out["current"] = int(open(f"{cg}/memory.current").read()) / GiB
        if os.path.exists(f"{cg}/memory.peak"):
            out["peak"] = int(open(f"{cg}/memory.peak").read()) / GiB
        mx = open(f"{cg}/memory.max").read().strip()
        out["max"] = int(mx) / GiB if mx != "max" else None
    except (OSError, ValueError):
        pass
    return out


@functools.lru_cache(maxsize=1)
def systemd_scope_available() -> bool:
    exe = shutil.which("systemd-run")
    if not exe:
        return False
    try:
        r = subprocess.run([exe, "--user", "--scope", "--quiet", "--collect", "true"], capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0


def glibc_version() -> str:
    try:
        return os.confstr("CS_GNU_LIBC_VERSION") or "?"
    except (ValueError, OSError):
        return "?"


def fs_type(path) -> str:
    path = os.path.realpath(path)
    best, kind = "", "?"
    try:
        for line in open("/proc/mounts"):
            _dev, mnt, fs = line.split()[:3]
            if (path == mnt or path.startswith(mnt.rstrip("/") + "/")) and len(mnt) >= len(best):
                best, kind = mnt, fs
    except OSError:
        pass
    return kind


def free_bytes(path) -> int:
    p = os.path.realpath(path)
    while not os.path.exists(p):
        p = os.path.dirname(p)
    st = os.statvfs(p)
    return st.f_bavail * st.f_frsize


def check_odirect(path) -> None:
    """The expert store is read with O_DIRECT, which tmpfs, overlayfs and most network filesystems refuse. A file
    gets a 4 KiB O_DIRECT read, a directory a write + read of a probe file."""
    path = str(path)
    block = 4096
    buf = mmap.mmap(-1, block)   # page-aligned, as O_DIRECT needs
    probe = os.path.join(path, f".odirect-probe-{os.getpid()}") if os.path.isdir(path) else None
    try:
        if probe:
            fd = os.open(probe, os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_DIRECT, 0o600)
            try:
                buf.write(b"\x5a" * block)
                if os.pwritev(fd, [buf], 0) != block:
                    raise OSError("short O_DIRECT write")
                if os.preadv(fd, [buf], 0) != block:
                    raise OSError("short O_DIRECT read")
            finally:
                os.close(fd)
                os.unlink(probe)
        else:
            fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
            try:
                os.preadv(fd, [buf], 0)
            finally:
                os.close(fd)
    except OSError as e:
        what = "writes" if probe else "reads"
        if e.errno == errno.EINVAL:
            raise BoxError(f"{path} ({fs_type(path)}) does not support O_DIRECT {what}; put the expert store on a local "
                           "NVMe filesystem that does (ext4, xfs): AutoBox(home=...) or BIGMOE_HOME") from None
        raise BoxError(f"O_DIRECT {what} at {path} ({fs_type(path)}) failed: {e.strerror or e}") from None


def report(home=None) -> dict:
    cores = cpu_cores()
    out = {
        "gpus": [asdict(g) for g in gpus()],
        "gpu_compute_apps": gpu_compute_apps(),
        "cpu": {"model": cpu_model(), "logical": sum(len(c.threads) for c in cores), "physical": len(cores),
                "smt_groups": [list(c.threads) for c in cores],
                "avx2_fma_f16c": {"avx2", "fma", "f16c"} <= cpu_flags()},
        "memory_gib": {k: round(v, 1) for k, v in meminfo().items()},
        "glibc": glibc_version(),
        "systemd_user_scope": systemd_scope_available(),
    }
    if home is not None:
        root = str(home)
        os.makedirs(root, exist_ok=True)
        try:
            check_odirect(root)
            od = True
        except BoxError as e:
            od = str(e)
        out["home"] = {"path": root, "fs": fs_type(root), "free_gb": round(free_bytes(root) / 1e9, 1), "odirect": od}
    return out
