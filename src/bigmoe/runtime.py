"""The server runs in its own pinned environment, created with uv under <home>/envs, so the application importing
bigmoe needs nothing heavy. Compiling happens in a "-build" twin (the same pins + the CUDA compiler from pip), which
keeps the toolchain out of the runtime; uv hardlinks packages from its cache, so the twin costs little disk.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from . import kernels, log
from .errors import BoxError, PrepareError

@dataclass(frozen=True)
class SourceBuild:
    """Installed with --no-deps: the runtime requirements pin its dependencies."""
    name: str
    version: str
    url: str  # source tarball


@dataclass(frozen=True)
class RuntimeSpec:
    name: str
    python: str
    requirements: Path
    build_requirements: Path  # added on top in the build twin
    index_url: str = None
    extra_index_urls: tuple = ()
    index_strategy: str = None
    source_builds: tuple = ()
    cuda_subdir: str = "nvidia/cu13"  # where the pip CUDA wheels install inside site-packages
    smoke_test: str = "import torch"

    def key(self) -> str:
        h = hashlib.sha256()
        parts = [self.name, self.python, _pins(self.requirements), _pins(self.build_requirements), self.index_url or "",
                 " ".join(self.extra_index_urls), self.index_strategy or ""]
        parts += [f"{s.name}=={s.version}@{s.url}" for s in self.source_builds]
        for p in parts:
            h.update(p.encode() + b"\0")
        return h.hexdigest()[:10]


def _pins(path) -> str:
    lines = (l.split("#", 1)[0].strip() for l in Path(path).read_text().splitlines())
    return "\n".join(l for l in lines if l)


# not `python -m bigmoe.kernels`: importing bigmoe already imports bigmoe.kernels, and runpy warns about that
_BUILD_KERNELS = "import sys; from bigmoe.kernels import main; main(sys.argv[1:])"


def find_uv() -> str:
    try:
        from uv import find_uv_bin
        return find_uv_bin()
    except Exception:
        pass
    for c in (shutil.which("uv"), str(Path.home() / ".local/bin/uv"), str(Path.home() / ".cargo/bin/uv")):
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    raise BoxError("uv not found (pip install uv); bigmoe uses it to create the server environments")


def patch_cuda_glibc_rsqrt(include_dir: Path) -> bool:
    """CUDA 13.0's crt/math_functions.h declares rsqrt/rsqrtf without the noexcept(true) that glibc >= 2.41 uses in
    C++, so nvcc fails with "exception specification is incompatible" (e.g. Ubuntu 26.04's glibc 2.43). This matches
    glibc's declaration; on older glibc the macro expands to nothing."""
    p = Path(include_dir) / "crt" / "math_functions.h"
    if not p.is_file():
        return False
    s = p.read_text()
    if "__NV_RSQRT_NOEXCEPT" in s:
        return False
    a = "extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 rsqrt(double x);"
    b = "extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  rsqrtf(float x);"
    if s.count(a) != 1 or s.count(b) != 1:
        log.warn("%s: rsqrt declarations not found, not patched", p)
        return False
    g = ("#if defined(__cplusplus) && defined(__GLIBC__) && (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 41))\n"
         "#define __NV_RSQRT_NOEXCEPT noexcept(true)\n#else\n#define __NV_RSQRT_NOEXCEPT\n#endif\n")
    p.write_text(s.replace(a, g + a[:-1] + " __NV_RSQRT_NOEXCEPT;").replace(b, b[:-1] + " __NV_RSQRT_NOEXCEPT;"))
    return True


def strip_debug(path: Path):
    """Strips a copy and renames it into place: uv installs packages as hardlinks into its cache, and strip rewrites a
    hardlinked file in place, which would change the cache and every other environment sharing the file."""
    if not shutil.which("strip"):
        return
    tmp = path.with_name(path.name + ".bigmoe-strip")
    shutil.copy2(path, tmp)
    if subprocess.run(["strip", "--strip-debug", str(tmp)], capture_output=True).returncode == 0:
        os.replace(tmp, path)
    else:
        tmp.unlink(missing_ok=True)


def _wheel_libraries(wheel: Path) -> list:
    import zipfile
    with zipfile.ZipFile(wheel) as z:
        return [n for n in z.namelist() if n.endswith(".so") and ".data/" not in n]


def _site_packages(env: Path) -> Path:
    found = sorted(Path(env).glob("lib/python3*/site-packages"))
    if not found:
        raise BoxError(f"no site-packages in {env}")
    return found[0]


def link_bigmoe(env: Path):
    pkg = Path(__file__).resolve().parent
    link = _site_packages(env) / "bigmoe"
    if link.is_symlink():
        if link.resolve() == pkg:
            return
        link.unlink()
    elif link.exists():
        return  # bigmoe was installed into that environment on purpose
    link.symlink_to(pkg, target_is_directory=True)


class Runtime:
    MARKER = ".bigmoe-runtime.json"

    def __init__(self, spec: RuntimeSpec, home):
        self.spec = spec
        self.home = home
        self.path = Path(home.envs) / f"{spec.name}-py{spec.python}-{spec.key()}"
        self.build_path = self.path.with_name(self.path.name + "-build")

    def __repr__(self):
        return f"Runtime({str(self.path)!r})"

    @property
    def python(self) -> Path:
        return self.path / "bin" / "python"

    def info(self) -> dict:
        try:
            m = json.loads((self.path / self.MARKER).read_text())
        except (OSError, ValueError):
            return {}
        return m if m.get("key") == self.spec.key() else {}

    def is_ready(self) -> bool:
        return bool(self.info()) and self.python.exists()

    @property
    def purelib(self) -> Path:
        return Path(self.info().get("purelib") or _site_packages(self.path))

    def _uv(self, *args, env=None):
        cmd = [find_uv(), *map(str, args)]
        log.info("$ %s", " ".join(cmd[1:]))
        r = subprocess.run(cmd, env=env)
        if r.returncode:
            raise PrepareError(f"uv {args[0]} {args[1] if len(args) > 1 else ''} failed (exit {r.returncode})")

    def _index_args(self) -> list:
        a = []
        if self.spec.index_url:
            a += ["--index-url", self.spec.index_url]
        for u in self.spec.extra_index_urls:
            a += ["--extra-index-url", u]
        if self.spec.index_strategy:
            a += ["--index-strategy", self.spec.index_strategy]
        return a

    def _create(self, path: Path, requirement_files):
        shutil.rmtree(path, ignore_errors=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._uv("venv", "--python", self.spec.python, path)
        args = ["pip", "install", "--python", path / "bin" / "python"]
        for f in requirement_files:
            args += ["-r", f]
        self._uv(*args, *self._index_args())

    def _probe(self, python) -> dict:
        code = ("import json, sys, sysconfig, importlib.metadata as m; print(json.dumps({'python': sys.version.split()[0], "
                "'python_tag': sys.implementation.cache_tag, 'purelib': sysconfig.get_paths()['purelib'], "
                "'torch': m.version('torch')}))")
        return json.loads(subprocess.run([str(python), "-c", code], check=True, capture_output=True, text=True).stdout)

    def ensure(self, arch: str):
        if self.is_ready():
            link_bigmoe(self.path)
            return
        t0 = time.time()
        log.info("creating the %s runtime environment %s", self.spec.name, self.path)
        self._create(self.path, [self.spec.requirements])
        wheels = [self.wheel(sb, arch) for sb in self.spec.source_builds]
        if wheels:
            self._uv("pip", "install", "--python", self.python, "--no-deps", *wheels)
            sp = _site_packages(self.path)
            for w in wheels:  # source builds carry debug info (exllamav3_ext: 380 MB -> 261 MB)
                for name in _wheel_libraries(w):
                    if (sp / name).is_file():
                        strip_debug(sp / name)
        link_bigmoe(self.path)
        info = self._probe(self.python)
        if self.spec.smoke_test:
            r = subprocess.run([str(self.python), "-c", self.spec.smoke_test], capture_output=True, text=True)
            if r.returncode:
                raise PrepareError(f"smoke test of {self.path} failed:\n{r.stderr[-3000:]}")
        info.update(key=self.spec.key(), name=self.spec.name, created=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    wheels=[str(w) for w in wheels], seconds=round(time.time() - t0))
        (self.path / self.MARKER).write_text(json.dumps(info, indent=1))
        log.info("runtime environment ready in %.0f s (torch %s, Python %s)", time.time() - t0, info["torch"],
                 info["python"])

    def ensure_build_env(self) -> Path:
        marker = self.build_path / self.MARKER
        try:
            if json.loads(marker.read_text()).get("key") == self.spec.key():
                link_bigmoe(self.build_path)
                return self.build_path
        except (OSError, ValueError):
            pass
        log.info("creating the build environment %s (CUDA compiler from pip)", self.build_path)
        self._create(self.build_path, [self.spec.requirements, self.spec.build_requirements])
        sp = _site_packages(self.build_path)
        cu = sp / self.spec.cuda_subdir
        if patch_cuda_glibc_rsqrt(cu / "include"):
            log.info("patched %s for glibc >= 2.41", cu / "include/crt/math_functions.h")
        # CUDA_HOME layout that torch.utils.cpp_extension expects, on top of the pip CUDA wheels
        lib64 = cu / "lib64"
        if not lib64.exists():
            lib64.symlink_to("lib")
        home = self.build_path / "cuda"
        shutil.rmtree(home, ignore_errors=True)
        (home / "lib64").mkdir(parents=True)
        for d in ("bin", "include", "nvvm"):
            if (cu / d).exists():
                (home / d).symlink_to(cu / d)
        for f in sorted((cu / "lib").iterdir()):
            (home / "lib64" / f.name).symlink_to(f)
        for f in sorted((home / "lib64").iterdir()):
            if ".so." in f.name:
                plain = home / "lib64" / (f.name.split(".so.")[0] + ".so")
                if not plain.exists():
                    plain.symlink_to(f)
        link_bigmoe(self.build_path)
        marker.write_text(json.dumps({"key": self.spec.key(), "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")}))
        return self.build_path

    def build_environ(self, arch: str) -> dict:
        b = self.build_path
        env = dict(os.environ)
        env.update(CUDA_HOME=str(b / "cuda"), PATH=f"{b / 'bin'}:{b / 'cuda' / 'bin'}:{env.get('PATH', '')}",
                   TORCH_CUDA_ARCH_LIST=arch, MAX_JOBS=str(max(1, (os.cpu_count() or 2) - 2)))
        env.pop("VIRTUAL_ENV", None)
        return env

    def wheel(self, sb: SourceBuild, arch: str) -> Path:
        key = hashlib.sha256(f"{self.spec.key()}|{sb.name}|{sb.version}|{sb.url}|{arch}".encode()).hexdigest()[:10]
        out = Path(self.home.wheels) / f"{sb.name}-{sb.version}-{key}"
        found = sorted(out.glob("*.whl")) if out.is_dir() else []
        if found:
            return found[-1]
        self.ensure_build_env()
        log.info("building %s %s from source for sm_%s (a few minutes)", sb.name, sb.version, arch.replace(".", ""))
        tarball = self._download(sb.url, f"{sb.name}-{hashlib.sha256(sb.url.encode()).hexdigest()[:12]}.tar.gz")
        tmp_root = Path(self.home.root) / "tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
            with tarfile.open(tarball) as tf:
                if hasattr(tarfile, "data_filter"):
                    tf.extractall(tmp, filter="data")
                else:
                    tf.extractall(tmp)
            tops = [p for p in Path(tmp).iterdir() if p.is_dir()]
            if len(tops) != 1:
                raise PrepareError(f"{sb.url}: expected one top-level directory, got {tops}")
            stage = Path(tmp) / "wheel"
            env = self.build_environ(arch)
            cmd = [find_uv(), "build", "--wheel", "--no-build-isolation", "--python", str(self.build_path / "bin/python"),
                   "--out-dir", str(stage), str(tops[0])]
            log.info("$ %s", " ".join(cmd[1:]))
            if subprocess.run(cmd, env=env).returncode:
                raise PrepareError(f"building {sb.name} failed")
            built = sorted(stage.glob("*.whl"))
            if len(built) != 1:
                raise PrepareError(f"building {sb.name} produced {built}")
            out.mkdir(parents=True, exist_ok=True)
            shutil.move(str(built[0]), out / built[0].name)
        return sorted(out.glob("*.whl"))[-1]

    def _download(self, url: str, filename: str) -> Path:
        d = Path(self.home.downloads)
        d.mkdir(parents=True, exist_ok=True)
        dst = d / filename
        if dst.is_file() and dst.stat().st_size > 0:
            return dst
        log.info("downloading %s", url)
        tmp = dst.with_suffix(".part")
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
        os.replace(tmp, dst)
        return dst

    def kernel_dirs(self, specs, arch: str) -> dict:
        info = self.info()
        if not info:
            raise BoxError(f"runtime {self.path} is not ready")
        return kernels.kernel_dirs(specs, self.home.kernels, torch_version=info["torch"],
                                   python_tag=info["python_tag"], arch=arch)

    def build_kernels(self, specs, arch: str):
        dirs = self.kernel_dirs(specs, arch)
        jobs = [{"name": s.name, "env_var": s.env_var, "module": s.module, "function": s.function,
                 "dir": str(dirs[s.name])} for s in specs if not kernels.is_built(s, dirs[s.name])]
        if not jobs:
            return
        self.ensure_build_env()
        r = subprocess.run([str(self.build_path / "bin/python"), "-c", _BUILD_KERNELS, "build", json.dumps(jobs)],
                           env=self.build_environ(arch))
        if r.returncode:
            missing = [s.name for s in specs if not kernels.is_built(s, dirs[s.name])]
            if missing:
                raise PrepareError(f"building kernels {missing} failed (exit {r.returncode})")
            log.warn("the kernel build process exited with %d after building every kernel", r.returncode)

    def remove_build_env(self):
        shutil.rmtree(self.build_path, ignore_errors=True)


class ExistingRuntime:
    """runtime="current" or a path to an interpreter: nothing is installed into it, and building kernels with it needs
    CUDA_HOME."""

    def __init__(self, python, home):
        self._python = Path(python)
        self.home = home
        self.path = self._python.parent.parent

    def __repr__(self):
        return f"ExistingRuntime({str(self._python)!r})"

    @property
    def python(self) -> Path:
        return self._python

    def info(self) -> dict:
        code = ("import json, sys, sysconfig, importlib.metadata as m; print(json.dumps({'python': sys.version.split()[0], "
                "'python_tag': sys.implementation.cache_tag, 'purelib': sysconfig.get_paths()['purelib'], "
                "'torch': m.version('torch')}))")
        r = subprocess.run([str(self._python), "-c", code], capture_output=True, text=True)
        return json.loads(r.stdout) if r.returncode == 0 else {}

    def is_ready(self) -> bool:
        return bool(self.info())

    @property
    def purelib(self) -> Path:
        return Path(self.info()["purelib"])

    def ensure(self, arch: str):
        if not self.is_ready():
            raise BoxError(f"{self._python} has no torch; install the backend's requirements there, or use "
                           "runtime='managed'")
        if Path(sys.executable).resolve() != self._python.resolve():
            link_bigmoe(self.path)

    def kernel_dirs(self, specs, arch: str) -> dict:
        info = self.info()
        return kernels.kernel_dirs(specs, self.home.kernels, torch_version=info["torch"],
                                   python_tag=info["python_tag"], arch=arch)

    def build_kernels(self, specs, arch: str):
        dirs = self.kernel_dirs(specs, arch)
        jobs = [{"name": s.name, "env_var": s.env_var, "module": s.module, "function": s.function,
                 "dir": str(dirs[s.name])} for s in specs if not kernels.is_built(s, dirs[s.name])]
        if jobs:
            env = dict(os.environ, TORCH_CUDA_ARCH_LIST=arch)
            subprocess.run([str(self._python), "-c", _BUILD_KERNELS, "build", json.dumps(jobs)], env=env)
            missing = [s.name for s in specs if not kernels.is_built(s, dirs[s.name])]
            if missing:
                raise PrepareError(f"building kernels {missing} failed (CUDA_HOME with nvcc needed in this environment)")
