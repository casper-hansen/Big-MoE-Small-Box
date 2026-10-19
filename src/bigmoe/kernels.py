"""Torch extensions are built once into <home>/kernels/<name>-<hash of sources, torch, Python ABI, GPU arch> and
handed to the engine through each module's build-directory variable. The server then imports the .so directly: stock
torch.utils.cpp_extension.load re-runs ninja + nvcc checks on every start, which would need the CUDA toolchain at
runtime.
"""
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class KernelSpec:
    name: str  # the build directory holds <name>.so
    env_var: str  # where the engine module reads its build directory from
    module: str  # module.function() builds / loads it with cpp_extension.load
    function: str
    sources: tuple = ()  # files hashed into the key (the module itself, extra .cu / .cpp / .h)


def kernel_key(spec: KernelSpec, *, torch_version: str, python_tag: str, arch: str) -> str:
    h = hashlib.sha256()
    for part in ("bigmoe-kernel-v1", spec.name, spec.module, spec.function, torch_version, python_tag, arch):
        h.update(part.encode() + b"\0")
    for src in sorted(Path(s) for s in spec.sources):
        h.update(src.name.encode() + b"\0" + src.read_bytes() + b"\0")
    return h.hexdigest()[:16]


def kernel_dirs(specs, root, *, torch_version: str, python_tag: str, arch: str) -> dict:
    return {s.name: Path(root) / f"{s.name}-{kernel_key(s, torch_version=torch_version, python_tag=python_tag, arch=arch)}"
            for s in specs}


def is_built(spec: KernelSpec, directory) -> bool:
    return (Path(directory) / f"{spec.name}.so").is_file()


def install_prebuilt_loader():
    """A missing .so still goes through the stock JIT build (which needs the CUDA toolchain)."""
    import importlib.util
    import torch.utils.cpp_extension as ce
    if getattr(ce.load, "_bigmoe_prebuilt", False):
        return
    jit = ce.load

    def load(name, sources, *a, build_directory=None, **kw):
        so = os.path.join(build_directory or "", name + ".so")
        if build_directory and os.path.isfile(so):
            spec = importlib.util.spec_from_file_location(name, so)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
        return jit(name, sources, *a, build_directory=build_directory, **kw)

    load._bigmoe_prebuilt = True
    ce.load = load


def build(jobs):
    """Builds into a temporary directory that is renamed into place, so an interrupted build never leaves a .so that
    looks finished."""
    for job in jobs:
        final = Path(job["dir"])
        tmp = final.with_name(f"{final.name}.tmp-{os.getpid()}")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        os.environ[job["env_var"]] = str(tmp)
        print(f"[bigmoe] building {job['name']} ({job['module']}.{job['function']}) -> {final}", flush=True)
        getattr(importlib.import_module(job["module"]), job["function"])()
        so = tmp / f"{job['name']}.so"
        if not so.is_file():
            raise RuntimeError(f"{job['module']}.{job['function']}() did not produce {so}")
        for f in tmp.iterdir():
            if f.suffix in (".so", ".cu", ".cpp"):
                continue
            shutil.rmtree(f) if f.is_dir() else f.unlink()
        if shutil.which("strip"):
            subprocess.run(["strip", "--strip-debug", str(so)], check=False)
        shutil.rmtree(final, ignore_errors=True)
        os.replace(tmp, final)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2 or argv[0] != "build":
        sys.exit("usage: build '<json list of {name, env_var, module, function, dir}>'")
    build(json.loads(argv[1]))
    # done: skip interpreter teardown, where unloading the freshly imported extensions next to torch / CUDA can
    # segfault (seen with glm53_nv2 on torch 2.13)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
