"""The server runtime of every backend: Python 3.12, the pinned package set in requirements/ (torch 2.13 + CUDA 13) and
exllamav3 built from source at one release, so GLM-5.3-Flash and DeepSeek-V4.1-Flash run on the same exllamav3.

frontend="vllm" (the default) adds vLLM 0.33.0 as the serving engine (requirements/runtime-vllm*.txt, bigmoe/vllm):
the same torch and the same exllamav3 wheel (reused from the native runtime's build), vLLM's frontend dependencies on
top. frontend="native" is the exllamav3 AsyncGenerator server the backends shipped before.

Each backend keeps its own environment (the key includes the backend's name), so either can be rebuilt alone.
"""
from pathlib import Path

from ..runtime import RuntimeSpec, SourceBuild

EXLLAMAV3 = SourceBuild("exllamav3", "1.6.6", "https://codeload.github.com/turboderp-org/exllamav3/tar.gz/"
                                              "bd1f519172406a9560bd5c90e61000691bb69a07")   # tag v1.6.6
REQUIREMENTS = Path(__file__).resolve().parent / "requirements"
VLLM_VERSION = "0.33.0"
DESCRIPTION = f"torch 2.13 + CUDA 13; exllamav3 {EXLLAMAV3.version} built from source"
FRONTENDS = ("vllm", "native")


def description(frontend: str = "vllm") -> str:
    return DESCRIPTION + (f"; vLLM {VLLM_VERSION} serving" if frontend == "vllm" else "")


def runtime_spec(name: str, smoke_test: str, frontend: str = "native") -> RuntimeSpec:
    native = RuntimeSpec(
        name=name, python="3.12", requirements=REQUIREMENTS / "runtime.txt",
        build_requirements=REQUIREMENTS / "build.txt",
        index_url="https://download.pytorch.org/whl/cu130", extra_index_urls=("https://pypi.org/simple",),
        index_strategy="unsafe-best-match", source_builds=(EXLLAMAV3,), cuda_subdir="nvidia/cu13",
        smoke_test=smoke_test)
    if frontend == "native":
        return native
    if frontend != "vllm":
        raise ValueError(f"frontend must be one of {FRONTENDS}, got {frontend!r}")
    return RuntimeSpec(
        name=f"{name}-vllm", python="3.12", requirements=REQUIREMENTS / "runtime-vllm.txt",
        build_requirements=REQUIREMENTS / "build.txt", nodeps_requirements=REQUIREMENTS / "runtime-vllm-nodeps.txt",
        index_url=native.index_url, extra_index_urls=native.extra_index_urls, index_strategy=native.index_strategy,
        source_builds=(EXLLAMAV3,), cuda_subdir=native.cuda_subdir, wheels_like=native,
        smoke_test=smoke_test + "; import vllm; from vllm.v1.core.sched.scheduler import Scheduler")
