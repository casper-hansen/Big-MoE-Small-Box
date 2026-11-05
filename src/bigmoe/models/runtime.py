"""The server runtime of every backend: Python 3.12, the pinned package set in requirements/ (torch 2.13 + CUDA 13) and
exllamav3 built from source at one release, so GLM-5.3-Flash and DeepSeek-V4.1-Flash run on the same exllamav3.

Each backend keeps its own environment (the key includes the backend's name), so either can be rebuilt alone.
"""
from pathlib import Path

from ..runtime import RuntimeSpec, SourceBuild

EXLLAMAV3 = SourceBuild("exllamav3", "1.6.6", "https://codeload.github.com/turboderp-org/exllamav3/tar.gz/"
                                              "bd1f519172406a9560bd5c90e61000691bb69a07")   # tag v1.6.6
REQUIREMENTS = Path(__file__).resolve().parent / "requirements"
DESCRIPTION = f"torch 2.13 + CUDA 13; exllamav3 {EXLLAMAV3.version} built from source"


def runtime_spec(name: str, smoke_test: str) -> RuntimeSpec:
    return RuntimeSpec(
        name=name, python="3.12", requirements=REQUIREMENTS / "runtime.txt",
        build_requirements=REQUIREMENTS / "build.txt",
        index_url="https://download.pytorch.org/whl/cu130", extra_index_urls=("https://pypi.org/simple",),
        index_strategy="unsafe-best-match", source_builds=(EXLLAMAV3,), cuda_subdir="nvidia/cu13",
        smoke_test=smoke_test)
