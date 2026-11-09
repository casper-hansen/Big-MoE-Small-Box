"""The server runtime of every backend: Python 3.12, vLLM 0.33.0 as the serving engine (requirements/runtime-vllm*.txt,
bigmoe/vllm) and exllamav3 built from source at one release as its model executor, so GLM-5.3-Flash and
DeepSeek-V4.1-Flash run on the same exllamav3.

The exllamav3 wheel is compiled against (and its cache keyed by) the torch 2.13 + CUDA 13 package set in
requirements/exllamav3-abi.txt, which the server runtime pins too. Each backend keeps its own environment (the key
includes the backend's name), so either can be rebuilt alone.
"""
from pathlib import Path

from ..runtime import RuntimeSpec, SourceBuild

EXLLAMAV3 = SourceBuild("exllamav3", "1.6.6", "https://codeload.github.com/turboderp-org/exllamav3/tar.gz/"
                                              "bd1f519172406a9560bd5c90e61000691bb69a07")   # tag v1.6.6
REQUIREMENTS = Path(__file__).resolve().parent / "requirements"
VLLM_VERSION = "0.33.0"
DESCRIPTION = f"torch 2.13 + CUDA 13; exllamav3 {EXLLAMAV3.version} built from source; vLLM {VLLM_VERSION} serving"
INDEX_URL = "https://download.pytorch.org/whl/cu130"
EXTRA_INDEX_URLS = ("https://pypi.org/simple",)


def description() -> str:
    return DESCRIPTION


def exllamav3_abi(name: str) -> RuntimeSpec:
    """The package set the exllamav3 wheel is built against: only its key is used (Runtime.wheel), it is never
    created as a server environment."""
    return RuntimeSpec(
        name=name, python="3.12", requirements=REQUIREMENTS / "exllamav3-abi.txt",
        build_requirements=REQUIREMENTS / "build.txt", index_url=INDEX_URL, extra_index_urls=EXTRA_INDEX_URLS,
        index_strategy="unsafe-best-match", source_builds=(EXLLAMAV3,), cuda_subdir="nvidia/cu13")


def runtime_spec(name: str, smoke_test: str) -> RuntimeSpec:
    abi = exllamav3_abi(name)
    return RuntimeSpec(
        name=f"{name}-vllm", python="3.12", requirements=REQUIREMENTS / "runtime-vllm.txt",
        build_requirements=REQUIREMENTS / "build.txt", nodeps_requirements=REQUIREMENTS / "runtime-vllm-nodeps.txt",
        index_url=abi.index_url, extra_index_urls=abi.extra_index_urls, index_strategy=abi.index_strategy,
        source_builds=(EXLLAMAV3,), cuda_subdir=abi.cuda_subdir, wheels_like=abi,
        smoke_test=smoke_test + "; import vllm; from vllm.v1.core.sched.scheduler import Scheduler")
