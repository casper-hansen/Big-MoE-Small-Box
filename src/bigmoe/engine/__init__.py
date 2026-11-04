"""Shared expert-offload engine of the exllamav3 backends (GLM-5.3-Flash, DeepSeek-V4.1-Flash).

Routed experts live in three tiers and every MoE layer call is served from wherever its experts are:
  VRAM   CLOCK cache of uniform slots in the free VRAM (tiers.Pool); an expert of layer li fills its slot in that
         layer's compact layout (gate | up | down), so layers with different expert sizes share one pool
  RAM    pinned, GPU-mapped host arena (nv2: exclusive of VRAM, LRU, optional small-slot class)
  NVMe   packed expert store read with O_DIRECT (store v1: one record size; v2: per-layer record sizes)
exllamav3's kernels read each expert through its live pointer tables, which the engine points at the expert's
current home; the bytes and the arithmetic are exllamav3's own. nv2.py adds the device-side publish / host planner
loop, the NVMe reader, prefetch, staged prefill and the CPU lane (AVX-512 VBMI / AVX2 trellis kernels).

Each backend sets BIGMOE_ENGINE_PREFIX (GLM53_, DSV41_) and the engine reads its knobs as <prefix>NAME.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def kernel_specs():
    """The engine's two torch extensions (one build per torch / Python / GPU arch, shared by every backend)."""
    from ..kernels import KernelSpec
    nv2_src = tuple(sorted((ROOT / "kernels" / "nv2").iterdir()))
    return [KernelSpec("bigmoe_ec", "BIGMOE_EC_BUILD", "bigmoe.engine.ec", "ext", (ROOT / "ec.py", ROOT / "env.py")),
            KernelSpec("bigmoe_nv2", "BIGMOE_NV2_BUILD", "bigmoe.engine.nv2", "ext", (ROOT / "nv2.py", ROOT / "env.py") + nv2_src)]
