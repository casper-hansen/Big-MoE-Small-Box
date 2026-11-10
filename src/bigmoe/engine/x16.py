"""X16 lane: CPU-computed experts that are bit-identical to the GPU's (mode nvme, NV_X16_HI leading MoE layers).

The ordinary CPU lane is close to the GPU's expert outputs but not equal, and some models (GLM-5.3-Flash) amplify
any change in their early MoE layers. In those layers the X16 lane computes the CPU's experts bit-exactly:
  CPU   gate / up GEMVs with rect_x16 (nv2/ft_x16.h: the tensor-core MMA's fp16-accumulator rounding and the kernel's
        fold / warp summation order) -> fp16 rows; after the GPU's activation epilogue, the down GEMVs -> fp32 rows
        (host-mapped buffers)
  GPU   x16_epa: exllamav3's kernel-A epilogue on those rows; x16_red: kernel B's token reduction over every slot (the
        GPU's own down outputs and the CPU's) -> the layer output the full GPU kernel would have produced
Every start self-tests each X16 layer (nv2.NV.x16_selftest) and leaves the lane off unless it is bit-identical. An X16
expert costs several times the CPU time of an ordinary one. This module builds / loads the device half
(kernels/x16/x16_dev.cu) with exllamav3's nvcc flags. Details: docs/x16.md.
"""
import os

from .env import build_dir

ACT = {"silu": 0, "gelu": 1, "relu2": 2, "swiglu_oai": 3}   # exllamav3 MOE_COOP_ACT_*

_EXT = None


def ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        root = os.path.dirname(os.path.abspath(__file__))
        here = os.path.join(root, "kernels", "x16")
        d = build_dir("X16", os.path.join(root, "build", "x16"))
        os.makedirs(d, exist_ok=True)
        # exllamav3's own flags (setup.py): the epilogue / reduction must compile to the same instructions
        _EXT = load(name="bigmoe_x16", sources=[os.path.join(here, "x16_dev.cu")], build_directory=d,
                    extra_cflags=["-O3"], extra_cuda_cflags=["-lineinfo", "-O3", "--use_fast_math"], verbose=False)
    return _EXT


def unsupported(m):
    """Why the X16 lane can't serve this BlockSparseMLP (None: it can): the bsz-1 path must be exllamav3's fused decode
    kernel with the configuration x16_epa / x16_red replicate."""
    if getattr(m, "bc", None) is None or getattr(m, "experts_cfg", None) is None:
        return "no fused decode kernel"
    if not getattr(m, "gated", True):
        return "gateless experts"
    if not all(getattr(mm, "mul1", False) for mm in (m.multi_gate, m.multi_up, m.multi_down)):
        return "codebook other than mul1"
    if m.activation_fn not in ACT:
        return f"activation {m.activation_fn!r}"
    if any(ls and ls[0].inner.bias is not None for ls in (m.gates, m.ups, m.downs)):
        return "expert biases"
    if getattr(m, "latent_in", None) is not None or getattr(m, "tp_reduce", False):
        return "latent / tensor-parallel MoE"
    if getattr(m.experts_cfg, "min_expert", -1) not in (-1, None):
        return "expert-parallel range"
    if m.experts_cfg.interm_g.shape[-1] % 128 or m.experts_cfg.out_d.shape[-1] % 128:
        return "widths not multiples of 128"
    return None
