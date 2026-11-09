"""What vLLM's API server and scheduler see of the two models: attention-free stub classes (vLLM allocates no K/V cache
and never runs them: the BoxWorker computes with exllamav3) and an `exl3` quantization config, so vLLM's ModelConfig
accepts the checkpoints' quantization_config. The multimodal processors live in mm.py and are attached here.
"""
import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS, register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig


class _NotRun(RuntimeError):
    pass


def register_quantization():
    if "exl3" in QUANTIZATION_METHODS:
        return

    @register_quantization_config("exl3")
    class Exl3Config(QuantizationConfig):
        """EXL3 trellis quantization: computed by exllamav3 in the BoxWorker, never by vLLM layers."""

        def __init__(self, config: dict | None = None):
            super().__init__()
            self.config = dict(config or {})

        def get_name(self):
            return "exl3"

        def get_supported_act_dtypes(self):
            return [torch.float16, torch.bfloat16]

        @classmethod
        def get_min_capability(cls) -> int:
            return 80

        @staticmethod
        def get_config_filenames() -> list[str]:
            return []

        @classmethod
        def from_config(cls, config):
            return cls(config)

        def get_quant_method(self, layer, prefix):
            return None


class _BoxStub(nn.Module):
    """vLLM model interface without weights. is_attention_free: no K/V cache groups, no vLLM prefix caching (the
    engine has its own)."""
    is_attention_free = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config

    def embed_input_ids(self, input_ids: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        raise _NotRun("bigmoe: the stub model is not executed by vLLM")

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, *args, **kwargs):
        raise _NotRun("bigmoe: the stub model is not executed by vLLM")

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        raise _NotRun("bigmoe: the stub model is not executed by vLLM")

    def load_weights(self, weights):
        return set()


class _BoxMMStub(_BoxStub):
    def embed_multimodal(self, **kwargs):
        raise _NotRun("bigmoe: the stub model is not executed by vLLM")


def _multimodal(cls):
    """Image input unless the server runs with BIGMOE_MM=0 (then vLLM refuses images itself)."""
    from .. import multimodal
    if not multimodal.enabled():
        return cls
    from . import mm
    return mm.attach(cls)


@_multimodal
class BigmoeDeepseekV41ForCausalLM(_BoxMMStub):
    pass


@_multimodal
class BigmoeGlm53ForCausalLM(_BoxMMStub):
    pass
