"""vLLM serves; exllamav3 over the expert engine (bigmoe.engine) computes. Overview: docs/architecture.md.

  API server   vLLM's OpenAI / Anthropic server (rendering, tool and reasoning parsers, image preprocessing,
               detokenization, stop strings) plus routes.py (/stats, /server_info, /generate, diagnostics)
  EngineCore   vLLM's scheduler (BoxScheduler, scheduler.py) and one in-process worker (BoxWorker, worker.py) that runs
               each scheduled step on exllamav3 (engine.py)

vLLM cannot run either model's attention on sm86 and has no EXL3 quantization, so the forwards and the K/V cache stay
exllamav3's: the model classes vLLM sees are attention-free stubs (models.py), and BoxKVCacheManager answers vLLM's
capacity and prefix-cache questions from exllamav3's page table.

register() is the vllm.general_plugins entry point; vLLM calls it in every process it starts.
"""

ARCHS = {
    # backend -> the architecture name vLLM's config / processors see (models.py)
    "dsv41": "BigmoeDeepseekV41ForCausalLM",
    "glm53": "BigmoeGlm53ForCausalLM",
}


def register():
    from vllm import ModelRegistry
    from . import models
    for arch in ARCHS.values():
        if arch not in ModelRegistry.get_supported_archs():
            ModelRegistry.register_model(arch, f"bigmoe.vllm.models:{arch}")
    models.register_quantization()
    reserve_output_room()


def reserve_output_room():
    """The context rule (serving.output_budget): a prompt needs room for min(max_tokens, 1024) new
    tokens plus the drafter's window, else it is refused with "This model's maximum context length is N tokens ..."
    (a 400 that agent clients answer by compacting) instead of being admitted for a handful of tokens. vLLM's own
    limit is the whole context; its TokenizeParams.max_input_tokens becomes the context minus that room."""
    import os
    reserve = os.environ.get("BIGMOE_CTX_RESERVE")
    if reserve is None:
        return
    from vllm.renderers.params import TokenizeParams
    from .. import serving
    if getattr(TokenizeParams, "_bigmoe_room", None) is not None:
        return
    room = int(reserve)

    def max_input_tokens(self):
        if self.max_total_tokens is None:
            return None
        if self.max_output_tokens == 0:      # pooling: no output
            return self.max_total_tokens
        want = self.max_output_tokens or serving.MIN_OUTPUT_TOKENS
        return max(1, self.max_total_tokens - room - min(want, serving.MIN_OUTPUT_TOKENS))
    TokenizeParams.max_input_tokens = property(max_input_tokens)
    TokenizeParams._bigmoe_room = room
