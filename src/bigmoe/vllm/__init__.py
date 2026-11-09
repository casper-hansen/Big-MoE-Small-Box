"""vLLM as the serving engine in front of the exllamav3 + shared-engine compute (both backends).

vLLM 0.33 cannot run either model's attention on sm86 (DeepSeek-V4.1's sparse-MLA backends are sm90/100/120 only,
GLM-5.3's sparse indexer needs DeepGEMM) and has no EXL3 quantization, so vLLM runs everything except the forward
passes:

  API server process   vLLM's OpenAI server (chat / completions / responses, tool + reasoning parsers, the model's
                       renderer or chat template, multimodal input processing, detokenization, stop strings, metrics)
                       plus this package's routes (routes.py: /stats, /server_info, /generate, KL tools)
  EngineCore process   vLLM's scheduler (BoxScheduler: admission, stop checks, outputs) and one in-process worker
                       (BoxWorker) whose runner is exllamav3's own Generator over the shared expert engine
                       (bigmoe.engine), stepped once per vLLM step (engine.py).

The engine is the source of truth for progress: exllamav3 keeps its paged K/V cache, prefix reuse (GPU pages, the
pinned-RAM and disk tiers, recurrent checkpoints), chunked / staged / warm prefill and speculative decoding exactly as
the native servers ran them, and after every step the scheduler takes each request's computed-token count and cached
prefix from it (scheduler.py). vLLM keeps no K/V cache of its own (the stub model classes are attention-free).

register() is the vllm.general_plugins entry point: vLLM calls it in every process it starts.
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
    """The native servers' admission rule (serving.output_budget): a prompt needs room for min(max_tokens, 1024) new
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
