"""vLLM is the serving engine of both backends; exllamav3 over the shared expert engine is its model executor.

vLLM 0.33 cannot run either model's attention on sm86 (DeepSeek-V4.1's sparse-MLA backends are sm90/100/120 only,
GLM-5.3's sparse indexer needs DeepGEMM) and has no EXL3 quantization, so vLLM runs everything except the forward
passes and the K/V memory:

  API server process   vLLM's OpenAI / Anthropic server (chat / completions / responses, tool + reasoning parsers, the
                       model's renderer or chat template, multimodal input processing, detokenization, stop strings,
                       metrics) plus this package's routes (routes.py: /stats, /server_info, /generate, KL tools)
  EngineCore process   vLLM's scheduler (BoxScheduler: admission and preemption, chunked prefill and the token budget,
                       draft-token scheduling and acceptance, stop checks, outputs) and one in-process worker (BoxWorker)
                       that executes each scheduled step on exllamav3 (engine.py): prefill chunks, one batched
                       sampling / verify forward, the drafter's next drafts.

vLLM's KV block manager is replaced by BoxKVCacheManager (scheduler.py), which answers vLLM's capacity and
prefix-cache questions from exllamav3's page table (GPU pages, the pinned-RAM and disk tiers, recurrent checkpoints).
The stub model classes are attention-free (models.py), so vLLM allocates no K/V of its own.

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
