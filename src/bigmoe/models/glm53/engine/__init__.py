"""GLM-5.3-Flash's model-specific pieces of the exllamav3 executor (from glm-flash-lite 6fe077d): the cost-aware n-gram
drafting policy (ngram_policy.py, speculation="ngram"). The expert offload is the shared engine (bigmoe.engine); the
serving engine is vLLM (bigmoe.vllm). exllamav3's own fused hyper-connection sites (EXL3_HC_FOLD=1, set by the backend)
replace the custom decode kernels the 1.5.1 runtime needed."""

# upstream glm-flash-lite modules not included here: the executor refuses to start with their switches on
NOT_INCLUDED = {"GLM53_ZC_VRAM": "exl3_tiers.py", "GLM53_CPU_TIER": "cpu_tier.py", "GLM53_K_OVL": "k_overlap.py",
                "GLM53_NVTX": "n136_prof.py", "GLM53_PROF": "n136_prof.py"}
