"""vLLM worker whose model runner is the exllamav3 executor (engine.py). Runs in the EngineCore process
(UniProcExecutor).

Configured through the environment the serve entry point (serve.py) sets before vLLM starts:
  BIGMOE_BACKEND        dsv41 | glm53
  BIGMOE_EXL_ARGV       JSON list: the engine's command line (exllamav3 model_init arguments)
  BIGMOE_VLLM_ENGINE    exl (default) | fake (engine.FakeEngine: no GPU, for tests of the plumbing)
"""
import json
import os

from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, DraftTokenIds, ModelRunnerOutput
from vllm.v1.worker.worker_base import CompilationTimes, WorkerBase

from . import state


def make_engine():
    kind = os.environ.get("BIGMOE_VLLM_ENGINE", "exl")
    if kind == "fake":
        from .engine import FakeEngine
        cfg = json.loads(os.environ.get("BIGMOE_FAKE_ENGINE", "{}"))
        return FakeEngine(**cfg)
    from .engine import ExlEngine
    e = ExlEngine(os.environ["BIGMOE_BACKEND"], json.loads(os.environ["BIGMOE_EXL_ARGV"]))
    e.load()
    return e


class BoxWorker(WorkerBase):

    def init_device(self):
        if os.environ.get("BIGMOE_VLLM_ENGINE", "exl") != "fake":
            import torch
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(self.device)
        self.engine = None

    def load_model(self, *, load_dummy_weights: bool = False):
        # serve.py pinned the API server (and so this process) to the main CPUs; loading uses every core, and the
        # engine pins its threads again once loaded (engine.pin_unpinned_threads)
        os.sched_setaffinity(0, set(range(os.cpu_count() or 1)))
        self.engine = make_engine()
        state.set_engine(self.engine)

    def get_model(self):
        return None

    def get_kv_cache_spec(self):
        return {}                    # attention-free for vLLM: the K/V cache is exllamav3's (BoxKVCacheManager)

    def determine_available_memory(self) -> int:
        return 0

    def initialize_from_config(self, kv_cache_configs) -> None:
        return None

    def compile_or_warm_up_model(self) -> CompilationTimes:
        return CompilationTimes(language_model=0.0, encoder=0.0)

    def get_supported_tasks(self):
        return ("generate",)

    def get_cache_block_size_bytes(self) -> int:
        return 0

    def take_draft_token_ids(self):
        """The drafter's tokens from the last step, which vLLM's scheduler verifies in the next one."""
        drafts = self.engine.take_drafts() if self.engine is not None else {}
        if not drafts:
            return None
        return DraftTokenIds(req_ids=list(drafts), draft_token_ids=[list(d) for d in drafts.values()])

    def reset_mm_cache(self) -> None:
        return None

    def reset_encoder_cache(self) -> None:
        return None

    def check_health(self) -> None:
        return None

    def execute_model(self, scheduler_output):
        """Take the step vLLM scheduled; it runs in sample_tokens, which gets the step's grammar bitmask (vLLM
        computes it for structured-output requests between the two calls). Admission and release already happened
        through BoxKVCacheManager."""
        e = self.engine
        for rid in scheduler_output.finished_req_ids:
            e.release(rid)
        plan = state.take_plan()
        if not scheduler_output.num_scheduled_tokens:
            if getattr(e, "idle_pending", False):
                e.idle()
            state.put_progress({})
            return EMPTY_MODEL_RUNNER_OUTPUT
        self.step = (scheduler_output, {rid: plan[rid] for rid in scheduler_output.num_scheduled_tokens})
        return None

    def sample_tokens(self, grammar_output):
        """Run exactly the step vLLM scheduled: the ranges BoxScheduler.schedule recorded, the spec tokens it scheduled
        for verification, the grammar rows of structured-output requests."""
        scheduler_output, plan = self.step
        self.step = None
        result = self.engine.execute(plan, scheduler_output.scheduled_spec_decode_tokens, grammar_output)
        req_ids = list(plan)
        sampled = [result[rid][0] if rid in result else [] for rid in req_ids]
        state.put_progress({rid: result[rid][1] for rid in req_ids if rid in result})
        return ModelRunnerOutput(req_ids=req_ids, req_id_to_index={r: i for i, r in enumerate(req_ids)},
                                 sampled_token_ids=sampled)

    # ---- RPC targets for the API server's extra routes (routes.py), via collective_rpc ----
    def bigmoe_call(self, method: str, *args, **kwargs):
        return getattr(self.engine, method)(*args, **kwargs)

    def shutdown(self) -> None:
        return None
