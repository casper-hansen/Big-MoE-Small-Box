"""vLLM worker whose model runner is the exllamav3 engine (engine.py). Runs in the EngineCore process (UniProcExecutor).

Configured through the environment the serve entry point (serve.py) sets before vLLM starts:
  BIGMOE_BACKEND        dsv41 | glm53
  BIGMOE_EXL_ARGV       JSON list: the native server's command line (exllamav3 model_init arguments)
  BIGMOE_VLLM_ENGINE    exl (default) | fake (engine.FakeEngine: no GPU, for tests of the plumbing)
"""
import json
import os

from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
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
        self.pending: dict = {}      # req_id -> tokens sampled while the request was not in vLLM's batch
        self.progress: dict = {}

    def load_model(self, *, load_dummy_weights: bool = False):
        # serve.py pinned the API server (and so this process) to the main CPUs; loading uses every core, and the
        # engine pins its threads again once loaded (engine.pin_unpinned_threads)
        os.sched_setaffinity(0, set(range(os.cpu_count() or 1)))
        self.engine = make_engine()
        state.set_engine(self.engine)

    def get_model(self):
        return None

    def get_kv_cache_spec(self):
        return {}                    # attention-free for vLLM: the K/V cache is exllamav3's

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
        return None

    def reset_mm_cache(self) -> None:
        return None

    def reset_encoder_cache(self) -> None:
        return None

    def check_health(self) -> None:
        return None

    def execute_model(self, scheduler_output):
        e = self.engine
        for rid in scheduler_output.finished_req_ids:
            e.cancel(rid)
            self.pending.pop(rid, None)
            self.progress.pop(rid, None)
        for nr in scheduler_output.scheduled_new_reqs:
            e.add(nr.req_id, nr.prompt_token_ids, nr.sampling_params, self._mm_inputs(nr))
        scheduled = scheduler_output.num_scheduled_tokens
        if not scheduled:
            if getattr(e, "idle_pending", False):
                e.idle()
            state.put_progress({})
            return EMPTY_MODEL_RUNNER_OUTPUT
        for rid, (toks, prog) in e.step().items():
            if toks:
                self.pending.setdefault(rid, []).extend(toks)
            self.progress[rid] = prog
        req_ids = list(scheduled)
        sampled = [self.pending.pop(rid, []) for rid in req_ids]
        state.put_progress({rid: self.progress[rid] for rid in req_ids if rid in self.progress})
        return ModelRunnerOutput(req_ids=req_ids, req_id_to_index={r: i for i, r in enumerate(req_ids)},
                                 sampled_token_ids=sampled)

    def sample_tokens(self, grammar_output):
        raise RuntimeError("bigmoe: execute_model always returns the output")

    def _mm_inputs(self, nr):
        if not nr.mm_features:
            return []
        from . import mm
        return mm.features_of(nr)

    # ---- RPC targets for the API server's extra routes (routes.py), via collective_rpc ----
    def bigmoe_call(self, method: str, *args, **kwargs):
        return getattr(self.engine, method)(*args, **kwargs)

    def shutdown(self) -> None:
        return None
