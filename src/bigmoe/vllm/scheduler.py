"""vLLM's scheduler over the exllamav3 executor (engine.py). vLLM makes every scheduling decision; three adapters
connect it to the engine:

- BoxKVCacheManager replaces vLLM's K/V block manager. get_computed_blocks reserves the request's pages in the engine
  and returns the prompt tokens the prefix cache saves; allocate_slots fails when the engine has no free pages (the
  request waits, or vLLM preempts); free returns the pages.
- _reserve_prefill_lookahead, the last word on a chunk's size in both of vLLM's scheduling loops, shortens a prefill
  chunk to where exllamav3 can end a forward (engine.plan_chunk), so the engine computes exactly what was scheduled.
- update_from_output lets vLLM do its bookkeeping, then corrects each request's computed-token count where the engine's
  differs (counted in /stats: scheduler.corrections), and finishes requests the engine ended on its own.

schedule() records each scheduled request's range for the worker.
"""
from vllm.outputs import RequestError
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason
from vllm.v1.request import RequestStatus

from . import state

_FINISH = {"length": (RequestStatus.FINISHED_LENGTH_CAPPED, FinishReason.LENGTH),
           "stop": (RequestStatus.FINISHED_STOPPED, FinishReason.STOP),
           "error": (RequestStatus.FINISHED_ERROR, FinishReason.ERROR)}


class BoxKVCacheManager(KVCacheManager):
    """vLLM's KV cache manager interface over exllamav3's page table (installed on the scheduler's manager by
    BoxScheduler; vLLM's own block pool has no KV cache groups for the attention-free stubs and stays empty)."""

    @property
    def engine(self):
        e = state.engine()
        assert e is not None, "bigmoe: the engine is not loaded"
        return e

    @property
    def usage(self) -> float:
        return float(self.engine.page_usage())

    def prefix_cache_lookup_enabled(self, request) -> bool:
        return True                  # exllamav3's prefix cache, always on

    def get_computed_blocks(self, request, max_cache_hit_length=None):
        hit = self.engine.reserve(request)
        if hit is None:              # no free pages: allocate_slots refuses it too
            return self.empty_kv_cache_blocks, 0, 0
        limit = request.num_tokens - 1 if max_cache_hit_length is None else max_cache_hit_length
        return self.empty_kv_cache_blocks, max(0, min(int(hit), limit)), 0

    def get_computed_blocks_for_connector(self, request, max_cache_hit_length=None):
        return *self.get_computed_blocks(request, max_cache_hit_length), False

    def allocate_slots(self, request, num_new_tokens, *args, **kwargs):
        if not self.engine.ensure(request):
            return None
        return self.empty_kv_cache_blocks

    def free(self, request) -> None:
        self.engine.release(request.request_id)

    def pop_blocks_for_free(self, request):
        self.engine.release(request.request_id)
        return []

    def cache_blocks(self, request, num_computed_tokens: int) -> None:
        return None                  # exllamav3 hashes its own pages

    def estimate_cached_tokens(self, request) -> int:
        return min(self.engine.cached_of(request.request_id), request.num_prompt_tokens)

    def get_blocks(self, request_id: str):
        return self.empty_kv_cache_blocks

    def reset_prefix_cache(self) -> bool:
        return False                 # exllamav3's tiers are not dropped from the API


class BoxScheduler(Scheduler):

    def __init__(self, *args, vllm_config, **kwargs):
        # EngineCore turns chunked prefill off for a model without KV cache groups (it expects an encoder-only model);
        # these models' K/V is exllamav3's, and their prompts are prefilled in vLLM-sized chunks
        vllm_config.scheduler_config.enable_chunked_prefill = True
        super().__init__(*args, vllm_config=vllm_config, **kwargs)
        self.kv_cache_manager.__class__ = BoxKVCacheManager
        self.kv_cache_manager.enable_caching = True

    def has_requests(self) -> bool:
        # an idle engine with disk-tier pages to flush keeps stepping (the worker does one flush slice per empty step)
        return super().has_requests() or state.idle_work_pending()

    def _reserve_prefill_lookahead(self, request, num_computed_tokens: int, num_new_tokens: int) -> int:
        n = super()._reserve_prefill_lookahead(request, num_computed_tokens, num_new_tokens)
        e = state.engine()
        if e is None or n <= 0:
            return n
        return e.plan(request.request_id, num_computed_tokens, n, request.num_tokens)

    def update_draft_token_ids(self, draft_token_ids):
        # A request whose prefill stopped one token short (engine.plan_chunk, hold_last) takes its first drafts: its
        # next step is the last prompt token plus them, a decode step. vLLM would drop drafts for a prefill chunk.
        for rid in draft_token_ids.req_ids:
            r = self.requests.get(rid)
            if r is not None and r.is_prefill_chunk and r.num_computed_tokens >= r.num_tokens - 1:
                r.is_prefill_chunk = False
        super().update_draft_token_ids(draft_token_ids)

    def schedule(self, *args, **kwargs):
        out = super().schedule(*args, **kwargs)
        plan = {}
        for rid, n in out.num_scheduled_tokens.items():
            req = self.requests[rid]
            plan[rid] = (req.num_computed_tokens - n, n, req.num_tokens)   # computed already advanced by n
        state.put_plan(plan)
        return out

    def update_from_output(self, scheduler_output, model_runner_output):
        progress = state.take_progress()
        ended = []
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests.get(req_id)
            p = progress.get(req_id)
            if request is None or p is None or request.is_finished():
                continue
            # The engine knows the prefix it reused once the request is reserved; set it before super() can consume
            # the stats on this step's first token
            if p.cached is not None and request.prefill_stats is not None:
                request.prefill_stats.set(request.num_prompt_tokens, min(p.cached, request.num_prompt_tokens), 0)
            if p.finished:
                ended.append((req_id, p))
        outputs = super().update_from_output(scheduler_output, model_runner_output)
        for req_id, p in progress.items():
            request = self.requests.get(req_id)
            if request is None or request.is_finished() or p.finished:
                continue
            if request.num_computed_tokens != p.computed:
                kind = "prefill" if p.computed < request.num_prompt_tokens else "decode"
                kind += "_behind" if p.computed < request.num_computed_tokens else "_ahead"
                state.CORRECTIONS[kind] += 1
                request.num_computed_tokens = p.computed
                request.is_prefill_chunk = p.computed < request.num_tokens - 1
        for req_id, p in ended:
            self._end(req_id, p, outputs)
        return outputs

    def _end(self, req_id, p, outputs):
        """Finish a request the engine ended (its last tokens, if any, went out in this step's outputs)."""
        request = self.requests.get(req_id)
        if request is None or request.is_finished():
            return
        status, reason = _FINISH.get(p.finished, _FINISH["error"])
        prefill_stats = request.take_prefill_stats()
        if prefill_stats is not None:
            if p.cached is not None:
                prefill_stats.set(request.num_prompt_tokens, min(p.cached, request.num_prompt_tokens), 0)
            prefill_stats.finalize(0)
        error = None
        if p.finished == "error":
            # a request the engine refused (too long, an image without a vision tower) is a 400 with its message;
            # a failure while it ran is a server error that keeps its message
            error = RequestError(code="validation_error" if p.status == 400 else "engine_error",
                                 message=p.error or "the engine failed this request", retryable=p.status == 503)
        self.finish_requests(req_id, status)
        out = EngineCoreOutput(request_id=req_id, new_token_ids=[], finish_reason=reason,
                               stop_reason=p.error if p.finished == "error" else None, prefill_stats=prefill_stats,
                               error=error)
        eco = outputs.get(request.client_index)
        if eco is None:
            outputs[request.client_index] = EngineCoreOutputs(outputs=[out])
        else:
            eco.outputs.append(out)
