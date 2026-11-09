"""vLLM's scheduler with the engine as the source of truth for progress.

vLLM's Scheduler admits requests (max_num_seqs = the engine's batch size), schedules every running request every step,
checks stop conditions (EOS, stop strings via the detokenizer, max_tokens), builds the outputs and keeps the metrics.
The worker runs one exllamav3 Generator.iterate() per step, which decides on its own how much of each prompt it
prefills (its chunk size, cached prefix pages, recurrent checkpoints, last-page splits) and how many tokens a decode
step yields (speculative decoding: up to 1 + draft). So after each step, before vLLM's own bookkeeping:

- num_computed_tokens is set to what the engine reports (Progress.computed): the prefill position, or all but the last
  sampled token once decoding. A request therefore always has tokens left to schedule (a prefill remainder or the
  last sampled token) and is in every step's batch; vLLM's token budget is set so that it never binds.
- the prefill stats get the prefix the engine reused (Progress.cached), which vLLM reports as
  usage.prompt_tokens_details.cached_tokens.
- a request the engine ended on its own (the cache is full, a failed job) is finished here with finish_reason
  "length" / "error" after its last tokens went out.

vLLM keeps no K/V cache (attention-free stubs): no blocks, no preemption, no vLLM prefix caching.
"""
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason
from vllm.outputs import RequestError
from vllm.v1.request import RequestStatus

from . import state

_FINISH = {"length": (RequestStatus.FINISHED_LENGTH_CAPPED, FinishReason.LENGTH),
           "stop": (RequestStatus.FINISHED_STOPPED, FinishReason.STOP),
           "error": (RequestStatus.FINISHED_ERROR, FinishReason.ERROR)}


class BoxScheduler(Scheduler):

    def has_requests(self) -> bool:
        # an idle engine with disk-tier pages to flush keeps stepping (the worker does one flush slice per empty step)
        return super().has_requests() or state.idle_work_pending()

    def update_from_output(self, scheduler_output, model_runner_output):
        progress = state.take_progress()
        ended = []
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests.get(req_id)
            p = progress.get(req_id)
            if request is None or p is None or request.is_finished():
                continue
            # Set cached-prefix stats even when the engine ended this request on this very token: super() can hit
            # max_tokens on that token and consume the stats before _end() runs.
            if p.cached is not None and request.prefill_stats is not None:
                request.prefill_stats.set(request.num_prompt_tokens, min(p.cached, request.num_prompt_tokens), 0)
            if p.finished:
                ended.append((req_id, p))
                continue
            request.num_computed_tokens = p.computed
        outputs = super().update_from_output(scheduler_output, model_runner_output)
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
