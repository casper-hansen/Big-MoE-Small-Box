"""Hand-off between BoxScheduler / BoxKVCacheManager and BoxWorker. All of them live in the EngineCore process
(UniProcExecutor runs the worker in-process), and EngineCore calls them in turn on one thread:
schedule -> execute_model -> update_from_output -> take_draft_token_ids."""
import collections

_PROGRESS: dict = {}
_PLAN: dict = {}
_ENGINE = [None]
# where vLLM's count of a request's computed tokens and the engine's differed after a step, by cause
CORRECTIONS = collections.Counter()


def set_engine(engine):
    _ENGINE[0] = engine


def engine():
    return _ENGINE[0]


def put_plan(plan: dict):
    _PLAN.clear()
    _PLAN.update(plan)


def take_plan() -> dict:
    p = dict(_PLAN)
    _PLAN.clear()
    return p


def put_progress(progress: dict):
    _PROGRESS.clear()
    _PROGRESS.update(progress)


def take_progress() -> dict:
    p = dict(_PROGRESS)
    _PROGRESS.clear()
    return p


def idle_work_pending() -> bool:
    e = _ENGINE[0]
    if e is None:
        return False
    try:
        return bool(e.idle_pending)
    except AttributeError:
        return False
