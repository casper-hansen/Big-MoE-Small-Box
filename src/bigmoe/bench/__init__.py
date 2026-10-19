def __getattr__(name):   # lazy: `python -m bigmoe.bench.longctx` must not find longctx already imported
    if name in ("run", "compare_runs"):
        from . import longctx
        return getattr(longctx, name)
    if name == "verify":
        from .verification import verify
        return verify
    raise AttributeError(name)
