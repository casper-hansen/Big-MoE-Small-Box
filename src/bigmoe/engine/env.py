"""Engine knobs: <BIGMOE_ENGINE_PREFIX>NAME (the backend's own prefix, e.g. GLM53_NV_CPU / DSV41_NV_CPU)."""
import os

PREFIX = os.environ.get("BIGMOE_ENGINE_PREFIX") or "BIGMOE_"


def env(name, default=None):
    return os.environ.get(PREFIX + name, default)


def build_dir(name, default):
    """Kernel build directories are per engine, not per model: BIGMOE_<NAME>_BUILD (set by AutoBox)."""
    return os.environ.get(f"BIGMOE_{name}_BUILD") or default
