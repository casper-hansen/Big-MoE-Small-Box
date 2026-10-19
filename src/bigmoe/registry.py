from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

from .errors import UnknownModelError


@dataclasses.dataclass(frozen=True)
class ModelRef:
    repo_id: str | None = None
    revision: str | None = None  # as requested; the backend pins it to a commit
    local_dir: Path | None = None

    @property
    def label(self) -> str:
        if self.local_dir:
            return str(self.local_dir)
        return f"{self.repo_id}@{self.revision}" if self.revision else str(self.repo_id)


class ModelBackend:
    name: str = ""
    description: str = ""
    repos: tuple = ()
    aliases: dict = {}  # alias -> (repo_id, revision)
    Options = None  # dataclass; AutoBox(**options) fills it
    default_context: int = 131072
    default_memory_gb: float = 52.0
    default_port: int = 30000
    default_served_name: str = "model"

    def __repr__(self):
        return f"<{type(self).__name__} {self.name!r}>"

    def matches(self, ref: ModelRef, config: dict | None) -> bool:
        """`config` is the repo's config.json, fetched only when no backend claims the repo by name."""
        return ref.repo_id is not None and ref.repo_id.lower() in {r.lower() for r in self.repos}

    def resolve_commit(self, ref: ModelRef, token=None) -> str | None:
        if ref.local_dir is not None:
            return None
        from .hub import resolve_commit
        return resolve_commit(ref.repo_id, ref.revision or "main", token=token)

    def make_options(self, **kwargs):
        if self.Options is None:
            if kwargs:
                raise TypeError(f"backend {self.name!r} takes no options (got {sorted(kwargs)})")
            return None
        names = {f.name for f in dataclasses.fields(self.Options)}
        bad = sorted(set(kwargs) - names)
        if bad:
            raise TypeError(f"unknown option(s) {bad} for backend {self.name!r}; valid: {sorted(names)}")
        opts = self.Options(**kwargs)
        if hasattr(opts, "validate"):
            opts.validate()
        return opts

    def runtime_spec(self, box):
        """None: the server runs with the current interpreter."""
        return None

    def kernel_specs(self, box) -> list:
        return []

    def steps(self, box) -> list:
        """Steps must be idempotent: AutoBox runs every one whose done() is False, on every prepare."""
        return []

    def check(self, box) -> list:
        """Raise BoxError if the box cannot run the model; return warnings."""
        return []

    def launch_spec(self, box):
        raise NotImplementedError

    def bench_frame(self, n_tokens: int):
        """(head, tail) raw prompt text around the benchmark document, in the model's chat format."""
        return f"[document set {n_tokens}]\n", "\n\nSummarize the text above."

    def verify_runs(self) -> list:
        """[(label, options, prompt token counts, result file name)]"""
        return []

    def baseline_dir(self) -> Path | None:
        return None


_REGISTRY: dict = {}
_PLUGINS_LOADED = False


def register_backend(backend):
    """Takes a class or an instance; returns its argument, so it works as a class decorator."""
    inst = backend() if isinstance(backend, type) else backend
    if not inst.name:
        raise ValueError(f"{backend!r} has no name")
    _REGISTRY[inst.name] = inst
    return backend


def _load_plugins():
    global _PLUGINS_LOADED
    if _PLUGINS_LOADED:
        return
    _PLUGINS_LOADED = True
    from . import models  # noqa: F401  (registers the built-in backends)
    try:
        from importlib.metadata import entry_points
        for ep in entry_points(group="bigmoe.backends"):
            register_backend(ep.load())
    except Exception as e:   # a broken plugin must not break the built-ins
        from .log import warn
        warn("loading bigmoe.backends entry points failed: %s", e)


def list_backends() -> list:
    _load_plugins()
    return list(_REGISTRY.values())


def get_backend(name: str) -> ModelBackend:
    _load_plugins()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise UnknownModelError(f"no backend {name!r}; registered: {sorted(_REGISTRY)}") from None


def _known() -> str:
    rows = []
    for b in list_backends():
        rows.append(f"  {b.name}: {', '.join(list(b.aliases) + list(b.repos))}")
    return "\n".join(rows)


def resolve(model, revision=None, backend=None, token=None):
    _load_plugins()
    pinned = get_backend(backend) if backend else None
    cands = [pinned] if pinned else list_backends()
    m = str(model)
    if isinstance(model, os.PathLike) or m.startswith(("/", "./", "../", "~")) or os.path.isdir(os.path.expanduser(m)):
        d = Path(m).expanduser().resolve()
        cfg = d / "config.json"
        if not cfg.is_file():
            raise UnknownModelError(f"{d} is not a checkpoint directory (no config.json)")
        config = json.loads(cfg.read_text())
        ref = ModelRef(local_dir=d)
        for b in cands:
            if b.matches(ref, config):
                return b, ref
        raise UnknownModelError(f"no backend recognises the checkpoint at {d} (model_type "
                                f"{config.get('model_type')!r}); backends:\n{_known()}")
    for b in cands:
        for alias, (repo, rev) in b.aliases.items():
            if alias.lower() == m.lower():
                return b, ModelRef(repo_id=repo, revision=revision or rev)
    ref = ModelRef(repo_id=m, revision=revision)
    if pinned:
        return pinned, ref
    for b in cands:
        if b.matches(ref, None):
            return b, ref
    if "/" in m:
        from .hub import fetch_config
        config = fetch_config(m, revision, token=token)
        for b in cands:
            if b.matches(ref, config):
                return b, ref
    raise UnknownModelError(f"no backend for {m!r}; known models:\n{_known()}")
