import functools
import os
import sys
import time
from pathlib import Path

from . import log, registry, system
from .client import Client
from .errors import BoxError
from .launcher import Server
from .paths import Home
from .runtime import ExistingRuntime, Runtime
from .steps import run_steps


class AutoBox:
    def __init__(self, model, *, revision=None, backend=None, context=None, memory_gb=None, host="127.0.0.1",
                 port=None, served_name=None, home=None, runtime="managed", env=None, extra_args=(), token=None,
                 ready_timeout=900, **options):
        """AutoBox(...) only resolves the model and checks the options; nothing is downloaded or built until prepare()
        (or from_pretrained / start). memory_gb is the server's cgroup cap, which its RAM expert tier is sized from.
        runtime: "managed" (a pinned environment built by bigmoe), "current", or the path of an interpreter that already
        has the backend's packages. Other keywords are backend options (`bigmoe models` lists them)."""
        self.token = token
        self.backend, self.ref = registry.resolve(model, revision=revision, backend=backend, token=token)
        self.options = self.backend.make_options(**options)
        self.home = Home(home)
        self.context = int(context or self.backend.default_context)
        self.memory_gb = float(memory_gb or self.backend.default_memory_gb)
        self.host = host
        self.port = int(port or self.backend.default_port)
        self.served_name = served_name or self.backend.default_served_name
        self.runtime_mode = runtime
        self.env = dict(env or {})
        self.extra_args = list(extra_args)
        self.ready_timeout = ready_timeout
        self.commit = self.backend.resolve_commit(self.ref, token=token)
        self.server = Server(self.home, self.port, host)
        self._prepared = False
        self._started_here = False

    @classmethod
    def from_pretrained(cls, model, **kwargs) -> "AutoBox":
        box = cls(model, **kwargs)
        box.prepare()
        return box

    def __repr__(self):
        rev = f"@{self.commit[:12]}" if self.commit else ""
        return (f"AutoBox({self.ref.repo_id or self.ref.local_dir}{rev}, backend={self.backend.name}, "
                f"context={self.context}, port={self.port}, options={self.options})")

    @functools.cached_property
    def runtime(self):
        if self.runtime_mode == "managed":
            spec = self.backend.runtime_spec(self)
            return Runtime(spec, self.home) if spec is not None else ExistingRuntime(sys.executable, self.home)
        if self.runtime_mode == "current":
            return ExistingRuntime(sys.executable, self.home)
        return ExistingRuntime(self.runtime_mode, self.home)

    @functools.cached_property
    def gpu(self):
        gpus = system.gpus()
        if not gpus:
            raise BoxError("no NVIDIA GPU visible (nvidia-smi)")
        return gpus[0]

    @property
    def arch(self) -> str:
        return self.gpu.compute_cap

    def plan(self) -> list:
        return self.backend.steps(self)

    def prepare(self, *, dry_run=False, force=()) -> list:
        report = run_steps(self.plan(), dry_run=dry_run, force=tuple(force))
        if not dry_run:
            self._prepared = True
        return report

    def adopt(self, model_dir=None, store_dir=None, *, move=False, trust_sparse=False) -> dict:
        """Takes over a checkpoint / expert store that already exists on this box (hardlinks, or renames with move=True)
        and verifies it. Checkpoint files whose expert bytes were punched out cannot be hashed: trust_sparse=True accepts
        them, which is only safe if they were verified before they were slimmed."""
        out = {}
        if model_dir is not None:
            ck = self.backend.checkpoint(self) if hasattr(self.backend, "checkpoint") else None
            if ck is None:
                raise BoxError("this model has no downloadable checkpoint to adopt into")
            out["checkpoint"] = ck.adopt(model_dir, move=move, trust_sparse=trust_sparse)
        if store_dir is not None:
            store = self.backend.store(self)
            store.dir.mkdir(parents=True, exist_ok=True)
            for src in (Path(store_dir) / (store.name + ".bin"), Path(store_dir) / (store.name + ".json")):
                dst = store.dir / src.name
                if dst.exists() and os.path.samefile(src, dst):
                    continue
                dst.unlink(missing_ok=True)
                os.replace(src, dst) if move else os.link(src, dst)
            system.check_odirect(store.bin)
            bad = store.verify(threads=8, log=log.info)
            out["store"] = {"bin": str(store.bin), "bad_records": bad}
            if bad:
                raise BoxError(f"adopted store {store.bin} failed verification: {len(bad)} bad records")
        return out

    def verify_store(self, *, sample=None, threads=8) -> list:
        return self.backend.store(self).verify(threads=threads, sample=sample, log=log.info)

    def launch_spec(self):
        if not self._prepared:
            self.prepare()
        return self.backend.launch_spec(self)

    def _check(self):
        for w in self.backend.check(self):
            log.warn("%s", w)

    def start(self, *, restart=False) -> "AutoBox":
        """A server already running on this port with the same configuration is reused (unless restart=True); one with
        a different configuration is restarted."""
        spec = self.launch_spec()
        pid = self.server.pid()
        if pid:
            same = self.server.record().get("fingerprint") == spec.fingerprint()
            if same and not restart and self.server.healthy():
                log.info("already running with this configuration (pid %d) at %s", pid, self.url)
                return self
            log.info("restarting the server on port %d%s", self.port, "" if same else " with the new configuration")
            self.server.stop()
        self._check()
        self.server.start(spec, timeout=self.ready_timeout)
        self._started_here = True
        return self

    def serve(self) -> int:
        spec = self.launch_spec()
        self._check()
        return self.server.run_foreground(spec)

    def stop(self) -> bool:
        self._started_here = False
        return self.server.stop()

    def status(self) -> dict:
        return self.server.status()

    def logs(self, lines=50) -> str:
        return self.server.logs(lines)

    @property
    def running(self) -> bool:
        return self.server.pid() is not None

    def __enter__(self):
        if not self.running:
            self.start()
        return self

    def __exit__(self, *exc):
        if self._started_here:
            self.stop()

    @property
    def url(self) -> str:
        return self.server.url + "/v1"

    @property
    def client(self) -> Client:
        return Client(self.server.url, model=self.served_name)

    def chat(self, messages, **kwargs):
        return self.client.chat(messages, **kwargs)

    def generate(self, prompt: str, **kwargs):
        return self.client.complete(prompt, **kwargs)

    def bench(self, tokens=(1024, 8192, 32768), *, out=None, log=print) -> dict:
        from .bench import longctx
        if not (self.running and self.server.healthy()):
            self.start()
        if out is None:
            out = self.home.results / f"longctx-{self.backend.name}-{time.strftime('%Y%m%d-%H%M%S')}.json"
        return longctx.run(self.client, list(tokens), self.runtime.purelib, frame=self.backend.bench_frame,
                           out=str(out), log=log)
