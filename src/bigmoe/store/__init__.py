import json
import os
import time
from pathlib import Path

from .exl3_experts import compare, manifest, pack, paths, verify
from .slim import punch_experts

__all__ = ["ExpertStore", "pack", "verify", "compare", "manifest", "paths", "punch_experts"]


class ExpertStore:
    """<dir>/<name>.bin + .json. A state file remembers the last full verification, so prepare does not re-hash
    100+ GB to know the store is good."""

    def __init__(self, directory, name):
        self.dir = Path(directory)
        self.name = name

    def __repr__(self):
        return f"ExpertStore({str(self.bin)!r})"

    @property
    def bin(self) -> Path:
        return self.dir / (self.name + ".bin")

    @property
    def json(self) -> Path:
        return self.dir / (self.name + ".json")

    @property
    def state_path(self) -> Path:
        return self.dir / (self.name + ".state.json")

    def manifest(self) -> dict:
        return manifest(self.dir, self.name)

    def layer_keys(self) -> list:
        return [L["key"] for L in self.manifest()["layers"]]

    def _state(self) -> dict:
        try:
            return json.loads(self.state_path.read_text())
        except (OSError, ValueError):
            return {}

    def is_complete(self) -> bool:
        try:
            st = self.bin.stat()
            m = self.manifest()
        except (OSError, ValueError):
            return False
        s = self._state()
        return (st.st_size == m.get("total_bytes") and s.get("verified") is True and s.get("size") == st.st_size
                and s.get("mtime_ns") == st.st_mtime_ns)

    def mark_verified(self, how: str):
        st = self.bin.stat()
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"verified": True, "how": how, "size": st.st_size, "mtime_ns": st.st_mtime_ns,
                                   "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=1))
        os.replace(tmp, self.state_path)

    def verify(self, *, threads=8, sample=None, log=print) -> list:
        bad = verify(self.dir, self.name, threads=threads, sample=sample, log=log)
        if not bad and sample is None:
            self.mark_verified("full O_DIRECT sha256 of every record")
        return bad
