import concurrent.futures as cf
import fnmatch
import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path

from . import log
from .errors import PrepareError

COMMIT_RE = re.compile(r"[0-9a-f]{40}")


def resolve_commit(repo_id: str, revision: str = "main", token=None) -> str:
    if revision and COMMIT_RE.fullmatch(revision):
        return revision
    from huggingface_hub import HfApi
    return HfApi(token=token).model_info(repo_id, revision=revision).sha


def fetch_config(repo_id: str, revision=None, token=None) -> dict:
    from huggingface_hub import hf_hub_download
    return json.loads(Path(hf_hub_download(repo_id, "config.json", revision=revision, token=token)).read_text())


def sha256_file(path, bufsize=16 << 20) -> str:
    h = hashlib.sha256()
    buf = bytearray(bufsize)
    mv = memoryview(buf)
    with open(path, "rb", buffering=0) as f:
        while True:
            n = f.readinto(mv)
            if not n:
                break
            h.update(mv[:n])
    return h.hexdigest()


def git_blob_sha1(path) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def is_sparse(path, threshold=0.9) -> bool:
    st = os.stat(path)
    return st.st_size > 0 and st.st_blocks * 512 < threshold * st.st_size


def _lfs_sha256(sibling):
    lfs = getattr(sibling, "lfs", None)
    if lfs is None:
        return None
    if isinstance(lfs, dict):
        return lfs.get("sha256")
    return getattr(lfs, "sha256", None)


class Checkpoint:
    """A checkpoint (repo @ commit) in a local directory. Verification results are kept in .bigmoe/checkpoint.json
    with each file's size and mtime: re-checks are free, and a file whose expert bytes were punched out (it can no
    longer be hashed) still counts as verified, unless a full copy is asked for (need_full=True)."""
    STATE = ".bigmoe/checkpoint.json"

    def __init__(self, repo_id: str, commit: str, directory, *, ignore=(), token=None):
        if not COMMIT_RE.fullmatch(commit or ""):
            raise ValueError(f"Checkpoint needs a 40-hex commit, got {commit!r}")
        self.repo_id, self.commit, self.dir = repo_id, commit, Path(directory)
        self.ignore, self.token = tuple(ignore), token
        self._state = None

    def __repr__(self):
        return f"Checkpoint({self.repo_id}@{self.commit[:12]}, {str(self.dir)!r})"

    @property
    def state(self) -> dict:
        if self._state is None:
            p = self.dir / self.STATE
            st = json.loads(p.read_text()) if p.is_file() else {}
            if st.get("repo_id") != self.repo_id or st.get("commit") != self.commit:
                st = {}
            st.setdefault("repo_id", self.repo_id)
            st.setdefault("commit", self.commit)
            st.setdefault("files", {})
            self._state = st
        return self._state

    def _save(self):
        p = self.dir / self.STATE
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1, sort_keys=True))
        os.replace(tmp, p)

    def _record(self, name, **kw):
        st = (self.dir / name).stat()
        rec = dict(self.state["files"].get(name) or {})
        rec.update(size=st.st_size, mtime_ns=st.st_mtime_ns, **kw)
        self.state["files"][name] = rec

    def _ignored(self, name) -> bool:
        return any(fnmatch.fnmatch(name, p) for p in self.ignore)

    def remote_files(self) -> dict:
        st = self.state
        if "remote" not in st:
            from huggingface_hub import HfApi
            info = HfApi(token=self.token).model_info(self.repo_id, revision=self.commit, files_metadata=True)
            remote = {}
            for s in info.siblings:
                sha = _lfs_sha256(s)
                remote[s.rfilename] = {"size": s.size, "sha256": sha, "git_sha1": None if sha else s.blob_id}
            st["remote"] = remote
            self.dir.mkdir(parents=True, exist_ok=True)
            self._save()
        return {k: v for k, v in st["remote"].items() if not self._ignored(k)}

    def _status(self, name, meta, need_full) -> str:
        p = self.dir / name
        try:
            st = p.stat()
        except FileNotFoundError:
            return "download"
        if st.st_size != meta["size"]:
            return "download"
        rec = self.state["files"].get(name)
        if rec and rec.get("verified") and rec.get("mtime_ns") == st.st_mtime_ns and rec.get("size") == st.st_size:
            return "download" if need_full and rec.get("slimmed") else "ok"
        return "download" if is_sparse(p) else "verify"

    def status(self, need_full=False) -> dict:
        out = {"ok": [], "verify": [], "download": []}
        for name, meta in sorted(self.remote_files().items()):
            out[self._status(name, meta, need_full)].append(name)
        return out

    def is_complete(self, need_full=False) -> bool:
        s = self.status(need_full)
        return not s["verify"] and not s["download"]

    def slimmed(self) -> list:
        return sorted(n for n, r in self.state["files"].items() if r.get("slimmed"))

    def size_to_download(self, need_full=False) -> int:
        remote = self.remote_files()
        return sum(remote[n]["size"] for n in self.status(need_full)["download"])

    def ensure(self, need_full=False, verify=True, max_workers=8, threads=8):
        s = self.status(need_full)
        bad = self.verify(s["verify"], threads) if verify else self._trust(s["verify"])
        todo = sorted(set(s["download"]) | set(bad))
        if todo:
            remote = self.remote_files()
            log.info("downloading %d file(s), %.1f GB, of %s @ %s into %s", len(todo),
                     sum(remote[n]["size"] for n in todo) / 1e9, self.repo_id, self.commit[:12], self.dir)
            for n in todo:
                self._forget(n)
            self._save()
            t0 = time.time()
            self._download(todo, max_workers)
            log.info("downloaded in %.0f s; verifying", time.time() - t0)
            bad = self.verify(todo, threads) if verify else self._trust(todo)
            if bad:
                raise PrepareError(f"{self.repo_id}@{self.commit[:12]}: hash mismatch after download: {bad}")
        if self.is_complete(need_full):
            shutil.rmtree(self.dir / ".cache" / "huggingface", ignore_errors=True)
            try:
                (self.dir / ".cache").rmdir()
            except OSError:
                pass

    def _forget(self, name):
        # huggingface_hub skips a file whose local-dir metadata matches the commit, whatever its content
        (self.dir / name).unlink(missing_ok=True)
        meta = self.dir / ".cache" / "huggingface" / "download"
        for suffix in (".metadata", ".lock"):
            (meta / (name + suffix)).unlink(missing_ok=True)
        self.state["files"].pop(name, None)

    def _download(self, names, max_workers):
        from huggingface_hub import snapshot_download
        self.dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(self.repo_id, revision=self.commit, local_dir=str(self.dir), allow_patterns=list(names),
                          max_workers=max_workers, token=self.token)

    def verify(self, names, threads=8) -> list:
        remote = self.remote_files()

        def one(n):
            p, meta = self.dir / n, remote[n]
            if p.stat().st_size != meta["size"]:
                return n, False
            if meta.get("sha256"):
                return n, sha256_file(p) == meta["sha256"]
            if meta.get("git_sha1"):
                return n, git_blob_sha1(p) == meta["git_sha1"]
            return n, True

        bad = []
        if names:
            t0 = time.time()
            with cf.ThreadPoolExecutor(max(1, threads)) as ex:
                for n, ok in ex.map(one, names):
                    if ok:
                        self._record(n, verified=True, slimmed=False, trusted=False)
                    else:
                        bad.append(n)
            self._save()
            log.info("verified %d file(s) in %.0f s%s", len(names), time.time() - t0, f", BAD: {bad}" if bad else "")
        return bad

    def _trust(self, names) -> list:
        for n in names:
            self._record(n, verified=True, slimmed=False, trusted=True)
        self._save()
        return []

    def mark_slimmed(self, names):
        for n in names:
            self._record(n, slimmed=True)
        self._save()

    def adopt(self, src_dir, *, move=False, trust_sparse=False, threads=8) -> dict:
        """Hardlink (or move) the files of an existing copy into this directory and verify them. Sparse files cannot be
        hashed: they are accepted as slimmed only with trust_sparse=True."""
        src = Path(src_dir).expanduser()
        remote = self.remote_files()
        self.dir.mkdir(parents=True, exist_ok=True)
        adopted, full, sparse = [], [], []
        for name, meta in sorted(remote.items()):
            s, d = src / name, self.dir / name
            if not s.is_file() or s.stat().st_size != meta["size"]:
                continue
            d.parent.mkdir(parents=True, exist_ok=True)
            if not (d.exists() and os.path.samefile(s, d)):
                d.unlink(missing_ok=True)
                if move:
                    os.replace(s, d)
                else:
                    try:
                        os.link(s, d)
                    except OSError:
                        shutil.copy2(s, d)
            adopted.append(name)
            (sparse if is_sparse(d) else full).append(name)
        bad = self.verify(full, threads)
        for n in bad:
            self._forget(n)
        if sparse:
            if not trust_sparse:
                self._save()
                raise PrepareError(f"{len(sparse)} adopted file(s) are sparse (expert bytes punched out) and cannot be "
                                   f"verified: {sparse[:3]}...; pass trust_sparse=True if they were verified before "
                                   "they were slimmed")
            for n in sparse:
                self._record(n, verified=True, slimmed=True, trusted=True)
        self._save()
        return {"adopted": adopted, "verified": [n for n in full if n not in bad], "bad": bad, "sparse": sparse}
