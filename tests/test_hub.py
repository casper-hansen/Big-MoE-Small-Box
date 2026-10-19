import hashlib

import pytest

from bigmoe.hub import Checkpoint, git_blob_sha1, sha256_file

COMMIT = "a" * 40


def test_git_blob_sha1(tmp_path):
    p = tmp_path / "f"
    p.write_bytes(b"hello\n")
    assert git_blob_sha1(p) == "ce013625030ba8dba906f756967f9e9ca394464a"  # git hash-object


def fake(tmp_path, files):
    ck = Checkpoint("org/repo", COMMIT, tmp_path / "ckpt")
    remote = {}
    for name, data in files.items():
        lfs = name.endswith(".bin")
        remote[name] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest() if lfs else None,
                        "git_sha1": None if lfs else hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()}
    ck.state["remote"] = remote
    return ck


def test_status_verify_and_slim(tmp_path):
    files = {"a.bin": b"x" * 5000, "config.json": b"{}"}
    ck = fake(tmp_path, files)
    assert ck.status() == {"ok": [], "verify": [], "download": ["a.bin", "config.json"]}
    ck.dir.mkdir(parents=True)
    for n, d in files.items():
        (ck.dir / n).write_bytes(d)
    assert ck.status()["verify"] == ["a.bin", "config.json"]
    assert ck.verify(["a.bin", "config.json"]) == []
    assert ck.is_complete()
    ck2 = Checkpoint("org/repo", COMMIT, ck.dir)  # state survives a new object
    assert ck2.is_complete() and sha256_file(ck.dir / "a.bin") == ck2.remote_files()["a.bin"]["sha256"]
    ck2.mark_slimmed(["a.bin"])
    assert ck2.is_complete() and not ck2.is_complete(need_full=True)
    assert ck2.status(need_full=True)["download"] == ["a.bin"]
    (ck.dir / "config.json").write_bytes(b"[]")  # changed after verification: re-verified, and fails
    assert ck2.status()["verify"] == ["config.json"]
    assert ck2.verify(["config.json"]) == ["config.json"]


def test_adopt_hardlinks(tmp_path):
    files = {"a.bin": b"y" * 4096, "config.json": b"{}"}
    src = tmp_path / "old"
    src.mkdir()
    for n, d in files.items():
        (src / n).write_bytes(d)
    ck = fake(tmp_path, files)
    out = ck.adopt(src)
    assert sorted(out["verified"]) == ["a.bin", "config.json"] and out["bad"] == []
    assert (ck.dir / "a.bin").stat().st_ino == (src / "a.bin").stat().st_ino
    assert ck.is_complete()
