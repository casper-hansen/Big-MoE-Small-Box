from bigmoe.launcher import LaunchSpec, Server, scope_command
from bigmoe.paths import Home


def test_scope_command():
    assert scope_command("bigmoe-30000", 52) == ["systemd-run", "--user", "--scope", "--quiet", "--collect", "--unit",
                                                 "bigmoe-30000", "-p", "MemoryMax=52G", "-p", "MemorySwapMax=0"]
    assert "MemoryMax=53150220288" in scope_command("u", 49.5)


def test_fingerprint():
    a = LaunchSpec(["py", "-m", "x"], {"A": "1", "B": "2"}, "127.0.0.1", 1)
    b = LaunchSpec(["py", "-m", "x"], {"B": "2", "A": "1"}, "127.0.0.1", 1)
    c = LaunchSpec(["py", "-m", "x"], {"A": "1", "B": "3"}, "127.0.0.1", 1)
    assert a.fingerprint() == b.fingerprint() != c.fingerprint()


def test_status_of_an_idle_port(tmp_path):
    s = Server(Home(tmp_path), 39999)
    st = s.status()
    assert st["running"] is False and st["url"] == "http://127.0.0.1:39999/v1"
    assert s.stop() is False


def test_launch_files_are_copied_at_start(tmp_path, monkeypatch):
    from bigmoe import launcher
    monkeypatch.setattr(launcher.system, "systemd_scope_available", lambda: False)
    src = tmp_path / "src.bin"
    src.write_bytes(b"pristine")
    s = Server(Home(tmp_path / "home"), 39998)
    dst = s.run_dir / "state.bin"
    spec = LaunchSpec(["python", "-c", "pass"], {"X": "1"}, "127.0.0.1", 39998, memory_gb=8,
                      files={str(dst): str(src)})
    cmd, env, rec = s._prepare_launch(spec, None)
    assert dst.read_bytes() == b"pristine" and cmd == ["python", "-c", "pass"] and env["X"] == "1"
    assert rec["memory_capped"] is False and rec["fingerprint"] == spec.fingerprint()
