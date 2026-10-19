import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from . import log, system
from .errors import LaunchError


@dataclass
class LaunchSpec:
    argv: list
    env: dict  # on top of the current environment
    host: str
    port: int
    memory_gb: float = None  # cgroup cap, swap capped to 0 too; the engine sizes its RAM tier from it
    unset: tuple = ()
    health_path: str = "/health"
    served_name: str = ""
    record: dict = field(default_factory=dict)  # extra fields for launch.json
    files: dict = field(default_factory=dict)  # {destination: source}, copied right before the start

    def fingerprint(self) -> str:
        """A running server with the same fingerprint can be reused."""
        blob = json.dumps({"argv": self.argv, "env": self.env, "memory_gb": self.memory_gb, "unset": sorted(self.unset)},
                          sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def scope_command(unit: str, memory_gb: float) -> list:
    mem = f"{int(memory_gb)}G" if float(memory_gb).is_integer() else str(int(memory_gb * 1024 ** 3))
    return ["systemd-run", "--user", "--scope", "--quiet", "--collect", "--unit", unit,
            "-p", f"MemoryMax={mem}", "-p", "MemorySwapMax=0"]


def _stop_unit(unit: str):
    for action in ("stop", "reset-failed"):
        try:
            subprocess.run(["systemctl", "--user", action, f"{unit}.scope"], capture_output=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            pass


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:  # a zombie (our own exited child) is not alive
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return True


def _cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return ""


class Server:

    def __init__(self, home, port: int, host: str = "127.0.0.1"):
        self.home, self.port, self.host = home, int(port), host
        self.run_dir = Path(home.run_dir(port))

    def __repr__(self):
        return f"Server(port={self.port}, running={self.pid() is not None})"

    @property
    def url(self) -> str:
        h = "127.0.0.1" if self.host in ("0.0.0.0", "::", "") else self.host
        return f"http://{h}:{self.port}"

    @property
    def unit(self) -> str:
        return f"bigmoe-{self.port}"

    @property
    def pid_file(self) -> Path:
        return self.run_dir / "server.pid"

    @property
    def record_file(self) -> Path:
        return self.run_dir / "launch.json"

    @property
    def log_dir(self) -> Path:
        return self.run_dir / "logs"

    def pid(self):
        try:
            pid = int(self.pid_file.read_text().strip())
        except (OSError, ValueError):
            return None
        return pid if _alive(pid) and "bigmoe" in _cmdline(pid) else None

    def record(self) -> dict:
        try:
            return json.loads(self.record_file.read_text())
        except (OSError, ValueError):
            return {}

    def healthy(self, timeout=3.0, path="/health") -> bool:
        try:
            with urllib.request.urlopen(self.url + path, timeout=timeout) as r:
                return r.status == 200
        except Exception:
            return False

    def status(self) -> dict:
        pid, rec = self.pid(), self.record()
        out = {"port": self.port, "url": self.url + "/v1", "running": pid is not None, "pid": pid,
               "healthy": self.healthy() if pid else False}
        if pid and rec:
            out.update({k: rec.get(k) for k in ("backend", "model", "served_name", "options", "context", "memory_gb",
                                                "started", "log")})
        return out

    def _prepare_launch(self, spec: LaunchSpec, log_path):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        _stop_unit(self.unit)
        for dst, src in spec.files.items():
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
        cmd, capped = list(spec.argv), False
        if spec.memory_gb:
            if system.systemd_scope_available():   # systemd-run execs the command: the scope's pid is the server's
                cmd, capped = scope_command(self.unit, spec.memory_gb) + cmd, True
            else:
                log.warn("systemd-run --user --scope is not available here: the server runs without a memory cap")
        env = dict(os.environ)
        for k in spec.unset:
            env.pop(k, None)
        env.update({k: str(v) for k, v in spec.env.items()})
        rec = dict(spec.record, argv=cmd, env=spec.env, host=spec.host, port=spec.port, memory_gb=spec.memory_gb,
                   memory_capped=capped, served_name=spec.served_name, fingerprint=spec.fingerprint(),
                   started=time.strftime("%Y-%m-%dT%H:%M:%S%z"), log=str(log_path) if log_path else None)
        return cmd, env, rec

    def _write_state(self, pid, rec):
        rec["pid"] = pid
        self.pid_file.write_text(str(pid))
        tmp = self.record_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, indent=1))
        os.replace(tmp, self.record_file)

    def start(self, spec: LaunchSpec, *, timeout: float = 900) -> int:
        """The server keeps running after this process exits."""
        if self.pid():
            raise LaunchError(f"a server is already running on port {self.port} (pid {self.pid()})")
        if self.healthy():
            raise LaunchError(f"port {self.port} is taken by another server (not started by bigmoe)")
        log_path = self.log_dir / f"server-{time.strftime('%Y%m%d-%H%M%S')}.log"
        cmd, env, rec = self._prepare_launch(spec, log_path)
        log.info("starting: %s", " ".join(cmd))
        with open(log_path, "ab") as lf:
            proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT, env=env,
                                    cwd=self.run_dir, start_new_session=True)
        self._write_state(proc.pid, rec)
        latest = self.log_dir / "latest.log"
        latest.unlink(missing_ok=True)
        latest.symlink_to(log_path.name)
        self._rotate_logs()
        log.info("log: %s", log_path)
        self.wait_ready(proc, timeout, spec.health_path)
        return proc.pid

    def wait_ready(self, proc, timeout, health_path="/health"):
        t0 = time.time()
        while not self.healthy(path=health_path):
            if proc.poll() is not None:
                self.pid_file.unlink(missing_ok=True)
                raise LaunchError(f"server exited during start (code {proc.returncode}); last log lines:\n"
                                  f"{self.logs(30)}")
            if time.time() - t0 > timeout:
                raise LaunchError(f"server not ready after {timeout:.0f} s (log: {self.log_dir / 'latest.log'})")
            time.sleep(2)
        log.info("ready after %.0f s at %s/v1", time.time() - t0, self.url)

    def run_foreground(self, spec: LaunchSpec) -> int:
        if self.pid():
            raise LaunchError(f"a server is already running on port {self.port} (pid {self.pid()})")
        cmd, env, rec = self._prepare_launch(spec, None)
        log.info("running: %s", " ".join(cmd))
        proc = subprocess.Popen(cmd, env=env, cwd=self.run_dir)
        self._write_state(proc.pid, rec)
        try:
            return proc.wait()
        except KeyboardInterrupt:
            try:
                return proc.wait(60)
            except subprocess.TimeoutExpired:
                proc.kill()
                return proc.wait()
        finally:
            self.pid_file.unlink(missing_ok=True)
            _stop_unit(self.unit)

    def stop(self, timeout: float = 60) -> bool:
        pid = self.pid()
        if pid:
            log.info("stopping the server on port %d (pid %d)", self.port, pid)
            os.kill(pid, signal.SIGTERM)
            t0 = time.time()
            while _alive(pid) and time.time() - t0 < timeout:
                time.sleep(0.5)
            if _alive(pid):
                log.warn("pid %d did not exit after %.0f s: SIGKILL", pid, timeout)
                os.kill(pid, signal.SIGKILL)
                time.sleep(2)
        _stop_unit(self.unit)
        self.pid_file.unlink(missing_ok=True)
        return pid is not None

    def logs(self, lines: int = 50) -> str:
        p = self.log_dir / "latest.log"
        try:
            with open(p, "rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 256 * 1024))
                return b"\n".join(f.read().splitlines()[-lines:]).decode(errors="replace")
        except OSError:
            return ""

    def follow(self):
        p = self.log_dir / "latest.log"
        with open(p, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 8192))
            try:
                while True:
                    chunk = f.read()
                    if chunk:
                        sys.stdout.write(chunk.decode(errors="replace"))
                        sys.stdout.flush()
                    else:
                        time.sleep(0.5)
            except KeyboardInterrupt:
                pass

    def _rotate_logs(self, keep: int = 10):
        logs = sorted(self.log_dir.glob("server-*.log"))
        for old in logs[:-keep]:
            old.unlink(missing_ok=True)
