import socket
import sys

import pytest

from bigmoe import AutoBox, ModelBackend, registry
from bigmoe.launcher import LaunchSpec
from bigmoe.steps import Step

SERVER = r'''
import http.server, json, os, sys
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        body = json.dumps({"ok": True, "tag": os.environ.get("TAG")}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)
print("dummy server up", flush=True)
http.server.ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
'''


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def dummy(tmp_path):
    script = tmp_path / "bigmoe_dummy_server.py"
    script.write_text(SERVER)
    ran = []

    class Dummy(ModelBackend):
        name = "dummy-lifecycle"
        aliases = {"dummy": ("me/dummy", "0" * 40)}
        default_memory_gb = 1

        def steps(self, box):
            return [Step("touch", "create a marker", lambda: (box.home.root / "marker").exists(),
                         lambda: (ran.append(1), box.home.root.mkdir(parents=True, exist_ok=True),
                                  (box.home.root / "marker").write_text("x")))]

        def launch_spec(self, box):
            return LaunchSpec(argv=[sys.executable, str(script), str(box.port)], env=dict(box.env), host=box.host,
                              port=box.port, memory_gb=box.memory_gb)

    registry.register_backend(Dummy)
    yield ran
    registry._REGISTRY.pop("dummy-lifecycle")


def test_lifecycle(tmp_path, dummy):
    port = free_port()
    box = AutoBox.from_pretrained("dummy", home=tmp_path / "home", port=port, env={"TAG": "a"})
    assert dummy == [1]
    assert box.prepare() == [{"step": "touch", "status": "done", "seconds": 0.0}]
    try:
        box.start()
        pid = box.status()["pid"]
        assert box.running and box.status()["healthy"] and box.client.request("/health")["tag"] == "a"
        assert "dummy server up" in box.logs()
        AutoBox("dummy", home=tmp_path / "home", port=port, env={"TAG": "a"}).start()  # same config: reused
        assert box.status()["pid"] == pid
        b2 = AutoBox("dummy", home=tmp_path / "home", port=port, env={"TAG": "b"}).start()  # new config: restarted
        assert b2.status()["pid"] != pid and b2.client.request("/health")["tag"] == "b"
    finally:
        assert box.stop() is True
    assert not box.running and box.stop() is False
