import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from bigmoe.client import Client


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _sse(self, events):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for e in events:
            self.wfile.write(b"data: " + json.dumps(e).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")

    def do_GET(self):
        self._json({"ok": True})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/v1/chat/completions":
            assert body["chat_template_kwargs"]["enable_thinking"] is False
            if body.get("stream"):
                return self._sse([{"choices": [{"delta": {"role": "assistant", "content": ""}}]},
                                  {"choices": [{"delta": {"content": "Hel"}}]},
                                  {"choices": [{"delta": {"content": "lo"}}]}])
            return self._json({"choices": [{"message": {"content": "Hello", "reasoning_content": None}}]})
        if self.path == "/v1/completions":
            return self._json({"choices": [{"text": body["prompt"][::-1]}]})
        if self.path == "/tokenize":
            return self._json({"tokens": [ord(c) for c in body["prompt"]]})
        if self.path == "/generate":
            return self._sse([{"text": "a", "meta_info": {"completion_tokens": 1}},
                              {"text": "ab", "meta_info": {"completion_tokens": 2}}])


@pytest.fixture
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_client(server):
    c = Client(server + "/v1")
    assert c.health()
    assert c.chat("hi") == "Hello"
    assert "".join(c.chat([{"role": "user", "content": "hi"}], stream=True)) == "Hello"
    assert c.complete("abc") == "cba"
    assert c.tokenize("AB") == [65, 66]
    assert [e["text"] for e in c.generate(input_ids=[1, 2], stream=True)] == ["a", "ab"]
    assert not Client("http://127.0.0.1:9").health()
