"""bigmoe.bench.agent against a scripted OpenAI-compatible server (no GPU): streamed tool-call deltas are assembled,
tools run in the task's repository, the check decides, usage and cached tokens are recorded."""
import http.server
import json
import threading

from bigmoe.bench import agent
from bigmoe.client import Client

FIXED = agent.TASKS["fix_stats_bug"]["files"]["stats.py"].replace("s[len(s) // 2]", "(s[(len(s) - 1) // 2] + "
                                                                  "s[len(s) // 2]) / 2").replace("(len(xs) - 1)",
                                                                                                  "len(xs)")


def _events(turn):
    if turn == 0:      # one write_file call, its JSON arguments split over several deltas
        import textwrap
        args = json.dumps({"path": "stats.py", "content": textwrap.dedent(FIXED)})
        pieces = [args[i:i + 40] for i in range(0, len(args), 40)]
        yield {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "c1", "type": "function",
                                                                   "function": {"name": "write_file",
                                                                                "arguments": ""}}]}}]}
        for p in pieces:
            yield {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": p}}]}}]}
        yield {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}
    elif turn == 1:    # a malformed call, then a valid one in the same turn
        yield {"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "id": "c2", "type": "function", "function": {"name": "bash", "arguments": "{not json"}},
            {"index": 1, "id": "c3", "type": "function",
             "function": {"name": "bash", "arguments": json.dumps({"command": "python -m unittest -q"})}}]}}]}
        yield {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}
    else:
        yield {"choices": [{"index": 0, "delta": {"content": "Fixed median and variance."}}]}
        yield {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
    yield {"choices": [], "usage": {"prompt_tokens": 100 + 50 * turn, "completion_tokens": 20,
                                    "prompt_tokens_details": {"cached_tokens": 90 * turn}}}


class Handler(http.server.BaseHTTPRequestHandler):
    bodies = []

    def log_message(self, *a):
        pass

    def _json(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/v1/models":
            return self._json({"data": [{"id": "fake"}]})
        if self.path == "/server_info":
            return self._json({"context_length": 4096, "server_args": {"env": {}}})
        return self._json({})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Handler.bodies.append(body)
        turn = sum(1 for m in body["messages"] if m["role"] == "assistant")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for ev in _events(turn):
            self.wfile.write(b"data: " + json.dumps(ev).encode() + b"\n\n")
        self.wfile.write(b": keepalive\n\ndata: [DONE]\n\n")


def test_agent_bench_over_a_scripted_server():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        res = agent.run(Client(f"http://127.0.0.1:{srv.server_port}"), ["fix_stats_bug"], log=lambda m: None)
    finally:
        srv.shutdown()
    r = res["tasks"]["fix_stats_bug"]
    assert r["passed"] and r["status"] == "done" and r["turns"] == 3
    assert r["tool_calls"] == 3 and r["invalid_tool_calls"] == 1
    assert r["per_turn"][0]["tools"] == ["write_file"] and r["per_turn"][0]["arg_deltas"] > 3
    assert r["per_turn"][1]["cached_tokens"] == 90 and r["cache_hit"] == round(270 / 350, 3)
    assert res["summary"]["passed"] == 1
    # the tool results went back to the model, in the OpenAI shape agent clients use
    last = Handler.bodies[-1]
    tool_msgs = [m for m in last["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["c1", "c2", "c3"]
    assert tool_msgs[1]["content"].startswith("error: JSONDecodeError") and "OK" in tool_msgs[2]["content"]
    assert last["tools"] == agent.TOOLS and last["stream"] is True
