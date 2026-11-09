"""Agentic benchmark: small coding tasks solved through the OpenAI chat API with native tool calling, the way agent
clients (DeepSeek Harness, Codex-style CLIs) drive the servers. Stdlib only (it runs in the caller's Python).

Every task is a fresh git repository; the model gets three function tools (bash, read_file, write_file), streams every
turn (tool-call arguments arrive as deltas, parsed by vLLM's DSML / GLM tool parser) and its calls run locally. A turn
without a tool call ends the task, which is then checked by a command independent of what the model claims (the tests
pass, the tests were not touched, the requested files exist).

Per turn it records what an agent waits for: time to the first streamed event (prefill of the turn's new tokens on top
of the conversation's cached prefix), decode speed, prompt / cached / completion tokens from the server's usage
(usage.prompt_tokens_details.cached_tokens: vLLM's prefix-cache hits, i.e. the exllamav3 pages reused), malformed tool
calls. `parallel` runs tasks concurrently, which is what vLLM's scheduler then interleaves (admission, chunked prefill
beside the other agent's decode).

  python -m bigmoe.bench.agent --url http://127.0.0.1:30000 [--tasks ...] [--parallel 2] [--out res.json]
  bigmoe bench-agent deepseek-v4.1-flash                      # starts the server if needed
"""
import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

from ..client import Client, _sse

SYSTEM = ("You are a coding agent working in a git repository, which is the current directory. Use the tools to "
          "inspect files, edit them and run commands (python is available). Work step by step and verify your change "
          "by running the tests or the program. When the task is complete, reply with a one-paragraph summary and no "
          "tool call.")

TOOLS = [
    {"type": "function", "function": {
        "name": "bash", "description": "Run a shell command in the repository and return its exit code and output.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "the command"}},
                       "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "read_file", "description": "Return the contents of a file of the repository.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "relative path"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "Create or overwrite a file of the repository with the given content.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "relative path"},
            "content": {"type": "string", "description": "the complete new file content"}},
            "required": ["path", "content"]}}},
]

TASKS = {
    "fix_stats_bug": {
        "files": {
            "stats.py": '''\
                def mean(xs):
                    return sum(xs) / len(xs)


                def median(xs):
                    """Median of a non-empty list of numbers."""
                    s = sorted(xs)
                    return s[len(s) // 2]


                def variance(xs):
                    """Population variance."""
                    m = mean(xs)
                    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
                ''',
            "test_stats.py": '''\
                import unittest

                from stats import mean, median, variance


                class StatsTest(unittest.TestCase):
                    def test_mean(self):
                        self.assertEqual(mean([1, 2, 3, 4]), 2.5)

                    def test_median_odd(self):
                        self.assertEqual(median([3, 1, 2]), 2)

                    def test_median_even(self):
                        self.assertEqual(median([4, 1, 3, 2]), 2.5)

                    def test_variance(self):
                        self.assertEqual(variance([2, 4, 4, 4, 5, 5, 7, 9]), 4.0)


                if __name__ == "__main__":
                    unittest.main()
                ''',
        },
        "task": "The test suite in this repository fails (`python -m unittest -q`). Fix the bugs in stats.py so that "
                "it passes. Do not modify the tests.",
        "check": "python -m unittest -q && git diff --quiet HEAD -- test_stats.py",
    },
    "wordfreq_cli": {
        "files": {
            "sample.txt": '''\
                the quick brown fox jumps over the lazy dog
                The dog barks; the fox runs.
                A quick brown dog? The end.
                ''',
        },
        "task": "Write a Python script wordfreq.py that takes a file path and an optional `--top N` argument (default 3) "
                "and prints the N most frequent words, one per line as `word count`, most frequent first (ties broken "
                "alphabetically). Words are case-insensitive and consist of letters only (punctuation is ignored). "
                "For example `python wordfreq.py sample.txt --top 2` must print `the 5` and then `dog 3`.",
        "check": "test \"$(python wordfreq.py sample.txt --top 2)\" = \"$(printf 'the 5\\ndog 3')\" && "
                 "test \"$(python wordfreq.py sample.txt | wc -l)\" -eq 3 && "
                 "test \"$(python wordfreq.py sample.txt --top 4 | tail -1)\" = \"fox 2\"",
    },
    "refactor_config": {
        "files": {
            "app/__init__.py": "",
            "app/server.py": '''\
                import json


                def load_settings(path):
                    with open(path) as f:
                        data = json.load(f)
                    host = data.get("host", "127.0.0.1")
                    port = int(data.get("port", 8080))
                    return host, port


                def address(path):
                    host, port = load_settings(path)
                    return f"{host}:{port}"
                ''',
            "app/worker.py": '''\
                import json


                def load_settings(path):
                    with open(path) as f:
                        data = json.load(f)
                    host = data.get("host", "127.0.0.1")
                    port = int(data.get("port", 8080))
                    return host, port


                def worker_url(path):
                    host, port = load_settings(path)
                    return f"http://{host}:{port}/work"
                ''',
            "tests/__init__.py": "",
            "tests/test_app.py": '''\
                import json
                import os
                import tempfile
                import unittest

                from app.server import address
                from app.worker import worker_url


                def write(data):
                    fd, p = tempfile.mkstemp(suffix=".json")
                    with os.fdopen(fd, "w") as f:
                        json.dump(data, f)
                    return p


                class AppTest(unittest.TestCase):
                    def test_defaults(self):
                        p = write({})
                        self.assertEqual(address(p), "127.0.0.1:8080")
                        self.assertEqual(worker_url(p), "http://127.0.0.1:8080/work")

                    def test_values(self):
                        p = write({"host": "example.org", "port": "9000"})
                        self.assertEqual(address(p), "example.org:9000")
                        self.assertEqual(worker_url(p), "http://example.org:9000/work")
                ''',
        },
        "task": "app/server.py and app/worker.py both contain an identical load_settings function. Refactor: move it into "
                "a new module app/config.py and make both modules import it from there (no copy may remain in server.py "
                "or worker.py). The tests must still pass with `python -m unittest -q`. Do not modify the tests.",
        "check": "python -m unittest -q && test -f app/config.py && grep -q 'def load_settings' app/config.py && "
                 "! grep -q 'def load_settings' app/server.py app/worker.py && git diff --quiet HEAD -- tests",
    },
}

OUTPUT_LIMIT = 8000


def make_repo(root: Path, spec: dict) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for rel, text in spec["files"].items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(text))
    sh = lambda c: subprocess.run(c, shell=True, cwd=root, check=True, capture_output=True)
    sh("git init -q && git add -A && git -c user.name=bench -c user.email=bench@localhost commit -qm init")
    return root


def tool_env(bin_dir: Path) -> dict:
    """`python` is the benchmark's own interpreter (stdlib unittest is all the tasks need)."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    for name in ("python", "python3"):
        link = bin_dir / name
        if not link.exists():
            link.symlink_to(sys.executable)
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ.get('PATH', '')}", PYTHONDONTWRITEBYTECODE="1",
               GIT_PAGER="cat", PAGER="cat")
    env.pop("VIRTUAL_ENV", None)
    return env


def _clip(text: str) -> str:
    return text if len(text) <= OUTPUT_LIMIT else text[:OUTPUT_LIMIT // 2] + "\n[... output truncated ...]\n" + \
        text[-OUTPUT_LIMIT // 2:]


def run_tool(name: str, args: dict, cwd: Path, env: dict) -> str:
    def inside(p):
        q = (cwd / p).resolve()
        if cwd.resolve() not in q.parents and q != cwd.resolve():
            raise ValueError(f"{p}: outside the repository")
        return q
    if name == "bash":
        try:
            r = subprocess.run(args["command"], shell=True, cwd=cwd, env=env, capture_output=True, text=True,
                               timeout=60, executable="/bin/bash")
            return _clip(f"exit code {r.returncode}\n{r.stdout}{r.stderr}")
        except subprocess.TimeoutExpired:
            return "error: the command timed out after 60 s"
    if name == "read_file":
        return _clip(inside(args["path"]).read_text())
    if name == "write_file":
        p = inside(args["path"])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(args["content"])
        return f"wrote {len(args['content'])} characters to {args['path']}"
    raise ValueError(f"unknown tool {name!r}")


def chat_turn(client: Client, model: str, messages: list, *, thinking: bool, max_tokens: int) -> dict:
    """One streamed turn: the assistant message (content, tool calls assembled from their deltas) and its timings."""
    body = {"model": model, "messages": messages, "tools": TOOLS, "tool_choice": "auto", "temperature": 0.0,
            "max_tokens": max_tokens, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": bool(thinking)}}
    t0 = time.perf_counter()
    first = last = None
    content, reasoning, calls, usage, finish, n_arg_deltas = "", "", {}, None, None, 0
    for ev in _sse(client.request("/v1/chat/completions", body, stream=True)):
        if ev.get("error"):
            raise RuntimeError(f"server error event: {ev['error']}")
        if ev.get("usage"):
            usage = ev["usage"]
        for ch in ev.get("choices") or []:
            d = ch.get("delta") or {}
            piece = d.get("content") or ""
            think = d.get("reasoning_content") or d.get("reasoning") or ""
            tcs = d.get("tool_calls") or []
            if piece or think or tcs:
                now = time.perf_counter()
                first = first or now
                last = now
            content += piece
            reasoning += think
            for tc in tcs:
                c = calls.setdefault(tc.get("index", 0), {"id": None, "name": "", "arguments": ""})
                c["id"] = tc.get("id") or c["id"]
                fn = tc.get("function") or {}
                c["name"] += fn.get("name") or ""
                if fn.get("arguments"):
                    c["arguments"] += fn["arguments"]
                    n_arg_deltas += 1
            finish = ch.get("finish_reason") or finish
    end = time.perf_counter()
    usage = usage or {}
    n_out = int(usage.get("completion_tokens") or 0)
    return {"content": content, "reasoning": reasoning, "calls": [calls[i] for i in sorted(calls)], "finish": finish,
            "ttft_s": round((first or end) - t0, 2), "turn_s": round(end - t0, 2),
            "decode_tok_s": round((n_out - 1) / (last - first), 2) if first and last and last > first and n_out > 1
            else None,
            "prompt_tokens": int(usage.get("prompt_tokens") or 0), "completion_tokens": n_out,
            "cached_tokens": int(((usage.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0),
            "arg_deltas": n_arg_deltas}


def run_task(client: Client, model: str, name: str, spec: dict, work: Path, *, max_turns=30, max_tokens=4096,
             thinking=False, log=print) -> dict:
    repo = make_repo(work / name, spec)
    env = tool_env(work / ".bin")
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": spec["task"]}]
    turns, invalid, n_calls, status = [], 0, 0, "turn limit"
    t0 = time.time()
    for _ in range(max_turns):
        try:
            t = chat_turn(client, model, messages, thinking=thinking, max_tokens=max_tokens)
        except Exception as e:      # noqa: BLE001 - a failed request ends the task, the benchmark goes on
            status = f"error: {type(e).__name__}: {e}"[:500]
            break
        msg = {"role": "assistant", "content": t["content"] or None}
        if t["calls"]:
            msg["tool_calls"] = [{"id": c["id"] or f"call_{len(turns)}_{i}", "type": "function",
                                  "function": {"name": c["name"], "arguments": c["arguments"]}}
                                 for i, c in enumerate(t["calls"])]
        messages.append(msg)
        results = []
        for tc in msg.get("tool_calls", []):
            n_calls += 1
            fn = tc["function"]
            try:
                args = json.loads(fn["arguments"] or "{}")
                if not isinstance(args, dict):
                    raise ValueError("arguments are not a JSON object")
                out = run_tool(fn["name"], args, repo, env)
            except (ValueError, KeyError, TypeError, OSError) as e:
                invalid += isinstance(e, (ValueError, KeyError, TypeError))
                out = f"error: {type(e).__name__}: {e}"
            results.append({"role": "tool", "tool_call_id": tc["id"], "content": out})
        messages += results
        rec = {k: t[k] for k in ("ttft_s", "turn_s", "decode_tok_s", "prompt_tokens", "cached_tokens",
                                 "completion_tokens", "finish", "arg_deltas")}
        rec["tools"] = [c["name"] for c in t["calls"]]
        turns.append(rec)
        log(f"  {name} turn {len(turns)}: prompt {t['prompt_tokens']} (cached {t['cached_tokens']}), "
            f"ttft {t['ttft_s']} s, out {t['completion_tokens']} at {t['decode_tok_s']} tok/s, "
            f"tools {rec['tools'] or '-'}")
        if not t["calls"]:
            status = "done"
            break
    wall = time.time() - t0
    chk = subprocess.run(spec["check"], shell=True, cwd=repo, env=env, capture_output=True, text=True,
                         executable="/bin/bash")
    later = turns[1:]
    new = [x["prompt_tokens"] - x["cached_tokens"] for x in later]
    return {"status": status, "passed": chk.returncode == 0, "check_output": (chk.stdout + chk.stderr)[-600:],
            "turns": len(turns), "tool_calls": n_calls, "invalid_tool_calls": invalid, "wall_s": round(wall, 1),
            "prompt_tokens": sum(x["prompt_tokens"] for x in turns),
            "cached_tokens": sum(x["cached_tokens"] for x in turns),
            "completion_tokens": sum(x["completion_tokens"] for x in turns),
            "cache_hit": round(sum(x["cached_tokens"] for x in later) / max(1, sum(x["prompt_tokens"] for x in later)),
                               3),
            "mean_new_tokens": round(sum(new) / len(new), 1) if new else None,
            "mean_ttft_s": round(sum(x["ttft_s"] for x in later) / len(later), 2) if later else None,
            "per_turn": turns, "transcript": messages}


def run(client: Client, tasks=None, *, parallel: int = 1, out=None, thinking=False, max_turns=30,
        max_tokens=4096, log=print) -> dict:
    names = list(tasks or TASKS)
    model = (client.models() or [{}])[0].get("id") or "default"
    info = client.server_info()
    res = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "model": model, "parallel": parallel,
           "thinking": thinking, "env": (info.get("server_args") or {}).get("env"), "tasks": {}}
    work = Path(tempfile.mkdtemp(prefix="bigmoe-agent-"))
    t0 = time.time()
    try:
        with concurrent.futures.ThreadPoolExecutor(max(1, parallel)) as ex:
            futs = {ex.submit(run_task, client, model, n, TASKS[n], work, max_turns=max_turns, max_tokens=max_tokens,
                              thinking=thinking, log=log): n for n in names}
            for f in concurrent.futures.as_completed(futs):
                n = futs[f]
                r = res["tasks"][n] = f.result()
                log(f"{n}: {'PASSED' if r['passed'] else 'FAILED'} ({r['status']}), {r['turns']} turns, "
                    f"{r['tool_calls']} tool calls ({r['invalid_tool_calls']} malformed), {r['wall_s']} s, "
                    f"cache hit {r['cache_hit']:.0%} after the first turn, mean ttft {r['mean_ttft_s']} s")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    rs = list(res["tasks"].values())
    res["summary"] = {"passed": sum(r["passed"] for r in rs), "tasks": len(rs), "wall_s": round(time.time() - t0, 1),
                      "turns": sum(r["turns"] for r in rs), "tool_calls": sum(r["tool_calls"] for r in rs),
                      "invalid_tool_calls": sum(r["invalid_tool_calls"] for r in rs),
                      "cache_hit": round(sum(sum(x["cached_tokens"] for x in r["per_turn"][1:]) for r in rs) /
                                         max(1, sum(sum(x["prompt_tokens"] for x in r["per_turn"][1:]) for r in rs)),
                                         3)}
    try:
        res["stats_end"] = client.stats()
    except Exception:        # noqa: BLE001
        pass
    res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    s = res["summary"]
    log(f"TOTAL: {s['passed']}/{s['tasks']} tasks solved, {s['turns']} turns, {s['tool_calls']} tool calls "
        f"({s['invalid_tool_calls']} malformed), cache hit {s['cache_hit']:.0%}, {s['wall_s']} s")
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(res, indent=1))
    return res


def compare_runs(base: dict, now: dict) -> list:
    rows = []
    for k in sorted(set(base.get("tasks", {})) | set(now.get("tasks", {}))):
        a, b = base.get("tasks", {}).get(k), now.get("tasks", {}).get(k)
        if a is None or b is None:
            rows.append(f"  {k}: only in {'this run' if a is None else 'the baseline'}")
            continue
        f = lambda r: (f"{'pass' if r['passed'] else 'FAIL'} {r['turns']:>2} turns {r['wall_s']:>6.1f} s "
                       f"hit {r['cache_hit']:.0%} ttft {r['mean_ttft_s']}")
        rows.append(f"  {k:16s} | {f(a)} -> {f(b)}")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="agentic benchmark of a running box (OpenAI tool calling)")
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--tasks", nargs="*", default=list(TASKS), choices=sorted(TASKS))
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--max-turns", type=int, default=30)
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2, metavar=("BASE", "NOW"))
    a = ap.parse_args(argv)
    if a.compare:
        base, now = (json.load(open(p)) for p in a.compare)
        print("\n".join(compare_runs(base, now)))
        return 0
    res = run(Client(a.url), a.tasks, parallel=a.parallel, out=a.out, thinking=a.thinking, max_turns=a.max_turns,
              log=lambda m: print(m, flush=True))
    return 0 if res["summary"]["passed"] == res["summary"]["tasks"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
