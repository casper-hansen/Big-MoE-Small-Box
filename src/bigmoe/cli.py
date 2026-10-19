import argparse
import dataclasses
import json
import os
import sys

from . import system
from .errors import BoxError
from .paths import Home

DEFAULT_MODEL = os.environ.get("BIGMOE_MODEL", "glm-5.3-flash")
DEFAULT_PORT = 30000


def _value(v: str):
    try:
        return json.loads(v)
    except ValueError:
        return v


def _add_box_args(p):
    p.add_argument("model", nargs="?", default=DEFAULT_MODEL,
                   help="alias, Hugging Face repo id or local checkpoint directory (default: %(default)s)")
    g = p.add_argument_group("box")
    g.add_argument("--revision", help="branch / tag / commit (default: the backend's validated one)")
    g.add_argument("--backend", help="force a backend")
    g.add_argument("--context", type=int, help="context length in tokens")
    g.add_argument("--memory-gb", type=float, help="memory cap of the server (GiB)")
    g.add_argument("--host", default="127.0.0.1")
    g.add_argument("--port", type=int)
    g.add_argument("--served-name")
    g.add_argument("--home", help="artifact directory (default $BIGMOE_HOME or ~/.cache/bigmoe)")
    g.add_argument("--runtime", default="managed", help="managed | current | /path/to/python")
    g.add_argument("--mode", help="backend option (glm53: nvme | nvme-exact)")
    g.add_argument("--cache-quant", type=int, help="backend option: KV-cache bits (glm53)")
    g.add_argument("--chunk-size", type=int, help="backend option: prefill chunk (glm53)")
    g.add_argument("-o", "--option", action="append", default=[], metavar="KEY=VALUE",
                   help="any backend option (JSON values: -o vram_expert_slots=null)")
    g.add_argument("-e", "--env", action="append", default=[], metavar="KEY=VALUE", help="extra server environment")
    g.add_argument("--arg", action="append", default=[], dest="extra_args", metavar="ARG", help="extra server argument")


def _box(a):
    from .autobox import AutoBox
    opts = {}
    for item in a.option:
        k, _, v = item.partition("=")
        opts[k.strip()] = _value(v)
    for name in ("mode", "cache_quant", "chunk_size"):
        if getattr(a, name, None) is not None:
            opts[name] = getattr(a, name)
    env = dict(kv.split("=", 1) for kv in a.env)
    return AutoBox(a.model, revision=a.revision, backend=a.backend, context=a.context, memory_gb=a.memory_gb,
                   host=a.host, port=a.port, served_name=a.served_name, home=a.home, runtime=a.runtime, env=env,
                   extra_args=a.extra_args, **opts)


def _server(a):
    from .launcher import Server
    return Server(Home(a.home), a.port or DEFAULT_PORT)


def cmd_models(a):
    from .registry import list_backends
    for b in list_backends():
        print(f"{b.name}: {b.description}")
        for alias, (repo, rev) in b.aliases.items():
            print(f"  alias {alias:16s} -> {repo} @ {rev}")
        print(f"  defaults: context {b.default_context}, memory_gb {b.default_memory_gb:g}, port {b.default_port}")
        if b.Options is not None:
            print("  options:")
            for f in dataclasses.fields(b.Options):
                print(f"    {f.name} = {f.default!r}")


def cmd_doctor(a):
    from .cpus import plan_layout
    r = system.report(Home(a.home).root)
    for g in r["gpus"] or [{"name": "none found"}]:
        print(f"GPU      {g.get('name')}  {g.get('memory_total_mib', '?')} MiB (used {g.get('memory_used_mib', '?')}), "
              f"compute {g.get('compute_cap', '?')}")
    if r["gpu_compute_apps"]:
        print(f"         other GPU processes: {r['gpu_compute_apps']}")
    c = r["cpu"]
    print(f"CPU      {c['model']}: {c['physical']} cores / {c['logical']} threads, AVX2+FMA+F16C {c['avx2_fma_f16c']}")
    cores = system.cpu_cores()
    print(f"         SMT groups {c['smt_groups']}")
    print(f"         layout (CPU lane on): {plan_layout(cores, True).describe()}")
    m = r["memory_gib"]
    print(f"RAM      {m.get('MemTotal', 0):.1f} GiB total, {m.get('MemAvailable', 0):.1f} GiB available, "
          f"swap {m.get('SwapTotal', 0):.1f} GiB")
    print(f"system   {r['glibc']}, systemd user scopes (memory cap): {r['systemd_user_scope']}")
    h = r.get("home") or {}
    print(f"home     {h.get('path')} ({h.get('fs')}), {h.get('free_gb')} GB free, O_DIRECT: {h.get('odirect')}")


def cmd_prepare(a):
    box = _box(a)
    rep = box.prepare(dry_run=a.dry_run, force=a.force)
    for r in rep:
        extra = f"  {r['title']}\n{'':12}{r['detail']}" if a.dry_run else f"  {r.get('seconds', 0):.0f} s"
        print(f"{r['step']:9s} {r['status']:7s}{extra}")


def cmd_serve(a):
    sys.exit(_box(a).serve())


def cmd_start(a):
    box = _box(a).start(restart=a.restart)
    print(box.url)


def cmd_stop(a):
    print("stopped" if _server(a).stop() else "not running")


def cmd_status(a):
    print(json.dumps(_server(a).status(), indent=1))


def cmd_logs(a):
    s = _server(a)
    if a.follow:
        s.follow()
    else:
        print(s.logs(a.lines))


def cmd_chat(a):
    from .client import Client
    c = Client(_server(a).url)
    if not c.health():
        raise BoxError(f"no server at {c.url} (bigmoe start)")
    kw = dict(thinking=a.thinking, temperature=a.temperature, max_tokens=a.max_tokens)
    if a.prompt:
        for piece in c.chat(" ".join(a.prompt), stream=True, **kw):
            print(piece, end="", flush=True)
        print()
        return
    print(f"chatting with {c.url} (/reset clears the history, Ctrl-D quits)")
    history = []
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not q:
            continue
        if q == "/reset":
            history = []
            continue
        history.append({"role": "user", "content": q})
        answer = ""
        for piece in c.chat(history, stream=True, **kw):
            answer += piece
            print(piece, end="", flush=True)
        print()
        history.append({"role": "assistant", "content": answer})


def cmd_bench(a):
    box = _box(a)
    box.bench(a.tokens, out=a.out, log=lambda m: print(m, flush=True))


def cmd_verify(a):
    from .bench.verification import verify
    box_kw = {k: v for k, v in dict(revision=a.revision, backend=a.backend, context=a.context, memory_gb=a.memory_gb,
                                    host=a.host, port=a.port, home=a.home, runtime=a.runtime).items() if v is not None}
    verify(a.model, out_dir=a.out, baseline_dir=a.baseline, log=lambda m: print(m, flush=True), **box_kw)


def cmd_verify_store(a):
    bad = _box(a).verify_store(sample=a.sample, threads=a.threads)
    sys.exit(1 if bad else 0)


def cmd_adopt(a):
    out = _box(a).adopt(model_dir=a.model_dir, store_dir=a.store_dir, move=a.move, trust_sparse=a.trust_sparse)
    print(json.dumps({k: {kk: (len(vv) if isinstance(vv, list) else vv) for kk, vv in v.items()} for k, v in out.items()},
                     indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bigmoe", description="Big MoE, Small Box")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("models", help="registered backends, aliases and options")
    p.set_defaults(fn=cmd_models)
    p = sub.add_parser("doctor", help="what this box has")
    p.add_argument("--home")
    p.set_defaults(fn=cmd_doctor)
    p = sub.add_parser("prepare", help="download / convert / build everything (idempotent)")
    _add_box_args(p)
    p.add_argument("--dry-run", action="store_true", help="only show what would run")
    p.add_argument("--force", action="append", default=[], metavar="STEP", help="re-run a step even if done")
    p.set_defaults(fn=cmd_prepare)
    p = sub.add_parser("serve", help="run the server in the foreground")
    _add_box_args(p)
    p.set_defaults(fn=cmd_serve)
    p = sub.add_parser("start", help="start the server in the background")
    _add_box_args(p)
    p.add_argument("--restart", action="store_true", help="restart even if it already runs with this configuration")
    p.set_defaults(fn=cmd_start)
    for name, fn, hlp in (("stop", cmd_stop, "stop the server"), ("status", cmd_status, "server status")):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("--port", type=int)
        p.add_argument("--home")
        p.set_defaults(fn=fn)
    p = sub.add_parser("logs", help="server log")
    p.add_argument("--port", type=int)
    p.add_argument("--home")
    p.add_argument("-f", "--follow", action="store_true")
    p.add_argument("-n", "--lines", type=int, default=50)
    p.set_defaults(fn=cmd_logs)
    p = sub.add_parser("chat", help="ask the running server")
    p.add_argument("prompt", nargs="*")
    p.add_argument("--port", type=int)
    p.add_argument("--home")
    p.add_argument("--thinking", action="store_true", help="let the model think first (reasoning is not printed)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int)
    p.set_defaults(fn=cmd_chat)
    p = sub.add_parser("bench", help="long-context benchmark (starts the server if needed)")
    _add_box_args(p)
    p.add_argument("--tokens", type=int, nargs="+", default=[1024, 8192, 32768])
    p.add_argument("--out", help="result JSON (default: <home>/results/...)")
    p.set_defaults(fn=cmd_bench)
    p = sub.add_parser("verify", help="speed + outputs against the reference results (~15 min for glm53)")
    _add_box_args(p)
    p.add_argument("--out", help="result directory")
    p.add_argument("--baseline", help="reference directory (default: the backend's)")
    p.set_defaults(fn=cmd_verify)
    p = sub.add_parser("verify-store", help="re-check every expert-store record")
    _add_box_args(p)
    p.add_argument("--sample", type=int, help="only N random records (+ first / last of every layer)")
    p.add_argument("--threads", type=int, default=8)
    p.set_defaults(fn=cmd_verify_store)
    p = sub.add_parser("adopt", help="use an existing checkpoint / expert store")
    _add_box_args(p)
    p.add_argument("--model-dir", help="existing local copy of the checkpoint")
    p.add_argument("--store-dir", help="directory with an existing packed expert store")
    p.add_argument("--move", action="store_true", help="rename instead of hardlinking")
    p.add_argument("--trust-sparse", action="store_true",
                   help="accept files whose expert bytes were punched out (verified before slimming)")
    p.set_defaults(fn=cmd_adopt)

    a = ap.parse_args(argv)
    try:
        a.fn(a)
    except BoxError as e:
        print(f"bigmoe: error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
