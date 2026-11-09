"""Entry point of the vLLM-fronted servers: python -m bigmoe.vllm.serve --backend {dsv41,glm53} <native server args>.

Takes the native servers' command line (exllamav3's model_init arguments plus --host / --port / --served-name /
--max-batch-size), hands it to the worker through BIGMOE_EXL_ARGV and starts vLLM's OpenAI server with:

  the stub architecture of the backend (models.py, hf_overrides), BoxWorker / BoxScheduler, one in-process worker,
  max_model_len = the exllamav3 cache (-cs), max_num_seqs = --max-batch-size, a token budget that never binds
  (the engine chunks prefills itself), the backend's tool-call and reasoning parsers, the SSE keepalive middleware and
  this package's extra routes (routes.py).

Extra vLLM arguments go after "--" (e.g. -- --enable-log-requests).
"""
import argparse
import json
import os
import sys

from . import ARCHS

PARSERS = {
    # backend -> (tool-call parser, reasoning parser, tokenizer mode)
    "dsv41": ("deepseek_v41", "deepseek_v41", "hf"),
    "glm53": ("glm47", "glm47", "hf"),
}
# the native servers' default: DeepSeek answers in chat mode (thinking on request), GLM-5.3 thinks
CHAT_DEFAULTS = {"dsv41": {"enable_thinking": False}, "glm53": {}}


# GLM-5.3's template has no thinking switch (its generation prompt always opens <think>); the native server closed the
# block itself when a request turned thinking off. The overlay template does the same, and with "enable_thinking" in
# the template vLLM's GLM parser honours the switch (vllm/parser/glm47_moe.py: _glm53_always_thinks).
GLM_THINKING_OFF = ("\n{%- if add_generation_prompt and enable_thinking is defined and enable_thinking is not none "
                    "and not enable_thinking -%}</think>{%- endif -%}\n")


# GLM-5.3's template renders an image block as a "you are unable to process this image" reminder (the checkpoint's
# text-only default); with image input the block becomes the image's placeholder (bigmoe/vllm/mm.py), where the native
# server put the image embedding's alias.
GLM_REMINDER = ('{{- "<reminder>You are unable to process this " ~ media_type ~ " because you don\'t have multi-modal '
                'input ability. Try different methods.</reminder>" }}')
GLM_IMAGE = ("{%- if media_type == 'image' -%}<|begin_of_image|><|image|><|end_of_image|>{%- else -%}" + GLM_REMINDER
             + "{%- endif -%}")


def glm_template(text: str, images: bool) -> str:
    if "enable_thinking" not in text:
        text = text.rstrip("\n") + GLM_THINKING_OFF
    if images:
        if GLM_REMINDER not in text:
            raise RuntimeError("bigmoe: GLM-5.3 chat template without the expected image reminder; cannot enable images")
        text = text.replace(GLM_REMINDER, GLM_IMAGE)
    return text


def tokenizer_overlay(backend: str, model_dir: str, root: str) -> str | None:
    """A tokenizer directory beside the checkpoint with the backend's template changes (GLM: the thinking switch,
    image blocks): symlinks to every small file of the checkpoint, the chat template rewritten. None when nothing
    changes."""
    import hashlib
    from pathlib import Path
    from .. import multimodal
    src = Path(model_dir)
    tpl = src / "chat_template.jinja"
    if backend != "glm53" or not tpl.is_file():
        return None
    text = tpl.read_text()
    new = glm_template(text, multimodal.enabled())
    if new == text:
        return None
    key = hashlib.sha1((str(src.resolve()) + "\0" + new).encode()).hexdigest()[:16]
    d = Path(root) / "vllm-tokenizers" / (src.name + "-" + key)
    d.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.suffix == ".safetensors" or f.name == "chat_template.jinja" or not f.is_file():
            continue
        link = d / f.name
        if link.is_symlink() and link.resolve() != f.resolve():
            link.unlink()
        if not link.is_symlink():
            link.symlink_to(f.resolve())
    out = d / "chat_template.jinja"
    if not out.is_file() or out.read_text() != new:
        out.write_text(new)
    return str(d)


def split_argv(argv):
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1:]
    return argv, []


def vllm_argv(backend: str, a, extra: list, tokenizer: str = None) -> list:
    tool, reasoning, tok_mode = PARSERS[backend]
    ctx = int(a.cache_size)
    batch = int(a.max_batch_size)
    argv = ["serve", a.model_dir,
            "--served-model-name", a.served_name or backend,
            "--host", a.host, "--port", str(a.port),
            "--hf-overrides", json.dumps({"architectures": [ARCHS[backend]]}),
            "--worker-cls", "bigmoe.vllm.worker.BoxWorker",
            "--scheduler-cls", "bigmoe.vllm.scheduler.BoxScheduler",
            "--distributed-executor-backend", "uni",
            "--max-model-len", str(ctx), "--max-num-seqs", str(batch),
            # every running request is in every step; the engine decides how much of a prompt a step prefills
            "--max-num-batched-tokens", str(ctx * batch), "--enable-chunked-prefill",
            "--no-async-scheduling", "--enforce-eager", "--disable-prefix-caching",
            "--tokenizer-mode", tok_mode, "--trust-remote-code",
            # the native servers did not inherit sampling defaults from generation_config.json (GLM pins top_p .95)
            "--generation-config", "vllm",
            "--enable-auto-tool-choice", "--tool-call-parser", tool, "--reasoning-parser", reasoning,
            "--middleware", "bigmoe.vllm.routes.keepalive_middleware",
            "--middleware", "bigmoe.vllm.routes.model_alias_middleware",
            "--middleware", "bigmoe.vllm.routes.context_error_middleware",
            "--enable-prompt-tokens-details",
            "--disable-uvicorn-access-log", "--disable-log-stats",
            "--gpu-memory-utilization", "0.01"]
    if tokenizer:
        argv += ["--tokenizer", tokenizer]
    # content blocks go to the template as blocks: it places images (and joins DeepSeek's text parts) itself
    argv += ["--chat-template-content-format", "openai", "--mm-processor-cache-gb", "0.25", "--skip-mm-profiling"]
    if os.environ.get("BIGMOE_MM", "1") in ("0", "", "false", "False"):
        argv += ["--limit-mm-per-prompt", json.dumps({"image": 0})]
    if CHAT_DEFAULTS[backend]:
        argv += ["--default-chat-template-kwargs", json.dumps(CHAT_DEFAULTS[backend])]
    return argv + list(extra)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ours, extra = split_argv(argv)
    ap = argparse.ArgumentParser(prog="bigmoe.vllm.serve", add_help=True, allow_abbrev=False)
    ap.add_argument("--backend", required=True, choices=sorted(ARCHS))
    ap.add_argument("-m", "--model_dir", required=True)
    ap.add_argument("-cs", "--cache_size", type=int, default=131072)
    ap.add_argument("--max-batch-size", type=int, default=4)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--served-name", default=None)
    # exllamav3 single-dash flags that would otherwise parse as "-m <rest>" (the engine parses them all again)
    ap.add_argument("-mtp", "--mtp", action="store_true")
    ap.add_argument("-ndt", "--num_draft_tokens", type=int, default=None)
    a, _rest = ap.parse_known_args(ours)
    exl = [x for x in ours]
    i = exl.index("--backend")
    del exl[i:i + 2]
    os.environ["BIGMOE_BACKEND"] = a.backend
    os.environ["BIGMOE_EXL_ARGV"] = json.dumps(exl)
    os.environ["BIGMOE_SERVED_NAME"] = a.served_name or a.backend
    # tokens a job keeps free past the prompt besides the answer: the drafter's window (serving.output_budget's reserve)
    ndt = a.num_draft_tokens or 8
    os.environ["BIGMOE_CTX_RESERVE"] = str(ndt + 2)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ["VLLM_PLUGINS"] = "bigmoe,bigmoe_routes"   # endpoint plugins load only when allowlisted
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    os.environ.setdefault("VLLM_DO_NOT_TRACK", "1")
    # The API server / detokenizer is a second process. Keep it off the model's logical CPU but otherwise let the OS
    # place it: pinning it to MAIN_CPUS stalls the model thread, and forcing it onto nv2's controller stalls the
    # planner. BIGMOE_API_CPUS overrides the mask for benchmarking; the EngineCore worker pins the model thread after
    # load, while vLLM's other I/O threads stay unpinned within this mask.
    from .engine import _cpus
    main = os.environ.get(("DSV41_" if a.backend == "dsv41" else "GLM53_") + "MAIN_CPUS")
    available = set(range(os.cpu_count() or 1)) - set(_cpus(main)) if main else set(range(os.cpu_count() or 1))
    cpus = set(_cpus(os.environ["BIGMOE_API_CPUS"])) if os.environ.get("BIGMOE_API_CPUS") else available
    if cpus:
        os.sched_setaffinity(0, cpus)
    from . import register
    register()
    from vllm.entrypoints.cli.main import main as vllm_main
    from ..paths import default_home
    tok = tokenizer_overlay(a.backend, a.model_dir, str(default_home()))
    sys.argv = ["vllm"] + vllm_argv(a.backend, a, extra, tok)
    print(" -- vLLM: " + " ".join(sys.argv[1:]), flush=True)
    return vllm_main()


if __name__ == "__main__":
    sys.exit(main())
