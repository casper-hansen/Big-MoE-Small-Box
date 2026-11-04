"""OpenAI-compatible server for DeepSeek-V4.1-Flash over stock exllamav3 1.6.6 (AsyncGenerator), configured through
DSV41_* variables. Expert offload: the shared engine (bigmoe.engine: VRAM cache, RAM tier, NVMe store, planner, CPU
lane), enabled by DSV41_NV_STORE.
No output caps: max_tokens defaults to the remaining context. /generate and /tokenize are SGLang-shaped (benchmark).
Quality tooling: DSV41_KL_CAPTURE=1 records the top-256 logprobs of every forward step (/kl_capture); /kl_teacher runs
one teacher-forced forward over given ids.
"""
import argparse, asyncio, json, os, sys, time, uuid
import torch
from exllamav3 import model_init
from exllamav3.generator import AsyncGenerator, AsyncJob
from exllamav3.generator.sampler import ArgmaxSampler, ComboSampler
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

from bigmoe.kernels import install_prebuilt_loader

install_prebuilt_loader()  # AutoBox prebuilt the kernels into the DSV41_*_BUILD dirs: no toolchain at runtime

app = FastAPI()
G = {"gen": None, "stats": {"requests": 0, "completion_tokens": 0}}
KL = {"on": os.environ.get("DSV41_KL_CAPTURE") == "1", "prompt": None, "gen": [], "steps": []}
BOS, USER, ASSISTANT, SYSTEM = "<｜begin▁of▁sentence｜>", "<｜User｜>", "<｜Assistant｜>", "<｜System｜>"


def load():
    """All model loading lives here (nothing heavy at import time)."""
    global args, model, config, cache, tok, draft_model, draft_cache, EOS, hf_tok, CTX, t_load
    ap = argparse.ArgumentParser()
    model_init.add_args(ap, cache=True, add_draft_model_args=True, default_chunk_size=2048)
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--max-batch-size", type=int, default=8)
    ap.add_argument("--served-name", default="deepseek-v4.1-flash")
    ap.add_argument("--max-q-size", type=int, default=8)
    args = ap.parse_args()

    tiers = None
    if os.environ.get("DSV41_NV_STORE"):
        from bigmoe.engine import tiers
        tiers.install()
    t_load = time.time()
    model, config, cache, tok, draft_model, _draft_config, draft_cache = model_init.init(args)
    if tiers is not None:
        tiers.attach(model)
    t_load = time.time() - t_load
    print(f" -- loaded in {t_load:.1f} s", flush=True)

    EOS = set()
    for e in (getattr(config, "eos_token_id_list", None) or []):
        EOS.add(int(e))
    if tok.eos_token_id is not None:
        EOS.add(int(tok.eos_token_id))
    try:
        gc = json.load(open(os.path.join(args.model_dir, "generation_config.json")))
        for e in (gc.get("eos_token_id") if isinstance(gc.get("eos_token_id"), list) else [gc.get("eos_token_id")]):
            if e is not None:
                EOS.add(int(e))
    except Exception:
        pass
    EOS = sorted(EOS)
    hf_tok = None
    try:
        from transformers import AutoTokenizer
        hf_tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
        if not getattr(hf_tok, "chat_template", None):
            hf_tok = None
    except Exception as e:
        print(f" !! HF tokenizer unavailable ({e}); chat endpoint uses the built-in DeepSeek template", flush=True)
    CTX = args.cache_size


def chat_prompt(messages, enable_thinking=False, tools=None, template_kwargs=None):
    if hf_tok is not None:
        kw = dict(template_kwargs or {})
        kw.pop("enable_thinking", None)
        try:
            return hf_tok.apply_chat_template(messages, tools=tools or None, tokenize=False, add_generation_prompt=True,
                                              enable_thinking=enable_thinking, **kw)
        except Exception as e:
            print(f" !! chat template failed ({e}); built-in template", flush=True)
    p, sys_p = BOS, ""
    for m in messages:
        c = m.get("content") or ""
        if m["role"] == "system":
            sys_p += c
    if sys_p:
        p += SYSTEM + sys_p
    for m in messages:
        c = m.get("content") or ""
        if m["role"] == "user":
            p += USER + c
        elif m["role"] == "assistant":
            p += ASSISTANT + "</think>" + c + "<｜end▁of▁sentence｜>"
    return p + ASSISTANT + ("<think>" if enable_thinking else "</think>")


def make_sampler(sp):
    t = float(sp.get("temperature", 1.0) if sp.get("temperature") is not None else 1.0)
    if t <= 0:
        return ArgmaxSampler()
    return ComboSampler(temperature=t, top_p=float(sp.get("top_p") or 1.0), top_k=int(sp.get("top_k") or 0),
                        min_p=float(sp.get("min_p") or 0.0))


def _kl_install():
    orig = model.forward

    def fwd(*a, **kw):
        out = orig(*a, **kw)
        if KL["on"]:
            ids = kw["input_ids"] if "input_ids" in kw else (a[0] if a else None)
            params = kw["params"] if "params" in kw else (a[1] if len(a) > 1 else None)
            if ids is not None and ids.shape[0] == 1 and isinstance(params, dict) and out is not None:
                cs = params.get("cache_seqlens")
                p0 = int(cs[0]) if cs is not None else -1
                with torch.inference_mode():
                    v, ix = out[0, -min(out.shape[1], 64):].float().log_softmax(-1).topk(256, dim=-1)
                KL["steps"].append({"p0": p0, "n": int(ids.shape[-1]), "ids": [int(x) for x in ids[0, -64:].tolist()],
                                    "lp": [[round(x, 4) for x in row] for row in v.tolist()], "ix": ix.tolist()})
        return out
    model.forward = fwd


@app.on_event("startup")
async def startup():
    G["gen"] = AsyncGenerator(model=model, cache=cache, tokenizer=tok, max_batch_size=args.max_batch_size,
                              max_chunk_size=args.chunk_size, max_q_size=args.max_q_size,
                              draft_model=draft_model, draft_cache=draft_cache,
                              num_draft_tokens=args.num_draft_tokens, dynamic_draft_tokens=args.dynamic_draft,
                              draft_confidence=args.draft_confidence, ngram_match_min=args.ngram_match_min,
                              record_draft_stats=bool(draft_model is not None or args.ngram_match_min))
    _kl_install()


def ids_of(body):
    if body.get("input_ids") is not None:
        return torch.tensor([body["input_ids"]], dtype=torch.long)
    return tok.encode(body["text"], encode_special_tokens=True)


FORCE = {"ids": None}


async def run_job(ids, sp, stop_strings=None):
    """Yields (text so far, completion tokens, finish or None) once per generator step."""
    gen = G["gen"]
    max_new = sp.get("max_new_tokens")
    room = CTX - ids.shape[-1] - (args.num_draft_tokens or 8) - 2
    max_new = room if max_new is None else min(int(max_new), room)
    rq = int(os.environ.get("DSV41_MAX_RQ_TOKENS", "0")) or None
    job = AsyncJob(gen, input_ids=ids, max_new_tokens=max(1, max_new), sampler=make_sampler(sp),
                   stop_conditions=list(EOS) + list(stop_strings or []), decode_special_tokens=False,
                   **({"max_rq_tokens": rq} if rq else {}))
    if KL["on"]:
        KL["prompt"] = [int(x) for x in ids.flatten().tolist()]
        KL["gen"], KL["steps"] = [], []
    if FORCE["ids"] is not None:   # /kl_force: the decode path runs on these output tokens
        job.constrain_output_now(torch.tensor([FORCE["ids"]], dtype=torch.long))
        FORCE["ids"] = None
    G["stats"]["requests"] += 1
    n, text, done = 0, "", False
    try:
        async for r in job:
            if r.get("stage") != "streaming":
                continue
            chunk = r.get("text") or ""
            tids = r.get("token_ids")
            if KL["on"] and tids is not None:
                KL["gen"] += [int(x) for x in tids.flatten().tolist()]
            n += int(tids.numel()) if tids is not None else (1 if chunk else 0)
            text += chunk
            fin = None
            if r.get("eos"):
                n = max(n, int(r.get("new_tokens", 0)))
                if r.get("accepted_draft_tokens") is not None:   # speculative decoding (DSpark / n-gram)
                    sd = G["stats"].setdefault("draft", {"accepted": 0, "rejected": 0, "rounds": 0, "windows": {},
                                                         "accepted_hist": {}})
                    sd["accepted"] += int(r.get("accepted_draft_tokens") or 0)
                    sd["rejected"] += int(r.get("rejected_draft_tokens") or 0)
                    inner = getattr(job, "job", None)
                    for _pos, win, acc in (getattr(inner, "draft_stats", None) or []):
                        sd["rounds"] += 1
                        sd["windows"][str(win)] = sd["windows"].get(str(win), 0) + 1
                        sd["accepted_hist"][str(acc)] = sd["accepted_hist"].get(str(acc), 0) + 1
                reason = r.get("eos_reason")
                fin = {"type": "length" if reason == "max_new_tokens" else "stop", "reason": str(reason)}
                G["stats"]["completion_tokens"] += n
                done = True
            yield text, n, fin
        done = True
    finally:
        if not done:
            try:
                await job.cancel()
            except Exception:
                pass


def teacher_forced(ids, start, topn, want_ids):
    with torch.inference_mode():
        logits = model.forward(ids, {"attn_mode": "flash_attn_nc"}).float()[0]
    lp = torch.log_softmax(logits, -1)
    L = ids.shape[-1]

    def dist(row):
        v, ix = lp[row].topk(topn)
        top = [[float(a), int(b), None] for a, b in zip(v.tolist(), ix.tolist())]
        sel = [[float(lp[row, t]), int(t), None] for t in want_ids] if want_ids else []
        return top, sel
    in_top, in_ids = [], []
    for t in range(start, L):
        if t == 0:
            in_top.append(None); in_ids.append(None); continue
        a, b = dist(t - 1); in_top.append(a); in_ids.append(b)
    o_top, o_ids = dist(L - 1)
    return {"input_top_logprobs": in_top, "input_token_ids_logprobs": in_ids, "output_top_logprobs": [o_top],
            "output_token_ids_logprobs": [o_ids], "next_token": int(lp[L - 1].argmax())}


def sse(obj):
    return "data: " + json.dumps(obj) + "\n\n"


def watch_disconnect(req):
    gone = asyncio.Event()

    async def watch():
        while not gone.is_set():
            if await req.is_disconnected():
                gone.set()
                return
            await asyncio.sleep(0.1)
    return gone, asyncio.create_task(watch())


@app.post("/generate")
async def generate(req: Request):
    body = await req.json()
    sp = body.get("sampling_params") or {}
    ids = ids_of(body)
    if body.get("return_logprob"):
        while len(G["gen"].jobs):
            await asyncio.sleep(0.05)
        r = teacher_forced(ids, int(body.get("logprob_start_len", 0)), int(body.get("top_logprobs_num", 20)),
                           body.get("token_ids_logprob") or [])
        meta = {k: r[k] for k in ("input_top_logprobs", "input_token_ids_logprobs", "output_top_logprobs",
                                  "output_token_ids_logprobs")}
        meta.update({"prompt_tokens": ids.shape[-1], "completion_tokens": 1, "finish_reason": {"type": "length"}})
        return JSONResponse({"text": "", "output_ids": [r["next_token"]], "meta_info": meta})
    P = ids.shape[-1]
    if body.get("stream"):
        async def events():
            g = run_job(ids, sp)
            gone, wt = watch_disconnect(req)
            try:
                async for text, n, fin in g:
                    yield sse({"text": text, "meta_info": {"prompt_tokens": P, "completion_tokens": n, "finish_reason": fin}})
                    if fin is None and gone.is_set():
                        break
                yield "data: [DONE]\n\n"
            finally:
                gone.set(); wt.cancel()
                await g.aclose()
        return StreamingResponse(events(), media_type="text/event-stream")
    text, n, fin = "", 0, None
    async for text, n, fin in run_job(ids, sp):
        pass
    return JSONResponse({"text": text, "meta_info": {"prompt_tokens": P, "completion_tokens": n, "finish_reason": fin}})


@app.post("/tokenize")
async def tokenize(req: Request):
    body = await req.json()
    return JSONResponse({"tokens": tok.encode(body["prompt"], encode_special_tokens=True)[0].tolist()})


def openai_sp(body):
    return {"temperature": body.get("temperature", 1.0), "top_p": body.get("top_p", 1.0), "top_k": body.get("top_k"),
            "min_p": body.get("min_p"), "max_new_tokens": body.get("max_completion_tokens") or body.get("max_tokens")}


def stops_of(body):
    s = body.get("stop")
    return [s] if isinstance(s, str) else list(s or [])


def split_reasoning(text):
    if "</think>" in text:
        r, c = text.split("</think>", 1)
        return r.replace("<think>", "").strip(), c.strip()
    return None, text.strip()


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    kw = dict(body.get("chat_template_kwargs") or {})
    thinking = bool(kw.get("enable_thinking", kw.get("thinking", False)))
    ids = tok.encode(chat_prompt(body["messages"], thinking, None, kw), encode_special_tokens=True)
    sp, stops = openai_sp(body), stops_of(body)
    cid, created, P = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time()), ids.shape[-1]
    if body.get("stream"):
        async def events():
            g = run_job(ids, sp, stops)
            gone, wt = watch_disconnect(req)
            sent, in_think = 0, thinking
            base = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": args.served_name}
            yield sse(dict(base, choices=[{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]))
            n, fin = 0, None
            try:
                async for text, n, fin in g:
                    new = text[sent:]
                    hold = 0
                    if in_think and fin is None:
                        for k in range(min(len("</think>") - 1, len(new)), 0, -1):
                            if "</think>".startswith(new[-k:]):
                                hold = k; break
                    new = new[:len(new) - hold] if hold else new
                    sent += len(new)
                    while new:
                        if in_think:
                            r, sep, rest = new.partition("</think>")
                            if r:
                                yield sse(dict(base, choices=[{"index": 0, "delta": {"reasoning_content": r}, "finish_reason": None}]))
                            if not sep:
                                break
                            in_think, new = False, rest
                        else:
                            yield sse(dict(base, choices=[{"index": 0, "delta": {"content": new}, "finish_reason": None}]))
                            break
                    if fin is None and gone.is_set():
                        break
                usage = {"prompt_tokens": P, "completion_tokens": n, "total_tokens": P + n}
                yield sse(dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": (fin or {}).get("type", "stop")}], usage=usage))
                yield "data: [DONE]\n\n"
            finally:
                gone.set(); wt.cancel()
                await g.aclose()
        return StreamingResponse(events(), media_type="text/event-stream")
    text, n, fin = "", 0, None
    async for text, n, fin in run_job(ids, sp, stops):
        pass
    if thinking:
        reasoning, content = split_reasoning(text)
        if reasoning is None:
            reasoning, content = text.strip(), ""
    else:
        reasoning, content = None, text.strip()
    msg = {"role": "assistant", "content": content, "reasoning_content": reasoning}
    return JSONResponse({"id": cid, "object": "chat.completion", "created": created, "model": args.served_name,
                         "choices": [{"index": 0, "message": msg, "finish_reason": (fin or {}).get("type")}],
                         "usage": {"prompt_tokens": P, "completion_tokens": n, "total_tokens": P + n}})


@app.post("/v1/completions")
async def completions(req: Request):
    body = await req.json()
    prompt = body["prompt"] if isinstance(body["prompt"], str) else body["prompt"][0]
    ids = tok.encode(prompt, encode_special_tokens=True)
    sp, stops = openai_sp(body), stops_of(body)
    cid, created, P = "cmpl-" + uuid.uuid4().hex[:24], int(time.time()), ids.shape[-1]
    base = {"id": cid, "object": "text_completion", "created": created, "model": args.served_name}
    text, n, fin = "", 0, None
    async for text, n, fin in run_job(ids, sp, stops):
        pass
    return JSONResponse(dict(base, choices=[{"index": 0, "text": text, "finish_reason": (fin or {}).get("type")}],
                             usage={"prompt_tokens": P, "completion_tokens": n, "total_tokens": P + n}))


@app.get("/v1/models")
async def models():
    return JSONResponse({"object": "list", "data": [{"id": args.served_name, "object": "model", "owned_by": "local",
                                                     "max_model_len": CTX}]})


@app.get("/server_info")
async def server_info():
    sa = {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool, type(None)))}
    sa.update({"context_length": CTX, "max_running_requests": args.max_batch_size, "max_total_num_tokens": CTX,
               "chunked_prefill_size": args.chunk_size, "kv_cache_dtype": args.cache_quant or "fp16",
               "engine": "exllamav3", "env": {k: v for k, v in os.environ.items() if k.startswith(("DSV41_", "EXL3_"))}})
    return JSONResponse({"context_length": CTX, "server_args": sa, "load_seconds": round(t_load, 1)})


@app.get("/stats")
async def stats():
    st = dict(G["stats"])
    if os.environ.get("DSV41_NV_STORE"):
        from bigmoe.engine import tiers
        st["tiers"] = tiers.summary()
    return JSONResponse(st)


@app.get("/gpu_prof")
async def gpu_prof(secs: float = 3.0, path: str = "", top: int = 40):
    """CUDA kernel profile (torch.profiler / CUPTI) of whatever the GPU runs during the next `secs` seconds: start a
    decode first. Diagnostics only (CUPTI adds launch overhead)."""
    from torch.profiler import profile, ProfilerActivity
    prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
    prof.__enter__()
    try:
        await asyncio.sleep(secs)
    finally:
        prof.__exit__(None, None, None)
    if path:
        prof.export_chrome_trace(path)
    dev = lambda ev, k: float(getattr(ev, k, None) if getattr(ev, k, None) is not None else 0.0)
    rows = []
    for ev in prof.key_averages():
        t = dev(ev, "self_device_time_total") or dev(ev, "self_cuda_time_total")
        if t > 0:
            rows.append({"name": ev.key[:120], "count": int(ev.count), "self_dev_us": round(t, 1),
                         "us_per_call": round(t / max(1, ev.count), 2)})
    rows.sort(key=lambda r: -r["self_dev_us"])
    tot = sum(r["self_dev_us"] for r in rows)
    return JSONResponse({"secs": secs, "dev_us_total": round(tot, 1), "path": path, "top": rows[:top]})


@app.get("/nv2_rate")
async def nv2_rate(mbps: float = 0.0):
    """NVMe emulation (diagnostics): reads complete no faster than a single device of `mbps` MB/s would serve them
    (0 = off, the real drive). Only slower than the real drive can be emulated."""
    from bigmoe.engine import nv2
    nv2.ext().set_read_rate(float(mbps))
    return JSONResponse({"read_rate_mbps": mbps})


@app.get("/cpu")
async def cpu_switch(on: int = 1):
    """Runtime CPU-lane switch (mode nvme): on=0 puts every pick on the GPU path (bit-exact), for same-server A/B."""
    from bigmoe.engine import nv2
    if nv2.NV is None:
        return JSONResponse({"error": "nv2 off"})
    nv2.NV.set_cpu(bool(on))
    return JSONResponse({"cpu": bool(on) and nv2.NV.cpu_on})


@app.get("/nv_verify")
async def nv_verify(n: int = 32):
    from bigmoe.engine import tiers
    return JSONResponse(tiers.verify(n))


@app.get("/kl_capture")
async def kl_capture(on: int = -1):
    if on >= 0:
        KL["on"] = bool(on)
    d = {"prompt": KL["prompt"], "gen": KL["gen"], "steps": KL["steps"], "eos": list(EOS), "on": KL["on"]}
    KL["steps"] = []
    return JSONResponse(d)


@app.post("/kl_force")
async def kl_force(req: Request):
    """Decode `ref` after `prompt` with every output token forced (KL capture on): per-step decode logits of this
    engine configuration on a fixed token path, comparable across configurations."""
    b = await req.json()
    ids = torch.tensor([[int(x) for x in b["prompt"]]], dtype=torch.long)
    FORCE["ids"] = [int(x) for x in b["ref"]]
    on = KL["on"]
    KL["on"] = True
    t0 = time.time()
    n = 0
    try:
        async for _text, n, _fin in run_job(ids, {"temperature": 0, "max_new_tokens": len(b["ref"])}):
            pass
    finally:
        KL["on"] = on
    d = {"n": n, "secs": time.time() - t0, "prompt": KL["prompt"], "gen": KL["gen"], "steps": KL["steps"]}
    KL["steps"] = []
    return JSONResponse(d)


@app.post("/kl_teacher")
async def kl_teacher(req: Request):
    b = await req.json()
    while len(G["gen"].jobs):
        await asyncio.sleep(0.05)
    seq = [int(x) for x in b["ids"]]
    s0, cnt = int(b["start"]), int(b["count"])
    x = torch.tensor([seq], dtype=torch.long)
    with torch.inference_mode():
        logits = model.forward(x, {"attn_mode": "flash_attn_nc"})
        v, ix = logits[0, s0:s0 + cnt].float().log_softmax(-1).topk(256, dim=-1)
    del logits
    torch.cuda.empty_cache()
    return JSONResponse({"start": s0, "count": cnt, "lp": [[round(a, 4) for a in r] for r in v.tolist()],
                         "ix": ix.tolist()})


@app.get("/health")
async def health():
    return JSONResponse({"ok": True})


def main():
    import faulthandler, signal
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    load()
    if os.environ.get("DSV41_MAIN_CPUS"):   # GPU-feeding thread off the CPU-lane / reader cores
        cpus = []
        for part in os.environ["DSV41_MAIN_CPUS"].split(","):
            a, _, b = part.partition("-")
            cpus += list(range(int(a), int(b or a) + 1))
        os.sched_setaffinity(0, set(cpus))
        torch.set_num_threads(int(os.environ.get("DSV41_TORCH_THREADS", "1")))
        print(f" -- main thread on cpus {sorted(os.sched_getaffinity(0))}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
