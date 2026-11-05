"""DeepSeek-V4.1-Flash-1017 (EXL3 2.25 bpw) on one 24 GB GPU, on exllamav3 1.6.6: routed experts in a VRAM cache, a
pinned RAM tier sized from the memory cap and an NVMe store (store v2: per-layer record sizes, the quant mixes 2.5 bpw
outer layers with 2.0 bpw middle layers); mode "nvme" adds the CPU lane (AVX-512 VBMI trellis decode for K 2 / 2.5),
"nvme-exact" keeps every expert on the GPU (bit-exact with stock exllamav3). The 203 GB Engram n-gram tables stay in the
checkpoint and are streamed from disk by exllamav3 itself (row gathers through the page cache).
"""
import dataclasses
import hashlib
import os
from pathlib import Path

from ... import engine, log, system
from ...bench.longctx import deepseek_frame
from ...cpus import CpuLayout, format_cpus, plan_layout
from ...errors import BoxError, PrepareError
from ...hub import COMMIT_RE, Checkpoint, is_sparse
from ...kernels import KernelSpec, is_built
from ...launcher import LaunchSpec
from ...registry import ModelBackend, register_backend
from ...runtime import RuntimeSpec, SourceBuild
from ...steps import Step
from ...store import ExpertStoreV2, punch_experts
from ...store import exl3_store_v2 as V2
from ...store.slim import expert_ranges

PKG = Path(__file__).resolve().parent
ENGINE = PKG / "engine"
DATA = PKG / "data"
GLM_REQ = PKG.parent / "glm53" / "requirements"   # same pinned package set (torch 2.13.0+cu130, ...)

REPO = "turboderp/DeepSeek-V4.1-Flash-1017-exl3"
VARIANTS = {"2.25bpw": "9183eaaeb6f08dff6ee5b6e34b852911cf6733c9"}   # branch -> the commit the engine was validated on
DEFAULT_VARIANT = "2.25bpw"
IGNORE = ("README.md", ".gitattributes", "quantization_config.json")   # never read by the server
STORE_NAME = "dsv41_flash_exl3_2.25bpw_experts"
EXLLAMAV3 = SourceBuild("exllamav3", "1.6.6", "https://codeload.github.com/turboderp-org/exllamav3/tar.gz/"
                                              "bd1f519172406a9560bd5c90e61000691bb69a07")
MODES = ("nvme", "nvme-exact")
SPECULATION = ("dspark", "none")
# planner cost model "g0,thit,tzc,ca,cb,ctok,push,nvlat,nvdeep,nvone" (ms per expert), see nv2_host.cpp Pol
TUNED_PLANNER = "0.17,0.023,0.36,0.145,0.42,0.12,0.40,0.25,1.36,0.75"


@dataclasses.dataclass
class DSV41Options:
    # "nvme" (default): CPU lane on, near-exact (forced-token mean KL 0.0011 vs nvme-exact); "nvme-exact": every expert
    # on the GPU path (exllamav3's own kernels on the same bytes)
    mode: str = "nvme"
    cache_quant: int = None      # KV-cache bits (-cq); None = FP16
    chunk_size: int = 8192   # staged prefill streams every non-resident expert once per chunk: 8192 -> ~270-360 tok/s
    max_batch_size: int = 4
    vram_expert_slots: int = None
    planner_cost: str = TUNED_PLANNER
    cpus: object = "auto"
    slim: bool = True
    verify_weights: bool = True
    prefix_cache_gb: float = 0.5   # exllamav3's SWA-layer state checkpoints in host RAM (-rcs, GiB)
    # DSpark speculative decoding (the checkpoint's own MTP block drafter, exllamav3 -mtp): drafts up to 5 tokens per
    # step, cut where its confidence head drops below dspark_conf; the target verifies the block in one forward.
    # Measured in mode nvme: +12.5 % (0.5: 75 % of drafted tokens accepted, 2.83 tokens per verify), 0.25 no gain.
    # Costs ~3.4 GB of VRAM (the drafter's experts), i.e. ~350 fewer VRAM expert slots. "none" turns it off.
    speculation: str = "dspark"
    dspark_conf: float = 0.5
    # mode nvme: the first x16_layers MoE layers keep the CPU lane on its bit-exact X16 path (engine/x16.py); 0 = off.
    # Only single-token forwards use it (with DSpark on, verify forwards of those layers run their experts on the GPU)
    x16_layers: int = 0

    def validate(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.speculation not in SPECULATION:
            raise ValueError(f"speculation must be one of {SPECULATION}, got {self.speculation!r}")
        if not 0.0 <= float(self.dspark_conf) <= 1.0:
            raise ValueError(f"dspark_conf must be 0..1, got {self.dspark_conf!r}")
        if self.cache_quant is not None and not 2 <= int(self.cache_quant) <= 8:
            raise ValueError(f"cache_quant must be 2..8 bits or None, got {self.cache_quant!r}")
        if int(self.chunk_size) <= 0 or int(self.max_batch_size) <= 0:
            raise ValueError("chunk_size and max_batch_size must be positive")
        if not 0.0 <= float(self.prefix_cache_gb) <= 64:
            raise ValueError(f"prefix_cache_gb must be 0..64, got {self.prefix_cache_gb!r}")
        engine.check_x16_layers(self.x16_layers)
        if not (self.cpus == "auto" or isinstance(self.cpus, (CpuLayout, dict))):
            raise ValueError("cpus must be 'auto', a CpuLayout or a dict(main=, controller=, lane=, readers=)")


@register_backend
class DSV41(ModelBackend):
    name = "dsv41"
    description = ("DeepSeek-V4.1-Flash (763B MoE incl. 203 GB Engram tables), EXL3 2.25 bpw, on one 24 GB GPU: experts in "
                   "VRAM cache + pinned RAM tier + NVMe store, AVX-512 CPU lane")
    repos = (REPO,)
    aliases = {"deepseek-v4.1-flash": (REPO, DEFAULT_VARIANT), "dsv41": (REPO, DEFAULT_VARIANT)}
    Options = DSV41Options
    default_context = 65536
    default_port = 30000
    default_served_name = "deepseek-v4.1-flash"

    @property
    def default_memory_gb(self) -> float:
        total = system.meminfo().get("MemTotal", 64)
        return 56.0 if total >= 60 else float(max(15, int(total - 8)))

    def matches(self, ref, config):
        if super().matches(ref, None):
            return True
        tc = (config or {}).get("text_config") or {}
        return (config or {}).get("model_type") == "deepseek_v41" and tc.get("n_routed_experts") == 384

    def resolve_commit(self, ref, token=None):
        if ref.local_dir is not None:
            return None
        if ref.repo_id.lower() != REPO.lower():
            return super().resolve_commit(ref, token)
        rev = ref.revision or DEFAULT_VARIANT
        if rev in VARIANTS:
            return VARIANTS[rev]
        if COMMIT_RE.fullmatch(rev):
            if rev not in VARIANTS.values():
                log.warn("%s is not the validated %s commit", rev[:12], DEFAULT_VARIANT)
            return rev
        raise BoxError(f"revision {rev!r} of {REPO} is not supported (validated: {sorted(VARIANTS)})")

    def model_dir(self, box) -> Path:
        return box.ref.local_dir or box.home.model_dir(box.ref.repo_id, box.commit)

    def checkpoint(self, box):
        if box.ref.local_dir is not None:
            return None
        return Checkpoint(box.ref.repo_id, box.commit, self.model_dir(box), ignore=IGNORE, token=box.token)

    def store(self, box) -> ExpertStoreV2:
        if box.ref.local_dir is not None:
            d = box.home.stores / f"local-{hashlib.sha1(str(box.ref.local_dir).encode()).hexdigest()[:12]}"
        else:
            d = box.home.store_dir(box.ref.repo_id, box.commit)
        return ExpertStoreV2(d, STORE_NAME)

    def runtime_spec(self, box):
        return RuntimeSpec(
            name="dsv41", python="3.12", requirements=GLM_REQ / "runtime.txt", build_requirements=GLM_REQ / "build.txt",
            index_url="https://download.pytorch.org/whl/cu130", extra_index_urls=("https://pypi.org/simple",),
            index_strategy="unsafe-best-match", source_builds=(EXLLAMAV3,), cuda_subdir="nvidia/cu13",
            smoke_test="import torch, exllamav3; from exllamav3.ext import exllamav3_ext; "
                       "from exllamav3.architecture.deepseek_v41 import DeepseekV41Config")

    def kernel_specs(self, box) -> list:
        from ... import engine
        return engine.kernel_specs()

    def steps(self, box) -> list:
        o, rt = box.options, box.runtime
        specs, store, ck, mdir = self.kernel_specs(box), self.store(box), self.checkpoint(box), self.model_dir(box)

        def kernels_done():
            if not rt.is_ready():
                return False
            dirs = rt.kernel_dirs(specs, box.arch)
            return all(is_built(s, dirs[s.name]) for s in specs)

        steps = [
            Step("runtime", "create the server's Python environment (torch 2.13 + CUDA 13; exllamav3 1.6.6 built from "
                 "source)", rt.is_ready, lambda: rt.ensure(box.arch), detail=lambda: str(rt.path)),
            Step("kernels", "compile the shared engine's torch extensions (expert pool, nv2 + CPU lane)", kernels_done,
                 lambda: rt.build_kernels(specs, box.arch), detail=lambda: str(box.home.kernels)),
        ]
        if ck is not None:
            def need_full():
                return not store.is_complete()
            steps.append(Step(
                "weights", f"download + verify {box.ref.repo_id} @ {box.commit[:12]} (361 GB incl. 203 GB Engram tables)",
                lambda: ck.is_complete(need_full=need_full()),
                lambda: ck.ensure(need_full=need_full(), verify=o.verify_weights),
                detail=lambda: f"{mdir}: {ck.size_to_download(need_full()) / 1e9:.1f} GB to download"))
        else:
            steps.append(Step("weights", f"use the local checkpoint {mdir}", lambda: (mdir / "config.json").is_file(),
                              lambda: None, detail=lambda: str(mdir)))
        steps.append(Step("store", "pack + verify the NVMe expert store (146 GB for 2.25 bpw)", store.is_complete,
                          lambda: self._build_store(box, store), detail=lambda: str(store.bin)))
        if o.slim and ck is not None:
            steps.append(Step("slim", "punch the expert bytes out of the checkpoint (the store has them)",
                              lambda: self._slimmed(ck, store), lambda: self._slim(box, ck, store),
                              detail=lambda: str(mdir)))
        return steps

    def _build_store(self, box, store):
        mdir = self.model_dir(box)
        sparse = [p.name for p in sorted(mdir.glob("model-*.safetensors")) if is_sparse(p)]
        if sparse:
            raise PrepareError(f"the checkpoint in {mdir} is slimmed ({sparse[:2]}...); it cannot rebuild the store")
        store.dir.mkdir(parents=True, exist_ok=True)
        system.check_odirect(store.dir)
        V2.pack(mdir, store.dir, store.name, naming="deepseek", threads=8, log=log.info)
        if V2.compare(mdir, store.dir, store.name, sample=384, log=log.info):
            raise PrepareError(f"{store.bin}: records differ from the checkpoint")
        if store.verify(threads=8, log=log.info):
            raise PrepareError(f"{store.bin}: sha256 verification failed")

    def _expert_files(self, ck, store) -> list:
        st = ck.state
        if "expert_files" not in st:
            keys = set(store.layer_keys())
            st["expert_files"] = sorted(p.name for p in ck.dir.glob("model-*.safetensors") if expert_ranges(p, keys)[1])
            ck._save()
        return st["expert_files"]

    def _slimmed(self, ck, store) -> bool:
        if not store.is_complete() or not ck.is_complete():
            return False
        return set(self._expert_files(ck, store)) <= set(ck.slimmed())

    def _slim(self, box, ck, store):
        if not store.is_complete():
            raise PrepareError("the expert store is not verified: not slimming the checkpoint")
        freed = punch_experts(ck.dir, store.layer_keys(), log=log.info)
        ck.mark_slimmed(sorted(freed))

    def cpu_layout(self, box) -> CpuLayout:
        c = box.options.cpus
        if isinstance(c, CpuLayout):
            return c
        if isinstance(c, dict):
            return CpuLayout.from_dict(c)
        return plan_layout(system.cpu_cores(), cpu_lane=box.options.mode == "nvme")

    def check(self, box) -> list:
        warnings = []
        g = box.gpu
        if g.memory_total_mib < 23000:
            raise BoxError(f"{g.name} has {g.memory_total_mib} MiB; the dsv41 configuration needs a 24 GB GPU")
        others = system.gpu_compute_apps()
        if others:
            warnings.append(f"other GPU compute processes are running (the VRAM expert cache will be smaller): {others}")
        if box.options.mode == "nvme" and not {"avx512f", "avx512bw", "avx512vl", "avx512vbmi", "avx512_vnni"} <= system.cpu_flags():
            raise BoxError("the dsv41 CPU lane needs AVX-512 VBMI + VNNI (Zen 4, Ice Lake or later): use mode='nvme-exact'")
        if box.memory_gb < 15:
            raise BoxError(f"memory_gb={box.memory_gb} < 15")
        if box.context % 256:
            raise BoxError(f"context must be a multiple of 256 (got {box.context})")
        system.check_odirect(self.store(box).bin)
        return warnings

    def launch_spec(self, box) -> LaunchSpec:
        o, store = box.options, self.store(box)
        # shared engine (bigmoe.engine), knobs under DSV41_: staged prefill stages a whole layer (384 experts) per
        # buffer from 11 GB of elastic VRAM; 3 GB of VRAM stay with torch
        env = {"BIGMOE_ENGINE_PREFIX": "DSV41_", "DSV41_NV": "2", "DSV41_NV_STORE": str(store.bin),
               "DSV41_NV_CPU": "1" if o.mode == "nvme" else "0", "DSV41_NV_POL": o.planner_cost,
               "DSV41_EC_RESERVE_GB": "3.0", "DSV41_EC_STAGE_SLOTS": "384", "DSV41_EC_STAGE_MIN": "512",
               "DSV41_EC_ELASTIC_GB": "11", "DSV41_MAX_RQ_TOKENS": "4096",
               "TRITON_CACHE_DIR": str(box.home.triton), "PYTHONUNBUFFERED": "1"}
        warm = DATA / "stats_dec.json"
        if warm.is_file():
            env["DSV41_EC_WARM"] = str(warm)
        if o.vram_expert_slots is not None:
            env["DSV41_EC_MAX_SLOTS"] = str(int(o.vram_expert_slots))
        lay = self.cpu_layout(box)
        env.update(DSV41_MAIN_CPUS=format_cpus(lay.main), DSV41_NV_CTL_CPU=str(lay.controller),
                   DSV41_NV_READER_CPUS=format_cpus(lay.readers))
        if env["DSV41_NV_CPU"] == "1":
            env["DSV41_NV_CPU_CPUS"] = format_cpus(lay.lane)
            env.update(engine.x16_env("DSV41_", o.x16_layers))
        for spec in self.kernel_specs(box):
            env[spec.env_var] = str(box.runtime.kernel_dirs([spec], box.arch)[spec.name])
        if not system.systemd_scope_available():
            env["DSV41_NV_RAM_GB"] = f"{max(4.0, box.memory_gb - 12.5):.1f}"
        env["PATH"] = f"{box.runtime.python.parent}:{os.environ.get('PATH', '')}"
        env.update({k_: str(v) for k_, v in box.env.items()})
        argv = [str(box.runtime.python), "-m", "bigmoe.models.dsv41.engine.server", "-m", str(self.model_dir(box)),
                "-cs", str(box.context), "--max-batch-size", str(o.max_batch_size), "-chunk_size", str(o.chunk_size),
                "-ccs", "0", "-rcs", f"{float(o.prefix_cache_gb):g}"]
        if o.cache_quant:
            argv += ["-cq", str(o.cache_quant)]
        if o.speculation == "dspark":   # the checkpoint's MTP block drafter (mtp.* tensors, never slimmed: not in the store)
            argv += ["-mtp"]
            env.setdefault("EXL3_DSPARK_CONF", f"{float(o.dspark_conf):g}")   # an explicit env override wins
        argv += ["--host", box.host, "--port", str(box.port), "--served-name", box.served_name, *box.extra_args]
        unset = tuple(v for v in os.environ if v.startswith(("DSV41_", "BIGMOE_", "EXL3_", "EXLLAMAV3_")) and v not in env)
        unset += ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")
        return LaunchSpec(argv=argv, env=env, host=box.host, port=box.port, memory_gb=box.memory_gb, unset=unset,
                          served_name=box.served_name,
                          record={"backend": self.name, "model": box.ref.label, "commit": box.commit,
                                  "context": box.context,
                                  "options": {k_: str(v) for k_, v in dataclasses.asdict(o).items()},
                                  "cpu_layout": lay.describe()})

    def bench_frame(self, n_tokens: int):
        return deepseek_frame(n_tokens)

    def verify_runs(self) -> list:
        return [("mode nvme (decode near-exact)", {"mode": "nvme"}, [1024, 8192, 32768], "longctx_nvme.json"),
                ("mode nvme-exact (decode bit-exact)", {"mode": "nvme-exact"}, [1024, 8192], "longctx_exact.json")]

    def baseline_dir(self) -> Path:
        return PKG / "baselines"
