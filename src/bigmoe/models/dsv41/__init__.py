"""DeepSeek-V4.1-Flash-1017 (EXL3 2.25 bpw) on one 24 GB GPU: routed experts in a VRAM cache, a pinned RAM tier and an
NVMe store (store v2: the quant mixes 2.5 bpw outer layers with 2.0 bpw middle layers, so record sizes differ per
layer). The 203 GB Engram n-gram tables stay in the checkpoint; exllamav3 streams them from disk.
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
from ...kernels import is_built
from ...launcher import LaunchSpec
from ...registry import ModelBackend, register_backend
from ...steps import Step
from ..prefix_cache import cpu_tier_gb
from ..runtime import description as runtime_description, runtime_spec
from ..multimodal_env import multimodal_env
from ...store import ExpertStoreV2, punch_experts
from ...store import exl3_store_v2 as V2
from ...store.slim import expert_ranges

PKG = Path(__file__).resolve().parent
DATA = PKG / "data"

REPO = "turboderp/DeepSeek-V4.1-Flash-1017-exl3"
VARIANTS = {"2.25bpw": "9183eaaeb6f08dff6ee5b6e34b852911cf6733c9"}   # branch -> the commit the engine is validated on
DEFAULT_VARIANT = "2.25bpw"
IGNORE = ("README.md", ".gitattributes", "quantization_config.json")   # never read by the server
STORE_NAME = "dsv41_flash_exl3_2.25bpw_experts"
MODES = ("nvme", "nvme-exact")
SPECULATION = ("dspark", "none")
KV_BYTES_PER_TOKEN = 3209   # FP16 K/V plus the DSpark drafter's, per token of a cached page
PLANNER_COST = "0.17,0.023,0.36,0.145,0.42,0.12,0.40,0.25,1.36,0.75"   # ms per expert, see Pol in nv2_host.cpp
MAX_BATCH_SIZE = 4
RC_INTERVAL = 256     # mode nvme: tokens between recurrent checkpoints while decoding (exllamav3's own: 2048)
DSPARK_CONF = 0.5     # a DSpark draft is cut where the drafter's confidence falls below this


@dataclasses.dataclass
class DSV41Options:
    # "nvme": RAM-resident experts may be computed on the CPU lane (near-exact). "nvme-exact": every expert on the GPU,
    # exllamav3's own arithmetic on the same bytes.
    mode: str = "nvme"
    cache_quant: int = None        # K/V cache bits (2..8); None = FP16
    chunk_size: int = 8192         # prefill chunk, tokens
    cpus: object = "auto"          # or a CpuLayout, or {"main", "controller", "lane", "readers"}
    slim: bool = True              # punch the expert bytes out of the checkpoint once the store is verified
    verify_weights: bool = True
    prefix_cache_gb: float = 0.5   # host RAM for recurrent-state checkpoints (exllamav3 -rcs)
    # Prefix-cache tiers behind the GPU page cache: cpu_cache_gb of pinned RAM (taken from the RAM expert tier), and
    # disk_cache, a directory whose pages and checkpoints survive restarts. A prefix returns from disk through the RAM
    # tier, so with disk_cache the RAM tier defaults to the whole context (models/prefix_cache.py).
    cpu_cache_gb: float = None
    disk_cache: str = None
    speculation: str = "dspark"    # the checkpoint's own block drafter (up to 5 tokens per step); "none" turns it off
    x16_layers: int = 0            # mode nvme: leading MoE layers whose CPU lane is bit-exact (docs/x16.md)
    # RAM added to memory_gb for vLLM's API server process. Above 0 the RAM expert tier has a fixed size; at 0 it is
    # sized from the cgroup when the engine starts.
    frontend_extra_ram_gb: float = 4.5
    multimodal: bool = True        # image input
    image_tokens: int = None       # cap on tokens per image; None = the model's own budget

    def validate(self):
        if not 0.0 <= float(self.frontend_extra_ram_gb) <= 8.0:
            raise ValueError(f"frontend_extra_ram_gb must be 0..8, got {self.frontend_extra_ram_gb!r}")
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.speculation not in SPECULATION:
            raise ValueError(f"speculation must be one of {SPECULATION}, got {self.speculation!r}")
        if self.cache_quant is not None and not 2 <= int(self.cache_quant) <= 8:
            raise ValueError(f"cache_quant must be 2..8 bits or None, got {self.cache_quant!r}")
        if int(self.chunk_size) <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0.0 <= float(self.prefix_cache_gb) <= 64:
            raise ValueError(f"prefix_cache_gb must be 0..64, got {self.prefix_cache_gb!r}")
        if self.cpu_cache_gb is not None and not 0.0 <= float(self.cpu_cache_gb) <= 64:
            raise ValueError(f"cpu_cache_gb must be 0..64, got {self.cpu_cache_gb!r}")
        if self.disk_cache and self.cpu_cache_gb is not None and not float(self.cpu_cache_gb) > 0:
            raise ValueError("disk_cache needs cpu_cache_gb > 0 (exllamav3's disk tier sits behind the CPU tier)")
        engine.check_x16_layers(self.x16_layers)
        if self.image_tokens is not None and not 16 <= int(self.image_tokens) <= 16384:
            raise ValueError(f"image_tokens must be 16..16384 or None, got {self.image_tokens!r}")
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
    default_context = 131072
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
        return runtime_spec("dsv41", "import torch, exllamav3; from exllamav3.ext import exllamav3_ext; "
                                     "from exllamav3.architecture.deepseek_v41 import DeepseekV41Config")

    def kernel_specs(self, box) -> list:
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
            Step("runtime", f"create the server's Python environment ({runtime_description()})", rt.is_ready,
                 lambda: rt.ensure(box.arch), detail=lambda: str(rt.path)),
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
        nvme = o.mode == "nvme"
        # staged prefill holds a whole layer (384 experts) per staging buffer, in up to 11 GB of VRAM lent by the
        # expert cache; 3 GB of VRAM stay with torch
        env = {"BIGMOE_ENGINE_PREFIX": "DSV41_", "DSV41_NV_STORE": str(store.bin), "DSV41_NV_CPU": "1" if nvme else "0",
               "DSV41_NV_POL": PLANNER_COST, "DSV41_EC_RESERVE_GB": "3.0", "DSV41_EC_STAGE_SLOTS": "384",
               "DSV41_EC_STAGE_MIN": "512", "DSV41_EC_ELASTIC_GB": "11", "DSV41_MAX_RQ_TOKENS": "4096",
               "DSV41_EC_WARM": str(DATA / "stats_dec.json"), "TRITON_CACHE_DIR": str(box.home.triton),
               "PYTHONUNBUFFERED": "1"}
        lay = self.cpu_layout(box)
        env.update(DSV41_MAIN_CPUS=format_cpus(lay.main), DSV41_NV_CTL_CPU=str(lay.controller),
                   DSV41_NV_READER_CPUS=format_cpus(lay.readers))
        if nvme:
            env["DSV41_NV_CPU_CPUS"] = format_cpus(lay.lane)
            env["DSV41_EC_WARM_PREFILL"] = "1"     # engine/warm_prefill.py
            env["DSV41_NV_SLRU"] = "80"            # RAM tier: segmented LRU, 80 % of the slots for reused experts
            env["DSV41_RC_INTERVAL"] = str(RC_INTERVAL)
            env.update(engine.x16_env("DSV41_", o.x16_layers))
        for spec in self.kernel_specs(box):
            env[spec.env_var] = str(box.runtime.kernel_dirs([spec], box.arch)[spec.name])
        if float(o.frontend_extra_ram_gb) > 0:
            # a tier sized from the cgroup's free memory at start varies by several GiB between identical starts
            env["DSV41_NV_RAM_GB"] = "46.0" if nvme else "47.5"
        elif not system.systemd_scope_available():
            env["DSV41_NV_RAM_GB"] = f"{max(4.0, box.memory_gb - 12.5):.1f}"
        env.update(multimodal_env(box, self.name))
        env["PATH"] = f"{box.runtime.python.parent}:{os.environ.get('PATH', '')}"
        env.update({k_: str(v) for k_, v in box.env.items()})
        argv = [str(box.runtime.python), "-m", "bigmoe.vllm.serve", "--backend", "dsv41", "-m", str(self.model_dir(box)),
                "-cs", str(box.context), "--max-batch-size", str(MAX_BATCH_SIZE), "-chunk_size", str(o.chunk_size),
                "-ccs", f"{cpu_tier_gb(o.cpu_cache_gb, o.disk_cache, box.context, KV_BYTES_PER_TOKEN):g}",
                "-rcs", f"{float(o.prefix_cache_gb):g}"]
        if o.disk_cache:
            argv += ["-dcd", str(Path(o.disk_cache).expanduser())]
        if o.cache_quant:
            argv += ["-cq", str(o.cache_quant)]
        if o.speculation == "dspark":
            argv += ["-mtp"]
            env.setdefault("EXL3_DSPARK_CONF", f"{DSPARK_CONF:g}")   # an explicit env override wins
        argv += ["--host", box.host, "--port", str(box.port), "--served-name", box.served_name, *box.extra_args]
        # ambient engine variables would silently change the configuration: only the ones set here reach the server
        unset = tuple(v for v in os.environ if v.startswith(("DSV41_", "BIGMOE_", "EXL3_", "EXLLAMAV3_")) and v not in env)
        unset += ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")
        return LaunchSpec(argv=argv, env=env, host=box.host, port=box.port,
                          memory_gb=box.memory_gb + float(o.frontend_extra_ram_gb), unset=unset,
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
