"""GLM-5.3-Flash (EXL3 3.05 bpw) on one 24 GB GPU with glm-flash-lite's engine: routed experts in a VRAM cache, a
pinned RAM tier sized from the memory cap, and an NVMe store; mode "nvme" adds an AVX2 CPU lane (near-exact decode),
"nvme-exact" keeps every expert on the GPU (bit-exact with stock exllamav3).
"""
import dataclasses
import hashlib
import os
from pathlib import Path

from ... import log, system
from ...bench.longctx import glm_frame
from ...cpus import CpuLayout, format_cpus, plan_layout
from ...errors import BoxError, PrepareError
from ...hub import COMMIT_RE, Checkpoint, is_sparse
from ...kernels import KernelSpec, is_built
from ...launcher import LaunchSpec
from ...registry import ModelBackend, register_backend
from ...runtime import RuntimeSpec, SourceBuild
from ...steps import Step
from ...store import ExpertStore, compare, pack, punch_experts
from ...store.slim import expert_ranges

PKG = Path(__file__).resolve().parent
ENGINE = PKG / "engine"
DATA = PKG / "data"

REPO = "turboderp/GLM-5.3-Flash-exl3"
VARIANTS = {"3.05bpw": "332ab457b709b7ba30dd9a448be5de03b80a7ac9"}  # branch -> the commit the engine was validated on
DEFAULT_VARIANT = "3.05bpw"
IGNORE = ("README.md", ".gitattributes", "*.kate-swp", "quantization_config.json")  # never read by the server
STORE_NAME = "glm53_flash_exl3_3.05bpw_experts"
MTP_MODEL_LAYER = 45
EXLLAMAV3 = SourceBuild("exllamav3", "1.5.1", "https://codeload.github.com/turboderp-org/exllamav3/tar.gz/"
                                              "958ec933361b24eb8426ec7222e5b0062a679dcd")
# planner cost model "g0,thit,tzc,ca,cb,ctok,push,nvlat,nvdeep,nvone" (ms). Upstream's cb = 0.165 ms per CPU-lane expert
# is calibrated for 22 EPYC cores; with 5 Ryzen lane cores an expert costs ~0.65 ms and the stock model overloads the
# lane. Tuned on the RTX 3090 box (2026-10-19, C1 chat): cb 0.165 -> 9.5, 0.30 -> 10.1, 0.40 -> 10.1, 0.65 -> 9.7 tok/s.
TUNED_PLANNER = "0.17,0.023,0.38,0.145,0.40,0.10,0.40,0.12,0.395,0.75"
MODES = ("nvme", "nvme-exact")


@dataclasses.dataclass
class GLM53Options:
    mode: str = "nvme"
    cache_quant: int = None  # KV-cache bits (-cq); None = FP16
    chunk_size: int = 8192
    max_batch_size: int = 8
    vram_expert_slots: int = None  # None = all free VRAM (measured best); an int caps the expert cache
    planner_cost: str = TUNED_PLANNER
    cpus: object = "auto"  # or a CpuLayout, or {"main", "controller", "lane", "readers"}
    slim: bool = True
    verify_weights: bool = True

    def validate(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.cache_quant is not None and not 2 <= int(self.cache_quant) <= 8:
            raise ValueError(f"cache_quant must be 2..8 bits or None, got {self.cache_quant!r}")
        if int(self.chunk_size) <= 0 or int(self.max_batch_size) <= 0:
            raise ValueError("chunk_size and max_batch_size must be positive")
        if not (self.cpus == "auto" or isinstance(self.cpus, (CpuLayout, dict))):
            raise ValueError("cpus must be 'auto', a CpuLayout or a dict(main=, controller=, lane=, readers=)")


@register_backend
class GLM53(ModelBackend):
    name = "glm53"
    description = ("GLM-5.3-Flash (320B MoE, 18B active), EXL3 3.05 bpw, on one 24 GB GPU: experts in VRAM cache + "
                   "pinned RAM tier + NVMe store, AVX2 CPU lane (glm-flash-lite)")
    repos = (REPO,)
    aliases = {"glm-5.3-flash": (REPO, DEFAULT_VARIANT), "glm53": (REPO, DEFAULT_VARIANT)}
    Options = GLM53Options
    default_context = 131072
    default_port = 30000
    default_served_name = "glm-5.3-flash"

    @property
    def default_memory_gb(self) -> float:
        """56 GiB on a 60 GiB+ box (measured best: 4.8 GiB host headroom at peak); smaller hosts MemTotal - 8 GiB."""
        total = system.meminfo().get("MemTotal", 64)
        return 56.0 if total >= 60 else float(max(15, int(total - 8)))

    def matches(self, ref, config):
        if super().matches(ref, None):
            return True
        q = (config or {}).get("quantization_config") or {}
        return (config or {}).get("model_type") == "glm5_next" and q.get("quant_method") == "exl3" \
            and abs(float(q.get("bits") or 0) - 3.05) < 1e-6

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
                log.warn("%s is not the validated %s commit; the engine may reject its layout", rev[:12], DEFAULT_VARIANT)
            return rev
        raise BoxError(f"revision {rev!r} of {REPO} is not supported: the glm53 engine is built for the 3-bit experts "
                       f"of the {DEFAULT_VARIANT} quant (validated: {sorted(VARIANTS)})")

    def model_dir(self, box) -> Path:
        return box.ref.local_dir or box.home.model_dir(box.ref.repo_id, box.commit)

    def checkpoint(self, box):
        if box.ref.local_dir is not None:
            return None
        return Checkpoint(box.ref.repo_id, box.commit, self.model_dir(box), ignore=IGNORE, token=box.token)

    def store(self, box) -> ExpertStore:
        if box.ref.local_dir is not None:
            d = box.home.stores / f"local-{hashlib.sha1(str(box.ref.local_dir).encode()).hexdigest()[:12]}"
        else:
            d = box.home.store_dir(box.ref.repo_id, box.commit)
        return ExpertStore(d, STORE_NAME)

    def runtime_spec(self, box):
        return RuntimeSpec(
            name="glm53", python="3.12", requirements=PKG / "requirements" / "runtime.txt",
            build_requirements=PKG / "requirements" / "build.txt",
            index_url="https://download.pytorch.org/whl/cu130", extra_index_urls=("https://pypi.org/simple",),
            index_strategy="unsafe-best-match", source_builds=(EXLLAMAV3,), cuda_subdir="nvidia/cu13",
            smoke_test="import torch, exllamav3; from exllamav3.ext import exllamav3_ext")

    def kernel_specs(self, box) -> list:
        e = "bigmoe.models.glm53.engine."
        nv2_src = tuple(sorted((PKG / "kernels" / "nv2").iterdir()))
        return [KernelSpec("glm53_ec", "GLM53_EC_BUILD", e + "expert_cache", "_ext", (ENGINE / "expert_cache.py",)),
                KernelSpec("glm53_k101", "GLM53_K_BUILD", e + "k_hcfuse", "ext", (ENGINE / "k_hcfuse.py",)),
                KernelSpec("glm53_nv", "GLM53_NV_BUILD", e + "nv_tier", "_ext", (ENGINE / "nv_tier.py",)),
                KernelSpec("glm53_nv2", "GLM53_NV2_BUILD", e + "nv2", "ext", (ENGINE / "nv2.py",) + nv2_src)]

    def steps(self, box) -> list:
        o, rt = box.options, box.runtime
        specs, store, ck, mdir = self.kernel_specs(box), self.store(box), self.checkpoint(box), self.model_dir(box)

        def kernels_done():
            if not rt.is_ready():
                return False
            dirs = rt.kernel_dirs(specs, box.arch)
            return all(is_built(s, dirs[s.name]) for s in specs)

        steps = [
            Step("runtime", "create the server's Python environment (torch 2.13 + CUDA 13; exllamav3 1.5.1 built from "
                 "source)", rt.is_ready, lambda: rt.ensure(box.arch), detail=lambda: str(rt.path)),
            Step("kernels", "compile the engine's 4 torch extensions (expert cache, hyper-connections, NVMe tier, "
                 "nv2 + CPU lane)", kernels_done, lambda: rt.build_kernels(specs, box.arch),
                 detail=lambda: str(box.home.kernels)),
        ]
        if ck is not None:
            def need_full():
                return not store.is_complete()
            steps.append(Step(
                "weights", f"download + verify {box.ref.repo_id} @ {box.commit[:12]}",
                lambda: ck.is_complete(need_full=need_full()),
                lambda: ck.ensure(need_full=need_full(), verify=o.verify_weights),
                detail=lambda: f"{mdir}: {ck.size_to_download(need_full()) / 1e9:.1f} GB to download"))
        else:
            def local_ok():
                if not (mdir / "config.json").is_file():
                    raise PrepareError(f"no checkpoint at {mdir}")
            steps.append(Step("weights", f"use the local checkpoint {mdir}", lambda: (mdir / "config.json").is_file(),
                              local_ok, detail=lambda: str(mdir)))
        steps.append(Step("store", "pack + verify the NVMe expert store (117 GB for 3.05 bpw)", store.is_complete,
                          lambda: self._build_store(box, store), detail=lambda: str(store.bin)))
        if o.slim and ck is not None:
            steps.append(Step("slim", "punch the expert bytes out of the checkpoint (the store has them)",
                              lambda: self._slimmed(ck, store), lambda: self._slim(box, ck, store),
                              detail=lambda: str(mdir)))
        return steps

    def _build_store(self, box, store: ExpertStore):
        mdir = self.model_dir(box)
        sparse = [p.name for p in sorted(mdir.glob("*.safetensors")) if is_sparse(p)]
        if sparse:
            raise PrepareError(f"the checkpoint in {mdir} is slimmed (expert bytes punched out of {sparse[:2]}...); it "
                               "cannot rebuild the store")
        store.dir.mkdir(parents=True, exist_ok=True)
        system.check_odirect(store.dir)
        pack(mdir, store.dir, store.name, threads=8, include_mtp=True, mtp_model_layer=MTP_MODEL_LAYER,
             fmt="glm53-expert-store-v1", log=log.info)
        if compare(mdir, store.dir, store.name, sample=300, log=log.info):
            raise PrepareError(f"{store.bin}: records differ from the checkpoint")
        if store.verify(threads=8, log=log.info):
            raise PrepareError(f"{store.bin}: sha256 verification failed")

    def _expert_files(self, ck, store) -> list:
        """Cached in the checkpoint state: finding them reads every shard's header."""
        st = ck.state
        if "expert_files" not in st:
            keys = set(store.layer_keys())
            st["expert_files"] = sorted(p.name for p in ck.dir.glob("*.safetensors") if expert_ranges(p, keys)[1])
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
            raise BoxError(f"{g.name} has {g.memory_total_mib} MiB; the glm53 configuration needs a 24 GB GPU")
        others = system.gpu_compute_apps()
        if others:
            warnings.append(f"other GPU compute processes are running (the VRAM expert cache will be smaller): {others}")
        if box.options.mode == "nvme" and not {"avx2", "fma", "f16c"} <= system.cpu_flags():
            raise BoxError("this CPU lacks AVX2 / FMA / F16C for the CPU lane: use mode='nvme-exact'")
        if box.memory_gb < 15:
            raise BoxError(f"memory_gb={box.memory_gb} < 15")
        avail = system.meminfo().get("MemAvailable", 0)
        if avail < box.memory_gb:
            warnings.append(f"MemAvailable {avail:.0f} GiB < memory_gb {box.memory_gb:g} GiB")
        if box.context % 256:
            raise BoxError(f"context must be a multiple of 256 (got {box.context})")
        system.check_odirect(self.store(box).bin)
        return warnings

    def launch_spec(self, box) -> LaunchSpec:
        """Same environment and arguments as the measured serve.sh (glm-flash-lite's docker/entrypoint.sh, modes nvme /
        nvme-exact); tests/test_glm53_launch.py checks it against the reference runs."""
        o, store = box.options, self.store(box)
        tune = box.server.run_dir / "coop_autotune_1gpu.bin"  # exllamav3 appends to its tune cache: fresh copy per start
        env = {}
        if o.mode == "nvme-exact":
            reserve, k, mode_args = "1.0", "0", []
            env.update(GLM53_NV="2", GLM53_NV_CPU="0", GLM53_K_HCFUSE="1")
        else:
            reserve, k, mode_args = "1.5", "1", ["-ambs", "4"]
            env.update(GLM53_NV="2", GLM53_NV_CPU="1", GLM53_K_FTSPLIT="0", GLM53_K_OVL="0", GLM53_NV_VRING="24",
                       GLM53_NV_PREFETCH="1", GLM53_MAX_RQ_TOKENS="4096", GLM53_NV_CPU_KERN="1", GLM53_LA="1",
                       GLM53_NV_PUBFAST="1", GLM53_NV_PFSIDE="1", GLM53_LA_BTTRIM="1", GLM53_NV_POL=o.planner_cost)
        env["GLM53_EC_STAGE_MIN"] = "512"
        env["GLM53_NV_STORE"] = str(store.bin)
        for var in ("GLM53_K_HCFUSE", "GLM53_K_FTSPLIT", "GLM53_K_OVL"):
            env.setdefault(var, k)
        stats = str(DATA / "stats_own_dec.json")
        env.update(GLM53_ZC_STATS=stats, GLM53_EC="1", GLM53_EC_WARM=stats, GLM53_EC_RESERVE_GB=reserve,
                   GLM53_EC_STAGE_GB="2.6", GLM53_EC_ELASTIC_GB="10", GLM53_CPU_TIER="0",
                   GLM53_TRITON_PIN=str(DATA / "triton_pin_exact.json"), EXLLAMAV3_TUNE_CACHE=str(tune),
                   TRITON_CACHE_DIR=str(box.home.triton), PYTHONUNBUFFERED="1")
        if o.vram_expert_slots is not None:
            env["GLM53_EC_MAX_SLOTS"] = str(int(o.vram_expert_slots))
        lay = self.cpu_layout(box)
        env.update(GLM53_MAIN_CPUS=format_cpus(lay.main), GLM53_NV_CTL_CPU=str(lay.controller),
                   GLM53_NV_READER_CPUS=format_cpus(lay.readers))
        if env["GLM53_NV_CPU"] == "1":
            env["GLM53_NV_CPU_CPUS"] = format_cpus(lay.lane)
        small = []
        if box.memory_gb < 24:  # upstream's small-host budget
            env.update(GLM53_NV_MARGIN_GB="1.5", GLM53_NV_PF_RING="128", GLM53_NV_THREADS="48", GLM53_NV_PIECE_KB="1024")
            small = ["-rcs", "1"]
        for spec in self.kernel_specs(box):
            env[spec.env_var] = str(box.runtime.kernel_dirs([spec], box.arch)[spec.name])
        if not system.systemd_scope_available():  # no cgroup cap to size the RAM tier from: give it explicitly
            env["GLM53_NV_RAM_GB"] = f"{max(4.0, box.memory_gb - 12.5):.1f}"
        env["PATH"] = f"{box.runtime.python.parent}:{os.environ.get('PATH', '')}"
        env.update({k_: str(v) for k_, v in box.env.items()})
        argv = [str(box.runtime.python), "-m", "bigmoe.models.glm53.engine.server", "-m", str(self.model_dir(box)),
                "-cs", str(box.context), "--max-batch-size", str(o.max_batch_size), "-chunk_size", str(o.chunk_size),
                *mode_args, *small]
        if o.cache_quant:
            argv += ["-cq", str(o.cache_quant)]
        argv += ["--host", box.host, "--port", str(box.port), "--served-name", box.served_name, *box.extra_args]
        # ambient engine variables would silently change the configuration: only the ones set here reach the server
        unset = tuple(v for v in os.environ if v.startswith(("GLM53_", "EXL3_", "EXLLAMAV3_")) and v not in env)
        unset += ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")
        return LaunchSpec(argv=argv, env=env, host=box.host, port=box.port, memory_gb=box.memory_gb, unset=unset,
                          served_name=box.served_name, files={str(tune): str(DATA / "coop_autotune_1gpu.bin")},
                          record={"backend": self.name, "model": box.ref.label, "commit": box.commit,
                                  "context": box.context, "options": {k_: str(v) for k_, v in
                                                                      dataclasses.asdict(o).items()},
                                  "cpu_layout": lay.describe()})

    def bench_frame(self, n_tokens: int):
        return glm_frame(n_tokens)

    def verify_runs(self) -> list:
        return [("mode nvme (decode near-exact)", {"mode": "nvme"}, [1024, 8192, 32768, 120000], "longctx_nvme.json"),
                ("mode nvme-exact (decode bit-exact)", {"mode": "nvme-exact"}, [1024, 8192], "longctx_exact.json")]

    def baseline_dir(self) -> Path:
        return PKG / "baselines"
