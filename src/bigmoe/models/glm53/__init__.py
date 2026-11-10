"""GLM-5.3-Flash (EXL3 3.05 bpw) on one 24 GB GPU: routed experts in a VRAM cache, a pinned RAM tier and an NVMe store
(store v1: one record size).
"""
import dataclasses
import hashlib
import os
from pathlib import Path

from ... import engine, log, system
from ...bench.longctx import glm_frame
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
from ...store import ExpertStore, compare, pack, punch_experts
from ...store.slim import expert_ranges

PKG = Path(__file__).resolve().parent
DATA = PKG / "data"

REPO = "turboderp/GLM-5.3-Flash-exl3"
VARIANTS = {"3.05bpw": "332ab457b709b7ba30dd9a448be5de03b80a7ac9"}  # branch -> the commit the engine is validated on
DEFAULT_VARIANT = "3.05bpw"
IGNORE = ("README.md", ".gitattributes", "*.kate-swp", "quantization_config.json")  # never read by the server
STORE_NAME = "glm53_flash_exl3_3.05bpw_experts"
MTP_MODEL_LAYER = 45
MODES = ("nvme", "nvme-exact")
SPECULATION = ("none", "ngram")
KV_BYTES_PER_TOKEN = 17609   # FP16 K/V per token of a cached page
# Planner cost model, ms per expert (Pol in nv2_host.cpp), tuned for 5 Ryzen 7600X lane cores and an SN850X. With
# AVX-512 VBMI + VNNI the CPU lane decodes the trellis faster, so its per-expert cost (the fifth value) is lower.
PLANNER_COST = "0.17,0.023,0.38,0.145,0.40,0.10,0.40,0.25,1.45,0.75"
PLANNER_COST_VBMI = "0.17,0.023,0.38,0.145,0.29,0.10,0.40,0.25,1.45,0.75"
VBMI_FLAGS = {"avx512f", "avx512bw", "avx512vl", "avx512vbmi", "avx512_vnni"}
MAX_BATCH_SIZE = 8
NGRAM_MIN = 2        # speculation="ngram": shortest context match that yields a draft
DRAFT_TOKENS = 5     # speculation="ngram": drafted tokens verified per step


@dataclasses.dataclass
class GLM53Options:
    # "nvme": RAM-resident experts may be computed on the CPU lane (near-exact). "nvme-exact": every expert on the GPU,
    # exllamav3's own arithmetic on the same bytes.
    mode: str = "nvme"
    cache_quant: int = None        # K/V cache bits (2..8); None = FP16
    chunk_size: int = 8192         # prefill chunk, tokens
    cpus: object = "auto"          # or a CpuLayout, or {"main", "controller", "lane", "readers"}
    slim: bool = True              # punch the expert bytes out of the checkpoint once the store is verified
    verify_weights: bool = True
    speculation: str = "none"      # "ngram": drafts copied from the context, verified by the model
    # host RAM for recurrent-state checkpoints (exllamav3 -rcs, ~143 MB each); what it does not take is RAM expert tier
    prefix_cache_gb: float = 1.0
    # Prefix-cache tiers behind the GPU page cache: cpu_cache_gb of pinned RAM (taken from the RAM expert tier), and
    # disk_cache, a directory whose pages and checkpoints survive restarts. A prefix returns from disk through the RAM
    # tier, so with disk_cache the RAM tier defaults to the whole context (models/prefix_cache.py).
    cpu_cache_gb: float = None
    disk_cache: str = None
    x16_layers: int = 0            # mode nvme: leading MoE layers whose CPU lane is bit-exact (docs/x16.md)
    # RAM added to memory_gb for vLLM's API server process, which would otherwise come out of the RAM expert tier
    frontend_extra_ram_gb: float = 1.5
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
        engine.check_x16_layers(self.x16_layers)
        if not 0.25 <= float(self.prefix_cache_gb) <= 64:
            raise ValueError(f"prefix_cache_gb must be 0.25..64, got {self.prefix_cache_gb!r}")
        if self.cpu_cache_gb is not None and not 0.0 <= float(self.cpu_cache_gb) <= 64:
            raise ValueError(f"cpu_cache_gb must be 0..64, got {self.cpu_cache_gb!r}")
        if self.disk_cache and self.cpu_cache_gb is not None and not float(self.cpu_cache_gb) > 0:
            raise ValueError("disk_cache needs cpu_cache_gb > 0 (exllamav3's disk tier sits behind the CPU tier)")
        if self.image_tokens is not None and not 16 <= int(self.image_tokens) <= 16384:
            raise ValueError(f"image_tokens must be 16..16384 or None, got {self.image_tokens!r}")
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
        """56 GiB on a host with 60 GiB or more, otherwise MemTotal - 8 GiB."""
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
        return runtime_spec("glm53", "import torch, exllamav3; from exllamav3.ext import exllamav3_ext")

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
            Step("kernels", "compile the torch extensions (shared engine: expert pool, nv2 + CPU lane; GLM: "
                 "hyper-connections)", kernels_done, lambda: rt.build_kernels(specs, box.arch),
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
        o, store = box.options, self.store(box)
        nvme = o.mode == "nvme"
        tune = box.server.run_dir / "coop_autotune_1gpu.bin"  # exllamav3 appends to its tune cache: fresh copy per start
        stats = str(DATA / "stats_own_dec.json")
        env = {"BIGMOE_ENGINE_PREFIX": "GLM53_", "GLM53_NV_STORE": str(store.bin), "GLM53_NV_CPU": "1" if nvme else "0",
               "GLM53_EC_WARM": stats, "GLM53_EC_RESERVE_GB": "1.5" if nvme else "1.0", "GLM53_EC_STAGE_GB": "2.6",
               "GLM53_EC_STAGE_MIN": "512", "GLM53_EC_ELASTIC_GB": "10", "EXLLAMAV3_TUNE_CACHE": str(tune),
               "TRITON_CACHE_DIR": str(box.home.triton), "PYTHONUNBUFFERED": "1",
               "EXL3_HC_FOLD": "1"}   # exllamav3's fused hyper-connection sites: same results, faster decode
        lay = self.cpu_layout(box)
        env.update(GLM53_MAIN_CPUS=format_cpus(lay.main), GLM53_NV_CTL_CPU=str(lay.controller),
                   GLM53_NV_READER_CPUS=format_cpus(lay.readers))
        if nvme:
            planner = PLANNER_COST_VBMI if VBMI_FLAGS <= system.cpu_flags() else PLANNER_COST
            env.update(GLM53_NV_CPU_CPUS=format_cpus(lay.lane), GLM53_NV_POL=planner, GLM53_NV_VRING="24",
                       GLM53_NV_PREFETCH="1", GLM53_NV_PFSIDE="1", GLM53_NV_PUBFAST="1", GLM53_MAX_RQ_TOKENS="4096")
            env.update(engine.x16_env("GLM53_", o.x16_layers))
        rcs = float(o.prefix_cache_gb)
        if box.memory_gb < 24:   # small hosts: a larger memory margin, more reader threads, smaller read pieces
            env.update(GLM53_NV_MARGIN_GB="1.5", GLM53_NV_PF_RING="128", GLM53_NV_THREADS="48", GLM53_NV_PIECE_KB="1024")
            rcs = min(1.0, rcs)
        for spec in self.kernel_specs(box):
            env[spec.env_var] = str(box.runtime.kernel_dirs([spec], box.arch)[spec.name])
        if not system.systemd_scope_available():  # no cgroup cap to size the RAM tier from: give it explicitly
            env["GLM53_NV_RAM_GB"] = f"{max(4.0, box.memory_gb - 12.5):.1f}"
        env.update(multimodal_env(box, self.name))
        env["PATH"] = f"{box.runtime.python.parent}:{os.environ.get('PATH', '')}"
        env.update({k_: str(v) for k_, v in box.env.items()})
        argv = [str(box.runtime.python), "-m", "bigmoe.vllm.serve", "--backend", "glm53", "-m", str(self.model_dir(box)),
                "-cs", str(box.context), "--max-batch-size", str(MAX_BATCH_SIZE), "-chunk_size", str(o.chunk_size)]
        if nvme:
            argv += ["-ambs", "4"]
        argv += ["-rcs", f"{rcs:g}"]
        ccs = cpu_tier_gb(o.cpu_cache_gb, o.disk_cache, box.context, KV_BYTES_PER_TOKEN)
        if ccs:
            argv += ["-ccs", f"{ccs:g}"]
        if o.disk_cache:
            argv += ["-dcd", str(Path(o.disk_cache).expanduser())]
        if o.cache_quant:
            argv += ["-cq", str(o.cache_quant)]
        if o.speculation == "ngram":
            # one recurrent slot (the later -ambs wins): exllamav3 keeps a recurrent-state snapshot per drafted token
            # and slot for rewinding, which at four slots would cost 2.7 GB of VRAM expert cache
            argv += ["-ngram", str(NGRAM_MIN), "-ndt", str(DRAFT_TOKENS), "-ambs", "1"]
        argv += ["--host", box.host, "--port", str(box.port), "--served-name", box.served_name, *box.extra_args]
        # ambient engine variables would silently change the configuration: only the ones set here reach the server
        unset = tuple(v for v in os.environ if v.startswith(("GLM53_", "BIGMOE_", "EXL3_", "EXLLAMAV3_")) and v not in env)
        unset += ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")
        return LaunchSpec(argv=argv, env=env, host=box.host, port=box.port,
                          memory_gb=box.memory_gb + float(o.frontend_extra_ram_gb), unset=unset,
                          served_name=box.served_name, files={str(tune): str(DATA / "coop_autotune_1gpu.bin")},
                          record={"backend": self.name, "model": box.ref.label, "commit": box.commit,
                                  "context": box.context,
                                  "options": {k_: str(v) for k_, v in dataclasses.asdict(o).items()},
                                  "cpu_layout": lay.describe()})

    def bench_frame(self, n_tokens: int):
        return glm_frame(n_tokens)

    def verify_runs(self) -> list:
        return [("mode nvme (decode near-exact)", {"mode": "nvme"}, [1024, 8192, 32768, 120000], "longctx_nvme.json"),
                ("mode nvme-exact (decode bit-exact)", {"mode": "nvme-exact"}, [1024, 8192], "longctx_exact.json")]

    def baseline_dir(self) -> Path:
        return PKG / "baselines"
