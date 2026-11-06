import json
from pathlib import Path

import pytest

from bigmoe import AutoBox, system
from bigmoe.models import glm53

BASE = Path(glm53.__file__).parent / "baselines"
PATHS = ("GLM53_EC_WARM", "GLM53_ZC_STATS", "GLM53_NV_STORE")
# deliberate changes since the reference runs: the planner's NVMe arrival model (nvlat, nvdeep; see TUNED_PLANNER), and
# the exllamav3 1.5.1 runtime's own modules are gone with it (fused hyper-connection kernels, decode lookahead, Triton
# pin, prompt / tail checkpoint patch): exllamav3's fused sites (EXL3_HC_FOLD) and the engine's prefill replace them
CHANGED = {"nvme": {"GLM53_NV_POL": glm53.TUNED_PLANNER}, "nvme-exact": {}}
REMOVED = {"GLM53_K_HCFUSE", "GLM53_LA", "GLM53_LA_BTTRIM", "GLM53_TRITON_PIN", "GLM53_EXL3_PATCHES"}


class FakeRuntime:
    python = Path("/envs/glm53/bin/python")

    def kernel_dirs(self, specs, arch):
        return {s.name: Path(f"/k/{s.name}-key") for s in specs}


def spec_for(tmp_path, monkeypatch, mode, flags=frozenset({"avx2", "fma", "f16c"}), **kw):
    monkeypatch.setattr(system, "systemd_scope_available", lambda: True)
    monkeypatch.setattr(system, "cpu_flags", lambda: set(flags))
    box = AutoBox("glm-5.3-flash", home=tmp_path, mode=mode,
                  cpus={"main": "0", "controller": 6, "lane": "1-5", "readers": "7-11"}, **kw)
    box.__dict__["runtime"] = FakeRuntime()
    box.__dict__["gpu"] = system.GPU(0, "NVIDIA GeForce RTX 3090", 24576, 0, "8.6")
    return box, box.backend.launch_spec(box)


@pytest.mark.parametrize("mode,fname", [("nvme", "longctx_nvme.json"), ("nvme-exact", "longctx_exact.json")])
def test_env_matches_the_reference_run(tmp_path, monkeypatch, mode, fname):
    # the reference runs were measured with the old 52 GiB cap and a 1376-slot expert cache: pass them explicitly
    box, spec = spec_for(tmp_path, monkeypatch, mode, memory_gb=52.0, vram_expert_slots=1376)
    ref = dict(json.loads((BASE / fname).read_text())["env"], **CHANGED[mode])
    ref = {k: v for k, v in ref.items() if k not in REMOVED}
    ours = {k: v for k, v in spec.env.items() if k.startswith("GLM53_") and not k.endswith("_BUILD")}
    assert set(ours) == set(ref)
    for k, v in ref.items():
        if k in PATHS:
            assert Path(ours[k]).name == Path(v).name, k
        else:
            assert ours[k] == v, k
    assert spec.env["BIGMOE_ENGINE_PREFIX"] == "GLM53_"
    assert {spec.env[v] for v in ("BIGMOE_EC_BUILD", "BIGMOE_NV2_BUILD")} == {"/k/bigmoe_ec-key", "/k/bigmoe_nv2-key"}
    assert "GLM53_K_BUILD" not in spec.env and spec.env["EXL3_HC_FOLD"] == "1"
    assert spec.files == {spec.env["EXLLAMAV3_TUNE_CACHE"]: str(glm53.DATA / "coop_autotune_1gpu.bin")}
    assert not Path(spec.env["EXLLAMAV3_TUNE_CACHE"]).exists()


def test_argv(tmp_path, monkeypatch):
    monkeypatch.setattr(system, "meminfo", lambda: {"MemTotal": 60.9, "MemAvailable": 50.0})
    box, spec = spec_for(tmp_path, monkeypatch, "nvme")
    assert spec.argv[:3] == ["/envs/glm53/bin/python", "-m", "bigmoe.models.glm53.engine.server"]
    assert spec.argv[3:] == ["-m", str(box.backend.model_dir(box)), "-cs", "131072", "--max-batch-size", "8",
                             "-chunk_size", "8192", "-ambs", "4", "-rcs", "1", "--host", "127.0.0.1", "--port", "30000",
                             "--served-name", "glm-5.3-flash"]
    assert spec.memory_gb == 56.0   # default on a 60 GiB+ box (measured best)
    _, spec = spec_for(tmp_path, monkeypatch, "nvme-exact", context=524288, cache_quant=8, chunk_size=4096)
    assert "-ambs" not in spec.argv and spec.argv[spec.argv.index("-cq") + 1] == "8"
    assert spec.argv[spec.argv.index("-cs") + 1] == "524288" and spec.argv[spec.argv.index("-chunk_size") + 1] == "4096"


def test_overrides_and_ambient_env(tmp_path, monkeypatch):
    monkeypatch.setenv("GLM53_NV_THREADS", "99")
    _, spec = spec_for(tmp_path, monkeypatch, "nvme", env={"GLM53_NV_POL": "1,2,3"}, vram_expert_slots=None)
    assert spec.env["GLM53_NV_POL"] == "1,2,3"
    assert "GLM53_EC_MAX_SLOTS" not in spec.env
    assert "GLM53_NV_THREADS" in spec.unset


def test_vbmi_planner(tmp_path, monkeypatch):
    # AVX-512 VBMI CPUs: the CPU lane is faster, so the default planner gives it more experts; explicit costs win
    _, spec = spec_for(tmp_path, monkeypatch, "nvme", flags=glm53.VBMI_FLAGS | {"avx2", "fma", "f16c"})
    assert spec.env["GLM53_NV_POL"] == glm53.TUNED_PLANNER_VBMI
    _, spec = spec_for(tmp_path, monkeypatch, "nvme", flags=glm53.VBMI_FLAGS | {"avx2", "fma", "f16c"}, planner_cost="1,2,3")
    assert spec.env["GLM53_NV_POL"] == "1,2,3"
    _, spec = spec_for(tmp_path, monkeypatch, "nvme")
    assert spec.env["GLM53_NV_POL"] == glm53.TUNED_PLANNER


def test_small_host_budget(tmp_path, monkeypatch):
    _, spec = spec_for(tmp_path, monkeypatch, "nvme", memory_gb=20)
    assert spec.env["GLM53_NV_THREADS"] == "48" and spec.argv[spec.argv.index("-rcs") + 1] == "1"


def test_speculation_option(tmp_path, monkeypatch):
    monkeypatch.setattr(system, "meminfo", lambda: {"MemTotal": 60.9, "MemAvailable": 50.0})
    _, spec = spec_for(tmp_path, monkeypatch, "nvme")
    assert "-ngram" not in spec.argv and "-ndt" not in spec.argv   # off by default
    _, spec = spec_for(tmp_path, monkeypatch, "nvme", speculation="ngram")
    assert spec.argv[spec.argv.index("-ngram") + 1] == "2" and spec.argv[spec.argv.index("-ndt") + 1] == "5"
    # one recurrent slot with drafts (the rewind history would take 2.7 GB of expert cache at -ambs 4): the last -ambs wins
    i = len(spec.argv) - 1 - spec.argv[::-1].index("-ambs")
    assert spec.argv[i + 1] == "1" and i > spec.argv.index("-ambs")
    _, spec = spec_for(tmp_path, monkeypatch, "nvme", speculation="ngram", ngram_min=3, draft_tokens=4)
    assert spec.argv[spec.argv.index("-ngram") + 1] == "3" and spec.argv[spec.argv.index("-ndt") + 1] == "4"
    with pytest.raises(ValueError):
        spec_for(tmp_path, monkeypatch, "nvme", speculation="dflash")
    with pytest.raises(ValueError):
        spec_for(tmp_path, monkeypatch, "nvme", speculation="ngram", draft_tokens=0)


def test_prefix_cache_tiers(tmp_path, monkeypatch):
    _, spec = spec_for(tmp_path, monkeypatch, "nvme")
    assert "-ccs" not in spec.argv and "-dcd" not in spec.argv   # off by default
    _, spec = spec_for(tmp_path, monkeypatch, "nvme", cpu_cache_gb=1, disk_cache="~/pc")
    assert spec.argv[spec.argv.index("-ccs") + 1] == "1"
    assert spec.argv[spec.argv.index("-dcd") + 1] == str(Path("~/pc").expanduser())
    with pytest.raises(ValueError):   # the disk tier sits behind the CPU tier
        spec_for(tmp_path, monkeypatch, "nvme", disk_cache="/x")
