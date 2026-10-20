import json
from pathlib import Path

import pytest

from bigmoe import AutoBox, system
from bigmoe.models import glm53

BASE = Path(glm53.__file__).parent / "baselines"
PATHS = ("GLM53_EC_WARM", "GLM53_ZC_STATS", "GLM53_TRITON_PIN", "GLM53_NV_STORE")
# deliberate changes since the reference runs: the planner's NVMe arrival model (nvlat, nvdeep; see TUNED_PLANNER), and
# mode nvme-exact runs exllamav3's stock prefill (the prompt / tail checkpoint patch changes its greedy answers)
CHANGED = {"nvme": {"GLM53_NV_POL": glm53.TUNED_PLANNER}, "nvme-exact": {"GLM53_EXL3_PATCHES": "0"}}


class FakeRuntime:
    python = Path("/envs/glm53/bin/python")

    def kernel_dirs(self, specs, arch):
        return {s.name: Path(f"/k/{s.name}-key") for s in specs}


def spec_for(tmp_path, monkeypatch, mode, **kw):
    monkeypatch.setattr(system, "systemd_scope_available", lambda: True)
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
    ours = {k: v for k, v in spec.env.items() if k.startswith("GLM53_") and not k.endswith("_BUILD")}
    assert set(ours) == set(ref)
    for k, v in ref.items():
        if k in PATHS:
            assert Path(ours[k]).name == Path(v).name, k
        else:
            assert ours[k] == v, k
    assert {spec.env[f"GLM53_{x}_BUILD"] for x in ("EC", "K", "NV", "NV2")} == \
        {"/k/glm53_ec-key", "/k/glm53_k101-key", "/k/glm53_nv-key", "/k/glm53_nv2-key"}
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
