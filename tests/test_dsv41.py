import json
import struct
from pathlib import Path

import pytest

from bigmoe import AutoBox, system
from bigmoe.errors import BoxError
from bigmoe.models import dsv41
from bigmoe.store import exl3_store_v2 as V2

VBMI = {"avx512f", "avx512bw", "avx512vl", "avx512vbmi", "avx512_vnni", "avx2", "fma", "f16c"}


class FakeRuntime:
    python = Path("/envs/dsv41/bin/python")

    def kernel_dirs(self, specs, arch):
        return {s.name: Path(f"/k/{s.name}-key") for s in specs}


def spec_for(tmp_path, monkeypatch, flags=frozenset(VBMI), **kw):
    monkeypatch.setattr(system, "systemd_scope_available", lambda: True)
    monkeypatch.setattr(system, "cpu_flags", lambda: set(flags))
    monkeypatch.setattr(system, "meminfo", lambda: {"MemTotal": 60.9, "MemAvailable": 50.0})
    box = AutoBox("deepseek-v4.1-flash", home=tmp_path,
                  cpus={"main": "0", "controller": 6, "lane": "1-5", "readers": "7-11"}, **kw)
    box.__dict__["runtime"] = FakeRuntime()
    box.__dict__["gpu"] = system.GPU(0, "NVIDIA GeForce RTX 3090", 24576, 0, "8.6")
    return box, box.backend.launch_spec(box)


def test_alias_and_commit(tmp_path):
    box = AutoBox("deepseek-v4.1-flash", home=tmp_path)
    assert box.backend.name == "dsv41" and box.commit == dsv41.VARIANTS["2.25bpw"]
    assert box.options.mode == "nvme" and box.options.speculation == "dspark"
    with pytest.raises(BoxError):
        AutoBox("deepseek-v4.1-flash", home=tmp_path, revision="1.68bpw").commit


def test_launch_exact(tmp_path, monkeypatch):
    box, spec = spec_for(tmp_path, monkeypatch, mode="nvme-exact")
    e = spec.env
    assert e["DSV41_NV"] == "2" and e["DSV41_NV_CPU"] == "0" and e["DSV41_EC_STAGE_SLOTS"] == "384"
    assert (e["DSV41_EC_STAGE_MIN"], e["DSV41_EC_ELASTIC_GB"], e["DSV41_EC_RESERVE_GB"]) == ("512", "11", "3.0")
    assert "DSV41_NV_CPU_CPUS" not in e and e["DSV41_NV_READER_CPUS"] == "7-11" and e["DSV41_NV_CTL_CPU"] == "6"
    assert e["DSV41_NV_STORE"].endswith(dsv41.STORE_NAME + ".bin")
    assert e["BIGMOE_ENGINE_PREFIX"] == "DSV41_"
    assert {e["BIGMOE_EC_BUILD"], e["BIGMOE_NV2_BUILD"]} == {"/k/bigmoe_ec-key", "/k/bigmoe_nv2-key"}
    assert spec.argv[:3] == ["/envs/dsv41/bin/python", "-m", "bigmoe.models.dsv41.engine.server"]
    assert spec.argv[spec.argv.index("-chunk_size") + 1] == "8192" and spec.argv[spec.argv.index("-cs") + 1] == "65536"
    assert "-ccs" in spec.argv and spec.memory_gb == 56.0


def test_prefix_cache_tiers(tmp_path, monkeypatch):
    arg = lambda spec, k: spec.argv[spec.argv.index(k) + 1] if k in spec.argv else None
    _, spec = spec_for(tmp_path, monkeypatch)
    assert (arg(spec, "-ccs"), arg(spec, "-rcs"), arg(spec, "-dcd")) == ("0", "0.5", None)
    _, spec = spec_for(tmp_path, monkeypatch, prefix_cache_gb=1, cpu_cache_gb=1, disk_cache="~/pc")
    assert (arg(spec, "-ccs"), arg(spec, "-rcs")) == ("1", "1")
    assert arg(spec, "-dcd") == str(Path("~/pc").expanduser())
    with pytest.raises(ValueError):   # the disk tier sits behind the CPU tier
        spec_for(tmp_path, monkeypatch, disk_cache="/x")


def test_engine_reserves_host_caches(monkeypatch):
    from bigmoe.engine import tiers
    monkeypatch.setattr(tiers.sys, "argv", ["server", "-rcs", "1", "-ccs", "2"])
    assert tiers._rcs_bytes() == 3 * 1024 ** 3
    monkeypatch.setattr(tiers.sys, "argv", ["server"])
    assert tiers._rcs_bytes() == int(0.5 * 1024 ** 3)


def test_dspark_default_and_off(tmp_path, monkeypatch):
    box, spec = spec_for(tmp_path, monkeypatch)
    assert box.options.speculation == "dspark" and box.options.dspark_conf == 0.5
    assert "-mtp" in spec.argv and spec.env["EXL3_DSPARK_CONF"] == "0.5"
    _, spec = spec_for(tmp_path, monkeypatch, speculation="none")
    assert "-mtp" not in spec.argv and "EXL3_DSPARK_CONF" not in spec.env
    _, spec = spec_for(tmp_path, monkeypatch, dspark_conf=0.35)
    assert spec.env["EXL3_DSPARK_CONF"] == "0.35"
    _, spec = spec_for(tmp_path, monkeypatch, env={"EXL3_DSPARK_CONF": "0.6"})
    assert spec.env["EXL3_DSPARK_CONF"] == "0.6"
    with pytest.raises(ValueError):
        AutoBox("deepseek-v4.1-flash", home=tmp_path, speculation="ngram")


def test_launch_cpu_lane(tmp_path, monkeypatch):
    _, spec = spec_for(tmp_path, monkeypatch)   # default mode nvme
    assert spec.env["DSV41_NV_CPU"] == "1" and spec.env["DSV41_NV_CPU_CPUS"] == "1-5"
    box, _ = spec_for(tmp_path, monkeypatch, flags={"avx2", "fma", "f16c"}, mode="nvme")
    monkeypatch.setattr(system, "gpu_compute_apps", lambda: [])
    monkeypatch.setattr(system, "check_odirect", lambda p: None)
    with pytest.raises(BoxError, match="VBMI"):
        box.backend.check(box)


def _st(path, tensors):
    """Minimal safetensors writer: {name: (dtype, shape, bytes)}."""
    hdr, off, blobs = {}, 0, []
    for k, (dt, shape, data) in tensors.items():
        hdr[k] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + len(data)]}
        off += len(data); blobs.append(data)
    h = json.dumps(hdr).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def test_store_v2_mixed_sizes(odirect_dir, tmp_path):
    """Two layers with different expert sizes -> two classes; pack / verify / compare round-trip."""
    md = tmp_path / "ckpt"
    md.mkdir()
    t, wm = {}, {}
    for li, tile in ((0, 1024), (1, 768)):
        for e in range(3):
            for w in ("w1", "w2", "w3"):
                base = f"layers.{li}.ffn.experts.{e}.{w}"
                t[base + ".trellis"] = ("I16", [tile // 2], bytes([(li * 7 + e * 3 + len(w)) % 251]) * tile)
                t[base + ".suh"] = ("F16", [8], bytes([e + 1]) * 16)
                t[base + ".svh"] = ("F16", [8], bytes([e + 2]) * 16)
    t["layers.0.ffn.shared_experts.w1.trellis"] = ("I16", [8], b"\0" * 16)   # not a routed expert
    _st(md / "model-00001-of-00001.safetensors", t)
    for k in t:
        wm[k] = "model-00001-of-00001.safetensors"
    (md / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wm}))
    meta_path = V2.pack(md, odirect_dir, "s", naming="deepseek", threads=2, log=lambda m: None)
    meta = json.load(open(meta_path))
    assert meta["n_layers"] == 2 and meta["experts_per_layer"] == 3 and len(meta["classes"]) == 2
    L0, L1 = meta["layers"]
    assert (L0["trellis_bytes"], L1["trellis_bytes"]) == (3 * 1024, 3 * 768)
    assert L0["off_u"] == 1024 and L0["off_d"] == 2048 and L1["base"] == 3 * L0["rec_bytes"]
    assert V2.verify(odirect_dir, "s", log=lambda m: None) == []
    assert V2.compare(md, odirect_dir, "s", sample=6, log=lambda m: None) == []
