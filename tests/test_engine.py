"""Shared engine (bigmoe.engine): per-model env prefix, store manifests (v1 normalised to the v2 per-layer layout),
kernel specs. No GPU; the tiers module needs torch + numpy (the server runtime), so these skip without them."""
import importlib
import json

import pytest

from bigmoe import engine


def test_kernel_specs():
    specs = {s.name: s for s in engine.kernel_specs()}
    assert set(specs) == {"bigmoe_ec", "bigmoe_nv2", "bigmoe_x16"}
    assert specs["bigmoe_ec"].env_var == "BIGMOE_EC_BUILD" and specs["bigmoe_nv2"].env_var == "BIGMOE_NV2_BUILD"
    names = {p.name for p in specs["bigmoe_nv2"].sources}
    assert {"nv2.py", "nv2_host.cpp", "nv2_dev.cu", "nv2_shared.h", "ft_core.h", "ft_n135.h", "ft_x16.h", "penv.h"} <= names
    assert specs["bigmoe_x16"].env_var == "BIGMOE_X16_BUILD"
    assert {"x16.py", "x16_dev.cu"} <= {p.name for p in specs["bigmoe_x16"].sources}


def test_env_prefix(monkeypatch):
    from bigmoe.engine import env as E
    monkeypatch.setenv("BIGMOE_ENGINE_PREFIX", "GLM53_")
    monkeypatch.setenv("GLM53_NV_CPU", "1")
    monkeypatch.setenv("BIGMOE_NV2_BUILD", "/k/nv2")
    E = importlib.reload(E)
    try:
        assert E.env("NV_CPU") == "1" and E.env("NV_MISSING", "d") == "d"
        assert E.build_dir("NV2", "x") == "/k/nv2" and E.build_dir("EC", "x") == "x"
    finally:
        monkeypatch.delenv("BIGMOE_ENGINE_PREFIX")
        importlib.reload(E)


@pytest.fixture
def tiers(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("numpy")
    from bigmoe.engine import env as E
    monkeypatch.setenv("BIGMOE_ENGINE_PREFIX", "TEST_")
    importlib.reload(E)
    from bigmoe.engine import tiers as T
    T = importlib.reload(T)
    yield T
    monkeypatch.delenv("BIGMOE_ENGINE_PREFIX")
    importlib.reload(E)
    importlib.reload(T)


def test_manifest_v1_normalised(tiers, tmp_path, monkeypatch):
    E, R = 4, 9474048
    v1 = {"format": "glm53-expert-store-v1", "record_bytes": R, "experts_per_layer": E, "mtp_layer_li": 2,
          "layers": [{"li": i, "model_layer": 3 + i, "key": f"m.layers.{3 + i}.mlp"} for i in range(3)],
          "fields": [{"name": f"{m}.trellis", "offset": o, "bytes": 3145728} for m, o in
                     (("gate_proj", 0), ("up_proj", 3145728), ("down_proj", 6291456))],
          "vram_slot": {"off_u": 3145728, "off_d": 6291456, "slot_bytes": 9437184}, "total_bytes": 3 * E * R}
    (tmp_path / "s.json").write_text(json.dumps(v1))
    monkeypatch.setenv("TEST_NV_STORE", str(tmp_path / "s.bin"))
    m = tiers.manifest()
    assert m["mats"] == ["gate_proj", "up_proj", "down_proj"] and m["experts_per_layer"] == E
    assert [L["base"] for L in m["layers"]] == [0, E * R, 2 * E * R]
    assert all(L["trellis_bytes"] == 9437184 and (L["off_u"], L["off_d"]) == (3145728, 6291456) for L in m["layers"])
    assert [L["key"] for L in tiers.routed_layers()] == ["m.layers.3.mlp", "m.layers.4.mlp"]   # MTP layer excluded


def test_manifest_v2(tiers, tmp_path, monkeypatch):
    layers = [{"li": 0, "key": "layers.0.ffn", "base": 0, "rec_bytes": 11104256, "trellis_bytes": 11059200,
               "off_u": 3686400, "off_d": 7372800, "fields": [], "cls": 0},
              {"li": 1, "key": "layers.1.ffn", "base": 3 * 11104256, "rec_bytes": 8892416, "trellis_bytes": 8847360,
               "off_u": 2949120, "off_d": 5898240, "fields": [], "cls": 1}]
    v2 = {"format": "exl3-expert-store-v2", "experts_per_layer": 3, "mats": ["w1", "w3", "w2"], "layers": layers}
    (tmp_path / "s.json").write_text(json.dumps(v2))
    monkeypatch.setenv("TEST_NV_STORE", str(tmp_path / "s.bin"))
    m = tiers.manifest()
    assert m["mats"] == ["w1", "w3", "w2"] and [L["routed"] for L in m["layers"]] == [True, True]
    assert [L["trellis_bytes"] for L in tiers.routed_layers()] == [11059200, 8847360]
