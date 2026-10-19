import json
import os
import struct

import pytest

from bigmoe.store import ExpertStore, compare, manifest, pack, punch_experts, verify

LAYERS, E = (3, 4), 4


def write_safetensors(path, tensors):
    """tensors: {name: (dtype, shape, bytes)}"""
    hdr, off, blobs = {}, 0, []
    for name, (dt, shape, data) in tensors.items():
        hdr[name] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + len(data)]}
        blobs.append(data)
        off += len(data)
    h = json.dumps(hdr).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def make_checkpoint(d):
    d.mkdir()
    t, seed = {}, 0
    for l in LAYERS:
        t[f"model.layers.{l}.self_attn.q_proj.weight"] = ("F16", (64,), bytes([l]) * 128)
        for e in range(E):
            for m in ("gate_proj", "up_proj", "down_proj"):
                seed += 1
                k = f"model.layers.{l}.mlp.experts.{e}.{m}"
                t[k + ".trellis"] = ("I16", (2048,), bytes([seed % 251]) * 4096)
                t[k + ".suh"] = ("F16", (64,), bytes([(seed + 1) % 251]) * 128)
                t[k + ".svh"] = ("F16", (32,), bytes([(seed + 2) % 251]) * 64)
                t[k + ".mul1"] = ("I32", (), struct.pack("<i", 7))
    write_safetensors(d / "model-00001-of-00001.safetensors", t)
    (d / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {k: "model-00001-of-00001.safetensors" for k in t}}))
    return t


def test_pack_verify_compare_slim(tmp_path, odirect_dir):
    ck = tmp_path / "ckpt"
    tensors = make_checkpoint(ck)
    out = odirect_dir
    pack(ck, out, "x", threads=2, log=lambda m: None)
    m = manifest(out, "x")
    assert m["n_layers"] == 2 and m["experts_per_layer"] == E and m["trellis_bytes"] == 3 * 4096
    assert m["record_bytes"] == 4096 * 4 and m["vram_slot"] == {"off_u": 4096, "off_d": 8192, "slot_bytes": 12288}
    assert verify(out, "x", threads=2, log=lambda m: None) == []
    assert compare(ck, out, "x", sample=3, log=lambda m: None) == []
    st = ExpertStore(out, "x")
    assert not st.is_complete()
    assert st.verify(log=lambda m: None) == [] and st.is_complete()
    with open(st.bin, "r+b") as f:  # corrupt one byte of record 5
        f.seek(5 * m["record_bytes"] + 100)
        f.write(b"\xff")
    assert verify(out, "x", threads=2, log=lambda m: None) == [5]
    assert not st.is_complete()  # the .bin changed since its verification

    shard = ck / "model-00001-of-00001.safetensors"
    freed = punch_experts(ck, st.layer_keys(), log=lambda m: None)
    assert freed == {shard.name: len(LAYERS) * E * 3 * 4096}
    with open(shard, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        for k, v in hdr.items():
            f.seek(base + v["data_offsets"][0])
            data = f.read(v["data_offsets"][1] - v["data_offsets"][0])
            if k.endswith(".trellis"):
                assert data == bytes(len(data)), k
            else:
                assert data == tensors[k][2], k
