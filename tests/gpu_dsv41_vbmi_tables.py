"""Box-only check (needs the checkpoint + a GPU): the n135 VBMI decode tables for trellis formats K 2 / 2.5 / 3
(ft_n135.h init_vbmi, re-implemented here line by line) extract the same 16-bit states from real tiles as exllamav3's
own unpack (unpack_trellis_frac for K 2.5; the ring definition for integer K). Simulates vpermt2b + vpmultishiftqb.

  python tests/gpu_dsv41_vbmi_tables.py MODEL_DIR
"""
import json
import struct
import sys

import numpy as np
import torch

TF = {"K2": (2, 0x0, 64), "K25": (2, 0xAAAA, 80), "K3": (3, 0x0, 96)}


def tf_s(ka, mask, i):
    s = (i >> 4) * (16 * ka + bin(mask).count("1"))
    for j in range((i & 15) + 1):
        s += ka + ((mask >> j) & 1)
    return s


def mem_of_stream(q, tb):
    q = ((q % tb) + tb) % tb
    return 4 * (q // 4) + 3 - q % 4


def tables(ka, mask, tb):
    pidx = np.zeros((8, 64), np.int64)
    msh = np.zeros((8, 64), np.int64)
    for C in range(8):
        for m in range(8):
            gA, j0 = 4 * C + (0 if m < 4 else 2), 2 * (m & 3)
            PA = tf_s(ka, mask, 8 * gA + j0) - 16
            qs = PA // 8   # floor (python), = the C++ expression for negatives
            for i in range(8):
                pidx[C, 8 * m + i] = mem_of_stream(qs + 7 - i, tb)
            st = [(gA, j0), (gA + 1, j0), (gA, j0 + 1), (gA + 1, j0 + 1)]
            for k in range(4):
                r = tf_s(ka, mask, 8 * st[k][0] + st[k][1]) - 16 - 8 * qs
                assert 0 <= r <= 48, (C, m, k, r)
                msh[C, 8 * m + 2 * k] = 48 - r
                msh[C, 8 * m + 2 * k + 1] = 56 - r
    return pidx, msh


def simulate(tile_bytes, pidx, msh, ka, mask):
    """-> states[256] by ring position, from the table-driven extraction."""
    out = np.zeros(256, np.int64)
    for C in range(8):
        for m in range(8):
            q = 0
            for i in range(8):
                q |= int(tile_bytes[pidx[C, 8 * m + i]]) << (8 * i)
            gA, j0 = 4 * C + (0 if m < 4 else 2), 2 * (m & 3)
            st = [(gA, j0), (gA + 1, j0), (gA, j0 + 1), (gA + 1, j0 + 1)]
            M64 = (1 << 64) - 1
            rot = lambda v, s: ((v >> s) | ((v << (64 - s)) & M64)) & 0xFF if s else v & 0xFF   # vpmultishiftqb byte
            for k in range(4):
                lo = rot(q, int(msh[C, 8 * m + 2 * k]))
                hi = rot(q, int(msh[C, 8 * m + 2 * k + 1]))
                out[8 * st[k][0] + st[k][1]] = lo | (hi << 8)
    return out


def ring_states(tile_bytes, ka, mask, tb):
    """Reference: 16-bit window [S(i) - 16, S(i)) of the MSB-first 32-bit-word ring."""
    words = np.frombuffer(bytes(tile_bytes), dtype="<u4")
    R = len(words) * 32
    bits = np.zeros(R, np.int64)
    for p in range(R):
        bits[p] = (int(words[p >> 5]) >> (31 - (p & 31))) & 1
    out = np.zeros(256, np.int64)
    for i in range(256):
        s = tf_s(ka, mask, i)
        v = 0
        for p in range(s - 16, s):
            v = (v << 1) | int(bits[p % R])
        out[i] = v
    return out


def load(model_dir, key):
    idx = json.load(open(f"{model_dir}/model.safetensors.index.json"))["weight_map"]
    fn = f"{model_dir}/{idx[key]}"
    with open(fn, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
        a, b = h[key]["data_offsets"]
        f.seek(8 + n + a)
        return np.frombuffer(f.read(b - a), np.int16).reshape(h[key]["shape"])


def main():
    md = sys.argv[1]
    from exllamav3.ext import exllamav3_ext as ext
    ok = True
    for name, layer in (("K25", 0), ("K2", 10)):
        ka, mask, tb = TF[name]
        tr = load(md, f"layers.{layer}.ffn.experts.0.w1.trellis")
        assert tr.shape[-1] * 2 == tb, (name, tr.shape)
        pidx, msh = tables(ka, mask, tb)
        rng = np.random.default_rng(0)
        tiles = [tuple(rng.integers(0, s) for s in tr.shape[:2]) for _ in range(6)] + [(0, 0), (tr.shape[0] - 1, tr.shape[1] - 1)]
        for (kt, nt) in tiles:
            tile = tr[kt, nt].tobytes()
            sim = simulate(np.frombuffer(tile, np.uint8), pidx, msh, ka, mask)
            ref = ring_states(np.frombuffer(tile, np.uint8), ka, mask, tb)
            good = np.array_equal(sim, ref)
            if mask:
                packed = torch.from_numpy(tr[kt:kt + 1, nt:nt + 1].copy()).cuda()
                unp = torch.empty((1, 1, 256), dtype=torch.short, device="cuda")
                ext.unpack_trellis_frac(unp, packed, ka, mask)
                ex = unp.view(-1).cpu().numpy().astype(np.int64) & 0xFFFF
                good = good and np.array_equal(sim, ex)
            ok &= good
            print(f"{name} layer {layer} tile ({kt},{nt}): {'OK' if good else 'MISMATCH'}")
    print("ALL OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
