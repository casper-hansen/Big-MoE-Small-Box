"""Packs the routed experts of an EXL3 MoE checkpoint into one O_DIRECT-readable store + JSON manifest.

Record r = li * E + e (layer-major; with include_mtp the MTP layer of mtp.safetensors is the last li), 4096-aligned:
  [gate.trellis][up.trellis][down.trellis][gate.suh][gate.svh][up.suh][up.svh][down.suh][down.svh][pad]
The trellis part is byte-identical to one VRAM slot of the expert cache (gate | up @ off_u | down @ off_d), so an
NVMe -> RAM -> VRAM fill is one contiguous copy, and a reader that keeps suh/svh resident reads only that part.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import mmap
import os
import random
import re
import struct
import sys
import threading
import time

ALIGN = 4096
MATS = ("gate_proj", "up_proj", "down_proj")


def al(x, a=ALIGN):
    return (x + a - 1) // a * a


def st_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return 8 + n, h


class Source:

    def __init__(self, model_dir, include_mtp=True):
        self.dir = str(model_dir)
        idx = json.load(open(os.path.join(self.dir, "model.safetensors.index.json")))["weight_map"]
        files = sorted(set(idx.values()))
        if include_mtp and os.path.exists(os.path.join(self.dir, "mtp.safetensors")):
            files.append("mtp.safetensors")
        self.loc, self.fds = {}, {}
        for fn in files:
            base, h = st_header(os.path.join(self.dir, fn))
            for k, v in h.items():
                if ".mlp.experts." not in k:
                    continue
                a, b = v["data_offsets"]
                self.loc[k] = (fn, base + a, b - a, v["dtype"], tuple(v["shape"]))
            self.fds[fn] = os.open(os.path.join(self.dir, fn), os.O_RDONLY)
        layers = {}
        for k in self.loc:
            m = re.match(r"(.*\.layers\.(\d+)\.mlp)\.experts\.(\d+)\.", k)
            layers.setdefault(int(m.group(2)), [m.group(1), set()])[1].add(int(m.group(3)))
        self.layers = [(l, layers[l][0], len(layers[l][1])) for l in sorted(layers)]
        if not self.layers:
            raise ValueError(f"no routed experts (*.mlp.experts.*) in {self.dir}")

    def read(self, key, mv):
        fn, off, n, _, _ = self.loc[key]
        got = 0
        while got < n:
            r = os.preadv(self.fds[fn], [mv[got:n]], off + got)
            assert r > 0, (key, got, n)
            got += r
        return n

    def close(self):
        for fd in self.fds.values():
            os.close(fd)
        self.fds = {}


def layout(src):
    """Taken from layer 0 / expert 0; check_uniform asserts that every expert matches."""
    _, pfx, _ = src.layers[0]
    fields, off = [], 0
    for m in MATS:
        k = f"{pfx}.experts.0.{m}.trellis"
        n = src.loc[k][2]
        fields.append((f"{m}.trellis", off, n, src.loc[k][3], src.loc[k][4]))
        off += n
    trellis = off
    assert trellis % ALIGN == 0, trellis
    for m in MATS:
        for s in ("suh", "svh"):
            k = f"{pfx}.experts.0.{m}.{s}"
            n = src.loc[k][2]
            fields.append((f"{m}.{s}", off, n, src.loc[k][3], src.loc[k][4]))
            off += n
    return fields, trellis, al(off)


def check_uniform(src, fields, n_layers):
    mul1 = set()
    for li in range(n_layers):
        _, pfx, E = src.layers[li]
        for e in range(E):
            for name, _, n, dt, shp in fields:
                v = src.loc[f"{pfx}.experts.{e}.{name}"]
                assert v[2] == n and v[3] == dt and v[4] == shp, (pfx, e, name, v, n, dt, shp)
            for m in MATS:
                k = f"{pfx}.experts.{e}.{m}.mul1"
                if k in src.loc:
                    fn, off, n, dt, _ = src.loc[k]
                    mul1.add(os.pread(src.fds[fn], n, off).hex())
    return sorted(mul1)


def fill_record(src, pfx, e, fields, buf):
    mv = memoryview(buf)
    for name, off, n, _, _ in fields:
        src.read(f"{pfx}.experts.{e}.{name}", mv[off:off + n])
    return mv


def paths(out_dir, name):
    return os.path.join(str(out_dir), name + ".bin"), os.path.join(str(out_dir), name + ".json")


def pack(model_dir, out_dir, name, *, threads=8, include_mtp=True, mtp_model_layer=None,
         fmt="exl3-expert-store-v1", log=print):
    """mtp_model_layer: model layer index of the MTP layer (recorded as mtp_layer_li if it is packed last)."""
    out_bin, out_meta = paths(out_dir, name)
    assert os.path.isdir(out_dir), f"out_dir {out_dir} does not exist"
    assert os.path.realpath(out_dir) != os.path.realpath(model_dir), "out_dir must not be the checkpoint directory"
    t0 = time.time()
    src = Source(model_dir, include_mtp)
    try:
        fields, trellis, rec = layout(src)
        L = len(src.layers)
        E = src.layers[0][2]
        assert all(x[2] == E for x in src.layers)
        mul1 = check_uniform(src, fields, L)
        total = L * E * rec
        st = os.statvfs(out_dir)
        free = st.f_bavail * st.f_frsize
        log(f"layers {L} ({src.layers[0][0]}..{src.layers[-1][0]}), E {E}, rec {rec} B (trellis {trellis}), total "
            f"{total / 1e9:.2f} GB, free {free / 1e12:.2f} TB, mul1 markers {mul1}")
        assert total < free * 0.9, "not enough free space"
        fd = os.open(out_bin, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT, 0o644)
        os.posix_fallocate(fd, 0, total)
        hashes = [None] * (L * E)
        tl = threading.local()

        def job(r):
            li, e = divmod(r, E)
            if not hasattr(tl, "buf"):
                tl.buf = mmap.mmap(-1, rec)  # page-aligned, as an O_DIRECT source must be
            mv = fill_record(src, src.layers[li][1], e, fields, tl.buf)
            if rec > fields[-1][1] + fields[-1][2]:
                pad0 = fields[-1][1] + fields[-1][2]
                mv[pad0:rec] = bytes(rec - pad0)
            h = hashlib.sha256(mv).hexdigest()
            n = os.pwrite(fd, mv, r * rec)
            assert n == rec, (r, n)
            hashes[r] = h
            return r

        done = 0
        with cf.ThreadPoolExecutor(threads) as ex:
            for li in range(L):
                tl0 = time.time()
                list(ex.map(job, range(li * E, (li + 1) * E)))
                done += E
                dt = time.time() - tl0
                log(f"  layer {li:2d} (model {src.layers[li][0]}) {E} records {E * rec / 1e9:.2f} GB in {dt:.1f} s "
                    f"({E * rec / 1e9 / max(dt, 1e-6):.2f} GB/s), total {done * rec / 1e9:.1f} GB, {time.time() - t0:.0f} s")
        os.fsync(fd)
        os.close(fd)
        mtp_li = L - 1 if include_mtp and mtp_model_layer is not None and src.layers[-1][0] == mtp_model_layer else None
        meta = dict(format=fmt, model_dir=str(model_dir), created=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    record_bytes=rec, trellis_bytes=trellis, align=ALIGN, experts_per_layer=E, n_layers=L,
                    order="record = li * experts_per_layer + e (layer-major)",
                    layers=[dict(li=i, model_layer=l, key=p) for i, (l, p, _) in enumerate(src.layers)],
                    mtp_layer_li=mtp_li,
                    fields=[dict(name=n, offset=o, bytes=b, dtype=d, shape=list(s)) for n, o, b, d, s in fields],
                    vram_slot=dict(off_u=fields[1][1], off_d=fields[2][1], slot_bytes=trellis),
                    mul1_markers_hex=mul1, sha256=hashes, total_bytes=total)
        tmp = out_meta + ".tmp"
        with open(tmp, "w") as f:
            json.dump(meta, f)
        os.replace(tmp, out_meta)
        log(f"PACKED {L * E} records, {total / 1e9:.2f} GB in {time.time() - t0:.0f} s -> {out_bin}")
        return out_meta
    finally:
        src.close()


def manifest(out_dir, name) -> dict:
    return json.load(open(paths(out_dir, name)[1]))


def _reader(out_bin, rec):
    fd = os.open(out_bin, os.O_RDONLY | os.O_DIRECT)
    tl = threading.local()

    def rd(r):
        if not hasattr(tl, "buf"):
            tl.buf = mmap.mmap(-1, rec)
        n = os.preadv(fd, [tl.buf], r * rec)
        assert n == rec, (r, n)
        return tl.buf
    return fd, rd


def verify(out_dir, name, *, threads=8, sample=None, seed=0, log=print) -> list:
    """sample=None checks every record, else `sample` random ones + the first / last of every layer. Reads use
    O_DIRECT, so the page cache cannot hide a bad disk copy."""
    t0 = time.time()
    out_bin, _ = paths(out_dir, name)
    meta = manifest(out_dir, name)
    rec, E, L = meta["record_bytes"], meta["experts_per_layer"], meta["n_layers"]
    assert os.path.getsize(out_bin) == meta["total_bytes"] == rec * E * L, "store size != manifest"
    if sample is None:
        pick = range(L * E)
    else:
        rng = random.Random(seed)
        s = set(rng.sample(range(L * E), min(sample, L * E)))
        for li in range(L):
            s.update((li * E, li * E + E - 1))
        pick = sorted(s)
    fd, rd = _reader(out_bin, rec)
    bad = []
    try:
        def job(r):
            if hashlib.sha256(rd(r)).hexdigest() != meta["sha256"][r]:
                bad.append(r)
        with cf.ThreadPoolExecutor(threads) as ex:
            list(ex.map(job, pick))
    finally:
        os.close(fd)
    n, dt = len(pick), max(time.time() - t0, 1e-6)
    log(f"VERIFY {'OK' if not bad else 'FAIL'}: {n} records, {n * rec / 1e9:.2f} GB O_DIRECT re-read + sha256 in "
        f"{dt:.0f} s ({n * rec / 1e9 / dt:.2f} GB/s); bad {len(bad)} {sorted(bad)[:20]}")
    return sorted(bad)


def compare(model_dir, out_dir, name, *, sample=200, seed=116, include_mtp=True, log=print) -> list:
    """Byte compare against fresh checkpoint reads: a different code path from the packer's buffer reuse."""
    t0 = time.time()
    out_bin, _ = paths(out_dir, name)
    meta = manifest(out_dir, name)
    rec, E, L = meta["record_bytes"], meta["experts_per_layer"], meta["n_layers"]
    src = Source(model_dir, include_mtp)
    fields = [(f["name"], f["offset"], f["bytes"]) for f in meta["fields"]]
    rng = random.Random(seed)
    pick = set(rng.sample(range(L * E), min(sample, L * E)))
    for li in range(L):
        pick.update((li * E, li * E + E - 1))
    pick = sorted(pick)
    fd, rd = _reader(out_bin, rec)
    bad = []
    try:
        for r in pick:
            li, e = divmod(r, E)
            pfx = meta["layers"][li]["key"]
            got = rd(r)
            for fname, off, n in fields:
                fn, soff, sn, _, _ = src.loc[f"{pfx}.experts.{e}.{fname}"]
                assert sn == n
                if got[off:off + n] != os.pread(src.fds[fn], n, soff):
                    bad.append((r, fname))
            tail = fields[-1][1] + fields[-1][2]
            if any(got[tail:rec]):
                bad.append((r, "pad"))
    finally:
        os.close(fd)
        src.close()
    log(f"CMP {'OK' if not bad else 'FAIL'}: {len(pick)} records byte-compared against the checkpoint (every field + "
        f"zero pad), {time.time() - t0:.0f} s; mismatches {len(bad)} {bad[:20]}")
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=["pack", "verify", "cmp"])
    ap.add_argument("--model", required=True, help="EXL3 checkpoint directory")
    ap.add_argument("--out-dir", required=True, help="store directory (local NVMe, O_DIRECT capable)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--mtp", type=int, default=1, help="append the MTP layer's experts as the last layer")
    ap.add_argument("--mtp-model-layer", type=int, default=None)
    ap.add_argument("--format", default="exl3-expert-store-v1")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--seed", type=int, default=116)
    a = ap.parse_args(argv)
    if a.mode == "pack":
        pack(a.model, a.out_dir, a.name, threads=a.threads, include_mtp=bool(a.mtp), mtp_model_layer=a.mtp_model_layer,
             fmt=a.format, log=lambda m: print(m, flush=True))
        return 0
    if a.mode == "verify":
        return 1 if verify(a.out_dir, a.name, threads=a.threads, sample=a.sample, log=lambda m: print(m, flush=True)) else 0
    return 1 if compare(a.model, a.out_dir, a.name, sample=a.sample or 200, seed=a.seed, include_mtp=bool(a.mtp),
                        log=lambda m: print(m, flush=True)) else 0


if __name__ == "__main__":
    sys.exit(main())
