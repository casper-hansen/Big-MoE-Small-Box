"""Expert store v2: the routed experts of an EXL3 MoE checkpoint in one O_DIRECT-readable file, with PER-LAYER record
sizes (quants that mix bitrates across layers, e.g. DeepSeek-V4.1-Flash 2.25 bpw: 2.5 bpw experts in the outer layers,
2.0 bpw in the middle ones).

Layout: layer-major. Layer li holds E records of rec[li] bytes (4096-aligned) at base[li] + e * rec[li]:
  [gate.trellis][up.trellis][down.trellis][gate.suh][gate.svh][up.suh][up.svh][down.suh][down.svh][zero pad]
The trellis part (gate | up @ off_u | down @ off_d) is byte-identical to one VRAM slot of the expert cache, so an
NVMe -> RAM -> VRAM fill is one contiguous copy. Layers with the same record layout form a size class: the engine keeps
one VRAM / RAM arena per class.

Manifest (<name>.json): format "exl3-expert-store-v2", experts_per_layer, n_layers, layers[li] = {li, model_layer, key,
base, rec_bytes, trellis_bytes, off_u, off_d, fields[...], cls}, classes[c] = {rec_bytes, trellis_bytes, off_u, off_d,
layers[...]}, sha256[li][e], total_bytes.
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
FORMAT = "exl3-expert-store-v2"
# (gate, up, down) tensor names inside one expert, per checkpoint family
NAMINGS = {"glm": ("gate_proj", "up_proj", "down_proj"), "deepseek": ("w1", "w3", "w2")}


def al(x, a=ALIGN):
    return (x + a - 1) // a * a


def st_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return 8 + n, h


class Source:
    """Routed-expert tensors of a checkpoint: key -> (file, offset, bytes, dtype, shape). expert_re captures
    (layer prefix, layer index, expert index) from a tensor key."""

    def __init__(self, model_dir, expert_re=r"^((?:.*\.)?layers\.(\d+)\.(?:mlp|ffn))\.experts\.(\d+)\.",
                 extra_files=()):
        self.dir = str(model_dir)
        idx = json.load(open(os.path.join(self.dir, "model.safetensors.index.json")))["weight_map"]
        files = sorted(set(idx.values())) + [f for f in extra_files if os.path.exists(os.path.join(self.dir, f))]
        rx = re.compile(expert_re)
        self.loc, self.fds = {}, {}
        layers = {}
        for fn in files:
            base, h = st_header(os.path.join(self.dir, fn))
            for k, v in h.items():
                m = rx.match(k)
                if not m:
                    continue
                a, b = v["data_offsets"]
                self.loc[k] = (fn, base + a, b - a, v["dtype"], tuple(v["shape"]))
                layers.setdefault(int(m.group(2)), [m.group(1), set()])[1].add(int(m.group(3)))
            self.fds[fn] = os.open(os.path.join(self.dir, fn), os.O_RDONLY)
        self.layers = [(l, layers[l][0], len(layers[l][1])) for l in sorted(layers)]
        if not self.layers:
            raise ValueError(f"no routed experts matching {expert_re!r} in {self.dir}")

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


def layer_layout(src, pfx, mats):
    """Field layout of one layer, from expert 0: trellis x3 first (the VRAM slot image), then suh / svh x3."""
    fields, off = [], 0
    for m in mats:
        k = f"{pfx}.experts.0.{m}.trellis"
        n = src.loc[k][2]
        assert n % 256 == 0, (k, n)
        fields.append((f"{m}.trellis", off, n, src.loc[k][3], src.loc[k][4]))
        off += n
    trellis = off
    for m in mats:
        for s in ("suh", "svh"):
            k = f"{pfx}.experts.0.{m}.{s}"
            n = src.loc[k][2]
            fields.append((f"{m}.{s}", off, n, src.loc[k][3], src.loc[k][4]))
            off += n
    return fields, trellis, al(off)


def check_layer(src, pfx, E, fields, mats):
    """Every expert of the layer matches expert 0's layout; returns the set of mul1 markers (hex)."""
    mul1 = set()
    for e in range(E):
        for name, _, n, dt, shp in fields:
            v = src.loc[f"{pfx}.experts.{e}.{name}"]
            assert v[2] == n and v[3] == dt and v[4] == shp, (pfx, e, name, v, n, dt, shp)
        for m in mats:
            k = f"{pfx}.experts.{e}.{m}.mul1"
            if k in src.loc:
                fn, off, n, _, _ = src.loc[k]
                mul1.add(os.pread(src.fds[fn], n, off).hex())
    return mul1


def paths(out_dir, name):
    return os.path.join(str(out_dir), name + ".bin"), os.path.join(str(out_dir), name + ".json")


def plan(src, mats):
    """Per-layer layouts, bases and size classes (layers with identical field layouts share a class)."""
    layers, classes, base = [], [], 0
    E = src.layers[0][2]
    assert all(x[2] == E for x in src.layers), "every MoE layer must have the same expert count"
    for li, (ml, pfx, _) in enumerate(src.layers):
        fields, trellis, rec = layer_layout(src, pfx, mats)
        sig = [(n, o, b, d, list(s)) for n, o, b, d, s in fields]
        cls = next((c for c, k in enumerate(classes) if k["fields"] == sig), None)
        if cls is None:
            cls = len(classes)
            classes.append(dict(fields=sig, rec_bytes=rec, trellis_bytes=trellis, off_u=fields[1][1],
                                off_d=fields[2][1], layers=[]))
        classes[cls]["layers"].append(li)
        layers.append(dict(li=li, model_layer=ml, key=pfx, base=base, rec_bytes=rec, trellis_bytes=trellis,
                           off_u=fields[1][1], off_d=fields[2][1], cls=cls,
                           fields=[dict(name=n, offset=o, bytes=b, dtype=d, shape=list(s)) for n, o, b, d, s in fields]))
        base += E * rec
    return E, layers, classes, base


def pack(model_dir, out_dir, name, *, naming="deepseek", threads=8, log=print):
    mats = NAMINGS[naming]
    out_bin, out_meta = paths(out_dir, name)
    assert os.path.isdir(out_dir), f"out_dir {out_dir} does not exist"
    assert os.path.realpath(out_dir) != os.path.realpath(model_dir), "out_dir must not be the checkpoint directory"
    t0 = time.time()
    src = Source(model_dir)
    try:
        E, layers, classes, total = plan(src, mats)
        L = len(layers)
        mul1 = set()
        for Ly in layers:
            mul1 |= check_layer(src, Ly["key"], E, [(f["name"], f["offset"], f["bytes"], f["dtype"], tuple(f["shape"]))
                                                    for f in Ly["fields"]], mats)
        st = os.statvfs(out_dir)
        free = st.f_bavail * st.f_frsize
        log(f"layers {L} ({layers[0]['model_layer']}..{layers[-1]['model_layer']}), E {E}, classes "
            + ", ".join(f"[{c}] rec {k['rec_bytes']} B x {len(k['layers'])} layers" for c, k in enumerate(classes))
            + f", total {total / 1e9:.2f} GB, free {free / 1e12:.2f} TB, mul1 markers {sorted(mul1)}")
        assert total < free * 0.95, "not enough free space"
        fd = os.open(out_bin, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT, 0o644)
        os.posix_fallocate(fd, 0, total)
        hashes = [[None] * E for _ in range(L)]
        tl = threading.local()
        maxrec = max(Ly["rec_bytes"] for Ly in layers)

        def job(arg):
            li, e = arg
            Ly = layers[li]
            rec = Ly["rec_bytes"]
            if not hasattr(tl, "buf"):
                tl.buf = mmap.mmap(-1, maxrec)  # page-aligned, as an O_DIRECT source must be
            mv = memoryview(tl.buf)[:rec]
            end = 0
            for f in Ly["fields"]:
                src.read(f"{Ly['key']}.experts.{e}.{f['name']}", mv[f["offset"]:f["offset"] + f["bytes"]])
                end = f["offset"] + f["bytes"]
            mv[end:rec] = bytes(rec - end)
            hashes[li][e] = hashlib.sha256(mv).hexdigest()
            n = os.pwrite(fd, mv, Ly["base"] + e * rec)
            assert n == rec, (li, e, n)

        done = 0
        with cf.ThreadPoolExecutor(threads) as ex:
            for li, Ly in enumerate(layers):
                tl0 = time.time()
                list(ex.map(job, [(li, e) for e in range(E)]))
                nb = E * Ly["rec_bytes"]
                done += nb
                dt = max(time.time() - tl0, 1e-6)
                log(f"  layer {li:2d} (model {Ly['model_layer']}) {E} x {Ly['rec_bytes']} B = {nb / 1e9:.2f} GB in "
                    f"{dt:.1f} s ({nb / 1e9 / dt:.2f} GB/s), total {done / 1e9:.1f} GB, {time.time() - t0:.0f} s")
        os.fsync(fd)
        os.close(fd)
        meta = dict(format=FORMAT, naming=naming, mats=list(mats), model_dir=str(model_dir),
                    created=time.strftime("%Y-%m-%dT%H:%M:%S%z"), align=ALIGN, experts_per_layer=E, n_layers=L,
                    layers=layers, classes=classes, mul1_markers_hex=sorted(mul1), sha256=hashes, total_bytes=total)
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


def _picks(meta, sample, seed):
    L, E = meta["n_layers"], meta["experts_per_layer"]
    if sample is None:
        return [(li, e) for li in range(L) for e in range(E)]
    rng = random.Random(seed)
    s = set(rng.sample([(li, e) for li in range(L) for e in range(E)], min(sample, L * E)))
    for li in range(L):
        s.update(((li, 0), (li, E - 1)))
    return sorted(s)


def verify(out_dir, name, *, threads=8, sample=None, seed=0, log=print) -> list:
    """Every (or `sample` random + first / last per layer) record re-read with O_DIRECT and sha256-checked."""
    t0 = time.time()
    out_bin, _ = paths(out_dir, name)
    meta = manifest(out_dir, name)
    assert os.path.getsize(out_bin) == meta["total_bytes"], "store size != manifest"
    pick = _picks(meta, sample, seed)
    maxrec = max(Ly["rec_bytes"] for Ly in meta["layers"])
    fd = os.open(out_bin, os.O_RDONLY | os.O_DIRECT)
    tl = threading.local()
    bad, nbytes = [], [0]

    def job(arg):
        li, e = arg
        Ly = meta["layers"][li]
        if not hasattr(tl, "buf"):
            tl.buf = mmap.mmap(-1, maxrec)
        mv = memoryview(tl.buf)[:Ly["rec_bytes"]]
        n = os.preadv(fd, [mv], Ly["base"] + e * Ly["rec_bytes"])
        assert n == Ly["rec_bytes"]
        nbytes[0] += n
        if hashlib.sha256(mv).hexdigest() != meta["sha256"][li][e]:
            bad.append((li, e))
    try:
        with cf.ThreadPoolExecutor(threads) as ex:
            list(ex.map(job, pick))
    finally:
        os.close(fd)
    dt = max(time.time() - t0, 1e-6)
    log(f"VERIFY {'OK' if not bad else 'FAIL'}: {len(pick)} records, {nbytes[0] / 1e9:.2f} GB O_DIRECT re-read + sha256 "
        f"in {dt:.0f} s ({nbytes[0] / 1e9 / dt:.2f} GB/s); bad {len(bad)} {sorted(bad)[:20]}")
    return sorted(bad)


def compare(model_dir, out_dir, name, *, sample=384, seed=116, log=print) -> list:
    """Byte compare against fresh checkpoint reads (a different code path from the packer's buffer reuse)."""
    t0 = time.time()
    out_bin, _ = paths(out_dir, name)
    meta = manifest(out_dir, name)
    src = Source(model_dir)
    pick = _picks(meta, sample, seed)
    fd = os.open(out_bin, os.O_RDONLY | os.O_DIRECT)
    maxrec = max(Ly["rec_bytes"] for Ly in meta["layers"])
    buf = mmap.mmap(-1, maxrec)
    bad = []
    try:
        for li, e in pick:
            Ly = meta["layers"][li]
            rec = Ly["rec_bytes"]
            mv = memoryview(buf)[:rec]
            assert os.preadv(fd, [mv], Ly["base"] + e * rec) == rec
            end = 0
            for f in Ly["fields"]:
                fn, soff, sn, _, _ = src.loc[f"{Ly['key']}.experts.{e}.{f['name']}"]
                assert sn == f["bytes"]
                if bytes(mv[f["offset"]:f["offset"] + sn]) != os.pread(src.fds[fn], sn, soff):
                    bad.append((li, e, f["name"]))
                end = f["offset"] + sn
            if any(mv[end:rec]):
                bad.append((li, e, "pad"))
    finally:
        os.close(fd)
        src.close()
    log(f"CMP {'OK' if not bad else 'FAIL'}: {len(pick)} records byte-compared against the checkpoint (every field + "
        f"zero pad), {time.time() - t0:.0f} s; mismatches {len(bad)} {bad[:20]}")
    return bad


def expert_ranges_v2(model_dir, expert_re=r"^((?:.*\.)?layers\.(\d+)\.(?:mlp|ffn))\.experts\.(\d+)\.",
                     suffix=".trellis") -> dict:
    """{file: [(offset, bytes), ...]} of the routed-expert trellis tensors (what slimming punches out)."""
    src = Source(model_dir, expert_re)
    try:
        out = {}
        for k, (fn, off, n, _, _) in src.loc.items():
            if k.endswith(suffix):
                out.setdefault(fn, []).append((off, n))
        return {fn: sorted(v) for fn, v in out.items()}
    finally:
        src.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=["pack", "verify", "cmp", "plan"])
    ap.add_argument("--model", required=True, help="EXL3 checkpoint directory")
    ap.add_argument("--out-dir", required=True, help="store directory (local NVMe, O_DIRECT capable)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--naming", default="deepseek", choices=sorted(NAMINGS))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--seed", type=int, default=116)
    a = ap.parse_args(argv)
    lg = lambda m: print(m, flush=True)
    if a.mode == "plan":
        src = Source(a.model)
        E, layers, classes, total = plan(src, NAMINGS[a.naming])
        src.close()
        lg(f"E {E}, {len(layers)} layers, {total / 1e9:.2f} GB; classes: " + "; ".join(
            f"[{c}] rec {k['rec_bytes']} trellis {k['trellis_bytes']} layers {k['layers']}" for c, k in enumerate(classes)))
        return 0
    if a.mode == "pack":
        pack(a.model, a.out_dir, a.name, naming=a.naming, threads=a.threads, log=lg)
        return 0
    if a.mode == "verify":
        return 1 if verify(a.out_dir, a.name, threads=a.threads, sample=a.sample, log=lg) else 0
    return 1 if compare(a.model, a.out_dir, a.name, sample=a.sample or 384, seed=a.seed, log=lg) else 0


if __name__ == "__main__":
    sys.exit(main())
