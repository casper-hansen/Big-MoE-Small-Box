import ctypes
import ctypes.util
import glob
import json
import os
import re
import struct

PUNCH = 0x02 | 0x01   # FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE
TRELLIS = re.compile(r"^(.*)\.experts\.\d+\.(gate|up|down)_proj\.trellis$")


def expert_ranges(path, layer_keys):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    rng = []
    for k, v in hdr.items():
        m = TRELLIS.match(k)
        if m and m.group(1) in layer_keys:
            a, b = v["data_offsets"]
            rng.append([8 + n + a, 8 + n + b])
    rng.sort()
    merged = []
    for a, b in rng:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return len(rng), merged


def punch_experts(model_dir, layer_keys, *, dry_run=False, log=print) -> dict:
    """fallocate PUNCH_HOLE | KEEP_SIZE over the expert trellis tensors of `layer_keys`: the files keep their size and
    every other tensor, so the offload engine loads them unchanged, but the punched bytes read back as zeros. After this
    the checkpoint can neither rebuild the store nor run on stock exllamav3."""
    layer_keys = set(layer_keys)
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    libc.fallocate.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_long, ctypes.c_long]
    out, tot_n = {}, 0
    for path in sorted(glob.glob(os.path.join(str(model_dir), "*.safetensors"))):
        n_tensors, merged = expert_ranges(path, layer_keys)
        if not merged:
            continue
        nb = sum(b - a for a, b in merged)
        tot_n += n_tensors
        if not dry_run:
            fd = os.open(path, os.O_RDWR)
            try:
                for a, b in merged:
                    if libc.fallocate(fd, PUNCH, a, b - a) != 0:
                        e = ctypes.get_errno()
                        raise OSError(e, f"fallocate punch-hole failed on {path}: {os.strerror(e)}")
            finally:
                os.close(fd)
        st = os.stat(path)
        out[os.path.basename(path)] = nb
        log(f"{os.path.basename(path)}: {n_tensors} expert trellis tensors, {nb / 1e9:.2f} GB "
            f"{'would be ' if dry_run else ''}freed; on disk now {st.st_blocks * 512 / 1e9:.2f} of {st.st_size / 1e9:.2f} GB")
    log(f"total: {tot_n} tensors, {sum(out.values()) / 1e9:.1f} GB {'to free (dry run)' if dry_run else 'freed'}")
    return out
