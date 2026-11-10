"""Launch-time sizing of exllamav3's prefix-cache tiers (the backends' cpu_cache_gb / disk_cache options).

exllamav3 restores a prefix from its disk tier by promoting every page of it into the pinned-RAM tier before copying
it to the GPU, and promotions never evict each other: the RAM tier caps the longest prefix the disk can give back,
and a prefix longer than that comes back not at all.
"""
import math

from .. import log

GiB = 1024 ** 3
HEADROOM = 1.1


def cpu_tier_gb(cpu_cache_gb, disk_cache, context: int, kv_bytes_per_token: int) -> float:
    """-ccs for a launch. Without disk_cache: cpu_cache_gb, or 0. With it: cpu_cache_gb, or (None) enough for the
    whole context (+10 %, rounded up to 0.25 GiB); an explicit size below that is honoured with a warning."""
    if not disk_cache:
        return float(cpu_cache_gb or 0.0)
    need = context * kv_bytes_per_token * HEADROOM / GiB
    if cpu_cache_gb is None:
        return math.ceil(need * 4) / 4
    if float(cpu_cache_gb) < need:
        tokens = int(float(cpu_cache_gb) * GiB / HEADROOM / kv_bytes_per_token)
        log.warn("cpu_cache_gb=%g: the disk cache gives back prefixes of up to ~%d tokens (the %d-token context needs "
                 "%.2f GiB; leave cpu_cache_gb unset to size it for the context)", float(cpu_cache_gb), tokens, context,
                 need)
    return float(cpu_cache_gb)
