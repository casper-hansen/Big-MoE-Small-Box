from bigmoe import log
from bigmoe.models import prefix_cache as pc


def test_cpu_tier_gb(monkeypatch):
    warned = []
    monkeypatch.setattr(log, "warn", lambda msg, *a: warned.append(msg % a))
    assert pc.cpu_tier_gb(None, None, 65536, 3209) == 0.0 and pc.cpu_tier_gb(0.5, None, 65536, 3209) == 0.5
    assert pc.cpu_tier_gb(None, "/d", 65536, 3209) == 0.25          # 0.216 GiB -> 0.25
    assert pc.cpu_tier_gb(None, "/d", 131072, 17609) == 2.5          # 2.36 GiB -> 2.5
    assert pc.cpu_tier_gb(1.0, "/d", 131072, 17609) == 1.0     # honoured, with a warning: ~55k of 128k tokens
    assert len(warned) == 1 and "prefixes of up to ~55" in warned[0]
    assert pc.cpu_tier_gb(0.25, "/d", 65536, 3209) == 0.25 and len(warned) == 1   # enough: no warning
