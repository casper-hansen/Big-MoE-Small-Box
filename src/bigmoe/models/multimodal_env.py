"""Server environment for image input (bigmoe/multimodal.py), shared by the backends' launch_spec."""


def multimodal_env(box, backend_name: str) -> dict:
    """BIGMOE_MM_* for the backends' `multimodal`, `image_tokens` and `mm_cache_mb` options. The embedding cache's disk
    tier lives under the bigmoe home, one directory per backend (keys carry the checkpoint and the token budget)."""
    o = box.options
    if not getattr(o, "multimodal", True):
        return {"BIGMOE_MM": "0"}
    env = {"BIGMOE_MM": "1"}
    if getattr(o, "image_tokens", None):
        env["BIGMOE_MM_MAX_TOKENS"] = str(int(o.image_tokens))
    mb = int(getattr(o, "mm_cache_mb", 2048) or 0)
    if mb > 0:
        env["BIGMOE_MM_DISK_DIR"] = str(box.home / "mm-cache" / backend_name)
        env["BIGMOE_MM_DISK_MB"] = str(mb)
    return env
