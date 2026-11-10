"""Server environment for image input (bigmoe/multimodal.py), shared by the backends' launch_spec."""

DISK_CACHE_MB = 2048   # image-embedding cache on disk


def multimodal_env(box, backend_name: str) -> dict:
    """BIGMOE_MM_* for the backends' `multimodal` and `image_tokens` options. The embedding cache's disk tier lives
    under the bigmoe home, one directory per backend (keys carry the checkpoint and the token budget)."""
    o = box.options
    if not o.multimodal:
        return {"BIGMOE_MM": "0"}
    env = {"BIGMOE_MM": "1", "BIGMOE_MM_DISK_DIR": str(box.home / "mm-cache" / backend_name),
           "BIGMOE_MM_DISK_MB": str(DISK_CACHE_MB)}
    if o.image_tokens:
        env["BIGMOE_MM_MAX_TOKENS"] = str(int(o.image_tokens))
    return env
