"""GLM-5.3-Flash server and its model-specific accelerators (from glm-flash-lite 6fe077d: fused hyper-connection
sites, decode lookahead, Triton pins, exllamav3 prompt patches, n-gram policy). The expert offload is the shared engine
(bigmoe.engine); server.py drops the upstream modules not included here."""
