from bigmoe.kernels import KernelSpec, kernel_dirs, kernel_key


def test_key_tracks_sources_and_toolchain(tmp_path):
    src = tmp_path / "mod.py"
    src.write_text("_SRC = 'a'")
    s = KernelSpec("k", "K_BUILD", "pkg.mod", "_ext", (src,))
    kw = dict(torch_version="2.13.0+cu130", python_tag="cpython-312", arch="8.6")
    k1 = kernel_key(s, **kw)
    assert k1 == kernel_key(s, **kw)
    assert k1 != kernel_key(s, **{**kw, "torch_version": "2.14.0+cu130"})
    assert k1 != kernel_key(s, **{**kw, "arch": "8.9"})
    src.write_text("_SRC = 'b'")
    assert k1 != kernel_key(s, **kw)
    assert kernel_dirs([s], tmp_path, **kw)["k"].name == f"k-{kernel_key(s, **kw)}"
