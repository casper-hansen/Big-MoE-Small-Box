from bigmoe.runtime import RuntimeSpec


def test_key_ignores_comments(tmp_path):
    req, build = tmp_path / "runtime.txt", tmp_path / "build.txt"
    req.write_text("# pins\ntorch==2.13.0\n")
    build.write_text("ninja==1.13.2\n")
    key = RuntimeSpec("x", "3.12", req, build).key()
    req.write_text("torch==2.13.0  # measured\n\n")
    assert RuntimeSpec("x", "3.12", req, build).key() == key
    req.write_text("torch==2.14.0\n")
    assert RuntimeSpec("x", "3.12", req, build).key() != key
