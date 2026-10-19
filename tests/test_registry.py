import json

import pytest

from bigmoe import AutoBox, BoxError, ModelBackend, UnknownModelError
from bigmoe import registry
from bigmoe.registry import resolve

COMMIT = "332ab457b709b7ba30dd9a448be5de03b80a7ac9"


def test_alias_resolves_to_the_pinned_commit(tmp_path):
    b, ref = resolve("glm-5.3-flash")
    assert b.name == "glm53" and ref.repo_id == "turboderp/GLM-5.3-Flash-exl3" and ref.revision == "3.05bpw"
    box = AutoBox("glm-5.3-flash", home=tmp_path)
    assert box.commit == COMMIT and box.context == 131072 and box.port == 30000
    assert str(box.backend.model_dir(box)).endswith(f"turboderp--GLM-5.3-Flash-exl3@{COMMIT[:12]}")


def test_repo_id_and_commit(tmp_path):
    assert resolve("turboderp/GLM-5.3-Flash-exl3")[0].name == "glm53"
    assert AutoBox("turboderp/GLM-5.3-Flash-exl3", revision=COMMIT, home=tmp_path).commit == COMMIT


def test_unvalidated_variant_is_refused(tmp_path):
    with pytest.raises(BoxError, match="not supported"):
        AutoBox("glm-5.3-flash", revision="2.05bpw", home=tmp_path)


def test_options_are_validated(tmp_path):
    with pytest.raises(TypeError, match="unknown option"):
        AutoBox("glm-5.3-flash", home=tmp_path, nonsense=1)
    with pytest.raises(ValueError, match="mode"):
        AutoBox("glm-5.3-flash", home=tmp_path, mode="fast")
    box = AutoBox("glm-5.3-flash", home=tmp_path, mode="nvme-exact", vram_expert_slots=None)
    assert box.options.mode == "nvme-exact" and box.options.vram_expert_slots is None


def test_local_checkpoint_dir(tmp_path):
    d = tmp_path / "ckpt"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "glm5_next",
                                               "quantization_config": {"quant_method": "exl3", "bits": 3.05}}))
    b, ref = resolve(str(d))
    assert b.name == "glm53" and ref.local_dir == d.resolve()
    (d / "config.json").write_text(json.dumps({"model_type": "llama"}))
    with pytest.raises(UnknownModelError):
        resolve(str(d))


def test_unknown_name():
    with pytest.raises(UnknownModelError):
        resolve("no-such-model")


def test_custom_backend_plugs_in(tmp_path):
    class Dummy(ModelBackend):
        name = "dummy"
        repos = ("me/tiny-moe",)
        aliases = {"tiny": ("me/tiny-moe", "0" * 40)}

    registry.register_backend(Dummy)
    try:
        b, ref = resolve("tiny")
        assert b.name == "dummy" and ref.revision == "0" * 40
        assert AutoBox("tiny", home=tmp_path).commit == "0" * 40
        assert resolve("me/tiny-moe")[0].name == "dummy"
    finally:
        registry._REGISTRY.pop("dummy")


def test_import_is_light():
    import subprocess
    import sys
    code = "import sys, bigmoe; bigmoe.list_backends(); assert 'torch' not in sys.modules and 'numpy' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
