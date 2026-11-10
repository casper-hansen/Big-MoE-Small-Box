"""Image input (bigmoe/multimodal.py): message parsing, alias substitution, the content-keyed embedding cache and the
backends' launch environment. No GPU: the vision tower itself is exercised by the servers' startup self-test."""
import asyncio
import base64
import io
import os

import pytest

from bigmoe import AutoBox, multimodal as mm, system
from bigmoe.serving import RequestError

PNG = None


def png_bytes(color=(200, 30, 30), size=(64, 48)):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def data_url(b, mime="image/png"):
    return f"data:{mime};base64," + base64.b64encode(b).decode()


def test_image_sources_cover_openai_anthropic_and_responses_blocks():
    msgs = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": [
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA", "detail": "high"}},
            {"type": "image_url", "image_url": "https://example.com/a.png"},
            {"type": "input_image", "image_url": "data:image/jpeg;base64,BBBB"},
        ]},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/webp", "data": "CCCC"}},
            {"type": "tool_result", "tool_use_id": "t1", "content": [
                {"type": "text", "text": "screenshot:"},
                {"type": "image", "source": {"type": "url", "url": "https://example.com/b.png"}}]},
        ]},
    ]
    assert mm.has_images(msgs)
    assert mm.image_sources(msgs) == ["data:image/png;base64,AAAA", "https://example.com/a.png",
                                      "data:image/jpeg;base64,BBBB", "data:image/webp;base64,CCCC",
                                      "https://example.com/b.png"]
    out = mm.with_aliases(msgs, ["<A>", "<B>", "<C>", "<D>", "<E>"])
    assert out[1]["content"][1:] == [{"type": "text", "text": "<A>"}, {"type": "text", "text": "<B>"},
                                     {"type": "text", "text": "<C>"}]
    assert out[3]["content"][0] == {"type": "text", "text": "<D>"}
    assert out[3]["content"][1]["content"][1] == {"type": "text", "text": "<E>"}
    assert msgs[1]["content"][1]["type"] == "image_url"          # the request's own messages are not modified
    assert not mm.has_images(out)


def test_text_only_messages_pass_through_untouched():
    msgs = [{"role": "user", "content": "hi"}, {"role": "user", "content": [{"type": "text", "text": "x"}]},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]}]
    assert not mm.has_images(msgs)
    got, embs = asyncio.run(mm.prepare_chat(None, msgs))
    assert got is msgs and embs == []


def test_images_without_a_vision_tower_are_a_400():
    msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]}]
    with pytest.raises(RequestError) as e:
        asyncio.run(mm.prepare_chat(None, msgs))
    assert e.value.status == 400 and e.value.code == "image_input_not_supported"


def test_bad_image_blocks_are_400s(monkeypatch):
    with pytest.raises(RequestError):
        mm.image_source({"type": "image_url", "image_url": {}})
    with pytest.raises(RequestError):
        mm.fetch_bytes("ftp://x/y.png", 1000)
    with pytest.raises(RequestError):
        mm.fetch_bytes("data:image/png;base64", 1000)                # no comma
    with pytest.raises(RequestError):
        mm.fetch_bytes(data_url(b"x" * 5000), 1000)                   # too large
    with pytest.raises(RequestError):
        mm.fetch_bytes("file:///etc/hostname", 1000)                  # file URLs are off by default
    monkeypatch.setenv("BIGMOE_MM_FETCH", "0")
    with pytest.raises(RequestError):
        mm.fetch_bytes("https://example.com/a.png", 1000)
    with pytest.raises(RequestError):
        mm.open_image(b"not an image")


def test_fetch_and_open_a_data_url():
    b = png_bytes()
    assert mm.fetch_bytes(data_url(b), 10 ** 6) == b
    im = mm.open_image(b)
    assert im.size == (64, 48)


def test_file_urls_when_enabled(tmp_path, monkeypatch):
    p = tmp_path / "a.png"
    p.write_bytes(png_bytes())
    monkeypatch.setenv("BIGMOE_MM_FILES", "1")
    assert mm.fetch_bytes(f"file://{p}", 10 ** 6) == p.read_bytes()


def make_mme(n=6, dim=8, fill=1.0):
    torch = pytest.importorskip("torch")
    pytest.importorskip("exllamav3")
    from exllamav3.tokenizer import MMEmbedding
    ts = torch.tensor([[7] + [-1] * n + [9]], dtype=torch.long)       # start / end marker ids around the rows
    return MMEmbedding(embeddings=torch.full((n, dim), fill, dtype=torch.half), token_string=ts)


def test_rebase_moves_only_the_placeholder_ids():
    a, b = make_mme(), make_mme()
    assert b.first_index != a.first_index
    mm.rebase(b, a.first_index)
    assert b.token_list == a.token_list and b.text_alias == a.text_alias
    assert b.token_list[0] == 7 and b.token_list[-1] == 9
    assert (b.first_index, b.last_index) == (a.first_index, a.last_index)


def test_cache_keeps_ids_per_content_key_and_evicts_by_bytes(tmp_path):
    one = make_mme().embeddings.numel() * 2
    c = mm.EmbeddingCache(ram_bytes=2 * one, disk_dir=None)
    first = c.put("k1", make_mme())
    ids = first.token_list
    c.put("k2", make_mme())
    c.put("k3", make_mme())                    # over budget: k1 (least recent) leaves RAM
    assert c.get("k1") is None and c.get("k3") is not None
    again = c.put("k1", make_mme())            # re-encoded: same ids as the first time, so prefix pages match
    assert again.token_list == ids
    assert c.summary()["keys"] == 3


def test_disk_tier_round_trip_and_size_cap(tmp_path):
    torch = pytest.importorskip("torch")
    d = tmp_path / "mm"
    one = make_mme(n=64, dim=64).embeddings.numel() * 2
    c = mm.EmbeddingCache(ram_bytes=one, disk_dir=str(d), disk_bytes=10 ** 9)
    e = c.put("k1", make_mme(n=64, dim=64, fill=3.0))
    c.write_disk("k1", e)
    c.put("k2", make_mme(n=64, dim=64))        # pushes k1 out of RAM
    assert "k1" not in c.lru
    back = c.get("k1")                          # from disk, with the process's ids for k1
    assert back is not None and back.token_list == e.token_list
    assert torch.equal(back.embeddings, e.embeddings)
    assert c.stats["disk_hits"] == 1
    # a new process (no ids yet) gets fresh ids for the record
    c2 = mm.EmbeddingCache(ram_bytes=one, disk_dir=str(d), disk_bytes=10 ** 9)
    fresh = c2.get("k1")
    assert fresh is not None and torch.equal(fresh.embeddings, e.embeddings)
    # the size cap trims the oldest records
    small = mm.EmbeddingCache(ram_bytes=one, disk_dir=str(d), disk_bytes=1)
    small.write_disk("k9", make_mme(n=64, dim=64))
    assert len([f for f in os.listdir(d) if f.endswith(".pt")]) <= 1
    # a damaged record is a miss, and is removed
    (d / "bad.pt").write_bytes(b"junk")
    assert c2.get("bad") is None and not (d / "bad.pt").exists()


def test_text_of_for_fallback_templates():
    assert mm.text_of("abc") == "abc"
    assert mm.text_of([{"type": "text", "text": "a"}, {"type": "image_url"}, {"type": "text", "text": "b"}]) == "ab"
    assert mm.text_of(None) == ""


def test_model_card_fields():
    assert mm.model_card_fields(None)["input_modalities"] == ["text"]
    assert mm.model_card_fields(object())["capabilities"] == {"vision": True}


class FakeRuntime:
    from pathlib import Path
    python = Path("/envs/x/bin/python")

    def kernel_dirs(self, specs, arch):
        from pathlib import Path
        return {s.name: Path(f"/k/{s.name}") for s in specs}


@pytest.mark.parametrize("model", ["glm-5.3-flash", "deepseek-v4.1-flash"])
def test_launch_env_turns_multimodal_on_by_default(tmp_path, monkeypatch, model):
    monkeypatch.setattr(system, "systemd_scope_available", lambda: True)
    monkeypatch.setattr(system, "cpu_flags", lambda: {"avx2", "fma", "f16c", "avx512f", "avx512bw", "avx512vl",
                                                       "avx512vbmi", "avx512_vnni"})

    def spec(**kw):
        box = AutoBox(model, home=tmp_path, cpus={"main": "0", "controller": 6, "lane": "1-5", "readers": "7-11"}, **kw)
        box.__dict__["runtime"] = FakeRuntime()
        box.__dict__["gpu"] = system.GPU(0, "NVIDIA GeForce RTX 3090", 24576, 0, "8.6")
        return box.backend.launch_spec(box).env

    env = spec()
    assert env["BIGMOE_MM"] == "1" and "BIGMOE_MM_MAX_TOKENS" not in env
    assert env["BIGMOE_MM_DISK_DIR"].startswith(str(tmp_path)) and env["BIGMOE_MM_DISK_MB"] == "2048"
    assert spec(image_tokens=512)["BIGMOE_MM_MAX_TOKENS"] == "512"
    assert spec(multimodal=False)["BIGMOE_MM"] == "0"
    with pytest.raises(ValueError):
        spec(image_tokens=4)


def test_client_puts_images_before_the_last_user_text(tmp_path):
    from bigmoe.client import _messages, image_block
    p = tmp_path / "a.png"
    p.write_bytes(png_bytes())
    m = _messages("what is this?", [str(p), "https://example.com/b.png"])
    assert [b["type"] for b in m[0]["content"]] == ["image_url", "image_url", "text"]
    assert m[0]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert mm.fetch_bytes(m[0]["content"][0]["image_url"]["url"], 10 ** 6) == p.read_bytes()
    hist = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    m = _messages(hist + [{"role": "user", "content": "and this?"}], [png_bytes()])
    assert m[:2] == hist and m[2]["content"][-1] == {"type": "text", "text": "and this?"}
    assert hist[-1] == {"role": "assistant", "content": "hello"}
    with pytest.raises(TypeError):
        image_block(42)
