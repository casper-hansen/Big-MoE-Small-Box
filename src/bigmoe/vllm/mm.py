"""Image input through vLLM: the stub models' multimodal processor (API server) and the worker side.

API server (vLLM's chat parsing / media loading -> this processor):
  each image is preprocessed with exllamav3's own preprocess() (resize / normalize / patchify, the reference
  preprocessing; exllamav3 is imported on the first image only, ~0.1 GB) and its prompt placeholder is expanded to the
  tokens the embedding will occupy:
    DeepSeek-V4.1  the template's <｜deepseek_image｜> -> 2 + h * (w + 1) positions, all embedding rows (the block's
                   start / newline / end markers are learned embeddings too)
    GLM-5.3        <|begin_of_image|><|image|><|end_of_image|> (the overlay template renders image blocks so,
                   serve.py) -> <|image|> becomes t * h * w / merge^2 embedding rows between the real start / end tokens
  vLLM sends the preprocessed tensors (feature `bx_pixels` + `bx_meta`) and the item's content hash to the engine.

Worker (engine.ExlEngine.add -> embed()): the content hash keys the embedding cache (RAM, then disk; multimodal.Vision);
a miss loads the tower, encodes and unloads it (VRAM lent by the expert cache when short). The placeholder positions
are replaced by the embedding's ids (fixed per content hash for the process lifetime), so a resent image gives the same
token ids and the prefix cache reuses the prompt through it, as in the native servers.
"""
import threading
from collections.abc import Mapping

import torch

from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.multimodal.processing import (BaseDummyInputsBuilder, BaseMultiModalProcessor, BaseProcessingInfo,
                                        PromptReplacement, PromptUpdateDetails)

DS_IMAGE = "<｜deepseek_image｜>"
MAX_IMAGES = 16
_TOWERS = {}
_LOCK = threading.Lock()


def _backend(hf_config) -> str:
    return "dsv41" if getattr(hf_config, "model_type", "") == "deepseek_v41" else "glm53"


def tower(model_dir: str):
    """exllamav3's vision module for the checkpoint, constructed without weights: preprocess() reads only the config."""
    with _LOCK:
        t = _TOWERS.get(model_dir)
        if t is None:
            from exllamav3 import Config, Model
            from .. import multimodal
            config = Config.from_directory(model_dir)
            t = Model.from_config(config, component="vision")
            n = multimodal.env_int("BIGMOE_MM_MAX_TOKENS", 0)
            if n:   # the per-image token budget (multimodal.Vision._apply_token_budget, same rule)
                pp = getattr(config, "vision_pp", None)
                if pp is not None and hasattr(pp, "max_image_tokens"):
                    pp.max_image_tokens = max(int(getattr(pp, "min_image_tokens", 16) or 16), n)
                elif getattr(config, "vision", None) is not None and hasattr(config.vision, "max_n_token"):
                    config.vision.max_n_token = max(4, n)
            _TOWERS[model_dir] = t
        return t


def max_image_tokens(hf_config) -> int:
    """Upper bound of one image's prompt positions (encoder budget)."""
    import os
    n = int(os.environ.get("BIGMOE_MM_MAX_TOKENS") or 0)
    if _backend(hf_config) == "dsv41":
        vc = getattr(hf_config, "vision_config", None)
        get = (lambda k, d: vc.get(k, d)) if isinstance(vc, dict) else (lambda k, d: getattr(vc, k, d))
        return (n or int(get("max_image_tokens", 1024))) + 2
    return (n or 8000) + 2


def prepare(model_dir: str, backend: str, image):
    """(tensor, meta ints, prompt positions) of one PIL image."""
    t = tower(model_dir)
    w, h = image.size
    if backend == "dsv41":
        patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = type(t).preprocess(t, image)
        meta = [n_vit_h, n_vit_w, n_llm_h, n_llm_w, w, h]
        return patches, meta, 2 + n_llm_h * (n_llm_w + 1)
    x, (pw, ph), (gt, gh, gw) = type(t).preprocess(t, image)
    m = t.config.vision.spatial_merge_size
    meta = [pw, ph, gt, gh, gw, w, h]
    return x, meta, gt * gh * gw // (m * m)


def prepped_of(backend: str, pixels, meta):
    """The preprocess() return value rebuilt from the IPC fields, and the original (width, height)."""
    meta = [int(v) for v in (meta.tolist() if hasattr(meta, "tolist") else meta)]
    if backend == "dsv41":
        return (pixels, meta[0], meta[1], meta[2], meta[3]), (meta[4], meta[5])
    return (pixels, (meta[0], meta[1]), (meta[2], meta[3], meta[4])), (meta[5], meta[6])


class BoxProcessingInfo(BaseProcessingInfo):

    def get_supported_mm_limits(self):
        return {"image": MAX_IMAGES}

    def get_mm_max_tokens_per_item(self, seq_len, mm_counts):
        return {"image": max_image_tokens(self.get_hf_config())}

    def placeholder_ids(self) -> tuple[int, int]:
        """(prompt token the template emits for an image, token id repeated over its embedding rows)."""
        tok = self.get_tokenizer()
        if _backend(self.get_hf_config()) == "dsv41":
            i = tok.convert_tokens_to_ids(DS_IMAGE)
            return i, i
        i = int(self.get_hf_config().image_token_id)
        return i, i


class BoxDummyInputsBuilder(BaseDummyInputsBuilder[BoxProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        tok = self.info.get_tokenizer()
        target, _ = self.info.placeholder_ids()
        return tok.decode([target]) * mm_counts.get("image", 0)

    def get_dummy_mm_data(self, seq_len, mm_counts, mm_options=None):
        return {"image": self._get_dummy_images(width=640, height=480, num_images=mm_counts.get("image", 0),
                                                overrides=(mm_options or {}).get("image") if mm_options else None)}


class BoxMMProcessor(BaseMultiModalProcessor[BoxProcessingInfo]):

    def _call_hf_processor(self, hf_data, hf_kwargs):
        from transformers.feature_extraction_utils import BatchFeature
        images = hf_data.get("images") or []
        if not isinstance(images, (list, tuple)):
            images = [images]
        mc = self.info.ctx.model_config
        backend = _backend(self.info.get_hf_config())
        pix, meta, ntok = [], [], []
        for im in images:
            x, m, n = prepare(mc.model, backend, im)
            pix.append(x)
            meta.append(torch.tensor(m, dtype=torch.int64))
            ntok.append(n)
        return BatchFeature({"bx_pixels": pix, "bx_meta": meta,
                             "bx_ntok": torch.tensor(ntok, dtype=torch.int64)}, tensor_type=None)

    def _hf_processor_applies_updates(self, prompt_text, mm_items, hf_processor_mm_kwargs, tokenization_kwargs):
        return False

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        return {"bx_pixels": MultiModalFieldConfig.batched("image"),
                "bx_meta": MultiModalFieldConfig.batched("image"),
                "bx_ntok": MultiModalFieldConfig.batched("image")}

    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        target, pad = self.info.placeholder_ids()
        items = out_mm_kwargs.get("image", []) if hasattr(out_mm_kwargs, "get") else []

        def replacement(i):
            n = int(items[i]["bx_ntok"].data)
            return PromptUpdateDetails.select_token_id([pad] * n, embed_token_id=pad)
        return [PromptReplacement(modality="image", target=[target], replacement=replacement)]


def attach(cls):
    """Register the processor for a stub class and give it the SupportsMultiModal surface."""
    cls.supports_multimodal = True
    cls.merge_by_field_config = True

    @classmethod
    def get_placeholder_str(klass, modality: str, i: int):
        if modality.startswith("image"):
            return DS_IMAGE if klass.__name__.startswith("BigmoeDeepseek") else \
                "<|begin_of_image|><|image|><|end_of_image|>"
        raise ValueError("bigmoe: only image input is supported")
    cls.get_placeholder_str = get_placeholder_str
    return MULTIMODAL_REGISTRY.register_processor(BoxMMProcessor, info=BoxProcessingInfo,
                                                  dummy_inputs=BoxDummyInputsBuilder)(cls)


# ---- worker side ---------------------------------------------------------------------------------------------------
def features_of(nr) -> list:
    """[(identifier, offset, length, pixels, meta)] of a NewRequestData's image features, in prompt order."""
    out = []
    for f in sorted(nr.mm_features or [], key=lambda f: f.mm_position.offset):
        if f.modality != "image":
            raise ValueError(f"bigmoe: {f.modality} input is not supported")
        if f.data is None:
            raise RuntimeError("bigmoe: multimodal item without data (processor cache out of sync)")
        d = f.data.get_data()
        out.append((f.identifier, int(f.mm_position.offset), int(f.mm_position.length), d["bx_pixels"],
                    d["bx_meta"]))
    return out


def splice(prompt: list, feats: list, embs: list) -> list:
    """The prompt with each feature's embedding rows given its embedding's ids. DeepSeek's token string is all
    embedding rows; GLM's is <start> rows <end> around a range that covers only the rows."""
    ids = list(prompt)
    for (_ident, off, n, _p, _m), mme in zip(feats, embs):
        tl = list(mme.token_list)
        rows = [t for t in tl if t >= mme.first_index]
        if len(tl) == n:
            ids[off:off + n] = tl
        elif len(rows) == n:
            ids[off:off + n] = rows
        else:
            raise ValueError(f"bigmoe: image embedding has {len(rows)} rows, the prompt holds {n} positions")
    return ids
