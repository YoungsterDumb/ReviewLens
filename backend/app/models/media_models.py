"""Pretrained media models: Whisper (speech), CLIP (alignment) + BLIP (VQA), and the CVAE decoder."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import numpy as np
from PIL import Image

from ..core.cvae import CATEGORIES, LATENT_DIM, condition_vector

SAMPLE_RATE = 16_000

# FashionMNIST classes first (their index is the CVAE category), then "other" products.
CATEGORY_PROMPTS = [
    "a t-shirt", "a pair of trousers", "a pullover sweater", "a dress", "a coat",
    "a sandal", "a shirt", "a sneaker", "a bag", "an ankle boot",
]
OTHER_PROMPTS = [
    "a smartphone", "a laptop", "headphones", "an electronic device",
    "a cosmetic product", "food", "a piece of furniture", "a toy",
]
CONDITION_PROMPTS = (
    "a photo of a damaged, broken, torn or defective product",
    "a photo of a clean product in good, new condition",
)
AUTO_QUESTIONS = ("What is this?", "What color is it?", "Is it damaged?")


def _embedding(out):
    """CLIP features: a tensor in transformers 4.x, an output whose pooler_output is projected in 5.x."""
    return out.pooler_output if hasattr(out, "pooler_output") else out


class SpeechToText:
    def __init__(self, model_id: str):
        from transformers import pipeline

        self.pipe = pipeline("automatic-speech-recognition", model=model_id, chunk_length_s=30, device=-1)
        self._lock = threading.Lock()

    @staticmethod
    def decode(audio_bytes: bytes) -> np.ndarray:
        from transformers.pipelines.audio_utils import ffmpeg_read

        return ffmpeg_read(audio_bytes, SAMPLE_RATE)

    def transcribe(self, audio: np.ndarray) -> str:
        with self._lock:  # the pipeline is not thread-safe
            out = self.pipe({"raw": audio, "sampling_rate": SAMPLE_RATE}, generate_kwargs={"task": "transcribe"})
        return str(out["text"]).strip()


class VisionModels:
    """CLIP for zero-shot category / condition / aspect alignment, BLIP for VQA."""

    def __init__(self, clip_id: str, blip_id: str):
        self.clip_id, self.blip_id = clip_id, blip_id
        self._clip = self._blip = None
        self._text_cache: dict[tuple[str, ...], np.ndarray] = {}
        self._lock = threading.Lock()

    # --- CLIP -------------------------------------------------------------
    def _load_clip(self):
        if self._clip is None:
            from transformers import CLIPModel, CLIPProcessor

            self._clip = (CLIPModel.from_pretrained(self.clip_id).eval(), CLIPProcessor.from_pretrained(self.clip_id))
        return self._clip

    def _text_features(self, prompts: tuple[str, ...]) -> np.ndarray:
        import torch

        if prompts not in self._text_cache:
            model, proc = self._load_clip()
            enc = proc(text=list(prompts), return_tensors="pt", padding=True)
            with torch.inference_mode():
                f = _embedding(model.get_text_features(**enc))
            f = f / f.norm(dim=-1, keepdim=True)
            self._text_cache[prompts] = f.cpu().numpy()
        return self._text_cache[prompts]

    def _image_features(self, images: list) -> np.ndarray:
        import torch

        model, proc = self._load_clip()
        feats = []
        for b in range(0, len(images), 32):
            enc = proc(images=images[b:b + 32], return_tensors="pt")
            with torch.inference_mode():
                f = _embedding(model.get_image_features(**enc))
            feats.append((f / f.norm(dim=-1, keepdim=True)).cpu().numpy())
        return np.vstack(feats)

    def _zero_shot(self, images: list, prompts: tuple[str, ...]) -> np.ndarray:
        model, _ = self._load_clip()
        scale = float(model.logit_scale.exp().item())
        logits = scale * self._image_features(images) @ self._text_features(prompts).T
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def category(self, image: Image.Image) -> tuple[int | None, float]:
        prompts = tuple(f"a photo of {p}" for p in CATEGORY_PROMPTS + OTHER_PROMPTS)
        p = self._zero_shot([image], prompts)[0]
        k = int(p.argmax())
        return (k if k < len(CATEGORY_PROMPTS) else None), float(p[k])

    def condition_proba(self, images: list) -> np.ndarray:
        """n x 2 probabilities: [defective, good]."""
        return self._zero_shot(images, CONDITION_PROMPTS)

    def align(self, image: Image.Image, terms: list[str]) -> list[float]:
        if not terms:
            return []
        img = self._image_features([image])[0]
        txt = self._text_features(tuple(f"a photo of the {t} of a product" for t in terms))
        return [float(x) for x in txt @ img]

    # --- BLIP -------------------------------------------------------------
    def _load_blip(self):
        if self._blip is None:
            from transformers import BlipForQuestionAnswering, BlipProcessor

            self._blip = (BlipForQuestionAnswering.from_pretrained(self.blip_id).eval(), BlipProcessor.from_pretrained(self.blip_id))
        return self._blip

    def vqa(self, image: Image.Image, question: str) -> tuple[str, float | None]:
        import torch

        model, proc = self._load_blip()
        enc = proc(image, question, return_tensors="pt")
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=12)
        return proc.decode(out[0], skip_special_tokens=True).strip(), None

    def warm(self) -> None:
        self._load_clip()
        self._load_blip()


class VariantGenerator:
    """Decoder half of the conditional VAE, run with Keras on the torch backend."""

    def __init__(self, assets_dir: str):
        os.environ.setdefault("KERAS_BACKEND", "torch")
        import keras

        from ..core import cvae as _  # noqa: F401  registers custom layers

        root = Path(assets_dir)
        self.decoder = keras.saving.load_model(root / "decoder.keras", compile=False)
        meta = json.loads((root / "anchors.json").read_text(encoding="utf-8"))
        self.anchors = {k: np.asarray(meta["anchors"][k], dtype=np.float32) for k in ("negative", "neutral", "positive")}
        self.latent_dim = int(meta.get("latent_dim", LATENT_DIM))
        self._lock = threading.Lock()

    def _decode(self, z: np.ndarray, cond: np.ndarray) -> np.ndarray:
        import keras

        with self._lock:
            out = self.decoder([z.astype(np.float32), cond.astype(np.float32)], training=False)
        return np.clip(keras.ops.convert_to_numpy(out)[..., 0], 0.0, 1.0)

    def generate(self, category: int, probs: dict[str, float], n: int, seed: int) -> list[np.ndarray]:
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((n, self.latent_dim)).astype(np.float32)
        cond = np.tile(condition_vector(category, probs, self.anchors), (n, 1))
        return list(self._decode(z, cond))

    def sweep(self, category: int, seed: int, steps: int = 7) -> list[np.ndarray]:
        rng = np.random.default_rng(seed)
        z = np.tile(rng.standard_normal((1, self.latent_dim)).astype(np.float32), (steps, 1))
        conds = []
        for i in range(steps):
            t = i / (steps - 1)
            conds.append(condition_vector(category, {"negative": 1 - t, "neutral": 0.0, "positive": t}, self.anchors))
        return list(self._decode(z, np.vstack(conds)))


def category_name(index: int | None) -> str | None:
    return None if index is None else CATEGORIES[index]
