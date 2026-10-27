"""Deterministic stand-ins with the same interfaces as the real models.

Used by the test suite and for working on the UI without model weights
(REVIEWLENS_STUB=true). /health reports mode="stub" so the UI can say so.
"""
from __future__ import annotations

import re

import numpy as np
from PIL import Image

from ..core.cvae import brightness_gain, sentiment_score
from ..core.text import word_spans

ASPECT_WORDS = [
    "battery life", "battery", "screen", "keyboard", "camera", "price", "delivery", "service",
    "fan", "sound", "quality", "food", "staff", "customer support",
    "giao hàng", "đóng gói", "đường may", "màn hình", "màu", "giá", "pin", "đế", "camera", "loa",
]
POSITIVE = {"good", "great", "excellent", "love", "fast", "helpful", "gorgeous", "nice", "tốt", "nhanh",
            "êm", "đẹp", "cẩn", "thích", "tuyệt", "trâu", "ổn"}
NEGATIVE = {"bad", "terrible", "poor", "slow", "loud", "broken", "barely", "worst", "tệ", "kém", "bung",
            "tối", "hỏng", "chậm", "lỗi", "rè", "nóng", "cao"}
TOXIC = {"idiot", "stupid", "hate", "ngu", "đồ"}


class StubAspectExtractor:
    def count_tokens(self, text: str) -> int:
        return len(word_spans(text))

    def extract(self, text: str) -> list[dict]:
        found: list[dict] = []
        lowered = text.lower()
        for term in sorted(set(ASPECT_WORDS), key=len, reverse=True):
            for m in re.finditer(r"(?<!\w)" + re.escape(term) + r"(?!\w)", lowered):
                if all(m.end() <= f["start"] or m.start() >= f["end"] for f in found):
                    found.append({"start": m.start(), "end": m.end(), "term": text[m.start():m.end()],
                                  "category": None, "score": 0.9})
        return sorted(found, key=lambda f: f["start"])


class StubSentimentClassifier:
    labels = ("negative", "neutral", "positive")

    def predict_proba(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        rows = []
        for sentence, aspect in pairs:
            words = {w.lower() for w in re.findall(r"\w+", sentence)}
            pos, neg = len(words & POSITIVE), len(words & NEGATIVE)
            if pos == neg:
                rows.append([0.2, 0.6, 0.2])
            elif pos > neg:
                rows.append([0.05, 0.1, 0.85])
            else:
                rows.append([0.85, 0.1, 0.05])
        return np.asarray(rows, dtype=float).reshape(-1, 3)


class StubToxicityModel:
    labels = ("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")

    def count_tokens(self, text: str) -> int:
        return len(word_spans(text)) + 2

    def predict(self, text: str) -> tuple[dict[str, float], int]:
        hits = len({w.lower() for w in re.findall(r"\w+", text)} & TOXIC)
        score = min(0.95, 0.02 + 0.45 * hits)
        scores = {k: (score if k in ("toxic", "insult") else score / 4) for k in self.labels}
        return scores, self.count_tokens(text)


class StubSpeechToText:
    @staticmethod
    def decode(audio_bytes: bytes) -> np.ndarray:
        if len(audio_bytes) < 4:
            raise ValueError("audio could not be decoded")
        return np.zeros(int(len(audio_bytes) / 2), dtype=np.float32)  # pretend 16-bit PCM @16 kHz

    def transcribe(self, audio: np.ndarray) -> str:
        return "The battery is great but the delivery was slow."


class StubVisionModels:
    def category(self, image: Image.Image) -> tuple[int | None, float]:
        return 7, 0.9

    def condition_proba(self, images: list) -> np.ndarray:
        out = []
        for img in images:
            arr = np.asarray(img.convert("RGB") if isinstance(img, Image.Image) else img, dtype=float)
            red = arr[..., 0].mean() / 255.0
            p = float(np.clip(red, 0.05, 0.95))
            out.append([p, 1 - p])
        return np.asarray(out)

    def align(self, image: Image.Image, terms: list[str]) -> list[float]:
        return [0.2 for _ in terms]

    def vqa(self, image: Image.Image, question: str) -> tuple[str, float | None]:
        q = question.lower()
        if "color" in q or "màu" in q:
            return "gray", 0.8
        if q.startswith(("is", "are", "does", "có")):
            return "yes", 0.7
        return "shoe", 0.9

    def warm(self) -> None:
        return None


class StubVariantGenerator:
    latent_dim = 16

    def _img(self, rng: np.random.Generator, gain: float) -> np.ndarray:
        yy, xx = np.mgrid[0:28, 0:28]
        blob = np.exp(-(((yy - 16) / 7.0) ** 2 + ((xx - 14) / 11.0) ** 2))
        return np.clip(blob * gain + rng.normal(0, 0.02, blob.shape), 0, 1)

    def generate(self, category: int, probs: dict[str, float], n: int, seed: int) -> list[np.ndarray]:
        rng = np.random.default_rng(seed)
        gain = brightness_gain(sentiment_score(probs))
        return [self._img(rng, gain) for _ in range(n)]

    def sweep(self, category: int, seed: int, steps: int = 7) -> list[np.ndarray]:
        rng = np.random.default_rng(seed)
        return [self._img(rng, brightness_gain(-1 + 2 * i / (steps - 1))) for i in range(steps)]
