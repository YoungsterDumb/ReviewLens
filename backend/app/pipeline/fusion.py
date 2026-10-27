"""ABSA over a whole review, overall sentiment, support priority and batch summaries."""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict

import numpy as np

from ..core.text import sentence_spans
from ..models.text_models import SENTIMENTS, sentence_for

GENERAL_ASPECT = {"vi": "chung", "en": "general"}

# UIT-ViSD4SA categories -> readable labels (the UI translates them again).
CATEGORY_LABELS = {
    "SCREEN": "Màn hình", "CAMERA": "Camera", "FEATURES": "Tính năng", "BATTERY": "Pin",
    "PERFORMANCE": "Hiệu năng", "STORAGE": "Bộ nhớ", "DESIGN": "Thiết kế", "PRICE": "Giá",
    "GENERAL": "Tổng thể", "SER&ACC": "Dịch vụ & phụ kiện",
}

# Keyword -> FashionMNIST category for text-only reviews (Bag when nothing matches).
CATEGORY_KEYWORDS = [
    (9, ("boot", "bốt", "giày cổ cao")), (7, ("sneaker", "giày", "shoe")), (5, ("sandal", "dép")),
    (4, ("coat", "jacket", "áo khoác")), (2, ("sweater", "pullover", "áo len")),
    (6, ("shirt", "áo sơ mi")), (0, ("t-shirt", "tee", "áo thun", "áo phông")),
    (3, ("dress", "váy", "đầm")), (1, ("trouser", "pants", "jeans", "quần")),
    (8, ("bag", "túi", "balo", "backpack")),
]
DEFAULT_CATEGORY = 8


def probs_dict(row: np.ndarray) -> dict[str, float]:
    return {k: round(float(v), 4) for k, v in zip(SENTIMENTS, row)}


def run_absa(ate, atsc, text: str, language: str) -> tuple[list[dict], dict]:
    """Extract aspects, score each in its sentence, and combine into an overall verdict."""
    if not text.strip():
        return [], {"label": "neutral", "mixed": False, "probs": probs_dict(np.array([0.0, 1.0, 0.0]))}
    spans = ate.extract(text)
    sentences = sentence_spans(text)
    pairs = [(sentence_for(text, s["start"], s["end"], sentences), s["term"]) for s in spans]
    aspects: list[dict] = []
    if pairs:
        probs = atsc.predict_proba(pairs)
        for s, (sent, _), row in zip(spans, pairs, probs):
            k = int(np.argmax(row))
            aspects.append({
                "term": s["term"], "start": s["start"], "end": s["end"], "category": s.get("category"),
                "sentence": sent, "sentiment": SENTIMENTS[k], "confidence": round(float(row[k]), 4),
                "probs": probs_dict(row), "extraction_score": round(float(s["score"]), 4),
            })
        weights = probs.max(axis=1)
        mean = (probs * weights[:, None]).sum(axis=0) / weights.sum()
    else:
        mean = atsc.predict_proba([(text[:2000], GENERAL_ASPECT.get(language, "general"))])[0]
    labels = {a["sentiment"] for a in aspects}
    overall = {
        "label": SENTIMENTS[int(np.argmax(mean))],
        "mixed": {"positive", "negative"} <= labels,
        "probs": probs_dict(mean),
    }
    return aspects, overall


def priority(aspects: list[dict], toxicity: dict | None, condition: dict | None) -> dict:
    """High: toxic, >= 2 negative aspects or a defective-looking photo. Medium: any negative."""
    reasons: list[str] = []
    negatives = [a for a in aspects if a["sentiment"] == "negative"]
    if toxicity and toxicity["max_score"] >= 0.5:
        reasons.append(f"toxic_language:{toxicity['max_label']}")
    if condition and condition.get("defective", 0.0) >= 0.6:
        reasons.append("photo_defective")
    if len(negatives) >= 2:
        reasons.append(f"negative_aspects:{len(negatives)}")
    if reasons:
        return {"level": "high", "reasons": reasons}
    if negatives:
        return {"level": "medium", "reasons": [f"negative_aspects:{len(negatives)}"]}
    return {"level": "low", "reasons": []}


def pick_category(text: str) -> int:
    lowered = text.lower()
    for idx, words in CATEGORY_KEYWORDS:
        if any(re.search(r"(?<!\w)" + re.escape(w) + r"(?!\w)", lowered) for w in words):
            return idx
    return DEFAULT_CATEGORY


def aspect_key(aspect: dict) -> tuple[str, str, bool]:
    """Group by category when the model gives one (Vietnamese), else by normalised term."""
    if aspect.get("category"):
        cat = aspect["category"]
        return f"cat:{cat}", CATEGORY_LABELS.get(cat, cat.title()), True
    term = unicodedata.normalize("NFC", aspect["term"]).lower().strip(" .,!?;:\"'()[]")
    term = re.sub(r"\s+", " ", term)
    return f"term:{term}", term, False


def summarize(reviews: list[list[dict]]) -> list[dict]:
    rows: dict[str, dict] = {}
    for aspects in reviews:
        for a in aspects:
            key, label, is_cat = aspect_key(a)
            row = rows.setdefault(key, {"key": key, "label": label, "is_category": is_cat,
                                        "mentions": 0, "positive": 0, "neutral": 0, "negative": 0,
                                        "_best": (-1.0, "")})
            row["mentions"] += 1
            row[a["sentiment"]] += 1
            if a["confidence"] > row["_best"][0]:
                row["_best"] = (a["confidence"], a["term"])
    out = []
    for row in rows.values():
        row["net"] = round(100 * (row["positive"] - row["negative"]) / row["mentions"])
        row["example"] = row.pop("_best")[1]
        out.append(row)
    return sorted(out, key=lambda r: (-r["mentions"], r["label"]))


def counts_by_sentiment(aspects: list[dict]) -> dict[str, int]:
    c: dict[str, int] = defaultdict(int)
    for a in aspects:
        c[a["sentiment"]] += 1
    return dict(c)
