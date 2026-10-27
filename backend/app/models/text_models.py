"""Transformer wrappers: aspect extraction (DistilBERT), aspect sentiment (ALBERT),
long-context toxicity (Longformer)."""
from __future__ import annotations

import numpy as np

from ..core.bio import word_tags_to_spans
from ..core.text import Span, build_windows, word_spans

SENTIMENTS = ("negative", "neutral", "positive")
TOXICITY_LABELS = ("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class AspectExtractor:
    """Token classification with BIO tags; reads text in sentence windows so nothing is cut."""

    def __init__(self, model_id: str, window_tokens: int = 256, batch_size: int = 16, *, tokenizer=None, model=None):
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self.tok = tokenizer or AutoTokenizer.from_pretrained(model_id)
        self.model = (model or AutoModelForTokenClassification.from_pretrained(model_id)).eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.window_tokens = window_tokens
        self.batch_size = batch_size

    def count_tokens(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def extract(self, text: str) -> list[dict]:
        import torch

        if not text.strip():
            return []
        windows = build_windows(text, self.count_tokens, self.window_tokens)
        all_words = word_spans(text)
        groups = [[w for w in all_words if win.start <= w.start and w.end <= win.end] for win in windows]
        groups = [g for g in groups if g]
        found: list[dict] = []
        for b in range(0, len(groups), self.batch_size):
            batch = groups[b:b + self.batch_size]
            enc = self.tok(
                [[text[w.start:w.end] for w in g] for g in batch],
                is_split_into_words=True, truncation=True, max_length=512,
                padding=True, return_tensors="pt",
            )
            with torch.inference_mode():
                logits = self.model(**{k: v for k, v in enc.items() if k in ("input_ids", "attention_mask")}).logits
            probs = _softmax(logits.float().cpu().numpy())
            for i, words in enumerate(batch):
                tags, conf = ["O"] * len(words), [0.0] * len(words)
                seen: set[int] = set()
                for pos, wid in enumerate(enc.word_ids(i)):
                    if wid is None or wid in seen or wid >= len(words):
                        continue
                    seen.add(wid)
                    k = int(probs[i, pos].argmax())
                    tags[wid], conf[wid] = self.id2label[k], float(probs[i, pos, k])
                for span in word_tags_to_spans(words, tags):
                    idx = [j for j, w in enumerate(words) if span.start <= w.start and w.end <= span.end]
                    found.append({
                        "start": span.start, "end": span.end,
                        "term": text[span.start:span.end],
                        "category": None if span.type == "ASP" else span.type,
                        "score": float(np.mean([conf[j] for j in idx])) if idx else 0.0,
                    })
        return _dedupe(found)


def _dedupe(spans: list[dict]) -> list[dict]:
    """Windows of an over-long sentence overlap; keep the best of overlapping spans."""
    kept: list[dict] = []
    for s in sorted(spans, key=lambda s: -s["score"]):
        if all(s["end"] <= k["start"] or s["start"] >= k["end"] for k in kept):
            kept.append(s)
    return sorted(kept, key=lambda s: s["start"])


class SentimentClassifier:
    """Sentence-pair classifier: (sentence, aspect) -> negative / neutral / positive."""

    labels = SENTIMENTS

    def __init__(self, model_id: str, batch_size: int = 32, max_length: int = 256, *, tokenizer=None, model=None):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tok = tokenizer or AutoTokenizer.from_pretrained(model_id)
        self.model = (model or AutoModelForSequenceClassification.from_pretrained(model_id)).eval()
        names = [str(self.model.config.id2label[i]).lower() for i in range(self.model.config.num_labels)]
        self.order = [names.index(s) for s in SENTIMENTS] if set(SENTIMENTS) <= set(names) else [0, 1, 2]
        self.batch_size = batch_size
        self.max_length = max_length

    def predict_proba(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        import torch

        if not pairs:
            return np.zeros((0, 3))
        out = []
        for b in range(0, len(pairs), self.batch_size):
            chunk = pairs[b:b + self.batch_size]
            enc = self.tok(
                [s for s, _ in chunk], [a for _, a in chunk],
                truncation="only_first", max_length=self.max_length,
                padding=True, return_tensors="pt",
            )
            with torch.inference_mode():
                logits = self.model(**enc).logits.float().cpu().numpy()
            out.append(_softmax(logits)[:, self.order])
        return np.vstack(out)


class ToxicityModel:
    """Multi-label toxicity over the whole review, up to 4,096 tokens in one pass."""

    labels = TOXICITY_LABELS

    def __init__(self, model_id: str, max_length: int = 4096, *, tokenizer=None, model=None):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tok = tokenizer or AutoTokenizer.from_pretrained(model_id)
        self.model = (model or AutoModelForSequenceClassification.from_pretrained(model_id)).eval()
        self.max_length = max_length
        cfg = self.model.config.id2label
        names = [str(cfg[i]) for i in range(self.model.config.num_labels)]
        self.names = names if set(TOXICITY_LABELS) <= set(names) else list(TOXICITY_LABELS)[: len(names)]

    def count_tokens(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=True)["input_ids"])

    def predict(self, text: str) -> tuple[dict[str, float], int]:
        import torch

        n_tokens = self.count_tokens(text)
        enc = self.tok(text, truncation=True, max_length=self.max_length, return_tensors="pt")
        with torch.inference_mode():
            logits = self.model(**enc).logits.float().cpu().numpy()[0]
        probs = 1.0 / (1.0 + np.exp(-logits))
        return {n: float(p) for n, p in zip(self.names, probs)}, n_tokens


def sentence_for(text: str, start: int, end: int, sentences: list[Span]) -> str:
    """The sentence (or sentences) containing a span — the context ATSC reads."""
    covering = [s for s in sentences if s.start < end and s.end > start]
    if not covering:
        return text
    return text[covering[0].start:covering[-1].end]
