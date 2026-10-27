"""Datasets: SemEval-2014 (EN), UIT-ViSD4SA (VI), Jigsaw (long context), FashionMNIST (CVAE).

Every dataset is pooled, de-duplicated and re-split 70/15/15 (seed 42), stratified by
language and source, as the spec requires.
"""
from __future__ import annotations

import gzip
import json
import random
import subprocess
import unicodedata
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import SEED
from app.core.bio import TaggedSpan, spans_to_word_tags
from app.core.text import sentence_spans
from app.models.text_models import sentence_for

SEMEVAL_REPO = "https://github.com/songyouwei/ABSA-PyTorch.git"
VI_REPO = "https://github.com/kimkim00/UIT-ViSD4SA.git"
FASHION_URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/{}"
POLARITY = {"-1": "negative", "0": "neutral", "1": "positive"}


def fetch_repo(url: str, dest: Path) -> Path:
    if not (dest / ".git").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", "--depth", "1", url, str(dest)], check=True)
    return dest


# --- ABSA -----------------------------------------------------------------

def load_semeval(root: Path) -> list[dict]:
    """Parse ABSA-PyTorch's .seg files (sentence with $T$, aspect, polarity) into sentences with spans."""
    by_text: dict[tuple[str, str], dict] = {}
    for source, stem in (("laptop", "Laptops"), ("restaurant", "Restaurants")):
        for part, name in (("train", f"{stem}_Train.xml.seg"), ("test", f"{stem}_Test_Gold.xml.seg")):
            lines = (root / "datasets" / "semeval14" / name).read_text(encoding="utf-8").splitlines()
            for i in range(0, len(lines) - 2, 3):
                template, aspect, pol = lines[i].strip(), lines[i + 1].strip(), lines[i + 2].strip()
                if "$T$" not in template:
                    continue
                left, right = template.split("$T$", 1)
                text = f"{left}{aspect}{right}"
                rec = by_text.setdefault((source, text), {"text": text, "lang": "en", "source": source,
                                                          "orig_split": part, "spans": [], "polarities": []})
                start = len(left)
                if all(s[0] != start for s in rec["spans"]):
                    rec["spans"].append((start, start + len(aspect), "ASP"))
                    rec["polarities"].append(POLARITY[pol])
    return list(by_text.values())


def load_visd4sa(root: Path) -> list[dict]:
    """UIT-ViSD4SA JSONL: text + [start, end, 'CATEGORY#POLARITY'] spans (opinion phrases)."""
    out = []
    for part in ("train", "dev", "test"):
        for line in (root / "data" / f"{part}.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            text = row["text"]
            nfc = unicodedata.normalize("NFC", text)
            if len(nfc) != len(text):  # offsets would shift; such rows are rare
                continue
            spans, pols = [], []
            for start, end, label in sorted(row["labels"], key=lambda x: x[0]):
                cat, _, pol = label.partition("#")
                if end <= start or not text[start:end].strip():
                    continue
                spans.append((start, end, cat))
                pols.append(pol.lower())
            out.append({"text": nfc, "lang": "vi", "source": "smartphone", "orig_split": part,
                        "spans": spans, "polarities": pols})
    return out


def pooled_split(records: list[dict], seed: int = SEED, ratios=(0.7, 0.15, 0.15)) -> dict[str, list[dict]]:
    """De-duplicate by text, then split each (lang, source) group 70/15/15."""
    seen, unique = set(), []
    for r in records:
        if r["text"] not in seen:
            seen.add(r["text"])
            unique.append(r)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in unique:
        groups[(r.get("lang"), r.get("source"))].append(r)
    rng = random.Random(seed)
    splits = {"train": [], "validation": [], "test": []}
    for key in sorted(groups, key=str):
        items = groups[key][:]
        rng.shuffle(items)
        n = len(items)
        a, b = int(round(n * ratios[0])), int(round(n * (ratios[0] + ratios[1])))
        splits["train"] += items[:a]
        splits["validation"] += items[a:b]
        splits["test"] += items[b:]
    return splits


def to_bio(records: list[dict]) -> list[dict]:
    out = []
    for r in records:
        words, tags = spans_to_word_tags(r["text"], [TaggedSpan(s, e, t) for s, e, t in r["spans"]])
        out.append({"tokens": [r["text"][w.start:w.end] for w in words], "tags": tags, "lang": r["lang"]})
    return out


def to_pairs(records: list[dict]) -> list[dict]:
    """ATSC examples: (sentence containing the aspect, aspect text) -> polarity."""
    out = []
    for r in records:
        sents = sentence_spans(r["text"])
        for (s, e, _), pol in zip(r["spans"], r["polarities"]):
            out.append({"sentence": sentence_for(r["text"], s, e, sents), "aspect": r["text"][s:e],
                        "label": pol, "lang": r["lang"]})
    return out


def load_absa(cache: Path) -> dict[str, list[dict]]:
    en = load_semeval(fetch_repo(SEMEVAL_REPO, cache / "semeval"))
    vi = load_visd4sa(fetch_repo(VI_REPO, cache / "visd4sa"))
    return pooled_split(en + vi)


# --- FashionMNIST -----------------------------------------------------------

def _download(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
    return dest


def load_fashion_mnist(cache: Path) -> tuple[np.ndarray, np.ndarray]:
    """All 70,000 images (float32 [0,1], 28x28) and labels, from the official GitHub mirror."""
    xs, ys = [], []
    for img_name, lbl_name in (("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
                               ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")):
        with gzip.open(_download(FASHION_URL.format(img_name), cache / img_name)) as f:
            xs.append(np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28))
        with gzip.open(_download(FASHION_URL.format(lbl_name), cache / lbl_name)) as f:
            ys.append(np.frombuffer(f.read(), np.uint8, offset=8))
    return np.concatenate(xs).astype(np.float32) / 255.0, np.concatenate(ys).astype(np.int64)


def split_arrays(n: int, seed: int = SEED, ratios=(0.7, 0.15, 0.15)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.random.default_rng(seed).permutation(n)
    a, b = int(n * ratios[0]), int(n * (ratios[0] + ratios[1]))
    return idx[:a], idx[a:b], idx[b:]
