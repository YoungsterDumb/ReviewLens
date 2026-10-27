"""Data preparation on small fixtures: no network, numpy only."""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reviewlens_train.data import load_semeval, load_visd4sa, pooled_split, split_arrays, to_bio, to_pairs  # noqa: E402
from app.core.cvae import CATEGORIES, condition_vector  # noqa: E402


@pytest.fixture
def semeval_root(tmp_path: Path) -> Path:
    d = tmp_path / "datasets" / "semeval14"
    d.mkdir(parents=True)
    laptop = (
        "I charge it at night and skip taking the $T$ with me because of the good battery life .\ncord\n0\n"
        "I charge it at night and skip taking the cord with me because of the good $T$ .\nbattery life\n1\n"
        "The $T$ is too slow .\nboot time\n-1\n"
    )
    for name in ("Laptops_Train.xml.seg", "Laptops_Test_Gold.xml.seg"):
        (d / name).write_text(laptop if "Train" in name else "Great $T$ !\nscreen\n1\n", encoding="utf-8")
    for name in ("Restaurants_Train.xml.seg", "Restaurants_Test_Gold.xml.seg"):
        (d / name).write_text("The $T$ was cold .\nsoup\n-1\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def vi_root(tmp_path: Path) -> Path:
    d = tmp_path / "vi" / "data"
    d.mkdir(parents=True)
    rows = [
        {"text": "Pin trâu nhưng camera chụp tối.", "labels": [[15, 30, "CAMERA#NEGATIVE"], [0, 8, "BATTERY#POSITIVE"]]},
        {"text": "Giao hàng nhanh.", "labels": [[0, 15, "SER&ACC#POSITIVE"]]},
    ]
    for part in ("train", "dev", "test"):
        (d / f"{part}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return tmp_path / "vi"


def test_semeval_groups_aspects_per_sentence(semeval_root):
    records = load_semeval(semeval_root)
    s = next(r for r in records if "cord" in r["text"] and r["source"] == "laptop")
    assert [s["text"][a:b] for a, b, _ in s["spans"]] == ["cord", "battery life"]
    assert s["polarities"] == ["neutral", "positive"]
    assert {r["source"] for r in records} == {"laptop", "restaurant"}


def test_visd4sa_keeps_categories_sorted_by_offset(vi_root):
    records = load_visd4sa(vi_root)
    first = records[0]
    assert [t for _, _, t in first["spans"]] == ["BATTERY", "CAMERA"]
    assert first["polarities"] == ["positive", "negative"]


def test_bio_and_pairs(semeval_root, vi_root):
    records = load_semeval(semeval_root) + load_visd4sa(vi_root)
    bio = to_bio(records)
    en = next(b for b in bio if "battery" in b["tokens"])
    i = en["tokens"].index("battery")
    assert en["tags"][i:i + 2] == ["B-ASP", "I-ASP"]
    vi = next(b for b in bio if b["lang"] == "vi" and "Pin" in b["tokens"])
    assert vi["tags"][:2] == ["B-BATTERY", "I-BATTERY"]
    pairs = to_pairs(records)
    assert {"sentence", "aspect", "label", "lang"} <= set(pairs[0])
    assert any(p["aspect"] == "camera chụp tối" and p["label"] == "negative" for p in pairs)


def test_pooled_split_is_70_15_15_without_leakage():
    records = [{"text": f"t{i}", "lang": "en" if i % 3 else "vi", "source": "x"} for i in range(400)]
    records += records[:20]  # duplicates are dropped
    splits = pooled_split(records)
    sizes = {k: len(v) for k, v in splits.items()}
    assert sum(sizes.values()) == 400
    assert abs(sizes["train"] / 400 - 0.70) < 0.01 and abs(sizes["test"] / 400 - 0.15) < 0.01
    texts = [set(r["text"] for r in v) for v in splits.values()]
    assert not (texts[0] & texts[1]) and not (texts[0] & texts[2]) and not (texts[1] & texts[2])
    langs = Counter(r["lang"] for r in splits["test"])
    assert langs["vi"] > 0 and langs["en"] > 0


def test_array_split_and_condition_size():
    tr, va, te = split_arrays(1000)
    assert (len(tr), len(va), len(te)) == (700, 150, 150)
    assert len(set(tr) | set(va) | set(te)) == 1000
    anchors = {k: np.ones(50, dtype=np.float32) for k in ("negative", "neutral", "positive")}
    assert condition_vector(9, {"positive": 1}, anchors).shape == (len(CATEGORIES) + 50,)
