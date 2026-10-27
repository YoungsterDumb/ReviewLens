"""Train the CVAE on CPU and bundle it with the API (backend/assets/cvae).

Usage (from the repo root):  python training/scripts/train_cvae_local.py --epochs 20
Also writes backend/assets/metrics.json (local fallback for /metrics) and the
FashionMNIST sample photos the web app uses for its "try a sample" buttons.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from reviewlens_train.cvae_train import evaluate_cvae, export, sentiment_anchors, train_cvae, train_word2vec  # noqa: E402
from reviewlens_train.data import load_absa, load_fashion_mnist, split_arrays  # noqa: E402


def save_samples(x: np.ndarray, y: np.ndarray, test_idx: np.ndarray, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for label, name in ((7, "sneaker"), (8, "bag")):
        i = next(int(j) for j in test_idx if y[j] == label and x[j].mean() > 0.25)
        img = Image.fromarray((x[i] * 255).astype(np.uint8)).resize((280, 280), Image.BICUBIC).convert("RGB")
        img.save(out / f"{name}.png", optimize=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--cache", type=Path, default=ROOT / "training" / "data")
    args = ap.parse_args()

    x, y = load_fashion_mnist(args.cache / "fashion")
    splits = split_arrays(len(x))
    absa = load_absa(args.cache)
    corpus = [r["text"] for part in absa.values() for r in part]
    w2v = train_word2vec(corpus)
    anchors, used = sentiment_anchors(w2v)
    print("anchor words:", used)
    print("cos(pos, neg) = %.3f" % float(anchors["positive"] @ anchors["negative"]))

    _, encoder, decoder, history = train_cvae(x, y, splits, anchors, epochs=args.epochs, beta=args.beta)
    metrics = evaluate_cvae(encoder, decoder, x, y, splits[2], anchors)
    metrics["vae_total_loss"] = round(float(history["val_loss"][-1]), 5)
    print(json.dumps(metrics, indent=1))

    export(decoder, anchors, used, metrics, ROOT / "backend" / "assets" / "cvae")
    local = {
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metrics": {k: v for k, v in metrics.items() if not isinstance(v, list)},
        "cvae_repo": "bundled with the API (backend/assets/cvae)",
        "notes": "CVAE trained on CPU by training/scripts/train_cvae_local.py; other metrics come from the Colab notebooks.",
    }
    (ROOT / "backend" / "assets" / "metrics.json").write_text(json.dumps(local, indent=2), encoding="utf-8")
    save_samples(x, y, splits[2], ROOT / "frontend" / "public" / "samples")
    print("exported decoder, anchors, metrics and sample photos")


if __name__ == "__main__":
    main()
