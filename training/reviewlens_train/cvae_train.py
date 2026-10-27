"""Conditional VAE on FashionMNIST with a Word2Vec sentiment condition (Keras 3 / TensorFlow).

Run on CPU in a few minutes; on Colab the same code runs with the TensorFlow backend.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np

from . import SEED
from app.core.cvae import (CATEGORIES, LATENT_DIM, SENTIMENTS, brightness_gain, build_models,
                           condition_vector, random_sentiment_mix)

ANCHOR_WORDS = {
    "positive": ["good", "great", "excellent", "love", "perfect", "amazing", "tốt", "tuyệt", "đẹp", "thích", "xuất", "sắc"],
    "neutral": ["okay", "ok", "average", "normal", "fine", "decent", "bình", "thường", "tạm", "ổn"],
    "negative": ["bad", "terrible", "poor", "awful", "worst", "broken", "tệ", "kém", "lỗi", "hỏng", "chán", "dở"],
}


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def train_word2vec(sentences: list[str], dim: int = 50, seed: int = SEED):
    from gensim.models import Word2Vec

    return Word2Vec([tokenize(s) for s in sentences], vector_size=dim, window=5, min_count=2, sg=1,
                    epochs=20, seed=seed, workers=1)


def sentiment_anchors(w2v) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    anchors, used = {}, {}
    for k, words in ANCHOR_WORDS.items():
        vecs = [w2v.wv[w] / np.linalg.norm(w2v.wv[w]) for w in words if w in w2v.wv]
        if not vecs:
            raise ValueError(f"no anchor words for {k} in the Word2Vec vocabulary")
        v = np.mean(vecs, axis=0)
        anchors[k] = (v / np.linalg.norm(v)).astype(np.float32)
        used[k] = [w for w in words if w in w2v.wv]
    return anchors, used


def conditioned_set(x: np.ndarray, y: np.ndarray, anchors: dict, rng: np.random.Generator):
    """Pair every image with a random sentiment mix; darken it to match that mix."""
    mixes = random_sentiment_mix(rng, len(x))  # columns: negative, neutral, positive
    probs = [dict(zip(SENTIMENTS, m)) for m in mixes]
    gains = np.array([brightness_gain(p["positive"] - p["negative"]) for p in probs], dtype=np.float32)
    images = (x * gains[:, None, None])[..., None].astype(np.float32)
    conds = np.stack([condition_vector(int(c), p, anchors) for c, p in zip(y, probs)]).astype(np.float32)
    return images, conds, mixes


def train_cvae(x: np.ndarray, y: np.ndarray, splits, anchors: dict, epochs: int = 20, batch_size: int = 128,
               beta: float = 0.5, seed: int = SEED, verbose: int = 2):
    import keras

    keras.utils.set_random_seed(seed)
    tr, va, te = splits
    rng = np.random.default_rng(seed)
    xtr, ctr, _ = conditioned_set(x[tr], y[tr], anchors, rng)
    xva, cva, _ = conditioned_set(x[va], y[va], anchors, rng)
    cond_dim = ctr.shape[1]
    cvae, encoder, decoder = build_models(cond_dim, LATENT_DIM, beta=beta)
    cvae.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    history = cvae.fit([xtr, ctr], xtr, validation_data=([xva, cva], xva), epochs=epochs,
                       batch_size=batch_size, verbose=verbose,
                       callbacks=[keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5)])
    return cvae, encoder, decoder, history.history


def evaluate_cvae(encoder, decoder, x: np.ndarray, y: np.ndarray, test_idx, anchors: dict, seed: int = SEED + 1) -> dict:
    """Per-pixel reconstruction MSE (z = posterior mean), KL, and the brightness response."""
    import keras

    rng = np.random.default_rng(seed)
    xte, cte, _ = conditioned_set(x[test_idx], y[test_idx], anchors, rng)
    mse, kl, n = 0.0, 0.0, 0
    for b in range(0, len(xte), 512):
        mean, logvar = (keras.ops.convert_to_numpy(t) for t in encoder([xte[b:b + 512], cte[b:b + 512]], training=False))
        recon = keras.ops.convert_to_numpy(decoder([mean, cte[b:b + 512]], training=False))
        mse += float(((recon - xte[b:b + 512]) ** 2).mean(axis=(1, 2, 3)).sum())
        kl += float((-0.5 * (1 + logvar - mean ** 2 - np.exp(logvar)).sum(axis=1)).sum())
        n += len(mean)
    # Same z, sentiment swept negative -> positive: mean brightness must rise.
    z = np.random.default_rng(seed).standard_normal((len(CATEGORIES), LATENT_DIM)).astype(np.float32)
    levels = []
    for t in (0.0, 0.5, 1.0):
        c = np.stack([condition_vector(k, {"negative": 1 - t, "neutral": 0.0, "positive": t}, anchors) for k in range(len(CATEGORIES))])
        levels.append(float(keras.ops.convert_to_numpy(decoder([z, c], training=False)).mean()))
    return {"vae_recon_mse": round(mse / n, 5), "vae_kl": round(kl / n, 3),
            "vae_brightness_neg_mid_pos": [round(v, 4) for v in levels]}


def export(decoder, anchors: dict, used: dict, metrics: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    decoder.save(out_dir / "decoder.keras")
    meta = {"latent_dim": LATENT_DIM, "categories": CATEGORIES, "anchor_words": used,
            "anchors": {k: [round(float(v), 6) for v in anchors[k]] for k in SENTIMENTS},
            "backend_trained": os.environ.get("KERAS_BACKEND", "tensorflow"), "metrics": metrics}
    (out_dir / "anchors.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return out_dir
