"""Local fidelity of a LIME explanation.

fidelity = 1 - weighted mean |g(z) - f(z)|, weighted by LIME's exponential
kernel so that perturbations close to the original input count most.
"""
from __future__ import annotations

import numpy as np


def kernel(distances: np.ndarray, width: float) -> np.ndarray:
    """LIME's default kernel: sqrt(exp(-d^2 / width^2))."""
    d = np.asarray(distances, dtype=float)
    return np.sqrt(np.exp(-(d ** 2) / width ** 2))


def surrogate_predict(masks: np.ndarray, weights: dict[int, float], intercept: float) -> np.ndarray:
    """g(z) = intercept + sum_j w_j * z_j over the features LIME kept."""
    masks = np.asarray(masks, dtype=float)
    g = np.full(masks.shape[0], float(intercept))
    for j, w in weights.items():
        g += w * masks[:, j]
    return np.clip(g, 0.0, 1.0)


def local_fidelity(f: np.ndarray, g: np.ndarray, sample_weights: np.ndarray | None = None) -> float:
    f = np.asarray(f, dtype=float)
    g = np.asarray(g, dtype=float)
    err = np.abs(g - f)
    if sample_weights is None or float(np.sum(sample_weights)) == 0.0:
        return float(1.0 - err.mean())
    w = np.asarray(sample_weights, dtype=float)
    return float(1.0 - np.sum(w * err) / np.sum(w))


def random_masks(n_features: int, n_samples: int, rng: np.random.Generator, max_off: int | None = None) -> np.ndarray:
    """Binary masks (1 = feature kept) removing 1..max_off features at random."""
    max_off = max(1, min(max_off or n_features - 1, n_features - 1)) if n_features > 1 else 0
    masks = np.ones((n_samples, n_features), dtype=int)
    for row in masks:
        if max_off == 0:
            continue
        k = rng.integers(1, max_off + 1)
        row[rng.choice(n_features, size=k, replace=False)] = 0
    return masks


def cosine_distance_to_full(masks: np.ndarray) -> np.ndarray:
    """Cosine distance (x100, as in LIME text) between each mask and the all-ones vector."""
    masks = np.asarray(masks, dtype=float)
    n = masks.shape[1]
    kept = masks.sum(axis=1)
    cos = np.where(kept > 0, kept / (np.sqrt(kept) * np.sqrt(n)), 0.0)
    return (1.0 - cos) * 100.0
