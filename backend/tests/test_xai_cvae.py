"""LIME fidelity on a model whose answer we know, and the CVAE (definition + the bundled trained decoder)."""
import json
from pathlib import Path

import numpy as np
import pytest

from app.xai.explain import explain_text

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "cvae"


class LinearSentiment:
    """P(positive) is linear in the presence of 'good' and 'great' — LIME should recover it almost exactly."""

    def predict_proba(self, pairs):
        rows = []
        for sentence, _ in pairs:
            words = set(sentence.lower().split())
            p = 0.1 + 0.4 * ("good" in words) + 0.3 * ("great" in words)
            rows.append([1 - p - 0.1, 0.1, p])
        return np.asarray(rows)


def test_lime_text_is_faithful_on_a_linear_model():
    e = explain_text(LinearSentiment(), "good screen and great keyboard but meh speakers", "screen", num_samples=300)
    assert e["label"] == "positive"
    assert e["fidelity"] >= 0.95
    assert {w["word"] for w in e["weights"][:2]} == {"good", "great"}
    good = next(w["weight"] for w in e["weights"] if w["word"] == "good")
    assert good == pytest.approx(0.4, abs=0.05)


def test_lime_handles_a_one_word_review():
    e = explain_text(LinearSentiment(), "good", "it", num_samples=60)
    assert e["fidelity"] == 1.0 and e["label"] == "positive"


keras = pytest.importorskip("keras")


def test_cvae_definition_shapes_and_range():
    from app.core.cvae import build_models

    cvae, encoder, decoder = build_models(cond_dim=60)
    z = np.random.default_rng(0).standard_normal((3, 16)).astype("float32")
    c = np.zeros((3, 60), dtype="float32")
    out = keras.ops.convert_to_numpy(decoder([z, c]))
    assert out.shape == (3, 28, 28, 1)
    assert 0.0 <= out.min() and out.max() <= 1.0
    x = np.random.default_rng(1).random((3, 28, 28, 1)).astype("float32")
    assert keras.ops.convert_to_numpy(cvae([x, c])).shape == (3, 28, 28, 1)
    assert len(cvae.losses) == 1  # the KL term


@pytest.mark.skipif(not (ASSETS / "decoder.keras").exists(), reason="decoder not exported yet")
def test_bundled_decoder_turns_sentiment_into_brightness():
    from app.models.media_models import VariantGenerator

    meta = json.loads((ASSETS / "anchors.json").read_text(encoding="utf-8"))
    assert meta["metrics"]["vae_recon_mse"] < 0.01
    gen = VariantGenerator(str(ASSETS))
    for category in (0, 7, 8):
        neg = np.mean(gen.generate(category, {"negative": 1, "neutral": 0, "positive": 0}, 8, seed=1))
        pos = np.mean(gen.generate(category, {"negative": 0, "neutral": 0, "positive": 1}, 8, seed=1))
        assert pos > neg * 1.2, (category, pos, neg)
    sweep = [s.mean() for s in gen.sweep(7, seed=3)]
    assert len(sweep) == 7 and sweep[-1] > sweep[0]
    assert all(img.shape == (28, 28) for img in gen.generate(3, {"positive": 1}, 2, seed=0))
