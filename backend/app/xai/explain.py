"""LIME explanations for text (ATSC) and images (CLIP defect vs good), with local fidelity."""
from __future__ import annotations

import numpy as np
from PIL import Image

from ..core.fidelity import cosine_distance_to_full, kernel, local_fidelity, random_masks, surrogate_predict
from ..media import heatmap_overlay, to_data_uri
from ..models.text_models import SENTIMENTS

CONDITION_LABELS = ("defective", "good")
FIDELITY_SAMPLES = 100


def _score(exp, label: int) -> float:
    s = exp.score
    return float(s.get(label, 0.0) if isinstance(s, dict) else s)


def explain_text(atsc, text: str, aspect: str, num_samples: int = 300, num_features: int = 10, seed: int = 0) -> dict:
    from lime.lime_text import LimeTextExplainer

    def predict(texts: list[str]) -> np.ndarray:
        return atsc.predict_proba([(t, aspect) for t in texts])

    base = predict([text])[0]
    label = int(np.argmax(base))
    explainer = LimeTextExplainer(class_names=list(SENTIMENTS), bow=True, random_state=seed)
    exp = explainer.explain_instance(text, predict, labels=(label,), num_features=num_features, num_samples=num_samples)

    indexed = exp.domain_mapper.indexed_string
    n_words = indexed.num_words()
    weights = dict(exp.local_exp[label])
    if n_words > 1:
        rng = np.random.default_rng(seed + 1)
        masks = random_masks(n_words, FIDELITY_SAMPLES, rng)
        perturbed = [indexed.inverse_removing(np.flatnonzero(m == 0)) for m in masks]
        f = predict(perturbed)[:, label]
        g = surrogate_predict(masks, weights, float(exp.intercept[label]))
        fidelity = local_fidelity(f, g, kernel(cosine_distance_to_full(masks), 25.0))
    else:
        fidelity = 1.0
    return {
        "aspect": aspect,
        "label": SENTIMENTS[label],
        "probs": {k: round(float(v), 4) for k, v in zip(SENTIMENTS, base)},
        "weights": [{"word": w, "weight": round(float(v), 4)} for w, v in exp.as_list(label=label)],
        "fidelity": round(fidelity, 4),
        "r2": round(_score(exp, label), 4),
        "num_samples": num_samples,
    }


def explain_image(vision, image: Image.Image, num_samples: int = 150, seed: int = 0) -> dict:
    from lime import lime_image
    from lime.wrappers.scikit_image import SegmentationAlgorithm

    img = np.asarray(image.convert("RGB").resize((224, 224)))

    def predict(batch: np.ndarray) -> np.ndarray:
        return vision.condition_proba([Image.fromarray(np.asarray(x, dtype=np.uint8)) for x in batch])

    segmenter = SegmentationAlgorithm("slic", n_segments=40, compactness=10, sigma=1, start_label=0)
    explainer = lime_image.LimeImageExplainer(random_state=seed)
    exp = explainer.explain_instance(img, predict, top_labels=1, hide_color=None, num_samples=num_samples,
                                     segmentation_fn=segmenter, batch_size=32)
    label = int(exp.top_labels[0])
    weights = dict(exp.local_exp[label])
    segments = exp.segments
    n_seg = int(segments.max()) + 1

    # Fidelity on fresh perturbations, hiding segments the way LIME does (segment mean colour).
    fudged = img.copy()
    for s in range(n_seg):
        m = segments == s
        if m.any():
            fudged[m] = img[m].mean(axis=0)
    rng = np.random.default_rng(seed + 1)
    masks = random_masks(n_seg, FIDELITY_SAMPLES // 2, rng)
    batch = []
    for m in masks:
        x = img.copy()
        off = np.isin(segments, np.flatnonzero(m == 0))
        x[off] = fudged[off]
        batch.append(x)
    f = predict(np.stack(batch))[:, label]
    g = surrogate_predict(masks, weights, float(exp.intercept[label]))
    dist = 1.0 - masks.sum(axis=1) / (np.sqrt(np.maximum(masks.sum(axis=1), 1)) * np.sqrt(n_seg))
    fidelity = local_fidelity(f, g, kernel(dist, 0.25))

    top = dict(sorted(weights.items(), key=lambda kv: -abs(kv[1]))[:8])
    overlay = heatmap_overlay(img, segments, top)
    prob = float(predict(img[None])[0, label])
    return {
        "label": CONDITION_LABELS[label],
        "probability": round(prob, 4),
        "heatmap": to_data_uri(overlay),
        "fidelity": round(fidelity, 4),
        "r2": round(_score(exp, label), 4),
        "num_samples": num_samples,
    }
