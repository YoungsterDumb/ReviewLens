"""Evaluations for pretrained parts: BLIP VQA on VQAv2 (MS-COCO images), CLIP category mapping, LIME fidelity."""
from __future__ import annotations

import re

import numpy as np

VQA_DATASET = "Multimodal-Fatima/VQAv2_sample_validation"
_ARTICLES = {"a", "an", "the"}
_NUMBERS = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
            "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}


def normalize_answer(ans: str) -> str:
    """Simplified official VQA normalisation: lower-case, strip punctuation/articles, number words -> digits."""
    ans = re.sub(r"(?<=\d),(?=\d)", "", ans.lower().strip())
    ans = re.sub(r"[^\w\s]", " ", ans)
    words = [_NUMBERS.get(w, w) for w in ans.split() if w not in _ARTICLES]
    return " ".join(words)


def vqa_accuracy(pred: str, answers: list[str]) -> float:
    """Standard VQA accuracy: min(#humans who gave the answer / 3, 1)."""
    p = normalize_answer(pred)
    return min(1.0, sum(normalize_answer(a) == p for a in answers) / 3.0)


def evaluate_vqa(model_id: str = "Salesforce/blip-vqa-base", n: int = 1000, batch_size: int = 16) -> dict:
    import torch
    from datasets import load_dataset
    from transformers import BlipForQuestionAnswering, BlipProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = load_dataset(VQA_DATASET, split="validation").select(range(n))
    proc = BlipProcessor.from_pretrained(model_id)
    model = BlipForQuestionAnswering.from_pretrained(model_id).to(device).eval()
    scores, by_type = [], {}
    for b in range(0, len(ds), batch_size):
        rows = ds[b:b + batch_size]
        images = [im.convert("RGB") for im in rows["image"]]
        enc = proc(images, rows["question"], return_tensors="pt", padding=True).to(device)
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=10)
        for pred, answers, qtype in zip(proc.batch_decode(out, skip_special_tokens=True), rows["answers"], rows["answer_type"]):
            s = vqa_accuracy(pred, answers)
            scores.append(s)
            by_type.setdefault(qtype, []).append(s)
    return {"vqa_accuracy": round(float(np.mean(scores)), 4), "vqa_questions": len(scores),
            **{f"vqa_accuracy_{k.replace('/', '_')}": round(float(np.mean(v)), 4) for k, v in by_type.items()}}


def evaluate_clip_categories(x: np.ndarray, y: np.ndarray, model_id: str = "openai/clip-vit-base-patch32", n: int = 1000) -> dict:
    """How well CLIP maps product photos to the CVAE's FashionMNIST categories (zero-shot)."""
    from PIL import Image

    from app.models.media_models import VisionModels

    v = VisionModels(model_id, "unused")
    idx = np.random.default_rng(0).choice(len(x), size=min(n, len(x)), replace=False)
    hits = 0
    for i in idx:
        img = Image.fromarray((x[i] * 255).astype(np.uint8)).convert("RGB").resize((112, 112))
        k, _ = v.category(img)
        hits += int(k == int(y[i]))
    return {"clip_fashion_zero_shot_accuracy": round(hits / len(idx), 4)}


def evaluate_xai(atsc_model_id: str, test_pairs: list[dict], n: int = 200, num_samples: int = 300, seed: int = 0) -> dict:
    """Mean LIME local fidelity (and R²) of ATSC explanations on held-out (sentence, aspect) pairs."""
    from app.models.text_models import SentimentClassifier
    from app.xai.explain import explain_text

    clf = SentimentClassifier(atsc_model_id)
    rng = np.random.default_rng(seed)
    chosen = [test_pairs[i] for i in rng.choice(len(test_pairs), size=min(n, len(test_pairs)), replace=False)]
    fid, r2 = [], []
    for p in chosen:
        e = explain_text(clf, p["sentence"], p["aspect"], num_samples=num_samples, seed=seed)
        fid.append(e["fidelity"])
        r2.append(e["r2"])
    return {"xai_fidelity": round(float(np.mean(fid)), 4), "xai_r2": round(float(np.mean(r2)), 4),
            "xai_samples": len(fid), "xai_fidelity_min": round(float(np.min(fid)), 4)}
