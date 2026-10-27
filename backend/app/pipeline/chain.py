"""/analyze as one LangChain chain: ingest -> (ABSA | long context | vision) -> fuse -> generate.

LangChain is used only to compose runnables (no LLM). RunnableParallel runs the
three model branches on a thread pool; torch releases the GIL, so they overlap.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable

import numpy as np
from langchain_core.runnables import RunnableLambda, RunnableParallel

from ..core.cvae import CATEGORIES
from ..core.text import count_words, detect_language, normalize
from ..media import gray_to_data_uri
from ..models.media_models import AUTO_QUESTIONS
from ..registry import ModelUnavailable, Registry
from .fusion import pick_category, priority, run_absa

log = logging.getLogger("reviewlens.chain")
State = dict[str, Any]


def _timed(name: str, fn: Callable[[State], Any]) -> Callable[[State], Any]:
    def run(value: State) -> Any:
        # After RunnableParallel the input is {"absa": …, "state": …}; the trace lives on the state.
        state = value["state"] if "_trace" not in value and "state" in value else value
        start = time.perf_counter()
        result = fn(value)
        state["_trace"].append({
            "step": name,
            "start_ms": round((start - state["_t0"]) * 1000, 1),
            "ms": round((time.perf_counter() - start) * 1000, 1),
        })
        return result

    return run


def build_analyze_chain(registry: Registry, n_variants: int = 4):
    def ingest(state: State) -> State:
        text = normalize(state.get("text") or "")
        transcript = None
        if state.get("audio") is not None:
            transcript = registry.get("asr").transcribe(state["audio"]) or None
            if transcript:
                text = f"{text}\n{transcript}".strip() if text else transcript
        lang = state.get("lang") or "auto"
        language = lang if lang in ("vi", "en") else (detect_language(text) if text else "en")
        tokens = registry.get("ate").count_tokens(text) + 2 if text else 0
        state.update(text=text, transcript=transcript, language=language, tokens=tokens, words=count_words(text))
        return state

    def absa(state: State) -> tuple[list[dict], dict]:
        return run_absa(registry.get("ate"), registry.get("atsc"), state["text"], state["language"])

    def longctx(state: State) -> dict | None:
        if not state["text"]:
            return None
        try:
            scores, n_tokens = registry.get("longctx").predict(state["text"])
        except ModelUnavailable as exc:
            state["_warnings"].append(f"toxicity_unavailable:{exc.reason[:80]}")
            return None
        top = max(scores, key=scores.get)
        return {"scores": {k: round(v, 4) for k, v in scores.items()}, "max_label": top,
                "max_score": round(scores[top], 4), "tokens": n_tokens,
                "coverage": round(min(1.0, 4096 / max(n_tokens, 1)), 4)}

    def vision(state: State) -> dict | None:
        image = state.get("image")
        if image is None:
            return None
        try:
            v = registry.get("vision")
        except ModelUnavailable as exc:
            state["_warnings"].append(f"vision_unavailable:{exc.reason[:80]}")
            return None
        idx, conf = v.category(image)
        defective, good = (float(x) for x in v.condition_proba([image])[0])
        answers = []
        for q in AUTO_QUESTIONS:
            a, c = v.vqa(image, q)
            answers.append({"question": q, "answer": a, "confidence": c})
        return {"category": CATEGORIES[idx] if idx is not None else None, "category_index": idx,
                "category_confidence": round(conf, 4),
                "condition": {"defective": round(defective, 4), "good": round(good, 4)},
                "answers": answers, "aspect_alignment": []}

    def fuse(merged: dict) -> State:
        state: State = merged["state"]
        aspects, overall = merged["absa"]
        image_result, toxicity = merged["vision"], merged["longctx"]
        if image_result and aspects:
            terms = list(dict.fromkeys(a["term"] for a in aspects))[:8]
            scores = registry.get("vision").align(state["image"], terms)
            image_result["aspect_alignment"] = [{"term": t, "score": round(s, 4)} for t, s in zip(terms, scores)]
        state.update(
            aspects=aspects, overall=overall, toxicity=toxicity, image_result=image_result,
            priority=priority(aspects, toxicity, image_result["condition"] if image_result else None),
        )
        if state.get("category") is not None:
            state["gen_category"] = int(state["category"])
        elif image_result and image_result["category_index"] is not None:
            state["gen_category"] = image_result["category_index"]
        else:
            state["gen_category"] = pick_category(state["text"])
        return state

    def generate(state: State) -> State:
        state["variants"] = None
        if n_variants <= 0:
            return state
        try:
            gen = registry.get("generator")
        except ModelUnavailable as exc:
            state["_warnings"].append(f"generator_unavailable:{exc.reason[:80]}")
            return state
        cat, probs = state["gen_category"], state["overall"]["probs"]
        seed = int(state.get("seed") if state.get("seed") is not None else np.random.default_rng().integers(0, 2**31 - 1))
        images = gen.generate(cat, probs, n_variants, seed)
        state["variants"] = {"category": CATEGORIES[cat], "category_index": cat, "probs": probs,
                             "images": [gray_to_data_uri(a) for a in images]}
        return state

    return (
        RunnableLambda(_timed("ingest", ingest), name="ingest")
        | RunnableParallel(
            absa=RunnableLambda(_timed("absa", absa), name="absa"),
            longctx=RunnableLambda(_timed("long_context", longctx), name="long_context"),
            vision=RunnableLambda(_timed("vision", vision), name="vision"),
            state=RunnableLambda(lambda s: s, name="carry"),
        )
        | RunnableLambda(_timed("fuse", fuse), name="fuse")
        | RunnableLambda(_timed("generate", generate), name="generate")
    )


def run_analyze(chain, **inputs: Any) -> State:
    state: State = {**inputs, "_t0": time.perf_counter(), "_trace": [], "_warnings": []}
    out = chain.invoke(state)
    out["total_ms"] = round((time.perf_counter() - state["_t0"]) * 1000, 1)
    out["_trace"].sort(key=lambda t: t["start_ms"])
    return out
