"""ReviewLens API — FastAPI app factory and routes."""
from __future__ import annotations

import logging
import time

import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from . import schemas
from .config import Settings, get_settings
from .core.cvae import CATEGORIES
from .core.text import detect_language, normalize
from .media import BadImage, gray_to_data_uri, load_image
from .metrics import MetricsStore
from .pipeline.chain import build_analyze_chain, run_analyze
from .pipeline.fusion import priority, run_absa, summarize
from .registry import ModelUnavailable, Registry, build_registry
from .xai.explain import explain_image, explain_text

VERSION = "1.0.0"
log = logging.getLogger("reviewlens.api")


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def _read_upload(upload: UploadFile | None, max_mb: float, kind: str) -> bytes | None:
    if upload is None:
        return None
    limit = int(max_mb * 1024 * 1024)
    data = upload.file.read(limit + 1)
    if len(data) > limit:
        raise ApiError(413, f"{kind}_too_large", f"The {kind} is larger than {max_mb:g} MB.")
    if not data:
        return None
    return data


def _image(data: bytes | None) -> Image.Image | None:
    if data is None:
        return None
    try:
        return load_image(data)
    except BadImage as exc:
        raise ApiError(400, "bad_image", str(exc)) from exc


def create_app(settings: Settings | None = None, registry: Registry | None = None) -> FastAPI:
    settings = settings or get_settings()
    registry = registry or build_registry(settings)
    metrics = MetricsStore(settings)
    chain = build_analyze_chain(registry, settings.variants)

    app = FastAPI(title="ReviewLens API", version=VERSION,
                  description="Aspect-based sentiment, long-context, multimodal and explainable review analysis.")
    app.state.settings, app.state.registry = settings, registry
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.origins,
        allow_origin_regex=settings.allowed_origin_regex or None,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        max_age=600,
    )

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError):
        return _error(exc.status, exc.code, exc.message)

    @app.exception_handler(ModelUnavailable)
    async def _model_error(_: Request, exc: ModelUnavailable):
        return _error(503, "model_unavailable", f"The {exc.name} model is not available yet. Try again shortly.")

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        where = ".".join(str(p) for p in first.get("loc", []) if p != "body")
        return _error(422, "invalid_request", f"{where}: {first.get('msg', 'invalid input')}".strip(": "))

    @app.get("/")
    def root():
        return {"name": "ReviewLens API", "version": VERSION, "docs": "/docs", "health": "/health"}

    @app.get("/health", response_model=schemas.HealthResponse)
    def health():
        return {"status": "ok", "mode": registry.mode, "version": VERSION, "models": registry.status()}

    @app.post("/warmup", response_model=schemas.HealthResponse)
    def warmup():
        registry.warmup()
        return {"status": "ok", "mode": registry.mode, "version": VERSION, "models": registry.status()}

    @app.get("/metrics", response_model=schemas.MetricsResponse)
    def get_metrics():
        return metrics.get()

    @app.post("/analyze", response_model=schemas.AnalyzeResponse)
    def analyze(
        text: str | None = Form(default=None),
        lang: str = Form(default="auto"),
        category: int | None = Form(default=None, ge=0, le=9),
        seed: int | None = Form(default=None, ge=0),
        image: UploadFile | None = File(default=None),
        audio: UploadFile | None = File(default=None),
    ):
        if lang not in ("auto", "vi", "en"):
            raise ApiError(400, "bad_language", "lang must be auto, vi or en.")
        if text and len(text) > settings.max_text_chars:
            raise ApiError(413, "text_too_long", f"The review is longer than {settings.max_text_chars:,} characters.")
        img = _image(_read_upload(image, settings.max_image_mb, "image"))
        audio_bytes = _read_upload(audio, settings.max_audio_mb, "audio")
        waveform, seconds = None, None
        if audio_bytes is not None:
            try:
                waveform = registry.decode_audio(audio_bytes)
            except Exception as exc:  # noqa: BLE001 — ffmpeg raises various errors
                raise ApiError(400, "bad_audio", "The audio could not be decoded. Use WebM, MP3, WAV or M4A.") from exc
            seconds = round(len(waveform) / 16_000, 2)
            if seconds > settings.max_audio_seconds:
                raise ApiError(413, "audio_too_long", f"The recording is longer than {settings.max_audio_seconds // 60} minutes.")
        if not (text and text.strip()) and img is None and waveform is None:
            raise ApiError(400, "empty_input", "Add a review text, a photo or a voice note.")

        out = run_analyze(chain, text=text, lang=lang, image=img, audio=waveform, category=category, seed=seed)
        return {
            "language": out["language"], "text": out["text"], "transcript": out["transcript"],
            "audio_seconds": seconds, "tokens": out["tokens"], "words": out["words"],
            "long_context": out["tokens"] > 512, "aspects": out["aspects"], "overall": out["overall"],
            "toxicity": out["toxicity"], "image": out["image_result"], "priority": out["priority"],
            "variants": out["variants"], "trace": out["_trace"], "total_ms": out["total_ms"],
            "warnings": out["_warnings"],
        }

    @app.post("/explain/text", response_model=schemas.ExplainTextResponse)
    def explain_text_route(req: schemas.ExplainTextRequest):
        text = normalize(req.text)
        return explain_text(registry.get("atsc"), text, req.aspect.strip(),
                            num_samples=req.num_samples or settings.lime_text_samples)

    @app.post("/explain/image", response_model=schemas.ExplainImageResponse)
    def explain_image_route(image: UploadFile = File(...), num_samples: int | None = Form(default=None, ge=50, le=500)):
        img = _image(_read_upload(image, settings.max_image_mb, "image"))
        if img is None:
            raise ApiError(400, "empty_input", "Add a photo to explain.")
        return explain_image(registry.get("vision"), img, num_samples=num_samples or settings.lime_image_samples)

    @app.post("/vqa", response_model=schemas.VQAResponse)
    def vqa(image: UploadFile = File(...), question: str = Form(..., min_length=1, max_length=200)):
        img = _image(_read_upload(image, settings.max_image_mb, "image"))
        if img is None:
            raise ApiError(400, "empty_input", "Add a photo to ask about.")
        answer, conf = registry.get("vision").vqa(img, question.strip())
        return {"question": question.strip(), "answer": answer, "confidence": conf}

    @app.post("/generate", response_model=schemas.GenerateResponse)
    def generate(req: schemas.GenerateRequest):
        gen = registry.get("generator")
        seed = req.seed if req.seed is not None else int(np.random.default_rng().integers(0, 2**31 - 1))
        probs = req.sentiment.model_dump()
        images = [gray_to_data_uri(a) for a in gen.generate(req.category, probs, req.n, seed)]
        sweep = [gray_to_data_uri(a) for a in gen.sweep(req.category, seed)] if req.sweep else []
        total = sum(probs.values())
        return {"category": CATEGORIES[req.category], "category_index": req.category,
                "probs": {k: round(v / total, 4) for k, v in probs.items()}, "seed": seed,
                "images": images, "sweep": sweep}

    @app.post("/absa/batch", response_model=schemas.BatchResponse)
    def absa_batch(req: schemas.BatchRequest):
        if len(req.texts) > settings.max_batch:
            raise ApiError(413, "too_many_reviews", f"Send at most {settings.max_batch} reviews at a time.")
        t0 = time.perf_counter()
        ate, atsc = registry.get("ate"), registry.get("atsc")
        reviews = []
        for i, raw in enumerate(req.texts):
            text = normalize(raw)[: settings.max_batch_chars]
            if not text:
                continue
            language = detect_language(text)
            aspects, overall = run_absa(ate, atsc, text, language)
            reviews.append({"index": i, "language": language, "text": text, "aspects": aspects,
                            "overall": overall, "priority": priority(aspects, None, None)})
        if not reviews:
            raise ApiError(400, "empty_input", "None of the rows contained review text.")
        return {"reviews": reviews, "summary": summarize([r["aspects"] for r in reviews]),
                "total_ms": round((time.perf_counter() - t0) * 1000, 1)}

    return app


def get_app() -> FastAPI:
    """Factory for uvicorn: `uvicorn app.main:get_app --factory`."""
    return create_app()
