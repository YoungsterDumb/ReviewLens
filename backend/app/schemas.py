from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Sentiment = Literal["negative", "neutral", "positive"]


class Probs(BaseModel):
    negative: float
    neutral: float
    positive: float


class Aspect(BaseModel):
    term: str
    start: int
    end: int
    category: str | None = None
    sentence: str
    sentiment: Sentiment
    confidence: float
    probs: Probs
    extraction_score: float


class Overall(BaseModel):
    label: Sentiment
    mixed: bool
    probs: Probs


class Toxicity(BaseModel):
    scores: dict[str, float]
    max_label: str
    max_score: float
    tokens: int
    coverage: float


class QA(BaseModel):
    question: str
    answer: str
    confidence: float | None = None


class AlignmentItem(BaseModel):
    term: str
    score: float


class ImageResult(BaseModel):
    category: str | None
    category_index: int | None
    category_confidence: float
    condition: dict[str, float]
    answers: list[QA]
    aspect_alignment: list[AlignmentItem] = []


class Priority(BaseModel):
    level: Literal["low", "medium", "high"]
    reasons: list[str]


class Variants(BaseModel):
    category: str
    category_index: int
    probs: Probs
    images: list[str]


class TraceStep(BaseModel):
    step: str
    start_ms: float
    ms: float


class AnalyzeResponse(BaseModel):
    language: Literal["vi", "en"]
    text: str
    transcript: str | None
    audio_seconds: float | None = None
    tokens: int
    words: int
    long_context: bool
    aspects: list[Aspect]
    overall: Overall
    toxicity: Toxicity | None
    image: ImageResult | None
    priority: Priority
    variants: Variants | None
    trace: list[TraceStep]
    total_ms: float
    warnings: list[str] = []


class ExplainTextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    aspect: str = Field(min_length=1, max_length=300)
    num_samples: int | None = Field(default=None, ge=50, le=1000)


class WordWeight(BaseModel):
    word: str
    weight: float


class ExplainTextResponse(BaseModel):
    aspect: str
    label: Sentiment
    probs: Probs
    weights: list[WordWeight]
    fidelity: float
    r2: float
    num_samples: int


class ExplainImageResponse(BaseModel):
    label: Literal["defective", "good"]
    probability: float
    heatmap: str
    fidelity: float
    r2: float
    num_samples: int


class VQAResponse(BaseModel):
    question: str
    answer: str
    confidence: float | None = None


class GenerateRequest(BaseModel):
    category: int = Field(ge=0, le=9)
    sentiment: Probs
    n: int = Field(default=8, ge=1, le=16)
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)
    sweep: bool = False

    @field_validator("sentiment")
    @classmethod
    def _non_negative(cls, v: Probs) -> Probs:
        if min(v.negative, v.neutral, v.positive) < 0 or (v.negative + v.neutral + v.positive) <= 0:
            raise ValueError("sentiment shares must be >= 0 and not all zero")
        return v


class GenerateResponse(BaseModel):
    category: str
    category_index: int
    probs: Probs
    seed: int
    images: list[str]
    sweep: list[str] = []


class BatchRequest(BaseModel):
    texts: list[str] = Field(min_length=1)


class BatchReview(BaseModel):
    index: int
    language: Literal["vi", "en"]
    text: str
    aspects: list[Aspect]
    overall: Overall
    priority: Priority


class SummaryRow(BaseModel):
    key: str
    label: str
    is_category: bool
    mentions: int
    positive: int
    neutral: int
    negative: int
    net: int
    example: str


class BatchResponse(BaseModel):
    reviews: list[BatchReview]
    summary: list[SummaryRow]
    total_ms: float


class Metric(BaseModel):
    key: str
    value: float | None
    target: float
    direction: Literal["higher", "lower"]
    met: bool | None
    details: dict[str, float | str | None] = {}


class MetricsResponse(BaseModel):
    updated_at: str | None
    source: str
    metrics: list[Metric]
    models: dict[str, str]


class ModelStatus(BaseModel):
    state: Literal["idle", "loading", "ready", "error"]
    error: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"]
    mode: Literal["live", "stub"]
    version: str
    models: dict[str, ModelStatus]
