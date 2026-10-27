"""Lazy, thread-safe model loading. Each model loads on first use (or on /warmup)."""
from __future__ import annotations

import logging
import threading
from typing import Any, Callable

from .config import Settings

log = logging.getLogger("reviewlens.registry")

MODEL_NAMES = ("ate", "atsc", "longctx", "asr", "vision", "generator")


class ModelUnavailable(RuntimeError):
    def __init__(self, name: str, reason: str):
        super().__init__(f"{name}: {reason}")
        self.name, self.reason = name, reason


class Registry:
    mode = "live"
    on_load: Callable[[str], None] | None = None  # called once after each model loads

    def __init__(self, settings: Settings):
        self.settings = settings
        self._models: dict[str, Any] = {}
        self._state: dict[str, str] = {n: "idle" for n in MODEL_NAMES}
        self._errors: dict[str, str] = {}
        self._locks = {n: threading.Lock() for n in MODEL_NAMES}
        self._loaders: dict[str, Callable[[], Any]] = self._build_loaders()
        if settings.torch_threads:
            try:
                import torch

                torch.set_num_threads(settings.torch_threads)
            except ImportError:  # pragma: no cover
                pass

    def _build_loaders(self) -> dict[str, Callable[[], Any]]:
        s = self.settings

        def ate():
            from .models.text_models import AspectExtractor

            return AspectExtractor(s.repo("ate"), window_tokens=s.absa_window_tokens)

        def atsc():
            from .models.text_models import SentimentClassifier

            return SentimentClassifier(s.repo("atsc"))

        def longctx():
            from .models.text_models import ToxicityModel

            return ToxicityModel(s.repo("longctx"))

        def asr():
            from .models.media_models import SpeechToText

            return SpeechToText(s.whisper_model)

        def vision():
            from .models.media_models import VisionModels

            v = VisionModels(s.clip_model, s.blip_model)
            v.warm()
            return v

        def generator():
            from .models.media_models import VariantGenerator

            return VariantGenerator(s.cvae_dir)

        return {"ate": ate, "atsc": atsc, "longctx": longctx, "asr": asr, "vision": vision, "generator": generator}

    def get(self, name: str) -> Any:
        model = self._models.get(name)
        if model is not None:
            return model
        with self._locks[name]:
            if name in self._models:
                return self._models[name]
            self._state[name] = "loading"
            try:
                model = self._loaders[name]()
            except Exception as exc:  # noqa: BLE001 — surfaced to the client as 503
                log.exception("failed to load %s", name)
                self._state[name] = "error"
                self._errors[name] = f"{type(exc).__name__}: {exc}"[:300]
                raise ModelUnavailable(name, self._errors[name]) from exc
            self._models[name] = model
            self._state[name] = "ready"
            self._errors.pop(name, None)
            if self.on_load:
                try:
                    self.on_load(name)
                except Exception:  # noqa: BLE001 — hooks (e.g. cache persistence) are best effort
                    log.warning("on_load hook failed for %s", name)
            return model

    def decode_audio(self, data: bytes):
        """Decode any browser audio to 16 kHz mono float32 (ffmpeg) without loading Whisper."""
        from .models.media_models import SpeechToText

        return SpeechToText.decode(data)

    def status(self) -> dict[str, dict[str, str | None]]:
        return {n: {"state": self._state[n], "error": self._errors.get(n)} for n in MODEL_NAMES}

    def warmup(self, names: tuple[str, ...] = MODEL_NAMES) -> None:
        """Load models in a background thread, cheapest first, so /analyze is usable early."""

        def run():
            for n in names:
                try:
                    self.get(n)
                except ModelUnavailable:
                    continue

        threading.Thread(target=run, name="warmup", daemon=True).start()


class StubRegistry(Registry):
    mode = "stub"

    def _build_loaders(self) -> dict[str, Callable[[], Any]]:
        from .models import stubs

        return {
            "ate": stubs.StubAspectExtractor,
            "atsc": stubs.StubSentimentClassifier,
            "longctx": stubs.StubToxicityModel,
            "asr": stubs.StubSpeechToText,
            "vision": stubs.StubVisionModels,
            "generator": stubs.StubVariantGenerator,
        }

    def decode_audio(self, data: bytes):
        from .models.stubs import StubSpeechToText

        return StubSpeechToText.decode(data)


def build_registry(settings: Settings) -> Registry:
    return StubRegistry(settings) if settings.stub else Registry(settings)
