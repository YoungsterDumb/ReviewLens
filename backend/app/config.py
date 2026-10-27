from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


class Settings(BaseSettings):
    """All runtime configuration, read from REVIEWLENS_* environment variables."""

    model_config = SettingsConfigDict(env_prefix="REVIEWLENS_", env_file=".env", extra="ignore")

    # Where the fine-tuned models live on the Hugging Face Hub.
    hf_namespace: str = "DatDuc"
    ate_model: str = ""
    atsc_model: str = ""
    longctx_model: str = ""
    metrics_repo: str = ""
    # Pretrained models used as-is.
    whisper_model: str = "openai/whisper-small"
    blip_model: str = "Salesforce/blip-vqa-base"
    clip_model: str = "openai/clip-vit-base-patch32"
    cvae_dir: str = str(ASSETS_DIR / "cvae")

    # stub = deterministic fake models (tests, local UI work without weights).
    stub: bool = False

    allowed_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    allowed_origin_regex: str = r"https://[a-z0-9-]+\.vercel\.app"

    max_text_chars: int = 20_000
    max_image_mb: float = 8
    max_audio_mb: float = 25
    max_audio_seconds: int = 120
    max_batch: int = 200
    max_batch_chars: int = 3_000
    absa_window_tokens: int = 256
    lime_text_samples: int = 300
    lime_image_samples: int = 150
    variants: int = 4
    torch_threads: int = 0

    def repo(self, name: str) -> str:
        explicit = getattr(self, f"{name}_model", "") if name != "metrics" else self.metrics_repo
        return explicit or f"{self.hf_namespace}/reviewlens-{name.replace('_', '-')}"

    @property
    def origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
