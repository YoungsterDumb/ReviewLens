"""Deploy the ReviewLens API to Modal: `modal deploy modal_app.py` (from the backend/ folder).

One autoscaled CPU container serves every route; it sleeps after 10 idle minutes.
Model weights download from the Hugging Face Hub on first use and are cached in a
Modal Volume, so later cold starts skip the download.
"""
from __future__ import annotations

import os
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
APP_NAME = os.environ.get("REVIEWLENS_MODAL_APP", "reviewlens-api")

# Runtime configuration baked into the image; override at deploy time through the environment.
CONFIG = {
    "KERAS_BACKEND": "torch",
    "HF_HOME": "/cache/hf",
    "TOKENIZERS_PARALLELISM": "false",
    "REVIEWLENS_HF_NAMESPACE": os.environ.get("REVIEWLENS_HF_NAMESPACE", "DatDuc"),
    "REVIEWLENS_ALLOWED_ORIGINS": os.environ.get("REVIEWLENS_ALLOWED_ORIGINS", "http://localhost:5173"),
    "REVIEWLENS_ALLOWED_ORIGIN_REGEX": os.environ.get("REVIEWLENS_ALLOWED_ORIGIN_REGEX", r"https://[a-z0-9-]+\.vercel\.app"),
    "REVIEWLENS_TORCH_THREADS": "4",
}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install("torch==2.14.1", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install_from_requirements(str(HERE / "requirements.txt"))
    .env(CONFIG)
    .add_local_dir(str(HERE / "app"), "/root/app")
    .add_local_dir(str(HERE / "assets"), "/root/assets")
)

app = modal.App(APP_NAME, image=image)
hf_cache = modal.Volume.from_name("reviewlens-hf-cache", create_if_missing=True)


@app.function(
    cpu=4.0,
    memory=12288,
    timeout=600,
    scaledown_window=600,
    volumes={"/cache": hf_cache},
)
@modal.concurrent(max_inputs=8)
@modal.asgi_app(label="reviewlens-api")
def api():
    import sys

    sys.path.insert(0, "/root")
    from app.main import create_app
    from app.registry import build_registry
    from app.config import get_settings

    settings = get_settings()
    registry = build_registry(settings)
    registry.on_load = lambda _name: hf_cache.commit()  # keep downloaded weights for the next cold start
    return create_app(settings, registry)
