"""Evaluation metrics vs targets. Notebooks publish metrics.json to the Hub; a local copy is the fallback."""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

from .config import ASSETS_DIR, Settings

log = logging.getLogger("reviewlens.metrics")

# key -> (target, direction, keys copied into "details")
TARGETS: dict[str, tuple[float, str, tuple[str, ...]]] = {
    "ate_f1": (0.80, "higher", ("ate_f1_vi", "ate_f1_all")),
    "atsc_macro_f1": (0.80, "higher", ("atsc_accuracy", "atsc_macro_f1_en", "atsc_macro_f1_vi")),
    "long_context_coverage": (1.0, "higher", ("long_context_auc", "long_context_max_tokens")),
    "vae_recon_mse": (0.01, "lower", ("vae_kl", "vae_total_loss")),
    "vqa_accuracy": (0.70, "higher", ("vqa_questions",)),
    "xai_fidelity": (0.90, "higher", ("xai_samples", "xai_r2")),
}
_TTL_SECONDS = 600


class MetricsStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._cache: tuple[float, dict, str] | None = None
        self._lock = threading.Lock()

    def _load(self) -> tuple[dict, str]:
        local_path = Path(ASSETS_DIR) / "metrics.json"
        local = json.loads(local_path.read_text(encoding="utf-8")) if local_path.exists() else {"metrics": {}}
        if self.settings.stub:
            return local, "local"
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(self.settings.repo("metrics"), "metrics.json")
            remote = json.loads(Path(path).read_text(encoding="utf-8"))
            merged = {**local, **remote, "metrics": {**local.get("metrics", {}), **remote.get("metrics", {})}}
            return merged, "hub"
        except Exception as exc:  # noqa: BLE001 — the Hub may be unreachable or the repo not created yet
            log.warning("metrics: using local copy (%s)", exc)
            return local, "local"

    def get(self) -> dict:
        with self._lock:
            if self._cache and time.time() - self._cache[0] < _TTL_SECONDS:
                data, source = self._cache[1], self._cache[2]
            else:
                data, source = self._load()
                self._cache = (time.time(), data, source)
        values = data.get("metrics", {})
        out = []
        for key, (target, direction, extra) in TARGETS.items():
            v = values.get(key)
            met = None if v is None else (v >= target if direction == "higher" else v < target)
            out.append({"key": key, "value": v, "target": target, "direction": direction, "met": met,
                        "details": {k: values.get(k) for k in extra}})
        models = {n: self.settings.repo(n) for n in ("ate", "atsc", "longctx")}
        models.update(whisper=self.settings.whisper_model, blip=self.settings.blip_model,
                      clip=self.settings.clip_model, cvae=data.get("cvae_repo", "bundled with the API"))
        return {"updated_at": data.get("updated_at"), "source": source, "metrics": out, "models": models}
