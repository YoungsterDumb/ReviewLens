"""Publishing to the Hugging Face Hub: models and the shared metrics.json the API reads."""
from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def push_model(model, tokenizer, repo_id: str, message: str) -> str:
    model.push_to_hub(repo_id, commit_message=message, safe_serialization=True)
    if tokenizer is not None:
        tokenizer.push_to_hub(repo_id, commit_message=message)
    return f"https://huggingface.co/{repo_id}"


def push_folder(folder: Path, repo_id: str, message: str) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, exist_ok=True)
    api.upload_folder(folder_path=str(folder), repo_id=repo_id, commit_message=message)
    return f"https://huggingface.co/{repo_id}"


def update_metrics(repo_id: str, new: dict, extra: dict | None = None) -> dict:
    """Merge new values into metrics.json on the Hub (each notebook adds its own keys)."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    api.create_repo(repo_id, exist_ok=True)
    try:
        current = json.loads(Path(hf_hub_download(repo_id, "metrics.json", force_download=True)).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — first notebook to run creates the file
        current = {"metrics": {}}
    current["metrics"] = {**current.get("metrics", {}), **{k: v for k, v in new.items() if v is not None}}
    current.update(extra or {})
    current["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "metrics.json"
        p.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
        api.upload_file(path_or_fileobj=str(p), path_in_repo="metrics.json", repo_id=repo_id,
                        commit_message=f"metrics: {', '.join(sorted(new))}")
    return current
