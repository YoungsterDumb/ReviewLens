"""Every endpoint through FastAPI's TestClient, with deterministic stub models."""
import pytest

from app.config import Settings
from app.main import create_app
from app.registry import ModelUnavailable, StubRegistry

from .conftest import png_bytes

REVIEW_EN = "The screen is great. The keyboard is good. But the battery is terrible. The fan is loud."
REVIEW_VI = "Giao hàng nhanh, đóng gói cẩn thận. Nhưng đường may bị bung sau một tuần và màu tối hơn hình."


def test_health_and_root(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "stub" and body["status"] == "ok"
    assert set(body["models"]) == {"ate", "atsc", "longctx", "asr", "vision", "generator"}
    assert client.get("/").json()["docs"] == "/docs"


def test_warmup_loads_models(client):
    assert client.post("/warmup").status_code == 200


def test_analyze_text_only(client):
    r = client.post("/analyze", data={"text": REVIEW_EN})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["language"] == "en"
    terms = [(a["term"], a["sentiment"]) for a in body["aspects"]]
    assert ("screen", "positive") in terms and ("battery", "negative") in terms
    assert body["overall"]["mixed"] is True
    assert body["priority"]["level"] == "high"  # two negative aspects
    assert body["image"] is None and body["transcript"] is None
    assert body["toxicity"]["coverage"] == 1.0
    assert len(body["variants"]["images"]) == 4
    assert body["variants"]["images"][0].startswith("data:image/png;base64,")
    steps = {t["step"] for t in body["trace"]}
    assert {"ingest", "absa", "long_context", "vision", "fuse", "generate"} <= steps


def test_analyze_vietnamese_with_photo(client, red_png):
    r = client.post("/analyze", data={"text": REVIEW_VI, "seed": "7"},
                    files={"image": ("shoe.png", red_png, "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["language"] == "vi"
    assert body["image"]["category"] == "Sneaker"
    assert body["image"]["condition"]["defective"] > 0.6
    assert len(body["image"]["answers"]) == 3
    assert "photo_defective" in body["priority"]["reasons"]
    assert body["variants"]["category"] == "Sneaker"
    assert body["image"]["aspect_alignment"]


def test_analyze_voice_note_only(client):
    r = client.post("/analyze", files={"audio": ("note.webm", b"\x00\x01" * 8000, "audio/webm")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["transcript"] and body["text"] == body["transcript"]
    assert body["audio_seconds"] == pytest.approx(0.5)


def test_photo_only_review_still_generates_from_photo_category(client, blue_png):
    r = client.post("/analyze", files={"image": ("bag.png", blue_png, "image/png")})
    assert r.status_code == 200
    assert r.json()["aspects"] == [] and r.json()["variants"]["category"] == "Sneaker"


@pytest.mark.parametrize("data,files,status,code", [
    ({}, None, 400, "empty_input"),
    ({"text": "   "}, None, 400, "empty_input"),
    ({"text": "ok", "lang": "fr"}, None, 400, "bad_language"),
    ({"text": "x" * 5001}, None, 413, "text_too_long"),
    ({"text": "ok"}, {"image": ("a.png", b"not an image", "image/png")}, 400, "bad_image"),
    ({"text": "ok"}, {"image": ("a.png", b"0" * (1024 * 1024 + 5), "image/png")}, 413, "image_too_large"),
    ({"text": "ok", "category": "12"}, None, 422, "invalid_request"),
])
def test_analyze_rejects_bad_input(client, data, files, status, code):
    r = client.post("/analyze", data=data, files=files)
    assert r.status_code == status, r.text
    assert r.json()["error"]["code"] == code


def test_audio_too_long_is_rejected(stub_settings, stub_registry):
    from fastapi.testclient import TestClient

    settings = stub_settings.model_copy(update={"max_audio_seconds": 1})
    c = TestClient(create_app(settings, stub_registry))
    r = c.post("/analyze", files={"audio": ("long.wav", b"\x00" * 64000, "audio/wav")})
    assert r.status_code == 413 and r.json()["error"]["code"] == "audio_too_long"


def test_explain_text(client):
    r = client.post("/explain/text", json={"text": "The battery is terrible and slow.", "aspect": "battery"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["label"] == "negative"
    assert body["weights"] and {"word", "weight"} <= set(body["weights"][0])
    assert 0.0 <= body["fidelity"] <= 1.0
    assert any(w["word"] == "terrible" and w["weight"] > 0 for w in body["weights"])


def test_explain_text_validation(client):
    assert client.post("/explain/text", json={"text": "", "aspect": "x"}).status_code == 422
    assert client.post("/explain/text", json={"text": "ok", "aspect": "x", "num_samples": 5}).status_code == 422


def test_explain_image(client, red_png):
    r = client.post("/explain/image", files={"image": ("a.png", png_bytes((210, 50, 40), (96, 96)), "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["label"] == "defective"
    assert body["heatmap"].startswith("data:image/png;base64,")
    assert 0.0 <= body["fidelity"] <= 1.0


def test_vqa(client, red_png):
    r = client.post("/vqa", data={"question": "What color is it?"}, files={"image": ("a.png", red_png, "image/png")})
    assert r.status_code == 200 and r.json()["answer"] == "gray"
    assert client.post("/vqa", data={"question": ""}, files={"image": ("a.png", red_png, "image/png")}).status_code == 422


def test_generate_and_sweep(client):
    r = client.post("/generate", json={"category": 7, "sentiment": {"positive": 3, "neutral": 1, "negative": 0},
                                       "n": 6, "seed": 1, "sweep": True})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["images"]) == 6 and len(body["sweep"]) == 7
    assert body["probs"]["positive"] == pytest.approx(0.75)
    assert body["category"] == "Sneaker" and body["seed"] == 1


@pytest.mark.parametrize("payload", [
    {"category": 10, "sentiment": {"positive": 1, "neutral": 0, "negative": 0}},
    {"category": 1, "sentiment": {"positive": 0, "neutral": 0, "negative": 0}},
    {"category": 1, "sentiment": {"positive": -1, "neutral": 1, "negative": 1}},
    {"category": 1, "sentiment": {"positive": 1, "neutral": 0, "negative": 0}, "n": 50},
])
def test_generate_validation(client, payload):
    r = client.post("/generate", json=payload)
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_request"


def test_batch_summary(client):
    texts = [REVIEW_EN, "The battery is bad.", "", "Great screen, fast delivery."]
    r = client.post("/absa/batch", json={"texts": texts})
    assert r.status_code == 200, r.text
    body = r.json()
    assert [rv["index"] for rv in body["reviews"]] == [0, 1, 3]
    rows = {row["label"]: row for row in body["summary"]}
    assert rows["battery"]["mentions"] == 2 and rows["battery"]["negative"] == 2 and rows["battery"]["net"] == -100
    assert body["summary"][0]["mentions"] >= body["summary"][-1]["mentions"]


def test_batch_limits(client):
    assert client.post("/absa/batch", json={"texts": ["a"] * 201}).status_code == 413
    assert client.post("/absa/batch", json={"texts": ["", "  "]}).json()["error"]["code"] == "empty_input"
    assert client.post("/absa/batch", json={"texts": []}).status_code == 422


def test_metrics_lists_all_targets(client):
    body = client.get("/metrics").json()
    keys = [m["key"] for m in body["metrics"]]
    assert keys == ["ate_f1", "atsc_macro_f1", "long_context_coverage", "vae_recon_mse", "vqa_accuracy", "xai_fidelity"]
    vae = next(m for m in body["metrics"] if m["key"] == "vae_recon_mse")
    assert vae["direction"] == "lower" and vae["target"] == 0.01
    assert "ate" in body["models"]


@pytest.mark.parametrize("origin,allowed", [
    ("https://reviewlens.vercel.app", True),
    ("https://reviewlens-git-main-me.vercel.app", True),
    ("http://localhost:5173", True),
    ("https://evil.example.com", False),
])
def test_cors(client, origin, allowed):
    r = client.options("/analyze", headers={"Origin": origin, "Access-Control-Request-Method": "POST"})
    assert (r.headers.get("access-control-allow-origin") == origin) is allowed


def test_missing_optional_model_degrades_gracefully(stub_settings):
    from fastapi.testclient import TestClient

    class NoToxicity(StubRegistry):
        def _build_loaders(self):
            loaders = super()._build_loaders()

            def boom():
                raise RuntimeError("weights not published yet")

            loaders["longctx"] = boom
            return loaders

    reg = NoToxicity(stub_settings)
    c = TestClient(create_app(stub_settings, reg))
    r = c.post("/analyze", data={"text": REVIEW_EN})
    assert r.status_code == 200
    assert r.json()["toxicity"] is None
    assert r.json()["warnings"][0].startswith("toxicity_unavailable")
    assert c.get("/health").json()["models"]["longctx"]["state"] == "error"


def test_missing_core_model_returns_503(stub_settings):
    from fastapi.testclient import TestClient

    class NoAte(StubRegistry):
        def _build_loaders(self):
            loaders = super()._build_loaders()
            loaders["ate"] = lambda: (_ for _ in ()).throw(RuntimeError("missing"))
            return loaders

    c = TestClient(create_app(stub_settings, NoAte(stub_settings)), raise_server_exceptions=False)
    r = c.post("/analyze", data={"text": REVIEW_EN})
    assert r.status_code == 503 and r.json()["error"]["code"] == "model_unavailable"


def test_settings_repo_names():
    s = Settings(hf_namespace="me")
    assert s.repo("ate") == "me/reviewlens-ate"
    assert Settings(ate_model="org/custom").repo("ate") == "org/custom"
    assert s.repo("metrics") == "me/reviewlens-metrics"
    assert ModelUnavailable("ate", "x").name == "ate"
