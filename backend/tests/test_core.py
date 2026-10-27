"""Pure-Python core: text handling, BIO tagging, fidelity maths, CVAE conditions."""
import unicodedata

import numpy as np
import pytest

from app.core import bio, fidelity, text
from app.core.cvae import CATEGORIES, brightness_gain, condition_vector, random_sentiment_mix, sentiment_score


def test_normalize_composes_vietnamese_and_tidies_space():
    decomposed = unicodedata.normalize("NFD", "Pin  trâu\r\n\n\n\ncamera ổn ")
    out = text.normalize(decomposed)
    assert out == "Pin trâu\n\ncamera ổn"
    assert unicodedata.is_normalized("NFC", out)


@pytest.mark.parametrize("sample,lang", [
    ("Pin trâu nhưng camera chụp đêm bị nhoè", "vi"),
    ("Máy dùng ổn, pin hơi yếu", "vi"),
    ("The battery life is great but the fan is loud.", "en"),
    ("", "en"),
])
def test_detect_language(sample, lang):
    assert text.detect_language(sample) == lang


def test_sentence_spans_keep_offsets():
    s = "Great screen. Battery is bad!  Fan is loud\nok"
    spans = text.sentence_spans(s)
    assert [sp.slice(s) for sp in spans] == ["Great screen.", "Battery is bad!", "Fan is loud", "ok"]


def test_windows_cover_every_word_and_respect_limit():
    s = " ".join(f"w{i}" + ("." if i % 5 == 4 else "") for i in range(203))
    count = lambda t: len(text.word_spans(t))  # noqa: E731
    windows = text.build_windows(s, count, max_tokens=24)
    assert all(count(w.slice(s)) <= 24 for w in windows)
    covered = set()
    for w in windows:
        covered |= {ws for ws in text.word_spans(s) if w.start <= ws.start and ws.end <= w.end}
    assert covered == set(text.word_spans(s))


def test_overlong_sentence_is_split_with_overlap_not_dropped():
    s = " ".join(f"w{i}" for i in range(100))  # one sentence, no punctuation
    count = lambda t: len(text.word_spans(t))  # noqa: E731
    windows = text.build_windows(s, count, max_tokens=30, overlap_words=5)
    assert len(windows) >= 4
    assert windows[0].start == 0 and windows[-1].end == len(s)
    assert windows[1].start < windows[0].end  # overlap


def test_bio_round_trip_with_types():
    s = "Pin trâu nhưng camera chụp tối."
    spans = [bio.TaggedSpan(0, 8, "BATTERY"), bio.TaggedSpan(15, 30, "CAMERA")]
    words, tags = bio.spans_to_word_tags(s, spans)
    assert tags[:2] == ["B-BATTERY", "I-BATTERY"]
    assert "B-CAMERA" in tags
    decoded = bio.word_tags_to_spans(words, tags)
    assert [(d.start, d.end, d.type) for d in decoded] == [(0, 8, "BATTERY"), (15, 30, "CAMERA")]


def test_stray_inside_tag_starts_a_span():
    s = "battery life is good"
    words = text.word_spans(s)
    spans = bio.word_tags_to_spans(words, ["I-ASP", "I-ASP", "O", "O"])
    assert [(sp.start, sp.end) for sp in spans] == [(0, 12)]


def test_align_word_labels_labels_first_subtoken_only():
    assert bio.align_word_labels([None, 0, 0, 1, None], [1, 2]) == [-100, 1, -100, 2, -100]


def test_find_term_offsets_prefers_whole_words_and_handles_repeats():
    s = "The battery and the batteryless mode; battery again"
    assert bio.find_term_offsets(s, ["battery", "battery"]) == [(4, 11), (38, 45)]
    assert bio.find_term_offsets(s, ["missing"]) == [None]


def test_label_list_is_stable():
    assert bio.build_label_list(["ASP"]) == ["O", "B-ASP", "I-ASP"]
    assert bio.build_label_list(["PRICE", "BATTERY", "PRICE"]) == ["O", "B-BATTERY", "I-BATTERY", "B-PRICE", "I-PRICE"]


def test_fidelity_math():
    f = np.array([0.2, 0.5, 0.9])
    assert fidelity.local_fidelity(f, f) == 1.0
    assert fidelity.local_fidelity(f, f + 0.1) == pytest.approx(0.9)
    g = fidelity.surrogate_predict(np.array([[1, 0], [1, 1]]), {0: 0.3, 1: 0.5}, 0.1)
    assert g.tolist() == pytest.approx([0.4, 0.9])
    masks = fidelity.random_masks(6, 50, np.random.default_rng(0))
    assert masks.shape == (50, 6) and (masks.sum(axis=1) < 6).all() and (masks.sum(axis=1) >= 1).all()
    d = fidelity.cosine_distance_to_full(np.array([[1, 1, 1, 1], [1, 0, 0, 0]]))
    assert d[0] == pytest.approx(0.0) and d[1] == pytest.approx(50.0)


def test_cvae_condition_and_brightness():
    anchors = {k: np.eye(50, dtype=np.float32)[i] for i, k in enumerate(("negative", "neutral", "positive"))}
    c = condition_vector(7, {"positive": 2, "neutral": 1, "negative": 1}, anchors)
    assert c.shape == (len(CATEGORIES) + 50,)
    assert c[7] == 1.0 and c[:10].sum() == 1.0
    assert c[10:13].tolist() == pytest.approx([0.25, 0.25, 0.5])
    assert brightness_gain(-1) == pytest.approx(0.45) and brightness_gain(1) == pytest.approx(1.0)
    assert brightness_gain(-0.5) < brightness_gain(0) < brightness_gain(0.5)
    assert sentiment_score({"positive": 0.7, "negative": 0.2}) == pytest.approx(0.5)
    mix = random_sentiment_mix(np.random.default_rng(0), 10)
    assert mix.shape == (10, 3) and np.allclose(mix.sum(axis=1), 1.0)
