"""Real model wrappers on tiny, randomly initialised (or rigged) transformers — no downloads."""
import numpy as np
import pytest

from app.core.text import word_spans
from app.models.text_models import AspectExtractor, SentimentClassifier, ToxicityModel, sentence_for
from app.core.text import sentence_spans
from app.pipeline.fusion import run_absa, summarize, priority, pick_category

from .conftest import random_words


def test_aspect_extractor_decodes_spans(tiny_tokenizer, rigged_tagger):
    ate = AspectExtractor("unused", tokenizer=tiny_tokenizer, model=rigged_tagger)
    s = "The battery life is great. The screen is dim."
    spans = ate.extract(s)
    assert [x["term"] for x in spans] == ["battery life", "screen"]
    assert all(x["category"] is None and 0.9 < x["score"] <= 1.0 for x in spans)
    assert s[spans[0]["start"]:spans[0]["end"]] == "battery life"


def test_aspect_extractor_reads_long_text_without_truncation(tiny_tokenizer, rigged_tagger):
    ate = AspectExtractor("unused", window_tokens=64, batch_size=4, tokenizer=tiny_tokenizer, model=rigged_tagger)
    s = random_words(1500) + " The fan is loud."
    spans = ate.extract(s)
    assert rigged_tagger.max_seen <= 64 + 2  # every window fits, so nothing is cut
    assert spans[-1]["term"] == "fan"  # the very end of a 1,500-word review is still read
    expected = sum(1 for w in word_spans(s) if s[w.start:w.end].lower() in {"battery", "screen", "keyboard", "fan"})
    assert len(spans) == expected


def test_sentiment_classifier_on_tiny_albert(tiny_tokenizer):
    import torch
    from transformers import AlbertConfig, AlbertForSequenceClassification

    torch.manual_seed(0)
    cfg = AlbertConfig(vocab_size=tiny_tokenizer.vocab_size, embedding_size=16, hidden_size=32, num_attention_heads=2,
                       intermediate_size=37, num_hidden_layers=1, num_labels=3,
                       id2label={0: "positive", 1: "negative", 2: "neutral"},
                       label2id={"positive": 0, "negative": 1, "neutral": 2})
    model = AlbertForSequenceClassification(cfg)
    clf = SentimentClassifier("unused", batch_size=2, tokenizer=tiny_tokenizer, model=model)
    assert clf.order == [1, 2, 0]  # columns reordered to negative, neutral, positive
    probs = clf.predict_proba([("the screen is great", "screen"), ("bad fan", "fan"), ("it is dim", "it")])
    assert probs.shape == (3, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_toxicity_model_on_tiny_longformer_reads_past_512_tokens(tiny_tokenizer):
    import torch
    from transformers import LongformerConfig, LongformerForSequenceClassification

    torch.manual_seed(0)
    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    cfg = LongformerConfig(vocab_size=tiny_tokenizer.vocab_size, hidden_size=32, num_attention_heads=2, intermediate_size=37,
                           num_hidden_layers=1, attention_window=[16], max_position_embeddings=1100, pad_token_id=0,
                           num_labels=6, problem_type="multi_label_classification",
                           id2label=dict(enumerate(labels)), label2id={l: i for i, l in enumerate(labels)})
    model = LongformerForSequenceClassification(cfg)
    tox = ToxicityModel("unused", max_length=1024, tokenizer=tiny_tokenizer, model=model)
    scores, n = tox.predict(random_words(700))
    assert n > 512
    assert set(scores) == set(labels)
    assert all(0.0 <= v <= 1.0 for v in scores.values())


def test_sentence_for_picks_the_covering_sentence():
    s = "Great screen. Battery is bad. Fan ok."
    sents = sentence_spans(s)
    assert sentence_for(s, 14, 21, sents) == "Battery is bad."
    assert sentence_for(s, 0, len(s), sents) == s


def test_run_absa_with_stubs_and_overall(stub_registry):
    ate, atsc = stub_registry.get("ate"), stub_registry.get("atsc")
    aspects, overall = run_absa(ate, atsc, "The screen is great. The battery is terrible.", "en")
    assert [(a["term"], a["sentiment"]) for a in aspects] == [("screen", "positive"), ("battery", "negative")]
    assert overall["mixed"] is True
    assert pytest.approx(sum(overall["probs"].values()), abs=1e-3) == 1.0
    none, general = run_absa(ate, atsc, "I love it.", "en")
    assert none == [] and general["label"] in {"negative", "neutral", "positive"}


def test_priority_rules():
    neg = {"sentiment": "negative"}
    pos = {"sentiment": "positive"}
    assert priority([pos], None, None)["level"] == "low"
    assert priority([neg, pos], None, None)["level"] == "medium"
    assert priority([neg, neg], None, None)["level"] == "high"
    assert priority([pos], {"max_score": 0.8, "max_label": "insult"}, None)["level"] == "high"
    assert priority([pos], None, {"defective": 0.7})["reasons"] == ["photo_defective"]


def test_summary_groups_by_category_or_term():
    reviews = [
        [{"term": "Pin trâu", "category": "BATTERY", "sentiment": "positive", "confidence": 0.9},
         {"term": "Screen", "category": None, "sentiment": "negative", "confidence": 0.8}],
        [{"term": "pin yếu", "category": "BATTERY", "sentiment": "negative", "confidence": 0.95},
         {"term": "screen.", "category": None, "sentiment": "negative", "confidence": 0.7}],
    ]
    rows = {r["key"]: r for r in summarize(reviews)}
    assert rows["cat:BATTERY"]["mentions"] == 2 and rows["cat:BATTERY"]["net"] == 0
    assert rows["cat:BATTERY"]["label"] == "Pin" and rows["cat:BATTERY"]["example"] == "pin yếu"
    assert rows["term:screen"]["negative"] == 2 and rows["term:screen"]["net"] == -100


@pytest.mark.parametrize("review,idx", [
    ("Đôi giày này đi rất êm", 7), ("The sandal strap broke", 5), ("Áo khoác dày dặn", 4),
    ("Laptop is fast", 8),
])
def test_pick_category(review, idx):
    assert pick_category(review) == idx
