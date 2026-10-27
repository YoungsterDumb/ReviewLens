"""Shared fixtures. Nothing here touches the network or needs a GPU."""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

os.environ.setdefault("KERAS_BACKEND", "torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402
from app.registry import StubRegistry  # noqa: E402

VOCAB_WORDS = (
    "the a an is was and but very great good bad dim screen battery keyboard fan loud life "
    "price delivery fast slow it . , ! i love hate this laptop phone camera quality"
).split()


@pytest.fixture
def stub_settings() -> Settings:
    return Settings(stub=True, max_image_mb=1, max_text_chars=5000, lime_text_samples=80, lime_image_samples=60)


@pytest.fixture
def stub_registry(stub_settings) -> StubRegistry:
    return StubRegistry(stub_settings)


@pytest.fixture
def client(stub_settings, stub_registry):
    from fastapi.testclient import TestClient

    return TestClient(create_app(stub_settings, stub_registry))


def png_bytes(color=(200, 40, 40), size=(64, 48)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def red_png() -> bytes:
    return png_bytes()


@pytest.fixture
def blue_png() -> bytes:
    return png_bytes((30, 60, 200))


@pytest.fixture(scope="session")
def tiny_tokenizer():
    """A real fast WordPiece tokenizer built offline from a tiny vocabulary."""
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    vocab = {t: i for i, t in enumerate(specials + VOCAB_WORDS)}
    tok = Tokenizer(models.WordPiece(vocab=vocab, unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
    )
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]",
                                   cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]",
                                   model_max_length=4096)


class RiggedTagger:
    """Token classifier that tags known aspect words B-ASP (and 'life' after 'battery' I-ASP)."""

    def __init__(self, tokenizer, aspect_words=("battery", "screen", "keyboard", "fan")):
        import torch
        from transformers import DistilBertConfig

        self.config = DistilBertConfig(id2label={0: "O", 1: "B-ASP", 2: "I-ASP"}, label2id={"O": 0, "B-ASP": 1, "I-ASP": 2})
        self.b_ids = torch.tensor([tokenizer.convert_tokens_to_ids(w) for w in aspect_words])
        self.i_id = tokenizer.convert_tokens_to_ids("life")
        self.calls = 0
        self.max_seen = 0

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask=None):
        import torch

        self.calls += 1
        self.max_seen = max(self.max_seen, input_ids.shape[1])
        logits = torch.full((*input_ids.shape, 3), -5.0)
        logits[..., 0] = 5.0
        b = torch.isin(input_ids, self.b_ids)
        prev_b = torch.zeros_like(b)
        prev_b[:, 1:] = b[:, :-1]
        i = (input_ids == self.i_id) & prev_b
        logits[b] = torch.tensor([-5.0, 5.0, -5.0])
        logits[i] = torch.tensor([-5.0, -5.0, 5.0])
        return SimpleNamespace(logits=logits)


@pytest.fixture
def rigged_tagger(tiny_tokenizer):
    return RiggedTagger(tiny_tokenizer)


def random_words(n: int, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    words = rng.choice(VOCAB_WORDS[:30], size=n)
    return " ".join(f"{w}." if i % 17 == 16 else str(w) for i, w in enumerate(words))
