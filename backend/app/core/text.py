"""Pure-Python text utilities shared by the API and the training package.

No third-party imports here, so the training notebooks can reuse it as-is.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Iterable

# Letters that only occur in Vietnamese (tone marks + đ, ơ, ư, ă, â, ê, ô).
_VI_CHARS = re.compile(
    "[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]",
    re.IGNORECASE,
)
_VI_WORDS = {
    "và", "là", "của", "không", "rất", "nhưng", "được", "này", "mình", "cho",
    "với", "thì", "có", "hơi", "quá", "mua", "hàng", "giao", "máy", "pin",
}
_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)
_WORD = re.compile(r"\w+(?:[-'’]\w+)*|[^\w\s]", re.UNICODE)
_SENT_END = re.compile(r"(?<=[.!?…])\s+|\n+")


@dataclass(frozen=True)
class Span:
    start: int
    end: int

    def slice(self, text: str) -> str:
        return text[self.start:self.end]


def normalize(text: str) -> str:
    """NFC-normalise (Vietnamese diacritics) and tidy whitespace, keeping newlines."""
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_language(text: str) -> str:
    """Return 'vi' or 'en' using Vietnamese-only letters and common words."""
    letters = _LETTER.findall(text)
    if not letters:
        return "en"
    vi_ratio = len(_VI_CHARS.findall(text)) / len(letters)
    words = {w.lower() for w in re.findall(r"\w+", text)}
    vi_hits = len(words & _VI_WORDS)
    return "vi" if vi_ratio >= 0.03 or vi_hits >= 2 else "en"


def word_spans(text: str) -> list[Span]:
    """Words and punctuation marks with character offsets."""
    return [Span(m.start(), m.end()) for m in _WORD.finditer(text)]


def sentence_spans(text: str) -> list[Span]:
    """Split on sentence-final punctuation or newlines, keeping offsets."""
    spans: list[Span] = []
    start = 0
    for m in _SENT_END.finditer(text):
        if text[start:m.start()].strip():
            spans.append(_trim(text, start, m.start()))
        start = m.end()
    if text[start:].strip():
        spans.append(_trim(text, start, len(text)))
    return spans


def _trim(text: str, start: int, end: int) -> Span:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return Span(start, end)


def build_windows(
    text: str,
    count_tokens: Callable[[str], int],
    max_tokens: int = 256,
    overlap_words: int = 8,
) -> list[Span]:
    """Group sentences into windows of at most ``max_tokens`` model tokens.

    A sentence longer than the limit is cut into word chunks that overlap by
    ``overlap_words`` words, so no text is ever dropped.
    """
    windows: list[Span] = []
    current: Span | None = None
    for sent in sentence_spans(text):
        n = count_tokens(sent.slice(text))
        if n > max_tokens:
            if current:
                windows.append(current)
                current = None
            windows.extend(_split_long(text, sent, count_tokens, max_tokens, overlap_words))
            continue
        if current is None:
            current = sent
            continue
        merged = Span(current.start, sent.end)
        if count_tokens(merged.slice(text)) <= max_tokens:
            current = merged
        else:
            windows.append(current)
            current = sent
    if current:
        windows.append(current)
    return windows


def _split_long(
    text: str,
    sent: Span,
    count_tokens: Callable[[str], int],
    max_tokens: int,
    overlap_words: int,
) -> Iterable[Span]:
    words = [w for w in word_spans(text) if sent.start <= w.start and w.end <= sent.end]
    i = 0
    while i < len(words):
        j = i + 1
        while j < len(words) and count_tokens(text[words[i].start:words[j].end]) <= max_tokens:
            j += 1
        yield Span(words[i].start, words[j - 1].end)
        if j >= len(words):
            break
        i = max(i + 1, j - overlap_words)


def count_words(text: str) -> int:
    return len(re.findall(r"\w+", text))
