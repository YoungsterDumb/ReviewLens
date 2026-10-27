"""BIO tagging for aspect spans: char spans <-> word tags <-> sub-word labels.

Tags are ``O``, ``B-<TYPE>`` and ``I-<TYPE>``. English SemEval aspects use the
type ``ASP``; Vietnamese UIT-ViSD4SA spans keep their category (``BATTERY``…).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from .text import Span, word_spans

OUTSIDE = "O"
DEFAULT_TYPE = "ASP"
IGNORE_INDEX = -100


@dataclass(frozen=True)
class TaggedSpan:
    start: int
    end: int
    type: str = DEFAULT_TYPE


def build_label_list(types: Sequence[str]) -> list[str]:
    """``O`` first, then B/I pairs for each type in a stable order."""
    labels = [OUTSIDE]
    for t in sorted(set(types)):
        labels += [f"B-{t}", f"I-{t}"]
    return labels


def spans_to_word_tags(text: str, spans: Sequence[TaggedSpan]) -> tuple[list[Span], list[str]]:
    """Tag each word: the first word overlapping a span gets B-, the rest I-."""
    words = word_spans(text)
    tags = [OUTSIDE] * len(words)
    for span in sorted(spans, key=lambda s: (s.start, -(s.end - s.start))):
        first = True
        for i, w in enumerate(words):
            if w.end <= span.start or w.start >= span.end:
                continue
            if tags[i] != OUTSIDE:  # overlapping gold spans: keep the earlier one
                first = False
                continue
            tags[i] = f"{'B' if first else 'I'}-{span.type}"
            first = False
    return words, tags


def word_tags_to_spans(words: Sequence[Span], tags: Sequence[str]) -> list[TaggedSpan]:
    """Decode word tags to char spans. A stray ``I-`` starts a new span."""
    out: list[TaggedSpan] = []
    cur: list[int] | None = None  # [start, end]
    cur_type = DEFAULT_TYPE
    for w, tag in zip(words, tags):
        if tag == OUTSIDE or not tag:
            if cur:
                out.append(TaggedSpan(cur[0], cur[1], cur_type))
                cur = None
            continue
        prefix, _, typ = tag.partition("-")
        typ = typ or DEFAULT_TYPE
        if prefix == "B" or cur is None or typ != cur_type:
            if cur:
                out.append(TaggedSpan(cur[0], cur[1], cur_type))
            cur, cur_type = [w.start, w.end], typ
        else:
            cur[1] = w.end
    if cur:
        out.append(TaggedSpan(cur[0], cur[1], cur_type))
    return out


def align_word_labels(word_ids: Sequence[int | None], word_label_ids: Sequence[int]) -> list[int]:
    """Label only the first sub-token of each word; others get IGNORE_INDEX."""
    labels: list[int] = []
    prev: int | None = None
    for wid in word_ids:
        if wid is None or wid == prev:
            labels.append(IGNORE_INDEX)
        else:
            labels.append(word_label_ids[wid])
        prev = wid
    return labels


def find_term_offsets(text: str, terms: Sequence[str]) -> list[tuple[int, int] | None]:
    """Locate each term in order (case-insensitive), preferring whole-word hits."""
    out: list[tuple[int, int] | None] = []
    used: list[tuple[int, int]] = []
    lowered = text.lower()
    for term in terms:
        t = term.strip().lower()
        hit = None
        if t:
            for m in re.finditer(re.escape(t), lowered):
                s, e = m.start(), m.end()
                whole = (s == 0 or not lowered[s - 1].isalnum()) and (e == len(lowered) or not lowered[e].isalnum())
                if any(s < ue and e > us for us, ue in used):
                    continue
                if whole:
                    hit = (s, e)
                    break
                hit = hit or (s, e)
        if hit:
            used.append(hit)
        out.append(hit)
    return out
