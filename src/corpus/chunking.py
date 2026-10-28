"""Word-level chunking with overlap. Splits on sentence boundaries where
possible so chunks stay clinically coherent."""

from __future__ import annotations

import re
from typing import Iterator

_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def _sentences(text: str) -> list[str]:
    parts = _SENT_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_words(
    text: str, max_words: int = 300, overlap: int = 50
) -> Iterator[tuple[str, list[str]]]:
    """Yield (chunk_id_hint, token_list) pairs of <= max_words words with
    `overlap`-word overlap between successive chunks."""
    sents = _sentences(text)
    chunks: list[list[str]] = []
    current: list[str] = []

    for sent in sents:
        words = sent.split()
        if not words:
            continue
        # hard-split pathological sentences (tables, dense abstracts)
        while len(words) > max_words:
            chunks.append(words[:max_words])
            words = words[max(1, max_words - overlap):]
        if current and len(current) + len(words) > max_words:
            chunks.append(current)
            current = current[len(current) - min(overlap, len(current)):]
        current = current + words
    if current:
        chunks.append(current)

    for chunk in chunks:
        yield f"c{len(chunks)}", chunk
