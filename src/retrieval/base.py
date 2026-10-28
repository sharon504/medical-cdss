"""Shared retrieval data structures and KB loading utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Evidence:
    """One retrieved passage with provenance for citation traceability."""

    chunk_id: str
    title: str
    source_type: str
    text: str
    score: float = 0.0
    pmid: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def short_id(self) -> str:
        """Human-facing citation id, e.g. [E3]."""
        return self.chunk_id.split("::")[0]

    def as_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "title": self.title,
            "source_type": self.source_type,
            "score": round(self.score, 4),
            "pmid": self.pmid,
            "text": self.text,
            **self.meta,
        }


def load_chunks(kb_dir: Path) -> list[dict]:
    """Load chunks.jsonl (one JSON record per line) as a list of dicts."""
    with open(kb_dir / "chunks.jsonl") as fh:
        return [json.loads(line) for line in fh]
