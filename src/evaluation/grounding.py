"""Grounding / hallucination scoring of generated reports.

Metrics per report:
  - unsupported_ratio : fraction of content sentences with no lexical support
    in retrieved evidence   (see src/generation/postprocess.py)
  - citation_accuracy : fraction of [Ex] tags that actually point to passages
    lexically covering the sentence they are attached to
  - structure_score   : fraction of required report headings present

Benchmark file (data/processed/eval/grounding_dev.jsonl):
    {"patient_id": "...", "report": "...generated report...",
     "evidence": [{"chunk_id": "...", "text": "..."}]}
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..generation.postprocess import (
    _HEADER_RE,
    extract_citations,
    is_supported,
)


class _Slot:
    """Minimal duck-typed Evidence stand-in (only `.text` is used)."""

    def __init__(self, text: str):
        self.text = text


def content_sentences(report: str) -> list[str]:
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", report) if s.strip()]
    return [
        s for s in sents
        if not _HEADER_RE.match(s) and len(s.split()) >= 6
    ]


def citation_accuracy(report: str, evidence_by_id: dict[str, str]) -> float:
    """Fraction of cited [Ex] tags whose sentence is supported by that passage."""
    total = hits = 0
    for sent in content_sentences(report):
        _, ids = extract_citations(sent)
        for e_id in ids:
            total += 1
            passage = evidence_by_id.get(e_id, "")
            if passage and is_supported(sent, [_Slot(passage)]):
                hits += 1
    return hits / total if total else 1.0


def structure_score(report: str, required=("FINDINGS", "DIFFERENTIAL",
                                           "INVESTIGATIONS", "REFERENCES")) -> float:
    upper = report.upper()
    return sum(1 for r in required if r in upper) / len(required)


def evaluate_file(path: Path) -> dict:
    from ..generation.postprocess import unsupported_ratio as _unsup_ratio

    rows = [json.loads(l) for l in open(path)]
    per_report = []
    for row in rows:
        slots = [_Slot(ev["text"]) for ev in row["evidence"]]
        evidence_by_id = {f"E{i+1}": ev["text"]
                          for i, ev in enumerate(row["evidence"])}
        per_report.append({
            "patient_id": row.get("patient_id"),
            "citation_accuracy": round(citation_accuracy(row["report"], evidence_by_id), 4),
            "structure_score": round(structure_score(row["report"]), 4),
            "unsupported_ratio": round(_unsup_ratio(row["report"], slots), 4),
        })
    n = len(per_report) or 1
    summary = {
        "n_reports": len(per_report),
        "mean_citation_accuracy": round(sum(r["citation_accuracy"] for r in per_report) / n, 4),
        "mean_structure_score": round(sum(r["structure_score"] for r in per_report) / n, 4),
        "mean_unsupported_ratio": round(sum(r["unsupported_ratio"] for r in per_report) / n, 4),
    }
    return {"summary": summary, "per_report": per_report}


if __name__ == "__main__":
    import sys

    path = Path(sys.argv[1] if len(sys.argv) > 1
                else "data/processed/eval/grounding_dev.jsonl")
    print(json.dumps(evaluate_file(path), indent=2))
