"""Hybrid retriever: Reciprocal Rank Fusion of dense (vector) + sparse (BM25)
candidate lists. Produces the final top-k evidence with provenance."""

from __future__ import annotations

from pathlib import Path

import yaml

from .base import Evidence


def settings() -> dict:
    cfg = Path(__file__).resolve().parents[2] / "configs" / "settings.yaml"
    with open(cfg) as fh:
        return yaml.safe_load(fh)


def reciprocal_rank_fusion(
    *ranked_lists: list[Evidence], rrf_k: int = 60
) -> list[Evidence]:
    """Fuse ranked evidence lists with RRF: score(d) = sum 1/(rrf_k + rank).

    Returns a single list sorted by fused score (deduplicated by chunk_id).
    """
    fused: dict[str, tuple[float, Evidence]] = {}
    for lst in ranked_lists:
        for rank, ev in enumerate(lst, start=1):
            hit = fused.get(ev.chunk_id)
            new_score = 1.0 / (rrf_k + rank)
            if hit:
                hit[0] += new_score
                # keep the higher raw member score for reporting
                hit[1].score = max(hit[1].score, ev.score)
            else:
                ev.score = new_score  # replace raw score with fused score
                fused[ev.chunk_id] = (new_score, ev)
    fused_list = [ (score, ev) for score, ev in fused.values() ]
    fused_list.sort(key=lambda x: x[0], reverse=True)
    return [ev for _, ev in fused_list]


def hybrid_search(
    query: str,
    top_k: int | None = None,
    kb_dir: Path | None = None,
) -> list[Evidence]:
    """Retrieve with BM25 and dense search, fuse with RRF, return top_k."""
    from .bm25_index import search as bm25_search
    from .dense_index import DenseIndex

    cfg = settings()["retrieval"]
    top_k = top_k or cfg["final_top_k"]

    bm25_hits = bm25_search(query, top_k=cfg["top_k_bm25"])
    dense_hits = DenseIndex().search(query, top_k=cfg["top_k_dense"])
    fused = reciprocal_rank_fusion(dense_hits, bm25_hits, rrf_k=cfg["rrf_k"])
    return fused[:top_k]


def build_patient_query(patient: dict) -> str:
    """Turn a structured patient record into one retrieval query string."""
    parts = [patient.get("chief_complaint", "")]
    if patient.get("history"):
        parts.append(patient["history"])
    vitals = patient.get("vitals") or {}
    labs = patient.get("lab_values") or {}
    for label, value in {**vitals, **labs}.items():
        parts.append(f"{label.replace('_', ' ')} {value}")
    return "; ".join(p for p in parts if p)


if __name__ == "__main__":
    import json
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else build_patient_query(
        json.load(open(Path(__file__).resolve().parents[2]
                       / "configs" / "sample_patient.json"))
    )
    for ev in hybrid_search(query):
        print(f"[{ev.chunk_id}] {ev.score:.4f} {ev.title[:80]}")
