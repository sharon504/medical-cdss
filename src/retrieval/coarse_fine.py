"""Coarse-to-fine hybrid retrieval (ER-MedRAG Algorithm 2, adapted).

Step I  (coarse):  BM25 sparse scoring -> top-n candidate set.
Step II (fine):    dense re-ranking of candidates only -> top-m snippets.

This sequential pipeline is the system default (paper-faithful). The older
parallel BM25+dense RRF fusion remains available for ablation via
src/retrieval/hybrid.py.

Usage:
    python -m src.retrieval.coarse_fine [query] [--n 15] [--m 5]
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .base import Evidence


def settings() -> dict:
    cfg = Path(__file__).resolve().parents[2] / "configs" / "settings.yaml"
    with open(cfg) as fh:
        return yaml.safe_load(fh)


def coarse_fine_search(
    query: str,
    n: int | None = None,
    m: int | None = None,
) -> list[Evidence]:
    """Retrieve: BM25 top-n coarse candidates, dense re-rank to top-m.

    Dense scoring is lazy: if sentence-transformers/FAISS are unavailable AND
    a dense index is not built, we degrade to a BM25-only top-m so the
    pipeline always returns evidence (and logs the degradation).
    """
    from . import bm25_index

    cfg = settings()["retrieval"]
    n = n or cfg["coarse_top_n"]
    m = m or cfg["fine_top_m"]

    coarse = bm25_index.search(query, top_k=n)
    if not coarse:
        return []

    fine = rerank_dense(query, coarse)
    if fine is None:
        # graceful degradation: keep BM25 order, truncate to m
        return coarse[:m]
    return fine[:m]


def rerank_dense(query: str, candidates: list[Evidence]) -> list[Evidence] | None:
    """Dense re-ranking of coarse candidates; None if dense unavailable."""
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:  # noqa: BLE001 (missing heavy deps - offline environments)
        print("[coarse_fine] dense re-ranker unavailable - degrading to BM25 order")
        return None

    try:
        from .dense_index import settings as dense_settings

        model = SentenceTransformer(dense_settings()["retrieval"]["dense_model"])
    except Exception as exc:  # noqa: BLE001 (missing index file / offline model)
        print(f"[coarse_fine] dense re-rank skipped ({exc.__class__.__name__})")
        return None

    import numpy as np

    vecs = np.asarray(
        model.encode(
            [query] + [ev.text for ev in candidates],
            normalize_embeddings=True,
        ),
        dtype="float32",
    )
    scores = vecs[0] @ vecs[1:].T
    ranked = sorted(
        zip(range(len(candidates)), scores), key=lambda x: x[1], reverse=True
    )
    out: list[Evidence] = []
    for idx, score in ranked:
        ev = candidates[idx]
        ev.score = float(score)  # dense cosine replaces sparse score
        out.append(ev)
    return out


if __name__ == "__main__":
    import argparse

    from .hybrid import build_patient_query

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("query", nargs="?", default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--m", type=int, default=None)
    args = ap.parse_args()

    query = args.query
    if query is None:
        import json

        query = build_patient_query(
            json.load(open(Path(__file__).resolve().parents[2]
                           / "configs" / "sample_patient.json"))
        )
    for ev in coarse_fine_search(query, n=args.n, m=args.m):
        print(f"[{ev.chunk_id}] {ev.score:.4f} {ev.title[:80]}")
