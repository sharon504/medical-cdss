"""Dense-retrieval leg tests. These run only when sentence-transformers +
faiss are installed (they download an embedding model on first use);
otherwise they are skipped, matching the local BM25-only environment."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("faiss")
pytest.importorskip("sentence_transformers")


def try_model():
    """Load the configured dense model (downloads on first use)."""
    from sentence_transformers import SentenceTransformer
    from src.retrieval.dense_index import settings

    try:
        return SentenceTransformer(settings()["retrieval"]["dense_model"])
    except Exception as exc:  # noqa: BLE001 (offline/environments without HF access)
        pytest.skip(f"dense model unavailable: {exc.__class__.__name__}")


def test_dense_model_encodes():
    model = try_model()
    vec = model.encode(["low serum ferritin with microcytic anemia"],
                       normalize_embeddings=True)
    assert vec.shape[0] == 1 and vec.shape[1] >= 128


def test_dense_build_and_faiss_query(tmp_path):
    import faiss

    import numpy as np

    from src.retrieval.base import Evidence
    from src.retrieval.coarse_fine import rerank_dense

    model = try_model()
    get = lambda t: np.asarray(model.encode([t], normalize_embeddings=True), dtype="float32")[0]  # noqa: E731

    candidates = [
        Evidence(chunk_id="seed:a::c0", title="IDA", source_type="seed",
                 text="low serum ferritin and microcytic anemia in iron deficiency"),
        Evidence(chunk_id="seed:b::c0", title="RD", source_type="seed",
                 text="chronic kidney disease with elevated creatinine"),
    ]
    out = rerank_dense("menorrhagia, fatigue, low ferritin, low MCV", candidates)
    assert out is not None and len(out) == 2
    assert out[0].chunk_id == "seed:a::c0", [ (e.chunk_id, e.score) for e in out ]


def test_coarse_fine_dense_rerank_order(chunks):
    from src.retrieval.coarse_fine import coarse_fine_search

    hits = coarse_fine_search("menorrhagia pallor low ferritin microcytic anemia",
                              n=15, m=4)
    assert 1 <= len(hits) <= 4
    assert hits[0].chunk_id.startswith("seed:iron_deficiency_anemia")
