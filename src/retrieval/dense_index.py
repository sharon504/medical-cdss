"""Dense semantic retrieval: sentence-transformers embeddings + FAISS.

Build:  python -m src.retrieval.dense_index build
Query:  python -m src.retrieval.dense_index query "menorrhagia fatigue"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from .base import Evidence, load_chunks


def settings() -> dict:
    cfg = Path(__file__).resolve().parents[2] / "configs" / "settings.yaml"
    with open(cfg) as fh:
        return yaml.safe_load(fh)


def _model_name() -> str:
    return settings()["retrieval"]["dense_model"]


def _paths() -> tuple[Path, Path]:
    s = settings()
    kb_dir = Path(s["paths"]["kb_dir"])
    return kb_dir / "dense.faiss", kb_dir / "dense.meta.json"


class DenseIndex:
    """FAISS inner-product index over biomedical sentence embeddings."""

    def __init__(self, model_name: str | None = None):
        import faiss

        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name or _model_name())
        self.index = faiss.read_index(str(_paths()[0]))
        self.chunk_ids, self.titles, self.source_types, self.texts = load_meta()

    def encode(self, text: str) -> np.ndarray:
        vec = self.model.encode([text], normalize_embeddings=True)
        return np.asarray(vec, dtype="float32")

    def search(self, query: str, top_k: int = 20) -> list[Evidence]:
        scores, ids = self.index.search(self.encode(query), top_k)
        out: list[Evidence] = []
        for score, i in zip(scores[0], ids[0]):
            if i < 0 or score <= 0:
                continue
            out.append(
                Evidence(
                    chunk_id=self.chunk_ids[i],
                    title=self.titles[i],
                    source_type=self.source_types[i],
                    text=self.texts[i],
                    score=float(score),
                )
            )
        return out


def load_meta():
    """Return (chunk_ids, titles, source_types, texts) aligned with FAISS ids,
    read back from chunks.jsonl in build order."""
    chunks = load_chunks(Path(settings()["paths"]["kb_dir"]))
    ids = [c["chunk_id"] for c in chunks]
    titles = [c["title"] for c in chunks]
    types = [c["source_type"] for c in chunks]
    texts = [c["text"] for c in chunks]
    return ids, titles, types, texts


def build() -> None:
    import faiss

    from sentence_transformers import SentenceTransformer

    s = settings()
    kb_dir = Path(s["paths"]["kb_dir"])
    chunks = load_chunks(kb_dir)
    if not chunks:
        raise SystemExit("no chunks found - run `python -m src.corpus.build_kb` first")

    model = SentenceTransformer(_model_name())
    print(f"encoding {len(chunks)} chunks with {_model_name()} ...")
    vecs = np.asarray(
        model.encode(
            [c["text"] for c in chunks],
            normalize_embeddings=True,
            show_progress_bar=True,
        ),
        dtype="float32",
    )

    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, str(kb_dir / "dense.faiss"))
    (kb_dir / "dense.faiss.done").write_text("ok")
    print(f"built dense index ({vecs.shape[0]} x {vecs.shape[1]}) -> {kb_dir/'dense.faiss'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    q = sub.add_parser("query")
    q.add_argument("text")
    q.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    if args.cmd == "build":
        build()
    else:
        for ev in DenseIndex().search(args.text, args.top_k):
            print(f"{ev.score:6.3f}  {ev.chunk_id}  {ev.title[:70]}")
