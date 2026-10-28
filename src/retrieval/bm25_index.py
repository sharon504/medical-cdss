"""Sparse keyword retrieval with BM25 (Okapi).

Build:  python -m src.retrieval.bm25_index build
Query:  python -m src.retrieval.bm25_index query "menorrhagia fatigue"
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import yaml

from .base import Evidence

_WORD_RE = re.compile(r"[a-z0-9]+")

# minimal domain stoplist; BM25 idf handles most redundancy
_STOP = {
    "the", "a", "an", "of", "and", "or", "to", "in", "is", "are", "with",
    "for", "on", "as", "by", "than", "from", "at", "be", "was", "were",
    "this", "that", "which", "may", "can", "should", "must", "not",
}


def tokenize(text: str) -> list[str]:
    words = _WORD_RE.findall(text.lower())
    return [w for w in words if w not in _STOP]


def kb_paths_from_settings() -> dict:
    settings = _settings()
    paths = settings["paths"]
    kb_dir = Path(paths["kb_dir"])
    return {
        "chunks": Path(paths["chunks_file"]),
        "index": kb_dir / "bm25.pkl",
    }


def _settings() -> dict:
    cfg = Path(__file__).resolve().parents[2] / "configs" / "settings.yaml"
    with open(cfg) as fh:
        return yaml.safe_load(fh)


def build() -> None:
    from rank_bm25 import BM25Okapi

    paths = kb_paths_from_settings()
    with open(paths["chunks"]) as fh:
        chunks = [json.loads(line) for line in fh]
    if not chunks:
        sys.exit("no chunks found - run `python -m src.corpus.build_kb` first")

    corpus_tokens = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(corpus_tokens, k1=_bm25_k1(), b=_bm25_b())

    with open(paths["index"], "wb") as fh:
        pickle.dump({"chunks": chunks, "bm25": bm25}, fh)
    print(f"built BM25 over {len(chunks)} chunks -> {paths['index']}")


def _bm25_k1():
    return _settings()["retrieval"]["bm25_k1"]


def _bm25_b():
    return _settings()["retrieval"]["bm25_b"]


def search(
    query: str,
    top_k: int = 20,
    index_path: Path | None = None,
) -> list[Evidence]:
    """Load the pickled index and return ranked Evidence hits."""
    path = index_path or kb_paths_from_settings()["index"]
    with open(path, "rb") as fh:
        bundle = pickle.load(fh)
    scores = bundle["bm25"].get_scores(tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    out: list[Evidence] = []
    for i in ranked[:top_k]:
        c = bundle["chunks"][i]
        if scores[i] <= 0:
            break
        out.append(
            Evidence(
                chunk_id=c["chunk_id"],
                title=c["title"],
                source_type=c["source_type"],
                text=c["text"],
                score=float(scores[i]),
                pmid=str(c.get("pmid", "")),
                meta={k: c.get(k, "") for k in ("condition", "pmid")},
            )
        )
    return out


def search_corpora(query: str, chunks: list[dict], top_k: int = 20,
                   k1: float | None = None, b: float | None = None) -> list[Evidence]:
    """Rank an in-memory chunk list directly (no pickled index needed) -
    used by data synthesis and tests."""
    from rank_bm25 import BM25Okapi

    corpus_tokens = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(corpus_tokens, k1=k1 or _bm25_k1(), b=b or _bm25_b())
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    out: list[Evidence] = []
    for i in ranked[:top_k]:
        if scores[i] <= 0:
            break
        c = chunks[i]
        out.append(
            Evidence(
                chunk_id=c["chunk_id"],
                title=c["title"],
                source_type=c["source_type"],
                text=c["text"],
                score=float(scores[i]),
                pmid=str(c.get("pmid", "")),
                meta={k: c.get(k, "") for k in ("condition",)},
            )
        )
    return out


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
        for ev in search(args.text, args.top_k):
            print(f"{ev.score:6.2f}  {ev.chunk_id}  {ev.title[:70]}")
