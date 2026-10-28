"""Build both retrieval indices over the knowledge base.

Usage:  python -m src.retrieval.build_indices
"""

from __future__ import annotations

from .bm25_index import build as build_bm25


def main() -> None:
    build_bm25()
    from .dense_index import build as build_dense

    build_dense()


if __name__ == "__main__":
    main()
