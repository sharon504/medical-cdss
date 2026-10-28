"""Retrieval quality metrics: recall@k and MRR over a labeled dev set.

Dev-set format (data/processed/eval/retrieval_dev.jsonl), one JSON per line:
    {
      "query": "...patient text or question...",
      "relevant_chunk_ids": ["seed:iron_deficiency_anemia::c0", ...],
      "specificity": "system condition label, e.g. iron deficiency anemia"
    }

Scores:
    recall@k : fraction of labeled relevant passages found in top-k
    mrr      : 1/rank of the first relevant passage in the fused list
"""

from __future__ import annotations

import json
from pathlib import Path


def recall_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant) / len(relevant)


def mrr(ranked_ids: list[str], relevant: set[str]) -> float:
    for i, cid in enumerate(ranked_ids, start=1):
        if cid in relevant:
            return 1.0 / i
    return 0.0


def evaluate(
    dev_file: Path, get_ranked_ids, k_values: tuple[int, ...] = (1, 3, 5, 6)
) -> dict:
    """Run a retrieval function get_ranked_ids(query) -> list[str] on the dev
    set and return aggregate metrics."""
    items = [json.loads(l) for l in open(dev_file)]
    agg = {"n": len(items), **{f"recall@{k}": 0.0 for k in k_values}, "mrr": 0.0}
    for item in items:
        ranked = get_ranked_ids(item["query"])
        relevant = set(item["relevant_chunk_ids"])
        for k in k_values:
            agg[f"recall@{k}"] += recall_at_k(ranked, relevant, k)
        agg["mrr"] += mrr(ranked, relevant)
    n = agg.pop("n")
    for key in agg:
        agg[key] = round(agg[key] / n, 4)
    agg["n_queries"] = n
    return agg


if __name__ == "__main__":
    import sys

    from ..retrieval.hybrid import hybrid_search

    dev = Path(sys.argv[1] if len(sys.argv) > 1
               else "data/processed/eval/retrieval_dev.jsonl")
    results = evaluate(
        dev, lambda q: [ev.chunk_id for ev in hybrid_search(q)]
    )
    print(json.dumps(results, indent=2))
