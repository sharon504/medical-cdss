"""Fetch PubMed abstracts for each configured condition into data/raw/pubmed/.

Usage:
    python -m src.corpus.fetch_pubmed --limit 60
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from tqdm import tqdm

from .pubmed_client import PubmedClient


def load_settings() -> dict:
    cfg_path = Path(__file__).resolve().parents[2] / "configs" / "settings.yaml"
    with open(cfg_path) as fh:
        return yaml.safe_load(fh)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None,
                    help="override max abstracts per query")
    ap.add_argument("--force", action="store_true",
                    help="re-fetch even if file exists")
    args = ap.parse_args()

    settings = load_settings()
    corpus_cfg = settings["corpus"]
    limit = args.limit or corpus_cfg["max_abstracts_per_query"]

    raw_dir = Path(settings["paths"]["raw_dir"]) / "pubmed"
    raw_dir.mkdir(parents=True, exist_ok=True)

    client = PubmedClient(sleep_seconds=corpus_cfg["fetch_sleep_seconds"])

    for condition in tqdm(corpus_cfg["conditions"], desc="conditions"):
        slug = condition.replace(" ", "_").lower()
        out_file = raw_dir / f"{slug}.jsonl"
        if out_file.exists() and not args.force:
            print(f"skip {out_file} (exists)")
            continue

        pmids = client.search(condition, limit=limit)
        records = [a.as_dict() for a in client.fetch_abstracts(pmids)]
        # keep only records with an actual abstract body
        records = [r for r in records if len(r["abstract"]) > 200]

        with open(out_file, "w") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")
        print(f"{condition}: saved {len(records)} abstracts -> {out_file}")


if __name__ == "__main__":
    main()
