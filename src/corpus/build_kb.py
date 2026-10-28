"""Merge every raw source into one deduplicated, chunked knowledge base.

Inputs:
  - data/raw/pubmed/*.jsonl          (fetched abstracts)
  - knowledge_base/sources/*.txt     (seed guideline / report-style texts)

Outputs:
  - knowledge_base/chunks.jsonl      text chunks with provenance
  - knowledge_base/metadata.jsonl    per-chunk metadata aligned by index

Usage:
    python -m src.corpus.build_kb
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from .chunking import chunk_words


def load_settings() -> dict:
    cfg_path = Path(__file__).resolve().parents[2] / "configs" / "settings.yaml"
    with open(cfg_path) as fh:
        return yaml.safe_load(fh)


def iter_documents(settings: dict):
    """Yield (doc_id, source_type, title, text) tuples."""
    raw_dir = Path(settings["paths"]["raw_dir"]) / "pubmed"
    if raw_dir.exists():
        for path in sorted(raw_dir.glob("*.jsonl")):
            condition = path.stem.replace("_", " ").title()
            with open(path) as fh:
                for i, line in enumerate(fh):
                    rec = json.loads(line)
                    text = f"{rec['title']}\n\n{rec['abstract']}".strip()
                    if len(text) > 300:
                        yield (
                            f"pmid:{rec['pmid']}",
                            "pubmed_abstract",
                            f"PubMed abstract: {rec['title']} ({rec['year']})",
                            text,
                            {"condition": condition, "pmid": rec["pmid"]},
                        )

    sources_dir = Path(settings["paths"]["kb_dir"]) / "sources"
    if sources_dir.exists():
        for path in sorted(sources_dir.glob("*.txt")):
            with open(path) as fh:
                text = fh.read().strip()
            if text:
                yield (
                    f"seed:{path.stem}",
                    "seed_reference",
                    path.stem.replace("_", " ").title(),
                    text,
                    {"seed_file": path.name},
                )


def main() -> None:
    settings = load_settings()
    chunk_cfg = settings["chunking"]
    kb_dir = Path(settings["paths"]["kb_dir"])
    kb_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = Path(settings["paths"]["chunks_file"])
    meta_path = Path(settings["paths"]["metadata_file"])

    seen_texts: set[str] = set()
    n_chunks = 0

    with open(chunks_path, "w") as chunks_fh, open(meta_path, "w") as meta_fh:
        for doc_id, source_type, title, text, extra in iter_documents(settings):
            for i, (norm_text, chunk) in enumerate(
                chunk_words(
                    text,
                    max_words=chunk_cfg["max_words"],
                    overlap=chunk_cfg["overlap_words"],
                )
            ):
                if " ".join(chunk) in seen_texts:
                    continue
                seen_texts.add(" ".join(chunk))
                chunk_id = f"{doc_id}::c{i}"
                rec = {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "source_type": source_type,
                    "title": title,
                    "text": " ".join(chunk),
                    **extra,
                }
                chunks_fh.write(json.dumps(rec) + "\n")
                meta_fh.write(
                    json.dumps(
                        {"chunk_id": chunk_id, "title": title,
                         "source_type": source_type, **extra}
                    )
                    + "\n"
                )
                n_chunks += 1

    print(f"wrote {n_chunks} chunks -> {chunks_path}")


if __name__ == "__main__":
    main()
