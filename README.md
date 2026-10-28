# Medical CDSS — Evidence-Grounded Medical Observation Generation

A Retrieval-Augmented Clinical Decision Support System that generates structured,
evidence-grounded medical observation reports by separating **language generation**
(domain-adapted GPT-2) from **factual medical knowledge retrieval** (hybrid dense +
BM25 retrieval over an external, independently updatable knowledge base).

> **Safety boundary**: the system assists clinical review by generating observation
> reports with citation-backed evidence. It does **not** make autonomous diagnoses
> or treatment decisions. All outputs are subject to clinician verification.

Group 19, Dept. of CSE, College of Engineering Trivandrum.
Guided by Leena V A, Associate Professor.

## Architecture

```
Patient inputs (symptoms, history, vitals, labs)
        │
        ▼
┌────────────────────────┐      ┌──────────────────────────────┐
│  Hybrid Retriever      │ ←──  │  Knowledge Base (updatable)  │
│  • dense (FAISS)       │      │  • PubMed abstracts          │
│  • sparse (BM25)       │      │  • NCBI Bookshelf / guidelines│
│  • RRF fusion          │      │  • clinical report corpus    │
└───────────┬────────────┘      └──────────────────────────────┘
            ▼ top-k evidence passages
┌──────────────────────────────────────────────────────────────┐
│  Domain-Adaptive GPT-2 Generator                             │
│  continued clinical pretraining + SFT on report structure    │
└───────────┬──────────────────────────────────────────────────┘
            ▼
┌──────────────────────────────────────────────────────────────┐
│  Report Post-Processor                                       │
│  FINDINGS • DIFFERENTIAL Dx (ranked) • INVESTIGATIONS        │
│  + per-claim citation references (traceability)              │
└──────────────────────────────────────────────────────────────┘
```

## Repository Layout

```
data/raw/          raw downloads (pubmed xml, transcriptions, pamphlets)
data/processed/    cleaned text, SFT jsonl, eval sets
knowledge_base/    chunked passages + built indices (faiss, bm25, metadata)
src/corpus/        data collection + corpus construction (E-utilities)
src/retrieval/     chunking, BM25, dense (sentence-transformers + faiss), hybrid RRF
src/generation/    prompt building, GPT-2 inference pipeline, report post-processing
src/evaluation/    retrieval metrics (recall@k, MRR), grounding / hallucination scoring
notebooks/         Google Colab training notebooks (domain pretraining, RAG SFT)
scripts/           end-to-end CLI entry points
configs/           project settings (models, sources, paths)
docs/              architecture notes and evaluation writeups
```

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Build the knowledge base from public sources
python -m src.corpus.fetch_pubmed --limit 60
python -m src.corpus.build_kb

# 2. Build retrieval indices
python -m src.retrieval.build_indices

# 3. Generate an evidence-grounded observation report
python -m src.generation.cli --patient configs/sample_patient.json
```

Training uses Google Colab (free tier): open the notebooks in `notebooks/`,
store checkpoints in Google Drive, then copy the final model into `models/`.

## Public data sources (no credentialing required)

- PubMed abstracts via NCBI E-utilities (`src/corpus`)
- NCBI Bookshelf / public clinical guideline text
- Openly available medical transcription samples

## Evaluation

- **Retrieval**: recall@k, MRR on a labeled dev set
- **Grounding**: hallucination rate + claim-evidence support (citation precision)
- **Ablations**: vanilla GPT-2 vs domain-adapted (no retrieval) vs full RAG system

See `docs/` for the full write-up.
