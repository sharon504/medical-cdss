# Medical CDSS — GT-MedRAG: Evidence-Grounded Medical Observation Generation

A Retrieval-Augmented Clinical Decision Support System that generates structured,
evidence-grounded medical observation reports. Built as an adaptation of
**ER-MedRAG** (Shi et al., 2026 — multi-agent RL framework for clinical RAG)
from clinical QA to **open-ended report generation**, with novel features for
traceability, robustness and conflict-aware safety at GPT-2 scale:

1. **Coarse-to-fine hybrid retrieval** (BM25 top-15 → dense re-rank top-5)
1. **Reasoning-triplet evidence distillation** `(condition; relation; conclusion)` with claim-level citation provenance
1. **Grounding-as-Reward RL** — DPO extractor + Grounded-GRPO respondent rewarded
   by *verifiable citation grounding* instead of binary answer match
1. **Conflict-aware clinician escalation** with abstention under contradictory evidence
1. **Grounding Robustness Score** under adversarial noise curriculum

> **Safety boundary**: decision support only — no autonomous diagnosis, no
> prescriptions. Outputs carry per-claim citations and escalation notes and
> remain subject to clinician verification.

Group 19, Dept. of CSE, College of Engineering Trivandrum.
Guided by Leena V A, Associate Professor.

## Architecture

```
Patient input (symptoms, history, vitals, labs)
        │
        ▼
┌────────────────────────────────────────────┐      ┌──────────────────────────┐
│  Coarse→Fine Hybrid Retriever              │ ←──  │  Knowledge Base (open)   │
│  BM25 top-15 → dense re-rank top-5         │      │  PubMed abstracts        │
│  (ER-MedRAG Alg. 2; RRF kept for ablation) │      │  curated seed references │
└───────────┬────────────────────────────────┘      │  guidelines              │
            │  evidence [E1..E5]                    └──────────────────────────┘
            ▼
┌────────────────────────────────────────────┐
│  Extractor agent (DPO)                     │ ←── conflict detector flags
│  snippet → (condition; relation;           │     contradictory triplets
│  conclusion) triplets with [E#] provenance │
└───────────┬────────────────────────────────┘
            ▼ concatenated triplets
┌────────────────────────────────────────────┐
│  Respondent agent (Grounded-GRPO)          │
│  domain-adapted GPT-2, GaR reward:         │
│  support + citations + structure + conflict│
└───────────┬────────────────────────────────┘
            ▼
┌────────────────────────────────────────────┐
│  Deterministic verifier (post-processor)   │
│  suppresses ungrounded claims, checks      │
│  citations, renders report + references    │
└────────────────────────────────────────────┘
```

Training: Colab notebooks 01→04 (clinical pretraining, RAG SFT, DPO extractor,
Grounded-GRPO respondent). Outputs verified against `suppressed: notes`,
conflict escalations, and GRS noise curves.

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

Training uses Google Colab (free tier): open notebooks `01`→`04` in order,
store checkpoints in Google Drive, then copy the final model into `models/`.

| Notebook | Trains | Illustrates |
| --- | --- | --- |
| `01_domain_pretrain_gpt2.ipynb` | clinical continued pretraining | domain adaptation |
| `02_rag_sft_gpt2.ipynb` | evidence+summary→report SFT | report structure learning |
| `03_dpo_extractor.ipynb` | DPO triplet extractor | proxy Seva/Slen preference scores |
| `04_grpo_respondent.ipynb` | Grounded-GRPO | GaR reward + adversarial noise curriculum |

## Public data sources (no credentialing required)

- PubMed abstracts via NCBI E-utilities (`src/corpus`)
- NCBI Bookshelf / public clinical guideline text
- Openly available medical transcription samples

## Evaluation

- **Retrieval**: recall@k, MRR on a labeled dev set; coarse-fine vs RRF ablation
- **Grounding**: citation accuracy, unsupported-claim ratio, structure score
- **Robustness**: Grounding Robustness Score under drop/swap/contradict
  corruption at 10–50% rates; token-level output-entropy stability study
- **Ablations**: GT variant matrix —
  `full_rag_triplets` (proposed), `rag_triplets_no_conflict`,
  `rag_raw_snippets`, `rag_rrf_not_c2f`, `baseline_gpt2`, `domain_adapted`

See `docs/architecture.md` for the GT-MedRAG design notes and
`docs/roadmap.md` for the phase checklist. Reference paper:
ER-MedRAG (Shi et al., 2026, JKSU-CIS 38:428).
