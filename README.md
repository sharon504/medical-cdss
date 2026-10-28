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

## Step-by-step setup and testing (Windows)

Run these commands from the repository root
(`medical-cdss`) in **cmd**. A PowerShell equivalent is noted where the
syntax differs.

### 1. Create and activate the environment

```bat
python -m venv .venv
.venv\Scripts\activate
python --version
```

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
```

If PowerShell blocks the activation script, run
`Set-ExecutionPolicy -Scope Process Bypass` first, or call
`.\.venv\Scripts\python.exe` directly instead of activating.

### 2. Install dependencies

```bat
pip install -r requirements.txt
pip list
```

Expected packages include `requests`, `rank-bm25`, `sentence-transformers`,
`faiss-cpu`, `transformers`, `torch`, `pytest`, and `streamlit`.

### 3. Build the knowledge base

```bat
python -m src.corpus.fetch_pubmed --limit 60
python -m src.corpus.build_kb
```

Expected: ten files under `data/raw/pubmed/`, roughly 59–60 abstracts per
condition, followed by:

```text
wrote 657 chunks -> knowledge_base\chunks.jsonl
```

Check the output files:

```bat
dir data\raw\pubmed
find /c /v "" knowledge_base\chunks.jsonl
```

### 4. Build retrieval indices

```bat
python -m src.retrieval.build_indices
```

Expected (counts vary slightly as PubMed grows):

```text
built BM25 over 657 chunks -> knowledge_base\bm25.pkl
built dense index (657 x 768) -> knowledge_base\dense.faiss
```

Verify `knowledge_base\bm25.pkl`, `knowledge_base\dense.faiss`, and
`knowledge_base\metadata.jsonl` exist.

### 5. Run the report CLI without a trained checkpoint

```bat
python -m src.generation.cli --patient configs/sample_patient.json
```

Expected: five retrieved passages, extracted triplets,
`unsupported-claim ratio: n/a`, and the full generation prompt. Without
`models/gpt2-clinical-rag`, the CLI intentionally stops before generation.

### 6. Exercise the full generation stage

Smoke-test with a tiny public model (downloads on first use):

```bat
python -m src.generation.cli --patient configs/sample_patient.json --model sshleifer/tiny-gpt2 --out report.md
```

Expected: an unsupported-claim ratio, `saved -> report.md`, and a report
ending with a real reference list:

```text
EVIDENCE REFERENCES:
[E1] Iron Deficiency Anemia (curated seed reference)
...
```

Use the trained checkpoint instead when available:

```bat
python -m src.generation.cli --patient configs/sample_patient.json --model models\gpt2-clinical-rag --out report.md
```

### 7. Test the RRF ablation path

```bat
python -m src.generation.cli --patient configs/sample_patient.json --retrieval rrf
```

Expected: `retrieved 6 evidence passages (rrf)`.

### 8. Run the automated tests

```bat
python -m pytest tests -q
```

Expected:

```text
22 passed
```

### 9. Verify command exit codes

After each command above, check that it succeeded:

```bat
echo %ERRORLEVEL%
```

Expected: `0`.

PowerShell:

```powershell
$LASTEXITCODE
```

## Step-by-step setup and testing (Linux/macOS)

Run these commands from the repository root (`medical-cdss`). They apply to
Linux and macOS; use `python3` if `python` is not on your PATH.

### 1. Create and activate the environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python --version
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip list
```

Expected packages include `requests`, `rank-bm25`, `sentence-transformers`,
`faiss-cpu`, `transformers`, `torch`, `pytest`, and `streamlit`.

### 3. Build the knowledge base

```bash
python -m src.corpus.fetch_pubmed --limit 60
python -m src.corpus.build_kb
```

Expected: ten files under `data/raw/pubmed/`, roughly 59–60 abstracts per
condition, followed by:

```text
wrote 657 chunks -> knowledge_base/chunks.jsonl
```

Check the output files:

```bash
ls data/raw/pubmed
wc -l knowledge_base/chunks.jsonl
```

### 4. Build retrieval indices

```bash
python -m src.retrieval.build_indices
```

Expected (counts vary slightly as PubMed grows):

```text
built BM25 over 657 chunks -> knowledge_base/bm25.pkl
built dense index (657 x 768) -> knowledge_base/dense.faiss
```

Verify the index files exist:

```bash
ls -l knowledge_base/bm25.pkl knowledge_base/dense.faiss knowledge_base/metadata.jsonl
```

### 5. Run the report CLI without a trained checkpoint

```bash
python -m src.generation.cli --patient configs/sample_patient.json
```

Expected: five retrieved passages, extracted triplets,
`unsupported-claim ratio: n/a`, and the full generation prompt. Without
`models/gpt2-clinical-rag`, the CLI intentionally stops before generation.

### 6. Exercise the full generation stage

Smoke-test with a tiny public model (downloads on first use):

```bash
python -m src.generation.cli --patient configs/sample_patient.json --model sshleifer/tiny-gpt2 --out report.md
```

Expected: an unsupported-claim ratio, `saved -> report.md`, and a report
ending with a real reference list:

```text
EVIDENCE REFERENCES:
[E1] Iron Deficiency Anemia (curated seed reference)
...
```

Use the trained checkpoint instead when available:

```bash
python -m src.generation.cli --patient configs/sample_patient.json --model models/gpt2-clinical-rag --out report.md
```

### 7. Test the RRF ablation path

```bash
python -m src.generation.cli --patient configs/sample_patient.json --retrieval rrf
```

Expected: `retrieved 6 evidence passages (rrf)`.

### 8. Run the automated tests

```bash
python -m pytest tests -q
```

Expected:

```text
22 passed
```

### 9. Verify command exit codes

After each command above, check that it succeeded:

```bash
echo $?
```

Expected: `0`.

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
