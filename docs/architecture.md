# Architecture Notes — Medical CDSS

## Design principle

**Knowledge lives outside the model.** The GPT-2 generator learns clinical
language and report structure only; medical facts are retrieved at runtime
from an external, independently updatable knowledge base. Updating medical
knowledge never requires retraining the model.

## Pipeline

1. **Knowledge base** (`src/corpus/`)
   - Sources: PubMed abstracts (E-utilities), curated seed references,
     optionally guideline text.
   - Everything is chunked to ≤300 words with 50-word overlap
     (`src/corpus/chunking.py`) and stored as JSONL with provenance
     (PMID / source file).

2. **Hybrid retrieval** (`src/retrieval/`)
   - Sparse: BM25 (Okapi, k1=1.5, b=0.75) — exact keyword/term matching.
   - Dense: S-PubMedBERT sentence embeddings, normalized, FAISS inner product —
     conceptual/semantic matching.
   - Fusion: Reciprocal Rank Fusion (`1/(60+rank)`), top-6 passages returned
     with `[E1]..[E6]` citation ids and per-chunk provenance.

3. **Generation** (`src/generation/`, `notebooks/`)
   - Phase 1 (Colab): GPT-2-medium continued pretraining on clinical text
     corpus ("domain adaptation").
   - Phase 2 (Colab): supervised fine-tuning on
     `evidence + patient summary -> structured report` pairs
     (label mask supervises only the report tokens, not the prompt).
   - Prompt template (matching training format) instructs the model to cite
     `[E#]` ids next to claims.

4. **Grounding / post-processing** (`src/generation/postprocess.py`)
   - Every generated sentence is checked against retrieved passages with a
     deterministic lexical support heuristic (word-window containment).
   - Unsupported sentences are demoted to "(suppressed: …)" notes.
   - `EVIDENCE REFERENCES` footer maps `[E#]` → source titles/PMIDs.
   - Report sections: FINDINGS / DIFFERENTIAL DIAGNOSES (ranked) /
     RECOMMENDED INVESTIGATIONS / EVIDENCE REFERENCES.

5. **Evaluation** (`src/evaluation/`)
   - Retrieval: recall@1/3/5/6, MRR on `data/processed/eval/retrieval_dev.jsonl`.
   - Grounding: citation accuracy, unsupported-claim ratio, structure score.
   - Ablations: `baseline_gpt2` vs `domain_adapted` (no retrieval) vs
     `full_rag` — see `docs/ablation_results.md` (generated artifact).

## Safety boundaries

- No autonomous diagnosis, no prescriptions; reports are decision support.
- Unsupported claims are suppressed, not silently kept.
- All retrieved evidence stays visible to the clinician.

## Upgrade paths (documented for viva)

- Lexical support heuristic → NLI-based claim–evidence entailment checker.
- BM25+RRF → cross-encoder reranking of top candidates.
- GPT-2 → BioGPT / GPT-2-medium+LoRA as compute allows.
- Cox leakage: knowledge base can grow without touching trained weights.
