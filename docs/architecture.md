# Architecture Notes — GT-MedRAG (Grounding-driven Triplet MedRAG)

## Relationship to the base paper

This project adapts **ER-MedRAG** (Shi et al., 2026, *JKSU-CIS* 38:428 — a
multi-agent RL framework for clinical RAG) from closed clinical **QA**
(7B/70B LLMs, binary answer-match reward) to open-ended, evidence-grounded
**observation report generation** at GPT-2 scale. Positions adopted from
the paper: coarse-to-fine hybrid retrieval; extractor-respondent
separation; triplet `(condition; relation; conclusion)` evidence
distillation; two-stage RL (DPO extractor + GRPO respondent).

## What we changed and added (beyond the paper)

| # | Addition | Why (paper limitation addressed) |
| --- | --- | --- |
| 1 | **Grounding-as-Reward (GaR)** for the respondent: R_support 0.4 + R_citation 0.25 + R_structure 0.15 + R_conflict 0.2, computed by the repo's deterministic verifier | Paper's `Rresult` is a binary QA match — coarse, no faithfulness measurement, impossible for open-ended reports |
| 2 | **Triplet-level provenance**: every triplet carries its `[E#]`, so citations are claim-level | Paper's traceability stops at snippet level |
| 3 | **Adversarial noise curriculum + Grounding Robustness Score (GRS)**: corrupted-evidence prompts (drop/swap/contradict at 20/40%) in GRPO training + degradation-curve metric | Paper reports brittleness under noisy/contradictory retrieval but evaluates only clean retrieval |
| 4 | **Conflict detection & clinician escalation**: contradiction triplets, `CONFLICTS / UNCERTAINTY` section, abstention when confidence < 0.6 | Paper has no contradiction handling |
| 5 | **Resource-frugal validation**: full pipeline at GPT-2 scale on free Colab T4 | Paper uses 7B/8B/70B open models |

## Pipeline (v2)

```
Patient inputs
    |
    v
Coarse-to-fine hybrid retrieval (ER-MedRAG Alg. 2)   src/retrieval/coarse_fine.py
  BM25 top-15 coarse -> dense re-rank top-5          src/retrieval/hybrid.py (RRF kept for ablation)
    |
    v
Evidence [E1..E5] ----+---> conflict detector        src/generation/conflict.py
    |                 |
    v                 |
Extractor agent       | (noise-corrupted copies feed only eval / GRPO curriculum)
(DPO, notebook 03)    |
  triplets (condition; relation; conclusion) with [E#] provenance
    |
    v
Respondent agent (Grounded-GRPO, notebook 04) generates the grounded report
    |
    v
Post-processor (deterministic verifier) src/generation/postprocess.py
  sentence-level support check -> suppress ungrounded claims
  citation accuracy, structure validation, reference footer
    |
    v
Report: FINDINGS / DIFFERENTIAL (ranked) / INVESTIGATIONS /
        CONFLICTS-UNCERTAINTY / EVIDENCE REFERENCES
        (+ SYSTEM NOTE escalation when conflicts force abstention)
```

## Grounding verification (src/generation/postprocess.py)

- Lexical containment heuristic (word-window with >=4 content-word overlap) -
  deterministic claim-evidence support check (upgrade path: NLI model).
- Unsupported sentences are demoted to `(suppressed: ...)` notes, never
  silently dropped.
- `EVIDENCE REFERENCES` footer maps `[E#]` -> source titles / PMIDs.

## Evaluation artifacts

- `src/evaluation/retrieval_metrics.py` - recall@k, MRR (retrieval_dev.jsonl)
- `src/evaluation/grounding.py` - citation accuracy, unsupported ratio, structure
- `src/evaluation/ablation.py` - GT variant matrix:
  `full_rag_triplets` (proposed) / `rag_triplets_no_conflict` /
  `rag_raw_snippets` / `rag_rrf_not_c2f` / `baseline_gpt2` / `domain_adapted`
- `src/evaluation/robustness.py` - corruption harness (drop/swap/contradict),
  GRS, token-level output entropy stability study (paper Sec. 4.11 analogue)

## Evaluation results (so far)

- Retrieval (BM25-only degraded path): recall@1 = 0.90, recall@3 = 1.00, MRR = 0.95
- Corruption/GRS machinery verified on synthetic evidence (all 3 operators)
- Full matrix pending RL training (notebooks 03/04, Colab), then
  `python -m src.evaluation.ablation --aggregate` -> docs/ablation_results.md

## Safety boundaries (tightened vs v1)

- No autonomous diagnosis, no prescriptions; decision-support only.
- Conflict-heavy evidence triggers clinician escalation instead of a silent blend.
- All retrieved evidence remains visible in the report footer.

