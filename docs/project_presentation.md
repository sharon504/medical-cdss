# GT-MedRAG — Evidence-Grounded Medical Observation Generation
### Presentation Source Document (full project brief)

_Group 19, Dept. of CSE, College of Engineering Trivandrum_
Members: Amaljith M V (TVE23CS027) · Rohan A Jacob (TVE23CS118) · Roshin R (TVE23CS120) · Sharon P Shajan (TVE23CS129)
Guide: Leena V A, Associate Professor, Dept. of CSE

> Use this single document as the source for all slides: every section below maps
> 1:1 to a slide, with talk-track notes. Slide skeleton is at the end.

---

## 1. One-liner

**GT-MedRAG** is a Retrieval-Augmented Clinical Decision Support System that
generates structured, *evidence-grounded* medical observation reports. It
adapts the ER-MedRAG multi-agent RL framework (clinical QA) to open-ended
report generation at GPT-2 scale, and adds safety/traceability features
the paper lacks: grounding-based rewards, claim-level citations, adversarial
noise robustness scoring, and conflict-aware clinician escalation.

---

## 2. Problem statement

- LLMs applied directly to clinical records learn superficial
  symptom→diagnosis patterns and hallucinate unverifiable claims.
- Medical knowledge freezes inside model weights; updating requires
  expensive retraining.
- Existing clinical RAG systems (per ER-MedRAG) suffer:
  1. imprecise retrieval from long/complex medical documents,
  2. poor transformation of evidence into coherent reasoning,
  3. high sensitivity to retrieval noise.

**We answer**: generate a structured observation report where every major
claim is anchored to retrievable, cite-able evidence — with knowledge held
outside the model in an updatable retrieval layer.

---

## 3. Objectives → implementation status

1. Domain-adapt a lightweight GPT-2 to clinical language and report structure. ✅ notebooks 01/02
2. Build an external medical knowledge layer from public sources. ✅ `src/corpus`, `knowledge_base/`
3. Retrieve evidence with a hybrid retriever (BM25 + dense re-rank). ✅ `src/retrieval/coarse_fine.py` (paper Algorithm 2)
4. Distill evidence into reasoning triplets; generate the report from them. ✅ `src/generation/triplets.py`, notebooks 03/04
5. Reduce hallucination by verifying every claim against retrieved sources. ✅ `src/generation/postprocess.py` (suppression + citations) + GRS (`src/evaluation/robustness.py`)

---

## 4. Literature survey

| Work | Year | Why it matters here |
| --- | --- | --- |
| RAG — Lewis et al. | 2020 | Kept facts external; baseline we extend |
| BioGPT — Luo et al. | 2022 | Generative biomedical pretraining viability |
| GatorTron — Yang et al. | 2022 | EHR-scale domain adaptation (theirs) vs ours (GPT-2 scale) |
| Med-PaLM — Singhal et al. | 2023 | Clinical knowledge in LLMs; medical eval discipline |
| Clinical Camel — Toma et al. | 2023 | Open expert-level model stance |
| REPLUG — Shi et al. | 2024 | Retrieval supervision for black-box LLMs |
| **ER-MedRAG — Shi et al. (base paper)** | **2026** | Extractor–respondent + DPO/GRPO + coarse-to-fine retrieval — adapted here |

---

## 5. Base paper — ER-MedRAG (what we adapt)

- **Coarse-to-fine retrieval (Alg. 2)**: BM25 → top-n candidates → dense
  re-rank (MedCPT) → top-m fine snippets (paper: n=15, m=5).
- **Extractor agent**: per snippet outputs a reasoning triplet
  `(condition; relation; conclusion)`; preference pairs by
  `Sfinal = Sacc + Seva + Slen`; trained with **DPO** (β=0.1, lr 1e-5).
- **Respondent agent**: consumes only the *concatenated triplets*;
  trained with **GRPO** (G=5 group samples, KL β=0.01, lr 5e-6;
  reward = binary answer match + 0.2·format).
- **Results**: +3–6 pts on MedQA/MedMCQA/PubMedQA/MMLU-ProM/GPQA-M/
  MedXpertQA for 7B/8B models; reduced output entropy ⇒ more stable reasoning.
- **Admitted limitations**: coarse reward (no faithfulness measurement),
  brittleness under noisy/contradictory retrieval, QA-only framing.

---

## 6. What we changed — beyond the paper (the viva differentiators)

| # | Addition | Paper limitation addressed | Where in repo |
| --- | --- | --- | --- |
| 1 | **Grounding-as-Reward (GaR)** — GRPO respondent reward = R_support 0.4 + R_citation 0.25 + R_structure 0.15 + R_conflict 0.2, all measured by our deterministic verifier | binary QA match is coarse and silent about faithfulness; impossible for open-ended reports | `notebooks/04` (`grounded_reward`) |
| 2 | **Claim-level provenance**: every triplet carries its `[E#]`; citations survive into the report | traceability stops at snippet level | `src/generation/triplets.py`, `postprocess.py` |
| 3 | **Adversarial noise curriculum + Grounding Robustness Score (GRS)**: train on drop/swap/contradict-corrupted evidence (20/40%); evaluate grounding degradation curves | paper evaluates only clean retrieval despite admitting failure under noise | `scripts/make_sft_data.py` (60 noise prompts), `src/evaluation/robustness.py` |
| 4 | **Conflict detection & clinician escalation**: contradiction triplets, `CONFLICTS / UNCERTAINTY` section, SYSTEM NOTE withholding when confidence < 0.6 or ≥2 independent conflicts | paper has no contradiction handling | `src/generation/conflict.py` |
| 5 | **Resource-frugal validation**: whole framework at GPT-2 scale on free Colab T4 (deterministic proxy scorers replace the paper's 70B reward model) | paper needs 7B/8B/70B open LLMs | `notebooks/03` (`proxy_score`) |

---

## 7. System architecture

```
Patient input (symptoms, history, vitals, labs)
        │
        ▼
┌──────────────────────────────────────────┐    ┌───────────────────────────┐
│ Coarse→Fine Hybrid Retriever             │◄───│ Knowledge Base (public)   │
│ BM25 top-15 → dense re-rank top-5        │    │ PubMed abstracts (E-utils)│
│ (ER-MedRAG Alg.2; RRF kept for ablation) │    │ 10 curated seed refs      │
└──────────────┬───────────────────────────┘    └───────────────────────────┘
               │ evidence [E1..E5]
               ▼
┌──────────────────────────────────────────┐
│ Extractor agent (DPO, notebook 03)       │      conflict detector
│ snippet → (condition; relation;          │──►── src/generation/conflict.py
│ conclusion) triplets, each tagged [E#]   │
└──────────────┬───────────────────────────┘
               ▼ concatenated triplets (+ conflict triplets)
┌──────────────────────────────────────────┐
│ Respondent agent (Grounded-GRPO, nb 04)  │
│ domain-adapted GPT-2; GaR rewards        │
└──────────────┬───────────────────────────┘
               ▼
┌──────────────────────────────────────────┐
│ Deterministic verifier (post-processor)  │
│ sentence-level support check → suppress  │
│ ungrounded claims; citation accuracy;    │
│ structure validation; reference footer   │
└──────────────┬───────────────────────────┘
               ▼
Report: FINDINGS / DIFFERENTIAL (ranked) / INVESTIGATIONS /
        CONFLICTS-UNCERTAINTY / EVIDENCE REFERENCES
        (+ SYSTEM NOTE escalation when conflicts force abstention)
```

Module map (keep as "where the code lives" slide):
`src/corpus/` · `src/retrieval/{bm25_index,dense_index,coarse_fine,hybrid}.py` ·
`src/generation/{triplets,conflict,prompting,pipeline,postprocess,cli}.py` ·
`src/evaluation/{retrieval_metrics,grounding,ablation,robustness}.py` ·
`scripts/{make_sft_data,demo_app}.py` · `notebooks/01–04` ·
`configs/settings.yaml` (single source of truth).

---

## 8. Knowledge base & corpus

- **Curated seed references** (committed): 10 conditions — iron deficiency
  anemia, community-acquired pneumonia, AMI, ischemic stroke, hypo/hyper-
  thyroidism, T2DM, pulmonary tuberculosis, CKD, acute appendicitis; each with
  canonical Findings / Differential / Investigations sections (these parse
  into gold triplets and gold reports).
- **Optional PubMed enrichment** (live-verified): `python -m src.corpus.fetch_pubmed`
  pulls structured abstracts via NCBI E-utilities (3 req/s observed, no key).
- **Chunking**: ≤300 words, 50-word overlap, sentence-aligned (`chunking.py`).
- **Provenance**: every chunk has `chunk_id = doc_id::cN`; abstracts retain PMID.

---

## 9. Hybrid retrieval — design & measured quality

- **Sparse leg**: BM25 Okapi (k1=1.5, b=0.75) with a small domain stoplist.
- **Dense leg**: `pritamdeka/S-PubMedBert-MS-MARCO` (S-PubMedBERT), normalized
  embeddings + FAISS inner-product index.
- **Default pipeline** (paper-faithful): BM25 top-15 coarse → dense re-rank top-5.
  Dense leg degrades gracefully to BM25-only order when heavy deps absent.
- **Ablation leg**: parallel RRF (`1/(60+rank)`) fusion — kept to show the
  retrieval-stage choice matters.
- **Measured (BM25 path, 10-query labeled dev set)**:
  recall@1 = 0.90 · recall@3 = 1.00 · recall@5/6 = 1.00 · **MRR = 0.95**.
- Dev queries: `data/processed/eval/retrieval_dev.jsonl`.

---

## 10. Triplets → report generation

- **Triplet = (condition; relation; conclusion)** + `[E#]` + source chunk id.
- Gold triplets parsed deterministically from the seed references' clinical
  sentence patterns ("X suggested by Y", "X is diagnostic of Y", ...).
- Baseline extractor = transparent cue-based rules (demo'd locally).
- Learned extractor = GPT-2-medium after DPO (notebook 03).
- Respondent prompt = patient summary + **concatenated triplets only** —
  raw snippets never reach the generator (the noise-suppression insight of
  ER-MedRAG), with `CONFLICTS / UNCERTAINTY` section in the template.

---

## 11. Training pipeline (Colab, T4, all committed to repo)

| Notebook | Stage | Data | Key hyperparams | Output artifact |
| --- | --- | --- | --- | --- |
| 01 | Continued pretraining on clinical text | KB passages | gpt2-medium, 3 ep, lr 3e-4, bs 4×4 | `gpt2-clinical-base` |
| 02 | SFT: evidence+summary → report | 10 pairs | 4 ep, lr 2e-4, labels −100 on prompt | `gpt2-clinical-rag` (SFT) |
| 03 | DPO triplet extractor | 21+ preference pairs | β=0.1, lr 1e-5 (paper) | `gpt2-clinical-extractor` |
| 04 | Grounded-GRPO respondent | 10 clean + 60 noise prompts | G=5, β_KL=0.01, lr 5e-6 (paper) | final `gpt2-clinical-rag` |

Data artifacts: `sft_pairs.jsonl` (10), `triplet_pairs.jsonl` (21 DPO pairs),
`noise_variants.jsonl` (60 adversarial prompts @ rates 0.2/0.4 ×
drop/swap/contradict). Eval: 10 held-out patient JSONs (age-shifted) +
labeled retrieval dev set. Everything committed/reproducible, seeded (19).

---

## 12. Grounding verification (the anti-hallucination layer)

- Claim–evidence support check: word-window lexical containment (≥4
  content-word hits; stopword-guarded) — deterministic, auditable, fast
  (planned upgrade: NLI entailment model behind the same interface).
- Unsupported sentences → demoted to `(suppressed: …)`, never silently kept.
- `EVIDENCE REFERENCES` footer maps `[E#]` → title + PMID/source type.
- Line-level `unsupported_ratio` metric (per-line counting avoids the
  "unsupported sentence hiding under a header line" trap found in tests).
- Citation accuracy metric: fraction of cited tags whose sentence is really
  supported by the cited passage.

---

## 13. Conflict-aware safety

- Signals: negation flips (`not excluded` vs `confirms`) and frequency-hedging
  contradictions (`common` vs `rare`) across *different* evidence ids on
  shared condition keys.
- Confidence = 1 − Σ penalties (0.15 negation, 0.25 contradiction), floor 0.
- Escalation: SYSTEM NOTE "output withheld — escalate to clinician" when
  confidence < 0.6 or ≥ 2 independent conflict pairs.
- Designed conservatively (fewer false alarms), unit-tested (19-test suite).

---

## 14. Evaluation framework

**Retrieval**: recall@k (1/3/5/6), MRR — measured above.
**Grounding**: citation accuracy, unsupported-claim ratio, structure score
(4 required sections).
**Robustness (novel)**: GRS = 1 − mean relative grounding drop across
corruption rates (10–50%) — verified on all three operators.
**Stability (paper-faithful)**: token-level output entropy across configs
(raw vs triplet context; pre- vs post-GRPO) — expect post-GRPO < pre-GRPO.
**GT ablation matrix** (6 variants × 10 patients):
`full_rag_triplets` (proposed) · `rag_triplets_no_conflict` ·
`rag_raw_snippets` · `rag_rrf_not_c2f` · `baseline_gpt2` · `domain_adapted`.

Results table (fill after training completes):

| Metric | baseline_gpt2 | domain_adapted | rag_raw | no_conflict/triplets | full_rag_triplets |
| --- | --- | --- | --- | --- | --- |
| unsupported ratio | TBD | TBD | TBD | TBD | TBD |
| citation accuracy | TBD | TBD | TBD | TBD | TBD |
| GRS (drop/swap/contradict) | TBD | TBD | TBD | TBD | TBD |
| mean output entropy | TBD | TBD | TBD | TBD | TBD |

---

## 15. Demos

- CLI: `python -m src.generation.cli --mode triplets` → retrieval list
  (with scores), distilled triplets, final grounded markdown report.
- Streamlit: patient JSON selector → evidence cards, triplet list,
  conflict-confidence metric, full report; runs pre-model as a
  retrieval+prompt+triplet demo.
- Screenshot checklist: (1) retrieval hits with scores, (2) triplet block,
  (3) suppression of an unsupported sentence, (4) CONFLICTS/escalation demo,
  (5) GRS degradation curve, (6) entropy comparison.

---

## 16. Limitations (own them in the presentation)

- Lexical verifier is conservative — valid paraphrases can be suppressed
  (NLI upgrade path exists behind the same interface).
- Seed KB is deliberately small (10 conditions, 10 chunks); PubMed
  enrichment adds noisy abstract-level evidence.
- Reports/vignettes are synthetic; no real patient data (good ethics story).
- GPT-2 scale ⇒ fluency ceiling vs 7B+ (a deliberate trade we study; the
  paper's gains pattern is the hypothesis being tested).
- DPO/GRPO reward proxies are deterministic scorers, not a learned RM.

---

## 17. Future work

1. NLI-based claim–evidence entailment replacing the lexical check.
2. Cross-encoder re-ranking stage on top of coarse candidates.
3. Larger corpus (full MIMIC with PhysioNet credentialed access).
4. BioGPT/Llama-style backbone + LoRA (same GaR reward is model-agnostic).
5. Multi-document report synthesis; structured lab-table ingestion.
6. Human clinician pilot with a 2-observer grounded-vs-free checklist.

---

## 18. Division of work (suggested slide)

- Amaljith — corpus + retrieval build (E-utilities, BM25/FAISS, coarse-fine).
- Rohan — triplet extractor pipeline + DPO notebook 03.
- Roshin — respondent RAG SFT + Grounded-GRPO notebook 04.
- Sharon — grounding verifier, conflict module, robustness/GRS + evaluation suite.
(All four: repo scaffolding, ablation runs, final report docs.)

---

## 19. Slide skeleton (10 slides, ~10 min)

1. **Title** (30 s) — GT-MedRAG, team, one-liner, safety boundary quote.
2. **Problem** (1 min) — hallucinatory LLMs; frozen knowledge; the three RAG gaps.
3. **Related work** (45 s) — lit table; move fast to the base paper.
4. **Base paper ER-MedRAG** (90 s) — architecture sketch; DPO+GRPO; +3–6 pts; entropy drop.
5. **Our adaptations (beyond-paper table)** (90 s) — GaR, provenance, curriculum+GRS, conflicts, scale.
6. **Architecture + modules** (90 s) — pipeline diagram from §7; file map.
7. **Demo** (90 s) — live CLI/Streamlit + fallback screenshots; show suppression + citations.
8. **Evaluation plan + current numbers** (75 s) — recall/MRR measured now; matrix pending checkpoints; GRS sketch.
9. **Limitations & future work** (45 s) — §16/§17 compressed.
10. **Conclusion + safety boundary** (30 s) — decision support only; every claim verifiable.

Q&A cheat-sheet (likely questions → one-line answers):
- *Why GPT-2 and not a big LLM?* — the research question is whether the
  ER-MedRAG training recipe works at resource-frugal scale; knowledge lives in
  retrieval, not model size.
- *How is this different from plain RAG?* — triplets (structured reasoning
  distillation), RL training of both agents, and verifier-enforced grounding
  with conflict escalation.
- *Why not ChatGPT with prompting?* — reproducibility, cost, auditability,
  verifiable rewards; the whole pipeline is open and deterministic at eval.
- *How do you know citations aren't fake?* — our verifier computes citation
  accuracy on every report; faking a tag fails `R_citation` in training.
- *Is this a medical product?* — no; decision support with explicit
  escalation; clinician remains the decider.
- *Security/privacy?* — only public data + synthetic vignettes; no PHI.

---

## 20. References

1. Shi P., Shui Y., Jin R., Pan Q. ER-MedRAG: A multi-agent RL framework for
   reliable clinical retrieval-augmented reasoning. JKSU-CIS 38:428 (2026).
   https://doi.org/10.1007/s44443-026-00825-0
2. Lewis P. et al. Retrieval-Augmented Generation for Knowledge-Intensive
   NLP Tasks. NeurIPS 2020.
3. Luo R. et al. BioGPT. Briefings in Bioinformatics 2022.
4. Yang X. et al. A large language model for EHRs (GatorTron). npj Digital Medicine 2022.
5. Singhal K. et al. Large language models encode clinical knowledge (Med-PaLM). Nature 2023.
6. Toma A. et al. Clinical Camel. 2023.
7. Shi W. et al. REPLUG: Retrieval-augmented black-box language models. NAACL 2024.
8. Rafailov R. et al. Direct Preference Optimization. NeurIPS 2023.
9. Shao Z. et al. DeepSeekMath/GRPO. arXiv 2024.
10. NCBI E-utilities docs. https://www.ncbi.nlm.nih.gov/books/NBK25501/
