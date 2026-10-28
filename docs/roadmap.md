# Project Roadmap & Deliverables

## Status board

| Phase | Deliverable | Where | Status |
| --- | --- | --- | --- |
| 0 | Repo scaffold, configs, license | repo root | done |
| 1a | Knowledge base builder + seeds | `src/corpus/` | done |
| 1b | PubMed fetch (optional enrichment) | `src/corpus/fetch_pubmed.py` | ready (network) |
| 2 | BM25 + FAISS + RRF hybrid retriever | `src/retrieval/` | done (kept for ablation) |
| 2' | **Coarse→fine retriever (paper Alg. 2)** | `src/retrieval/coarse_fine.py` | done, system default |
| 3' | **Reasoning triplets + gold builder** | `src/generation/triplets.py` | done, claim-level `[E#]` provenance |
| 3'' | **Conflict detection + escalation** | `src/generation/conflict.py` | done, abstention < 0.6 confidence |
| 3a | Colab: clinical continued pretraining | `notebooks/01_domain_pretrain_gpt2.ipynb` | to run in Colab |
| 3b | Colab: RAG SFT report generator | `notebooks/02_rag_sft_gpt2.ipynb` | to run in Colab |
| 3c | Colab: **DPO triplet extractor** | `notebooks/03_dpo_extractor.ipynb` | to run in Colab |
| 3d | Colab: **Grounded-GRPO respondent** | `notebooks/04_grpo_respondent.ipynb` | to run in Colab |
| 4 | SFT/eval data (report + triplet + noise pairs) | `scripts/make_sft_data.py` | done |
| 5 | Evaluation suite + **GRS + entropy study** | `src/evaluation/` | done (runs dry until models land) |
| 6 | Streamlit demo | `scripts/demo_app.py` | done |
| 7 | Final report/viva | seminars | pending |

## Task checklist

### Local machine
- [x] `knowledge_base/chunks.jsonl` built from seed references
- [x] `data/processed/sft_pairs.jsonl` (report pairs) — 10 pairs
- [x] `data/processed/triplet_pairs.jsonl` (DPO preference pairs) — 21 pairs
- [x] `data/processed/noise_variants.jsonl` (adversarial curriculum) — 60 pairs
- [x] Eval patients + labeled retrieval dev set generated
- [x] GT ablation dry-run: 6 variants × 10 patients recorded (pending-model notes)
- [ ] Optional: `src/corpus/fetch_pubmed.py` run to enrich KB with abstracts
        (adds thousands of chunks; re-run build + indices afterwards)
- [ ] Optional: install `sentence-transformers` + `faiss-cpu` and run
        `python -m src.retrieval.build_indices` for dense retrieval

### Google Colab (free tier, T4)
- [ ] `notebooks/01_domain_pretrain_gpt2.ipynb` → `gpt2-clinical-base`
- [ ] `notebooks/02_rag_sft_gpt2.ipynb` → `gpt2-clinical-rag` (SFT baseline)
- [ ] `notebooks/03_dpo_extractor.ipynb` → `gpt2-clinical-extractor`
- [ ] `notebooks/04_grpo_respondent.ipynb` → final `gpt2-clinical-rag`
- [ ] Copy checkpoints into Drive, snapshot entropy + GRS curves

### After model lands
- [ ] `python -m src.generation.cli --patient configs/sample_patient.json --mode triplets --out report.md`
- [ ] GT matrix: `python -m src.evaluation.ablation --variant gt_all --model models/gpt2-clinical-rag`
- [ ] `python -m src.evaluation.ablation --aggregate` → `docs/ablation_results.md`
- [ ] Robustness: `python -m src.evaluation.robustness --patient configs/sample_patient.json`
- [ ] Streamlit demo: `streamlit run scripts/demo_app.py`
- [ ] Extend SFT pairs (more conditions, PubMed-based negatives) and retrain

## Known limitations to disclose in the report

- Support checker is lexical (containment), conservative — some correct
  paraphrases may be suppressed; NLI upgrade planned.
- Seed KB is deliberately small (10 conditions) and curated; PubMed
  enrichment may introduce noisy abstract-level evidence.
- Demo vignettes are synthetic; no real patient data is used.
