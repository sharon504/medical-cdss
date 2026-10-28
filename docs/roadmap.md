# Project Roadmap & Deliverables

## Status board

| Phase | Deliverable | Where | Status |
| --- | --- | --- | --- |
| 0 | Repo scaffold, configs, license | repo root | done |
| 1a | Knowledge base builder + seeds | `src/corpus/` | done |
| 1b | PubMed fetch (optional enrichment) | `src/corpus/fetch_pubmed.py` | ready (network) |
| 2 | BM25 + FAISS + RRF hybrid retriever | `src/retrieval/` | code done, dense build pending (needs sentence-transformers) |
| 3a | Colab: clinical continued pretraining | `notebooks/01_domain_pretrain_gpt2.ipynb` | to run in Colab |
| 3b | Colab: RAG supervised fine-tuning | `notebooks/02_rag_sft_gpt2.ipynb` | to run in Colab |
| 4 | SFT/eval data | `scripts/make_sft_data.py` | done |
| 5 | Evaluation suite | `src/evaluation/` | code done |
| 6 | Streamlit demo | `scripts/demo_app.py` | done |
| 7 | Final report/viva | seminars | pending |

## Task checklist

### Local machine
- [x] `knowledge_base/chunks.jsonl` built from seed references
- [x] `data/processed/sft_pairs.jsonl` synthesized (10 training pairs)
- [x] Eval patients + labeled retrieval dev set generated
- [ ] Optional: `src/corpus/fetch_pubmed.py` run to enrich KB with abstracts
        (adds thousands of chunks; re-run build + indices afterwards)
- [ ] Optional: install `sentence-transformers` + `faiss-cpu` and run
        `python -m src.retrieval.build_indices` for dense retrieval

### Google Colab (free tier, T4)
- [ ] Open `notebooks/01_domain_pretrain_gpt2.ipynb`, set repo URL of your
        GitHub account, run → produces `Drive:/medical-cdss/gpt2-clinical-base`
- [ ] Upload `data/processed/sft_pairs.jsonl` (or pull from repo via web),
        open `notebooks/02_rag_sft_gpt2.ipynb`, run → produces
        `gpt2-clinical-rag`
- [ ] Copy `gpt2-clinical-rag` into local `models/gpt2-clinical-rag`

### After model lands
- [ ] `python -m src.generation.cli --patient configs/sample_patient.json --out report.md`
- [ ] Eval patients: `python -m src.evaluation.ablation --variant all --model models/gpt2-clinical-rag`
- [ ] `python -m src.evaluation.ablation --aggregate` → `docs/ablation_results.md`
- [ ] Streamlit demo: `streamlit run scripts/demo_app.py`
- [ ] Extend SFT pairs (more conditions, PubMed-based negatives) and retrain

## Known limitations to disclose in the report

- Support checker is lexical (containment), conservative — some correct
  paraphrases may be suppressed; NLI upgrade planned.
- Seed KB is deliberately small (10 conditions) and curated; PubMed
  enrichment may introduce noisy abstract-level evidence.
- Demo vignettes are synthetic; no real patient data is used.
