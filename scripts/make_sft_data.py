"""Build the fine-tuning and evaluation data for Medical CDSS.

Creates from the seed reference texts (knowledge_base/sources/*.txt):

  1. data/processed/sft_pairs.jsonl          evidence + patient -> report (train)
  2. data/processed/eval/patients/*.json     held-out patient vignettes
  3. data/processed/eval/retrieval_dev.jsonl labeled retrieval dev queries

Deterministic (seeded); re-run with --force to rebuild. BM25 is used for
evidence selection so no GPU/embedding model is needed locally.

Usage:  python scripts/make_sft_data.py [--force]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieval.bm25_index import (  # noqa: E402
    kb_paths_from_settings,
    search_corpora,
)

random.seed(19)

SEED_HEADERS = {
    "findings": re.compile(r"Findings\.\s*(.+?)(?=\n\n)", re.S),
    "differential": re.compile(
        r"Differential diagnosis[^:]*:\s*(.+?)(?=\n\n)", re.S),
    "investigations": re.compile(
        r"Recommended investigations\.\s*(.+?)(?=\n\n)", re.S),
}


def parse_seed_section(text: str, key: str) -> str:
    """Parse a section from a seed reference file, joining wrapped lines."""
    m = SEED_HEADERS[key].search(text.replace("\r", ""))
    if not m:
        return ""
    return " ".join(m.group(1).split()).strip()


def report_target(seed_text: str) -> str | None:
    findings = parse_seed_section(seed_text, "findings")
    differential = parse_seed_section(seed_text, "differential")
    investigations = parse_seed_section(seed_text, "investigations")
    if not (findings and differential and investigations):
        return None
    return (
        "FINDINGS:\n"
        f"{findings}\n\n"
        "DIFFERENTIAL DIAGNOSES (ranked):\n"
        f"{differential}\n\n"
        "RECOMMENDED INVESTIGATIONS:\n"
        f"{investigations}"
    )


# ---- deterministic synthetic vignettes per reference file --------------------

VIGNETTES = {
    "iron_deficiency_anemia": dict(
        age=34, sex="female", complaint="Fatigue and exertional breathlessness for 3 months",
        history="Heavy menstrual bleeding for 2 years, vegetarian diet. No GI symptoms.",
        vitals={"pulse": "96/min regular", "bp": "110/70 mmHg"},
        lab_values={"hemoglobin": "8.2 g/dL (low)", "mcv": "67 fL (low)",
                    "serum_ferritin": "5 ng/mL (low)"},
    ),
    "community_acquired_pneumonia": dict(
        age=61, sex="male", complaint="Fever and productive cough for 4 days",
        history="Smoker. Right-sided pleuritic pain, mild confusion yesterday.",
        vitals={"temperature": "38.7 C", "pulse": "110/min",
                "respiratory_rate": "32/min"},
        lab_values={"wbc": "15500/mm3 (high)", "crp": "180 mg/L (high)",
                    "urea": "8.6 mmol/L (high)"},
    ),
    "acute_myocardial_infarction": dict(
        age=55, sex="female", complaint="Central chest pressure with sweating for 1 hour",
        history="Type 2 diabetes, hypertension. Epigastric discomfort, no arm radiation.",
        vitals={"pulse": "100/min", "bp": "140/90 mmHg", "spo2": "94% room air"},
        lab_values={"troponin_i": "8.2 ng/mL (high)", "ecg": "ST elevation V2-V4"},
    ),
    "ischemic_stroke": dict(
        age=68, sex="male", complaint="Sudden left arm weakness and slurred speech 90 minutes ago",
        history="Atrial fibrillation on irregular medication, smoker.",
        vitals={"pulse": "irregular 104/min", "bp": "165/95 mmHg", "glucose": "7.1 mmol/L"},
        lab_values={"ecg": "atrial fibrillation", "ct_head": "early ischemic change, no hemorrhage"},
    ),
    "hypothyroidism": dict(
        age=41, sex="female", complaint="Fatigue, weight gain and cold intolerance for 6 months",
        history="Constipation, dry skin, heavier periods. Diffuse goiter on examination.",
        vitals={"pulse": "56/min", "bp": "115/75 mmHg"},
        lab_values={"tsh": "28 mIU/L (high)", "free_t4": "0.6 ng/dL (low)",
                    "anti_tpo": "positive"},
    ),
    "type_2_diabetes_mellitus": dict(
        age=49, sex="male", complaint="Thirst, excessive urination and blurred vision for 2 months",
        history="Obesity, Father with diabetes, sedentary job.",
        vitals={"pulse": "82/min", "bp": "138/86 mmHg", "weight": "92 kg"},
        lab_values={"hba1c": "8.9%", "fasting_glucose": "168 mg/dL (high)"},
    ),
    "pulmonary_tuberculosis": dict(
        age=37, sex="male", complaint="Productive cough and night sweats for 6 weeks",
        history="Weight loss 6 kg. Contact with tuberculosis patient. Low-grade evening fever.",
        vitals={"temperature": "37.8 C", "pulse": "94/min"},
        lab_values={"chest_xray": "right upper lobe cavity", "xpert_mtb_rif": "detected, rifampicin sensitive"},
    ),
    "chronic_kidney_disease": dict(
        age=57, sex="female", complaint="Progressive tiredness, nocturia and ankle swelling",
        history="Diabetes 12 years, hypertension. NSAID use for back pain.",
        vitals={"bp": "150/92 mmHg", "pulse": "84/min"},
        lab_values={"eGFR": "28 mL/min (low)", "creatinine": "2.6 mg/dL (high)",
                    "urine_acr": "420 mg/g"},
    ),
    "acute_appendicitis": dict(
        age=22, sex="male", complaint="Pain that started near the navel and settled in the right lower abdomen over 24 hours",
        history="Anorexia, one episode of vomiting, pain worse on movement.",
        vitals={"temperature": "37.9 C", "pulse": "96/min"},
        lab_values={"wbc": "13400/mm3 (high)", "crp": "46 mg/L (high)",
                    "urinalysis": "unremarkable"},
    ),
    "hyperthyroidism_graves": dict(
        age=29, sex="female", complaint="Palpitations, weight loss and heat intolerance for 2 months",
        history="Tremor, loose stools. Diffuse goiter with proptosis.",
        vitals={"pulse": "118/min", "bp": "145/60 mmHg"},
        lab_values={"tsh": "<0.01 mIU/L (low)", "free_t4": "3.4 ng/dL (high)",
                    "trab": "positive"},
    ),
}


def vignette_patient(stem: str, spec: dict) -> dict:
    return {
        "patient_id": f"EVAL-{stem[:12].upper()}",
        "age": spec["age"],
        "sex": spec["sex"],
        "chief_complaint": spec["complaint"],
        "history": spec["history"],
        "vitals": spec["vitals"],
        "lab_values": spec["lab_values"],
    }


def chunk_index(seed_stem: str, chunks: list[dict]) -> list[str]:
    return [c["chunk_id"] for c in chunks if c["chunk_id"].startswith(f"seed:{seed_stem}")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="rebuild BM25 index before generating data")
    args = ap.parse_args()

    if not kb_paths_from_settings()["index"].exists() or args.force:
        import subprocess
        subprocess.run(
            [sys.executable, "-m", "src.retrieval.bm25_index", "build"], cwd=ROOT,
            check=True)

    processed = ROOT / "data" / "processed"
    eval_patients = processed / "eval" / "patients"
    eval_patients.mkdir(parents=True, exist_ok=True)

    sources = sorted((ROOT / "knowledge_base" / "sources").glob("*.txt"))
    chunks = [json.loads(l) for l in
              open(ROOT / "knowledge_base" / "chunks.jsonl")] \
        if (ROOT / "knowledge_base" / "chunks.jsonl").exists() else []
    if not chunks:
        import subprocess
        subprocess.run(["python3", "-m", "src.corpus.build_kb"], cwd=ROOT, check=True)
        chunks = [json.loads(l) for l in
                  open(ROOT / "knowledge_base" / "chunks.jsonl")]

    from src.generation.prompting import GenerationConfig, build_prompt

    cfg = GenerationConfig.from_settings()
    sft_out = processed / "sft_pairs.jsonl"
    n_pairs = n_skip = 0

    with open(sft_out, "w") as sft:
        for seed_path in sources:
            stem = seed_path.stem
            seed_text = seed_path.read_text()
            target = report_target(seed_text)
            if target is None:
                print(f"!! could not parse sections in {seed_path.name}, skipped")
                n_skip += 1
                continue

            spec = VIGNETTES.get(stem)
            if spec is None:
                continue
            patient = vignette_patient(stem, spec)

            # numbered top-k evidence list for the prompt (BM25 only, deterministic)
            evidence = search_corpora(
                json.dumps(patient, ensure_ascii=False), chunks, top_k=6
            )

            sft.write(json.dumps({
                "condition": stem,
                "patient": patient,
                "prompt": build_prompt(patient, evidence, cfg),
                "target": target,
                "evidence_ids": [ev.chunk_id for ev in evidence],
            }) + "\n")
            n_pairs += 1

    # ---- held-out eval patients (different age band) ----
    for seed_path in sources:
        spec = VIGNETTES.get(seed_path.stem)
        if spec:
            spec = {**spec, "age": spec["age"] + 8}
            with open(eval_patients / f"{seed_path.stem}.json", "w") as fh:
                json.dump(vignette_patient(seed_path.stem, spec), fh, indent=2)

    # ---- retrieval dev set ----
    with open(processed / "eval" / "retrieval_dev.jsonl", "w") as dev:
        for seed_path in sources:
            stem = seed_path.stem
            relevant = chunk_index(stem, chunks)
            if not relevant:
                continue
            spec = VIGNETTES[stem]
            query = (f"{spec['complaint']}; {spec['history']}; " +
                     "; ".join(f"{k.replace('_', ' ')} {v}"
                               for k, v in list(spec["lab_values"].items())))
            dev.write(json.dumps({
                "query": query,
                "relevant_chunk_ids": relevant,
                "condition": stem,
            }) + "\n")

    print(f"sft pairs: {n_pairs} (skipped seeds: {n_skip}) -> {sft}")
    print(f"eval patients -> {eval_patients}")
    print("retrieval dev -> data/processed/eval/retrieval_dev.jsonl")


if __name__ == "__main__":
    main()
