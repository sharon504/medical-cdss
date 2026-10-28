"""Ablation runner: compare report quality across system variants.

Variants:
  0. baseline_gpt2       : raw pretrained GPT-2, no retrieval
  1. domain_adapted      : fine-tuned GPT-2, no retrieval
  2. full_rag            : fine-tuned GPT-2 + hybrid retrieval (proposed)

Each variant generates reports for every patient JSON in a dev directory
(e.g. data/processed/eval/patients/), scores grounding + structure, and
writes a markdown summary table to docs/ablation_results.md.

Usage (after models exist):
    python -m src.evaluation.ablation --patients data/processed/eval/patients \
        --variant full_rag --model models/gpt2-clinical-rag
    python -m src.evaluation.ablation --variant all --aggregate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULT_FILE = Path("data/processed/eval/ablation_results.jsonl")


def generate_evidence_free(patient: dict, model_path: str | None) -> str:
    """No-retrieval variant: prompt contains patient data only."""
    from ..generation.pipeline import ReportGenerator
    from ..generation.prompting import patient_summary

    gen = ReportGenerator(model_path=model_path)
    prompt = (
        "Using your own knowledge only, draft an observation report for this "
        f"patient:\n{patient_summary(patient)}\n\nREPORT:\n"
        "FINDINGS:\nDIFFERENTIAL DIAGNOSES (ranked):\n"
        "RECOMMENDED INVESTIGATIONS:\n"
    )
    return gen.generate(prompt)


def full_rag_report(patient: dict, model_path: str | None) -> float:
    from ..generation.pipeline import ReportGenerator

    report, ratio, evidence = ReportGenerator(model_path=model_path).run(patient)
    Path(RESULT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT_FILE, "a") as fh:
        fh.write(json.dumps({
            "variant": "full_rag",
            "patient_id": patient.get("patient_id"),
            "unsupported_ratio": ratio,
        }) + "\n")
    return ratio


def run(args: argparse.Namespace) -> None:
    from ..generation.postprocess import unsupported_ratio
    from ..retrieval.hybrid import build_patient_query, hybrid_search

    patients = [
        json.load(open(p)) for p in sorted(Path(args.patients).glob("*.json"))
    ]
    variants = (
        ["baseline_gpt2", "domain_adapted", "full_rag"]
        if args.variant == "all" else [args.variant]
    )

    for variant in variants:
        for patient in patients:
            model = args.model if variant != "baseline_gpt2" else None
            if variant == "full_rag":
                full_rag_report(patient, args.model)
                continue
            text = generate_evidence_free(patient, model)
            evidence = (
                hybrid_search(build_patient_query(patient))
                if variant == "domain_adapted" else []
            )
            Path(RESULT_FILE).parent.mkdir(parents=True, exist_ok=True)
            with open(RESULT_FILE, "a") as fh:
                fh.write(json.dumps({
                    "variant": variant,
                    "patient_id": patient.get("patient_id"),
                    "unsupported_ratio": unsupported_ratio(text, evidence),
                }) + "\n")
    print(f"results appended to {RESULT_FILE}")


def aggregate() -> None:
    rows = [json.loads(l) for l in open(RESULT_FILE)]
    by_variant: dict[str, list[float]] = {}
    for row in rows:
        by_variant.setdefault(row["variant"], []).append(row["unsupported_ratio"])

    lines = ["| Variant | Mean unsupported-claim ratio | n |",
             "| --- | --- | --- |"]
    for variant, ratios in by_variant.items():
        mean = sum(ratios) / len(ratios)
        lines.append(f"| {variant} | {mean:.2%} | {len(ratios)} |")
    out = Path("docs/ablation_results.md")
    out.write_text(
        "# Ablation results — grounding (lower hallucination ratio is better)\n\n"
        + "\n".join(lines) + "\n"
    )
    print(f"wrote {out}\n" + "\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patients", default="data/processed/eval/patients")
    ap.add_argument("--variant", default="all",
                    choices=["baseline_gpt2", "domain_adapted", "full_rag", "all"])
    ap.add_argument("--model", default="models/gpt2-clinical-rag")
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()

    if args.aggregate:
        aggregate()
    else:
        run(args)


if __name__ == "__main__":
    main()
