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


# ---------------------------------------------------------------------------
# GT-MedRAG ablation matrices (paper-style variants for report generation)
# ---------------------------------------------------------------------------

GT_VARIANTS = (
    "full_rag_triplets",       # proposed: coarse-fine + triplets + grounded report
    "rag_triplets_no_conflict",  # triplets, conflict/abstention OFF
    "rag_raw_snippets",        # respondent on raw snippets (no extractor)
    "rag_rrf_not_c2f",         # parallel RRF retrieval instead of coarse->fine
    "baseline_gpt2",           # raw pretrained GPT-2, no retrieval
    "domain_adapted",          # fine-tuned GPT-2, no retrieval
)


def run_gt_ablation(args: argparse.Namespace) -> None:
    """GT-MedRAG variant matrix: retrieval x context x safety modules."""
    from ..generation.pipeline import ReportGenerator
    from ..generation.postprocess import finalize_report, unsupported_ratio
    from ..generation.prompting import (
        GenerationConfig,
        build_prompt,
        build_triplet_prompt,
    )
    from ..generation.triplets import rule_based_extract
    from ..retrieval.coarse_fine import coarse_fine_search
    from ..retrieval.hybrid import build_patient_query, hybrid_search

    patients = [json.load(open(p)) for p in sorted(Path(args.patients).glob("*.json"))]
    variants = (
        list(GT_VARIANTS) if args.variant == "gt_all"
        else [v.strip() for v in args.variant.split(",") if v.strip()]
    )
    model_path = args.model if Path(args.model).exists() else None
    import importlib.util

    has_transformers = importlib.util.find_spec("transformers") is not None
    can_generate = model_path is not None or has_transformers
    cfg = GenerationConfig.from_settings()

    for variant in variants:
        for patient in patients:
            # retrieval + prompt per variant
            if variant in ("baseline_gpt2", "domain_adapted"):
                if not can_generate:
                    record(variant, patient, 0.0, note="no model yet - pending Colab")
                    continue
                text = no_retrieval_text(patient, model_path)
                if variant == "domain_adapted":
                    evidence = hybrid_search(build_patient_query(patient))
                    ratio = unsupported_ratio(text, evidence)
                else:
                    ratio = unsupported_ratio(text, [])
                record(variant, patient, ratio)
                continue
            if variant == "rag_rrf_not_c2f":
                evidence = hybrid_search(build_patient_query(patient))
                prompt, _, _ = build_triplet_prompt(
                    patient, evidence, cfg, extractor=rule_based_extract
                )
            elif variant == "rag_raw_snippets":
                evidence = coarse_fine_search(build_patient_query(patient))
                prompt = build_prompt(patient, evidence, cfg)
            elif variant == "full_rag_triplets":
                evidence = coarse_fine_search(build_patient_query(patient))
                prompt, _, _ = build_triplet_prompt(
                    patient, evidence, cfg,
                    extractor=rule_based_extract, inject_conflicts=True,
                )
            elif variant == "rag_triplets_no_conflict":
                evidence = coarse_fine_search(build_patient_query(patient))
                prompt, _, _ = build_triplet_prompt(
                    patient, evidence, cfg,
                    extractor=rule_based_extract, inject_conflicts=False,
                )
            else:
                raise ValueError(f"unknown GT variant {variant}")

            gen = ReportGenerator(model_path=model_path, mode="raw")
            if gen.model_path is None:
                record(variant, patient, 0.0, note="no model yet - pending Colab")
                continue
            raw = gen.generate(prompt)
            report, _ = finalize_report(raw, evidence)
            record(variant, patient, unsupported_ratio(report, evidence))
    print(f"results appended to {RESULT_FILE}")


def no_retrieval_text(patient: dict, model_path: str | None) -> str:
    from ..generation.pipeline import ReportGenerator
    from ..generation.prompting import patient_summary

    gen = ReportGenerator(model_path=model_path, mode="raw")
    prompt = (
        "Using your own knowledge only, draft an observation report for this "
        f"patient:\n{patient_summary(patient)}\n\nREPORT:\n"
        "FINDINGS:\nDIFFERENTIAL DIAGNOSES (ranked):\n"
        "RECOMMENDED INVESTIGATIONS:\n"
    )
    return gen.generate(prompt)


def record(variant: str, patient: dict, ratio: float, note: str = "") -> None:
    Path(RESULT_FILE).parent.mkdir(parents=True, exist_ok=True)
    row = {"variant": variant, "patient_id": patient.get("patient_id"),
           "unsupported_ratio": round(ratio, 4)}
    if note:
        row["note"] = note
    with open(RESULT_FILE, "a") as fh:
        fh.write(json.dumps(row) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patients", default="data/processed/eval/patients")
    ap.add_argument("--variant", default="all",
                    help="legacy: baseline_gpt2|domain_adapted|full_rag|all; "
                         "GT matrix: gt_all or comma-separated GT variant names")
    ap.add_argument("--model", default="models/gpt2-clinical-rag")
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()

    if args.aggregate:
        aggregate()
    elif args.variant == "gt_all" or any(v in str(args.variant) for v in GT_VARIANTS):
        run_gt_ablation(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
