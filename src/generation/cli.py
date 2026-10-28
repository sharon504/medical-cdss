"""CLI: generate an evidence-grounded observation report for a patient file.

Usage:
    python -m src.generation.cli --patient configs/sample_patient.json \
        [--model models/gpt2-clinical-rag] [--out report.md]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patient", required=True, help="patient JSON file")
    ap.add_argument("--model", default=None, help="fine-tuned checkpoint dir")
    ap.add_argument("--out", default=None, help="write markdown report here")
    args = ap.parse_args()

    from ..retrieval.hybrid import build_patient_query, hybrid_search
    from .pipeline import ReportGenerator
    from .postprocess import finalize_report
    from .prompting import GenerationConfig, build_prompt

    with open(args.patient) as fh:
        patient = json.load(fh)

    evidence = hybrid_search(build_patient_query(patient))
    print(f"retrieved {len(evidence)} evidence passages:")
    for i, ev in enumerate(evidence, start=1):
        print(f"  [E{i}] {ev.score:.3f} {ev.title[:70]}")

    cfg = GenerationConfig.from_settings()
    prompt = build_prompt(patient, evidence, cfg)

    # If no model is available locally, emit retrieval-only report skeleton:
    if args.model is None and not Path("models").exists():
        report, ratio = finalize_report(
            "FINDINGS:\n(automatic generation disabled - no model checkpoint "
            "found; run scripts/train_colab instructions or pass --model)\n",
            evidence,
        )
    else:
        gen = ReportGenerator(model_path=args.model)
        raw = gen.generate(prompt)
        report, ratio = finalize_report(raw, evidence)

    print(f"\nunsupported-claim ratio: {ratio:.2%}\n")
    print(report)

    if args.out:
        Path(args.out).write_text(report)
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
