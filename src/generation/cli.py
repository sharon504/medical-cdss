"""CLI: generate an evidence-grounded observation report for a patient file.

Usage:
    python -m src.generation.cli --patient configs/sample_patient.json \
        [--model models/gpt2-clinical-rag] [--mode triplets|raw] \
        [--retrieval coarse_fine|rrf] [--out report.md]

Without --model (and no models/ checkpoint), the CLI prints the retrieval
results, distilled triplets, and the full generation prompt — useful for
demoing the proof of the retrieval and extraction stages.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patient", required=True, help="patient JSON file")
    ap.add_argument("--model", default=None, help="fine-tuned checkpoint dir")
    ap.add_argument("--mode", default="triplets",
                    choices=["triplets", "raw"],
                    help="respondent context: distilled triplets or raw snippets")
    ap.add_argument("--retrieval", default="coarse_fine",
                    choices=["coarse_fine", "rrf"])
    ap.add_argument("--out", default=None, help="write markdown report here")
    args = ap.parse_args()

    from ..retrieval.hybrid import build_patient_query, hybrid_search
    from .pipeline import ReportGenerator
    from .postprocess import finalize_report, unsupported_ratio
    from .prompting import GenerationConfig, build_prompt, build_triplet_prompt

    with open(args.patient) as fh:
        patient = json.load(fh)

    query = build_patient_query(patient)
    if args.retrieval == "rrf":
        evidence = hybrid_search(query)
    else:
        from ..retrieval.coarse_fine import coarse_fine_search

        evidence = coarse_fine_search(query)

    print(f"retrieved {len(evidence)} evidence passages ({args.retrieval}):")
    for i, ev in enumerate(evidence, start=1):
        print(f"  [E{i}] {ev.score:.3f} {ev.title[:70]}")

    cfg = GenerationConfig.from_settings()
    if args.mode == "triplets":
        prompt, triplets, evidence = build_triplet_prompt(patient, evidence, cfg)
        print(f"\nextracted {len(triplets)} reasoning triplets:")
        for t in triplets[:8]:
            print(f"  [{t.evidence_id}] ({t.condition[:44]} | "
                  f"{t.relation} | {t.conclusion[:44]})")
    else:
        prompt = build_prompt(patient, evidence, cfg)

    has_model = args.model is not None or Path("models/gpt2-clinical-rag").exists()
    if not has_model:
        print("\nunsupported-claim ratio: n/a (generation stage not loaded)\n")
        print(prompt)
    else:
        gen = ReportGenerator(model_path=args.model, mode=args.mode)
        raw = gen.generate(prompt)
        report, ratio = finalize_report(raw, evidence)
        print(f"\nunsupported-claim ratio: {ratio:.2%}\n")
        print(report)
        if args.out:
            Path(args.out).write_text(report)
            print(f"saved -> {args.out}")
        return

    if args.out:
        Path(args.out).write_text(prompt)
        print(f"saved prompt -> {args.out}")


if __name__ == "__main__":
    main()
