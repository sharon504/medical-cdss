"""Robustness + stability evaluation (GT-MedRAG novel evaluation layer).

ER-MedRAG admits its failure modes under noisy or contradictory retrieval
but evaluates only on clean retrieval. This module makes robustness a
first-class, measured property:

  A. Evidence corruption harness  - reproducible corruption of retrieved
     evidence at a given rate with three operators:
        drop        : remove a fraction of evidence passages
        swap        : replace passages with incoherent random ones
        contradict  : negate a phrase in the passage text (negation/antonym)
  B. Grounding Robustness Score (GRS) - area between the clean grounding
     score and the degradation curve across corruption rates (higher=better;
     range 0..1 via normalizing by the worst-case drop).
  C. Token-level output entropy  - replication of the paper's stability
     analysis (Fig.1 style), computed from generation logprobs, for each
     ablation configuration (raw snippets vs triplets etc.).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..retrieval.base import Evidence


# ---------------------------------------------------------------------------
# A. Corruption operators (seeded & reproducible)
# ---------------------------------------------------------------------------

_NEG_PREFIXES = ["not associated with ", "not indicated by "]


@dataclass
class CorruptionResult:
    corrupted: list[Evidence]
    n_original: int
    ops_applied: int


def corrupt_evidence(
    evidence: list[Evidence],
    rate: float,
    op: str = "drop",
    seed: int = 19,
    all_chunks: list[dict] | None = None,
) -> CorruptionResult:
    """Corrupt evidence by `rate` fraction using `op`.

    op:
      drop       - delete a fraction of passages (evidence starvation)
      swap       - replace them with random unrelated chunk texts (fabricated noise)
      contradict - negate relation cues inside kept passages (contradiction)
    """
    assert op in ("drop", "swap", "contradict"), f"unknown op {op}"
    rng = np.random.default_rng(seed)
    import copy

    evidence = [copy.deepcopy(ev) for ev in evidence]
    n_target = max(1, int(round(len(evidence) * rate)))
    n_applied = 0
    out: list[Evidence] = []
    if op == "drop":
        idxs = rng.choice(len(evidence), size=len(evidence) - n_target, replace=False)
        out = [evidence[i] for i in sorted(idxs)]
        return CorruptionResult(out, len(evidence), n_target)
    if op == "swap":
        keep_n = len(evidence) - n_target
        idxs = rng.choice(len(evidence), size=keep_n, replace=False)
        out = [evidence[i] for i in sorted(idxs)]
        pool = all_chunks or []
        for _ in range(n_target):
            if pool:
                pick = pool[int(rng.integers(len(pool)))]
                ev = evidence[-1]  # reuse last kept as scaffold
                decoy = Evidence(
                    chunk_id=pick.get("chunk_id", "corrupt:swap"),
                    title="Swapped noise passage",
                    source_type="corrupt_swap",
                    text=pick["text"][:400],
                    score=0.0,
                )
                out.append(decoy)
                n_applied += 1
        return CorruptionResult(out, len(evidence), n_applied)
    # contradict: negate relation cues / flip affirmative statements
    for ev in evidence:
        text = ev.text
        if n_applied < n_target:
            text2 = re.sub(
                r"\b(is|are) diagnostic of\b", r"\1 not diagnostic of", text, count=1
            )
            if text2 == text:
                text2 = re.sub(
                    r"(?m)\b(confirmed by|suggested by)\b",
                    lambda m: m.group(1) + " (counter-indicated)",
                    text, count=1,
                )
            if text2 != text:
                ev.text = text2
                n_applied += 1
        out.append(ev)
    return CorruptionResult(out, len(evidence), n_applied)


# ---------------------------------------------------------------------------
# B. Grounding Robustness Score
# ---------------------------------------------------------------------------

def grounding_score(report: str, evidence: list[Evidence]) -> float:
    """Clean grounding score for a report = 1 - unsupported_ratio."""
    from ..generation.postprocess import unsupported_ratio

    return 1.0 - unsupported_ratio(report, evidence)


def grounding_robustness_score(
    degradation: dict[float, float], clean_score: float
) -> float:
    """GRS = 1 - mean relative drop across corruption rates.

    degradation maps corruption rate -> grounding score at that rate.
    Reads as a single robustness value in [0, 1] (1 = no degradation).
    """
    drops = [
        max(0.0, clean_score - score) / max(clean_score, 1e-9)
        for rate, score in sorted(degradation.items())
    ]
    return round(1.0 - float(np.mean(drops)) if drops else 1.0, 4)


def robustness_curve(
    patient: dict,
    generator,
    ops: tuple[str, ...] = ("drop", "swap", "contradict"),
    rates: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5),
    seed: int = 19,
) -> dict:
    """Run one patient through corrupted-retrieval conditions.

    generator must expose .retrieve(patient) and .run(patient, mode=...);
    returns {op: {rate: grounding_score}} plus GRS summary.
    """
    evidence = generator.retrieve(patient)
    base_report, _, _, _ = generator.run(patient)
    clean = grounding_score(base_report, evidence)

    all_chunks = json.loads(
        (Path(__file__).resolve().parents[2] / "knowledge_base" / "chunks.jsonl")
        .read_text()
    )

    original_retrieve = generator.retrieve
    try:
        curve: dict[str, dict[float, float]] = {}
        for op in ops:
            curve[op] = {}
            for rate in rates:
                corr = corrupt_evidence(evidence, rate, op=op, seed=seed,
                                        all_chunks=all_chunks)
                generator.retrieve = (  # pin the corrupted set for this trial
                    lambda p, _c=corr.corrupted: _c
                )
                report, _, _, _ = generator.run(patient)
                curve[op][rate] = round(grounding_score(report, corr.corrupted), 4)
    finally:
        # restore the class-bound method patch
        if hasattr(original_retrieve, "__self__"):
            generator.retrieve = original_retrieve.__get__(generator)
        else:
            generator.retrieve = original_retrieve
    grs = {op: grounding_robustness_score(curve[op], clean) for op in ops}
    return {"clean_score": round(clean, 4), "curve": curve, "grs": grs}


# ---------------------------------------------------------------------------
# C. Output entropy stability (paper Sec. 4.11 / Fig.1 style, for reports)
# ---------------------------------------------------------------------------

def token_entropy(generator, prompt: str, n_samples: int = 8) -> float:
    """Mean token-level entropy of sampled report generations.

    generator must expose .cfg and transformer model; measures output
    distributional stability under the given context (lower = more stable).
    """
    import torch

    tokenizer, model = generator.ensure_model()
    n_new = 160  # max_new_tokens; keep prompt + generation <= 1024 positions
    enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                    max_length=1024 - n_new)
    entropies = []
    with torch.no_grad():
        for _ in range(n_samples):
            tokens = model.generate(
                **enc,
                max_new_tokens=n_new,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
            for step_scores in tokens.scores:
                probs = torch.softmax(step_scores[0], dim=-1)
                entropies.append(float(-(probs * torch.log(probs + 1e-9)).sum()))
    return round(float(np.mean(entropies)), 4) if entropies else 0.0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patient", default="configs/sample_patient.json")
    ap.add_argument("--model", default="models/gpt2-clinical-rag")
    args = ap.parse_args()

    from src.generation.pipeline import ReportGenerator

    gen = ReportGenerator(model_path=args.model, mode="triplets")
    result = robustness_curve(json.load(open(args.patient)), gen)
    print(json.dumps(result, indent=2))
