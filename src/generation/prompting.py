"""Prompt construction and inference settings for the evidence-grounded
GPT-2 report generator.

The prompt design follows the project's grounding principle: the model is
given (a) a structured patient summary and (b) numbered retrieved passages,
and must write the report reusing only supported claims. Unsupported
sentences are removed downstream by src/generation/postprocess.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from ..retrieval.base import Evidence


@dataclass
class GenerationConfig:
    base_model: str = "gpt2-medium"
    max_new_tokens: int = 320
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.2
    max_evidence_chars: int = 5200
    max_patient_chars: int = 900

    @classmethod
    def from_settings(cls) -> "GenerationConfig":
        cfg = Path(__file__).resolve().parents[2] / "configs" / "settings.yaml"
        with open(cfg) as fh:
            s = yaml.safe_load(fh)["generation"]
        return cls(
            base_model=s["pretrained_model"],
            max_new_tokens=s["max_new_tokens"],
            temperature=s["temperature"],
            top_p=s["top_p"],
            repetition_penalty=s["repetition_penalty"],
        )


def patient_summary(patient: dict, max_chars: int = 900) -> str:
    lines = [
        f"Patient {patient.get('patient_id', '')}: "
        f"{patient.get('age', '?')} year old {patient.get('sex', '?')}."
    ]
    lines.append(f"Chief complaint: {patient.get('chief_complaint', '-')}")
    if patient.get("history"):
        lines.append(f"History: {patient['history']}")
    for label, value in (patient.get("vitals") or {}).items():
        lines.append(f"Vitals - {label.replace('_', ' ')}: {value}")
    for label, value in (patient.get("lab_values") or {}).items():
        lines.append(f"Lab - {label.replace('_', ' ')}: {value}")
    text = "\n".join(lines)
    return text[:max_chars]


def build_prompt(patient: dict, evidence: list[Evidence], cfg: GenerationConfig) -> str:
    """Assemble the final generation prompt with numbered evidence blocks."""
    blocks, budget = [], cfg.max_evidence_chars
    for i, ev in enumerate(evidence, start=1):
        body = f"[E{i}] ({ev.title})\n{ev.text}"
        if budget - len(body) < 0:
            break
        budget -= len(body)
        blocks.append(body)
    evidence_block = "\n\n".join(blocks) or "[no evidence retrieved]"

    instruction = (
        "You are a clinical decision support assistant. Using ONLY the "
        "retrieved evidence provided, draft a structured observation report. "
        "Cite evidence ids like [E1] next to each claim. Do not state facts "
        "that are not supported by the evidence or the patient data.\n\n"
    )
    return (
        f"{instruction}"
        f"PATIENT SUMMARY:\n{patient_summary(patient, cfg.max_patient_chars)}\n\n"
        f"RETRIEVED EVIDENCE:\n{evidence_block}\n\n"
        f"REPORT:\n"
        f"FINDINGS:\n"
        f"DIFFERENTIAL DIAGNOSES (ranked):\n"
        f"RECOMMENDED INVESTIGATIONS:\n"
        f"EVIDENCE REFERENCES:\n"
    )


def build_triplet_prompt(
    patient: dict,
    evidence: list[Evidence],
    cfg: GenerationConfig,
    extractor=None,
    inject_conflicts: bool = False,
) -> tuple[str, list, list[Evidence]]:
    """Respondent prompt conditioned on distilled reasoning triplets
    (paper-faithful) instead of raw evidence snippets.

    inject_conflicts=True merges flagged contradiction triplets into the
    triplet block (GT-MedRAG conflict module) with aggregator rendering;
    ablation 'rag_triplets_no_conflict' sets it False.
    """
    from .triplets import concatenated_block, rule_based_extract

    extractor = extractor or rule_based_extract
    triplets: list = []
    for i, ev in enumerate(evidence, start=1):
        triplets.extend(extractor(ev, f"E{i}"))

    conflicts_section = None
    if inject_conflicts and triplets:
        from .conflict import detect_conflicts, conflict_triplet, aggregate_confidence

        flags = detect_conflicts(triplets)
        triplets = triplets + [conflict_triplet(f) for f in flags]
        if flags and aggregate_confidence(triplets, base=1.0) < 0.6:
            conflicts_section = (
                "SYSTEM NOTE: conflict pressure high - automatic output "
                "withheld; escalate to clinician for adjudication."
            )

    evidence_block = (
        concatenated_block(triplets)
        if triplets
        else "[no usable evidence triplets extracted]"
    )

    instruction = (
        "You are a clinical decision support assistant. Using ONLY the "
        "structured reasoning triplets below, draft a structured observation "
        "report. Cite the triplet's evidence id, e.g. [E1], next to each "
        "claim you take from it. If a triplet is flagged CONFLICT or a "
        "conclusion is uncertain, say so in the CONFLICTS / UNCERTAINTY "
        "section instead of asserting either side.\n\n"
    )
    label = "RETRIEVED EVIDENCE TRIPLETS:"
    prompt = (
        f"{instruction}"
        f"PATIENT SUMMARY:\n{patient_summary(patient, cfg.max_patient_chars)}\n\n"
        f"{label}\n{evidence_block}\n\n"
        f"REPORT:\n"
        f"FINDINGS:\n"
        f"DIFFERENTIAL DIAGNOSES (ranked):\n"
        f"RECOMMENDED INVESTIGATIONS:\n"
        f"CONFLICTS / UNCERTAINTY:\n"
        f"EVIDENCE REFERENCES:\n"
    )
    if conflicts_section:
        prompt = f"{prompt}\n{conflicts_section}\n"
    return prompt, triplets, evidence


def generate_kwargs(cfg: GenerationConfig) -> dict:
    return {
        "max_new_tokens": cfg.max_new_tokens,
        "do_sample": True,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "repetition_penalty": cfg.repetition_penalty,
        "pad_token_id": 50256,  # GPT-2 eos
    }
