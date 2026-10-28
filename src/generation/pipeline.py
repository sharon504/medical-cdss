"""End-to-end GT-MedRAG pipeline (ER-MedRAG adapted for report generation):

patient record -> coarse-fine retrieval -> evidence [E#] -> reasoning
triplets (+ provenance, + conflict flags) -> domain-adapted GPT-2 respondent
-> grounded report with triplet-level citations, CONFLICTS/UNCERTAINTY
handling and reference footer.

Usage (programmatic):
    from src.generation.pipeline import generate_report
    report, ratio, evidence = generate_report(patient_dict)
"""

from __future__ import annotations

from pathlib import Path

from ..retrieval.base import Evidence
from ..retrieval.hybrid import build_patient_query
from .conflict import system_conflicts_report
from .prompting import GenerationConfig
from .prompting import build_prompt as raw_evidence_prompt
from .prompting import build_triplet_prompt, generate_kwargs
from .postprocess import finalize_report
from .triplets import rule_based_extract as default_extract


class ReportGenerator:
    """Lazy-loads a (domain-adapted) GPT-2 respondent with transformers.

    mode:
      - "triplets" (default): evidence is distilled to reasoning triplets
        (paper-faithful Extractor-Respondent flow);
      - "raw": prompts on raw snippets (pre-paper baseline / ablation).
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        cfg: GenerationConfig | None = None,
        mode: str = "triplets",
        extractor=None,
    ):
        self.cfg = cfg or GenerationConfig.from_settings()
        # Keep HuggingFace repo ids (e.g. "org/model") intact on Windows;
        # only convert to a Path when the path actually exists locally.
        if model_path is None:
            self.model_path = None
        else:
            candidate = Path(model_path)
            self.model_path = candidate if candidate.exists() else str(model_path)
        self.mode = mode
        self.extractor = extractor or default_extract
        self._model = None
        self._tokenizer = None

    def ensure_model(self) -> tuple:
        if self._model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            name = str(self.model_path) if self.model_path else self.cfg.base_model
            print(f"loading generator: {name}")
            self._tokenizer = AutoTokenizer.from_pretrained(name)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            self._model = AutoModelForCausalLM.from_pretrained(name)
        return self._tokenizer, self._model

    def generate(self, prompt: str) -> str:
        tokenizer, model = self.ensure_model()
        encoding = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        )
        tokens = model.generate(**encoding, **generate_kwargs(self.cfg))
        return tokenizer.decode(tokens[0], skip_special_tokens=True)

    def retrieve_endpoints(self, patient: dict):
        """Evidence + prompt without invoking the model (for studies/eval)."""
        evidence = self.retrieve(patient)
        if self.mode == "triplets":
            prompt, triplets, evidence = build_triplet_prompt(
                patient, evidence, self.cfg, extractor=self.extractor
            )
            return prompt, triplets, evidence
        return raw_evidence_prompt(patient, evidence, self.cfg), [], evidence

    def retrieve(self, patient: dict):
        from ..retrieval.coarse_fine import coarse_fine_search

        return coarse_fine_search(build_patient_query(patient))

    def extract_triplets(self, evidence: list[Evidence]) -> list:
        from .triplets import concatenated_block

        triplets = []
        for i, ev in enumerate(evidence, start=1):
            triplets.extend(self.extractor(ev, f"E{i}"))
        return triplets

    def run(self, patient: dict, mode: str | None = None):
        """Returns (report, unsupported_ratio, evidence, trace: dict)."""
        mode = mode or self.mode
        evidence = self.retrieve(patient)

        if mode == "triplets":
            prompt, triplets, evidence = build_triplet_prompt(
                patient, evidence, self.cfg, extractor=self.extractor
            )
            conflicts_section, confidence = system_conflicts_report(triplets)
        else:
            prompt = raw_evidence_prompt(patient, evidence, self.cfg)
            conflicts_section, confidence = None, 1.0

        raw = self.generate(prompt)
        report, ratio = finalize_report(raw, evidence)
        if conflicts_section and confidence >= 0.6:
            report = f"{report}\n\n{conflicts_section}"
        elif conflicts_section:
            report = (
                "SYSTEM NOTE: conflict pressure high - automatic output "
                "withheld; escalate to clinician.\n\n" + conflicts_section
            )
        trace = {
            "mode": mode,
            "n_evidence": len(evidence),
            "confidence": round(confidence, 3),
        }
        return report, ratio, evidence, trace


def generate_report(
    patient: dict,
    model_path: str | None = None,
    mode: str = "triplets",
):
    """Convenience wrapper used by the CLI and the Streamlit demo."""
    report, ratio, evidence, trace = ReportGenerator(
        model_path=model_path, mode=mode
    ).run(patient)
    return report, ratio, evidence, trace


def _prompt_for_debug(patient: dict):
    """Keep the raw-evidence prompt accessible for ablation tooling."""
    cfg = GenerationConfig.from_settings()
    from ..retrieval.hybrid import build_patient_query as q
    from ..retrieval.coarse_fine import coarse_fine_search

    return raw_evidence_prompt(patient, coarse_fine_search(q(patient)), cfg)
