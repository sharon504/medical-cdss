"""End-to-end RAG pipeline: patient record -> hybrid retrieval -> GPT-2 ->
grounded report with citations.

Usage (programmatic):
    from src.generation.pipeline import generate_report
    report, ratio, evidence = generate_report(patient_dict)
"""

from __future__ import annotations

from pathlib import Path

from ..retrieval.base import Evidence
from ..retrieval.hybrid import build_patient_query, hybrid_search
from .postprocess import finalize_report
from .prompting import GenerationConfig, build_prompt, generate_kwargs


class ReportGenerator:
    """Lazy-loads a (domain-adapted) GPT-2 checkpoint with transformers."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        cfg: GenerationConfig | None = None,
    ):
        self.cfg = cfg or GenerationConfig.from_settings()
        self.model_path = Path(model_path) if model_path else None
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

    def retrieve(self, patient: dict) -> list[Evidence]:
        return hybrid_search(build_patient_query(patient))

    def run(self, patient: dict) -> tuple[str, float, list[Evidence]]:
        evidence = self.retrieve(patient)
        prompt = build_prompt(patient, evidence, self.cfg)
        raw = self.generate(prompt)
        report, ratio = finalize_report(raw, evidence)
        return report, ratio, evidence


def generate_report(patient: dict, model_path: str | None = None):
    """Convenience wrapper used by the CLI and the Streamlit demo."""
    report, ratio, evidence = ReportGenerator(model_path=model_path).run(patient)
    return report, ratio, evidence
