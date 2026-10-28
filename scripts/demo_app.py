"""Streamlit demo: evidence-grounded observation reports.

Local run:
    streamlit run scripts/demo_app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title="Medical CDSS", layout="wide")


@st.cache_resource
def get_generator():
    from src.generation.pipeline import ReportGenerator

    model_dir = ROOT / "models" / "gpt2-clinical-rag"
    if not model_dir.exists():
        return None
    return ReportGenerator(model_path=model_dir)


def demo_patient_files() -> list[Path]:
    files = sorted((ROOT / "configs").glob("*.json"))
    files += sorted((ROOT / "data" / "processed" / "eval" / "patients").glob("*.json"))
    return files


st.title("Evidence-Grounded Medical Observation Reports")
st.caption(
    "Retrieval-augmented clinical decision support — outputs are for clinician "
    "review only, not autonomous diagnosis."
)

sample_files = demo_patient_files()
chosen = st.selectbox(
    "Demo patient record", [p.name for p in sample_files],
    index=0 if sample_files else 0,
)
model_missing = get_generator() is None
if model_missing:
    st.info(
        "No fine-tuned model found at models/gpt2-clinical-rag — the demo below "
        "shows retrieval + prompt + evidence citations without generation. Train "
        "the model with the Colab notebooks first for full reports."
    )

if sample_files:
    chosen_index = [p.name for p in sample_files].index(chosen)
    patient = json.load(open(sample_files[chosen_index]))
    with st.expander("Patient record", expanded=True):
        st.json(patient)

    if st.button("Generate observation report", type="primary"):
        from src.retrieval.hybrid import build_patient_query, hybrid_search

        with st.spinner("Retrieving evidence ..."):
            evidence = hybrid_search(build_patient_query(patient))
        st.subheader(f"Retrieved evidence ({len(evidence)})")
        for i, ev in enumerate(evidence, start=1):
            st.markdown(
                f"**[E{i}] {ev.title}** "
                f"(_chunk {ev.chunk_id}, score {ev.score:.3f})\n\n"
                f"> {ev.text[:400]}..."
            )

        if not model_missing:
            with st.spinner("Generating grounded report ..."):
                report, ratio, _ = get_generator().run(patient)
            st.subheader(f"Observation report — unsupported claim ratio {ratio:.1%}")
            st.markdown(report)
        else:
            from src.generation.prompting import (
                GenerationConfig, build_prompt, patient_summary)

            cfg = GenerationConfig.from_settings()
            st.subheader("Prompt that would be sent to the model")
            st.code(build_prompt(patient, evidence, cfg), language="text")
            st.subheader("Patient summary extracted")
            st.code(patient_summary(patient, cfg.max_patient_chars), language="text")
else:
    st.warning("No sample patient JSON files found.")
