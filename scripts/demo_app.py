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
        from src.generation.conflict import system_conflicts_report
        from src.generation.pipeline import ReportGenerator
        from src.retrieval.hybrid import build_patient_query

        with st.spinner("Retrieving evidence (coarse-fine) ..."):
            gen = get_generator() or ReportGenerator(mode="triplets")
            evidence = gen.retrieve(patient)
        st.subheader(f"Retrieved evidence ({len(evidence)})")
        for i, ev in enumerate(evidence, start=1):
            st.markdown(
                f"**[E{i}] {ev.title}** "
                f"(_chunk {ev.chunk_id}, score {ev.score:.3f})\n\n"
                f"> {ev.text[:400]}..."
            )

        with st.spinner("Distilling reasoning triplets ..."):
            prompt, triplets, _ = gen.retrieve_endpoints(patient)
        st.subheader(f"Reasoning triplets ({len(triplets)})")
        for idx, t in enumerate(triplets, start=1):
            st.markdown(
                f"**T{idx}** ({t.condition} **; {t.relation};** {t.conclusion})"
                f" `[{t.evidence_id}]`"
            )

        conflicts, confidence = system_conflicts_report(triplets)
        st.metric("Conflict confidence", f"{confidence:.2f}")
        if conflicts:
            st.warning(conflicts)

        if get_generator() is not None:
            with st.spinner("Generating grounded report ..."):
                report, ratio, _, trace = gen.run(patient)
            st.subheader(
                f"Observation report — unsupported claim ratio {ratio:.1%}, "
                f"confidence {trace['confidence']:.2f} ({trace['mode']} mode)"
            )
            st.markdown(report)
        else:
            st.subheader("Respondent prompt (model checkpoint pending Colab training)")
            st.code(prompt, language="text")
else:
    st.warning("No sample patient JSON files found.")
