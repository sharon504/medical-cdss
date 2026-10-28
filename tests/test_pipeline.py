"""End-to-end test battery for the BM25-only local path of Medical CDSS /
GT-MedRAG. Dense legs (sentence-transformers/faiss) are exercised separately
because of heavy dependencies; here they must degrade gracefully."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.corpus.chunking import chunk_words  # noqa: E402
from src.generation.conflict import detect_conflicts, system_conflicts_report  # noqa: E402
from src.generation.pipeline import ReportGenerator  # noqa: E402
from src.generation.postprocess import (  # noqa: E402
    finalize_report,
    is_supported,
    unsupported_ratio,
)
from src.generation.triplets import (  # noqa: E402
    Triplet,
    concatenated_block,
    gold_triplets,
    parse_triplet_output,
    rule_based_extract,
)
from src.evaluation.retrieval_metrics import mrr, recall_at_k  # noqa: E402
from src.evaluation.robustness import (  # noqa: E402
    corrupt_evidence,
    grounding_robustness_score,
)
from src.retrieval.base import Evidence, load_chunks  # noqa: E402
from src.retrieval.bm25_index import search_corpora  # noqa: E402


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def chunks():
    return load_chunks(ROOT / "knowledge_base")


@pytest.fixture(scope="module")
def sample_patient():
    return json.load(open(ROOT / "configs" / "sample_patient.json"))


def ev(text, cid="seed:test::c0", eid="E1"):
    return Evidence(chunk_id=cid, title="test title", source_type="seed_reference",
                    text=text) if eid == "E1" else Evidence(
        chunk_id=cid, title="t", source_type="seed_reference", text=text,
        score=1.0)


# ---------------------------------------------------------------------------
# chunking
# ---------------------------------------------------------------------------

def test_chunking_respects_max_words_and_overlap():
    text = " ".join(f"Sentence number {i} about clinical findings here." for i in range(120))
    out = list(chunk_words(text, max_words=50, overlap=10))
    assert all(len(chunk) <= 50 for _, chunk in out)
    assert len(out) > 3
    # overlap means consecutive chunks share some words
    a = " ".join(out[0][1][-6:])
    b = " ".join(out[1][1][:6])
    assert any(w in out[1][1] for w in out[0][1][-10:])


# ---------------------------------------------------------------------------
# retrieval (BM25-only)
# ---------------------------------------------------------------------------

def test_bm25_search_ranks_relevant_first(chunks):
    hits = search_corpora("menorrhagia pallor low ferritin microcytic anemia",
                          chunks, top_k=5)
    assert hits, "no hits returned"
    assert hits[0].chunk_id.startswith("seed:iron_deficiency_anemia")


def test_bm25_scores_are_positive_and_sorted(chunks):
    hits = search_corpora("diabetes glucose hba1c fasting", chunks, top_k=5)
    assert hits and hits[0].chunk_id.startswith("seed:type_2_diabetes")
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_coarse_fine_degrades_without_dense(sample_patient, chunks):
    from src.retrieval.coarse_fine import coarse_fine_search
    from src.retrieval.hybrid import build_patient_query

    hits = coarse_fine_search(build_patient_query(sample_patient), n=15, m=3)
    assert len(hits) <= 3
    ids = [h.chunk_id for h in hits]
    assert "seed:iron_deficiency_anemia::c0" in ids


def test_hybrid_rrf_degrades_and_returns_evidence(sample_patient):
    from src.retrieval.hybrid import build_patient_query, hybrid_search

    hits = hybrid_search(build_patient_query(sample_patient), top_k=4)
    assert 0 < len(hits) <= 4


def test_retrieval_metrics_agree():
    ranked = ["a", "b", "c"]
    relevant = {"c", "d"}
    assert recall_at_k(ranked, relevant, k=3) == 0.5
    assert mrr(ranked, relevant) == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# triplets
# ---------------------------------------------------------------------------

def test_rule_based_extract_produces_parseable_triplets():
    text = ("Microcytic anemia with low serum ferritin is diagnostic of iron "
            "deficiency. Thalassemia trait is suggested by very low MCV.")
    e = Evidence(chunk_id="seed:t::c0", title="t", source_type="seed_reference", text=text)
    triplets = rule_based_extract(e, "E1")
    assert triplets, "no triplets from cue-rich text"
    assert all(t.relation for t in triplets)
    assert all(t.evidence_id == "E1" for t in triplets)
    rendered = concatenated_block(triplets)
    assert "T1:" in rendered and "[E1]" in rendered


def test_gold_triplets_from_seed_text(chunks):
    seed = next(c for c in chunks if c["chunk_id"].startswith("seed:iron_deficiency_anemia"))
    e = Evidence(chunk_id=seed["chunk_id"], title=seed["title"],
                 source_type=seed["source_type"], text=seed["text"])
    golds = gold_triplets(e, "E1")
    assert golds
    assert any(g.relation in ("suggested by", "confirmed by", "is diagnostic of",
                              "associated with") for g in golds)
    assert all(g.condition and g.conclusion for g in golds)


def test_parse_triplet_output():
    t = parse_triplet_output("(low ferritin; is diagnostic of; iron deficiency) [E3]", "E3")
    assert t is not None
    assert t.condition == "low ferritin"
    assert t.conclusion == "iron deficiency"
    assert parse_triplet_output("no triplet here") is None


# ---------------------------------------------------------------------------
# conflicts
# ---------------------------------------------------------------------------

def test_conflict_detects_negation_and_frequency():
    t_aff = Triplet("thalassemia trait anemia", "suggested by", "very low MCV", "E1", "c1")
    t_neg = Triplet("thalassemia trait anemia risk", "confirmed by", "not excluded by RDW", "E2", "c2")
    t_freq_a = Triplet("common presentation iron deficiency", "associated with",
                       "frequent cause of menorrhagia", "E1", "c1")
    t_freq_b = Triplet("common presentation iron deficiency two", "associated with",
                       "rare cause of menorrhagia", "E2", "c2")
    flags = detect_conflicts([t_aff, t_neg, t_freq_a, t_freq_b])
    assert flags, "no conflicts detected"
    reasons = {f.reason for f in flags}
    assert reasons & {"negation", "contradictory-conclusions"}


def test_clean_evidence_has_no_conflicts_and_full_confidence():
    t = Triplet("iron deficiency anemia", "suggested by", "low ferritin", "E1", "c1")
    section, confidence = system_conflicts_report([t])
    assert section is None
    assert confidence == 1.0


def test_low_confidence_triggers_escalation():
    t1 = Triplet("anemia of chronic disease here", "associated with", "common cause", "E1", "c1")
    t2 = Triplet("anemia of chronic disease also", "associated with", "rare cause", "E2", "c2")
    t3 = Triplet("anemia of chronic disease more", "confirmed by", "not excluded", "E3", "c3")
    section, confidence = system_conflicts_report([t1, t2, t3])
    assert section is not None
    assert "clinic" in section.lower() or "confidence" in section.lower()


# ---------------------------------------------------------------------------
# post-processing / grounding
# ---------------------------------------------------------------------------

EVIDENCE = [
    Evidence(chunk_id="seed:a::c0", title="t1", source_type="pubmed_abstract",
             text="microcytic anemia with low serum ferritin is diagnostic of "
                  "iron deficiency anemia in adults."),
    Evidence(chunk_id="seed:b::c0", title="t2", source_type="pubmed_abstract",
             text="oral iron therapy raises hemoglobin within four weeks "
                  "confirming the diagnosis."),
]


def test_supported_claim_recognized():
    assert is_supported("Microcytic anemia with low serum ferritin is diagnostic of "
                        "iron deficiency.", EVIDENCE)
    assert not is_supported("Totally unrelated arbeles claims about quantum "
                            "chromodynamics tomorrow.", EVIDENCE)


def test_unsupported_ratio_line_based():
    raw = ("FINDINGS:\nMicrocytic anemia with low serum ferritin is diagnostic of "
           "iron deficiency.\nEVIDENCE REFERENCES:\nThis one mentions "
           "completely invented arbeles numbers about nothing relevant at all.\n")
    ratio = unsupported_ratio(raw, EVIDENCE)
    assert 0.3 < ratio <= 1.0, ratio


def test_finalize_report_suppresses_and_keeps_header():
    raw = ("FINDINGS:\nMicrocytic anemia is diagnostic of iron deficiency anemia. "
           "Invented jabberwocky sincerely nonclinical nonsense everywhere.\n")
    report, ratio = finalize_report(raw, EVIDENCE)
    assert "FINDINGS:" in report
    assert "EVIDENCE REFERENCES" in report
    assert "(suppressed:" in report
    assert 0.0 <= ratio <= 1.0


# ---------------------------------------------------------------------------
# robustness
# ---------------------------------------------------------------------------

def test_corruption_operators_behave(chunks):
    evs = [
        Evidence(chunk_id=f"E{i}", title="t", source_type="s", text=t)
        for i, t in enumerate([
            "Microcytic anemia with low serum ferritin is diagnostic of iron deficiency.",
            "Thalassemia trait is suggested by very low MCV with normal RDW.",
            "Ferritin below 30 ug/L supports the diagnosis of iron deficiency.",
        ])
    ]
    drop = corrupt_evidence(evs, 0.34, op="drop")
    assert len(drop.corrupted) == 2 and drop.n_original == 3

    swap = corrupt_evidence(evs, 0.34, op="swap", all_chunks=chunks)
    assert len(swap.corrupted) == 3
    assert any("corrupt" in e.chunk_id or e.source_type == "corrupt_swap"
               for e in swap.corrupted), [e.chunk_id for e in swap.corrupted]

    contra = corrupt_evidence(evs, 0.5, op="contradict")
    assert any(" not " in e.text for e in contra.corrupted)


def test_grs_bounds_and_semantics():
    stable = grounding_robustness_score({0.1: 0.95, 0.3: 0.94}, 0.95)
    weak = grounding_robustness_score({0.1: 0.3, 0.3: 0.1}, 0.95)
    assert 1.0 >= stable > 0.9
    assert weak < 0.8
    assert grounding_robustness_score({}, 0.9) == 1.0


# ---------------------------------------------------------------------------
# pipeline (BM25 path with stub generation)
# ---------------------------------------------------------------------------

def test_pipeline_run_triplets_mode(sample_patient, monkeypatch):
    good = ("FINDINGS:\nMicrocytic anemia with low serum ferritin is diagnostic of "
            "iron deficiency.\nDIFFERENTIAL DIAGNOSES (ranked):\n(1) Iron "
            "deficiency anemia [E1]\n(2) Thalassemia trait [E1]\n"
            "RECOMMENDED INVESTIGATIONS:\nSerum iron studies and CRP [E1]\n"
            "CONFLICTS / UNCERTAINTY:\nNone\nEVIDENCE REFERENCES:\n"
            "Invented gibberish unrelated to anything retrieved anywhere here.\n")

    gen = ReportGenerator(mode="triplets")
    gen.generate = lambda prompt: good  # stub the respondent
    report, ratio, evidence, trace = gen.run(sample_patient)

    assert trace["mode"] == "triplets"
    assert len(evidence) == 5  # coarse-fine m=5
    assert "(suppressed:" in report
    assert "EVIDENCE REFERENCES" in report
    assert ratio >= 0.0


def test_pipeline_run_raw_mode(sample_patient, monkeypatch):
    raw_ok = ("FINDINGS:\nMicrocytic anemia is diagnostic of iron deficiency "
              "anemia here.\nEVIDENCE REFERENCES:\nsomething merely decorative. "
              "This tail adds ungrounded jabberwocky fabrications clearly.\n")
    gen = ReportGenerator(mode="raw")
    gen.generate = lambda prompt: raw_ok
    report, ratio, evidence, trace = gen.run(sample_patient)
    assert trace["mode"] == "raw"
    assert 0 < len(evidence) <= 5


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
