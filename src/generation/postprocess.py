"""Report post-processing: split generated text into sections, strip lines
whose claims are not supported by the retrieved evidence, and attach citation
references. This is the traceability / anti-hallucination layer.

Support check = lexical containment between the claim and evidence passage
(cheap, deterministic, no extra model needed). An NLI-based support checker
can replace `is_supported` later without changing the interface.
"""

from __future__ import annotations

import re

from ..retrieval.base import Evidence

_HEADER_RE = re.compile(
    r"^(FINDINGS|DIFFERENTIAL DIAGNOSES?( \(ranked\))?|RECOMMENDED INVESTIGATIONS|"
    r"EVIDENCE REFERENCES)",
    re.IGNORECASE,
)

# split into overlapping word windows
def _windows(text: str, size: int = 10, step: int = 6) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= size:
        return [" ".join(words)]
    return [" ".join(words[i:i + size]) for i in range(0, len(words) - size + 1, step)]


def _normalized(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", text.lower())


def _covers(claim_window: str, passage: str, min_hits: int = 4) -> bool:
    """True if at least `min_hits` claim words (excluding stopwords) appear in
    the evidence passage - a conservative lexical grounding heuristic."""
    stop = {
        "the", "a", "an", "of", "and", "or", "to", "in", "is", "are", "with",
        "for", "on", "may", "be", "was", "were", "patients", "patient",
        "associated", "common", "typically", "usually", "such", "as",
    }
    claim_words = [
        w for w in _normalized(claim_window).split() if w not in stop
    ]
    if not claim_words:
        return False
    passage_norm = _normalized(passage)
    return sum(w in passage_norm for w in claim_words) >= min_hits


def is_supported(claim: str, evidence: list[Evidence]) -> bool:
    """A claim is grounded if some window of it is covered by some passage."""
    return any(
        _covers(w, ev.text) for ev in evidence for w in _windows(claim)
    )


def extract_citations(claim: str) -> tuple[str, set[str]]:
    """Strip [Ex] style citation tags from a claim; return clean text + ids."""
    ids = set(re.findall(r"\[(E\d+)\]", claim))
    clean = re.sub(r"\s*\[(E\d+)\]", "", claim).strip(" ,;")
    return clean, ids


def unsupported_ratio(text: str, evidence: list[Evidence]) -> float:
    """Fraction of generated content sentences with no supporting passage.

    Counted line-by-line so that an unsupported sentence following a
    structural header line (e.g. 'EVIDENCE REFERENCES:') is not merged
    into the header and silently skipped.
    """
    supported_flags: list[bool] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _HEADER_RE.match(stripped) and stripped.endswith(":"):
            continue  # structural header line
        for sentence in re.split(r"(?<=[.!?])\s+", stripped):
            content = sentence.strip()
            if not content or len(content.split()) < 6:
                continue
            supported_flags.append(is_supported(content, evidence))
    if not supported_flags:
        return 0.0
    return 1 - sum(supported_flags) / len(supported_flags)


def strip_unsupported(
    report_text: str, evidence: list[Evidence], attribution: str = "suppressed: "
) -> str:
    """Return the report with ungrounded sentences demoted to suppressed
    notes, keeping the traceability promise transparent to the clinician."""
    lines = report_text.strip().splitlines()
    out = []
    for line in lines:
        s = line.strip()
        if not s or _HEADER_RE.match(s):
            out.append(line if line.strip() else line)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", s)
        kept = []
        for sent in sentences:
            _, _ = extract_citations(sent)  # normalize
            if sent.strip() and (len(sent.split()) < 6 or is_supported(sent, evidence)):
                kept.append(sent.strip())
            else:
                kept.append(f"({attribution}{sent.strip()})")
        out.append(" ".join(kept))
    return "\n".join(out)


def build_references(evidence: list[Evidence]) -> str:
    """Format the EVIDENCE REFERENCES footer mapping [Ex] -> provenance."""
    lines = []
    for i, ev in enumerate(evidence, start=1):
        provenance = ev.title
        if ev.chunk_id.startswith("pmid:"):
            provenance += f" (PMID {ev.pmid or ''})".replace(" ()", "")
        elif ev.source_type == "seed_reference":
            provenance += " (curated seed reference)"
        lines.append(f"[E{i}] {provenance}")
    return "\n".join(lines)


def finalize_report(raw_text: str, evidence: list[Evidence]) -> tuple[str, float]:
    """Post-process a generated report: ground each sentence, add references.
    Returns (final_report, unsupported_ratio)."""
    grounded = strip_unsupported(raw_text, evidence)
    ratio = unsupported_ratio(raw_text, evidence)
    footer = build_references(evidence)
    if "EVIDENCE REFERENCES" in grounded.upper():
        grounded = re.sub(
            r"EVIDENCE REFERENCES[:\s]*$[\s\S]*",
            footer,
            grounded,
            flags=re.IGNORECASE,
        )
    else:
        grounded = f"{grounded}\n\nEVIDENCE REFERENCES:\n{footer}"
    return grounded, ratio
