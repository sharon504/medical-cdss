"""Reasoning triplets — the Extractor-Respondent evidence bridge (GT-MedRAG).

A triplet is (condition; relation; conclusion) — a minimal medical reasoning
unit distilled from one evidence snippet (ER-MedRAG, Sec. 3.2). This module

  1. parses GOLD triplets from our structured seed references (for SFT/DPO
     training pairs without any large reward model),
  2. provides a rule-based baseline extractor (deterministic, offline),
  3. builds the extractor prompt and the respondent evidence block (the
     "concatenated reasoning triplet" of the paper), and
  4. carries per-triplet provenance [E#] so citations stay claim-level.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..retrieval.base import Evidence


@dataclass
class Triplet:
    condition: str
    relation: str
    conclusion: str
    evidence_id: str          # e.g. "E3" — provenance for claim-level citations
    source_chunk: str = ""    # chunk_id of the source passage

    def render(self, idx: int) -> str:
        return (f"({self.condition}; {self.relation}; {self.conclusion})"
                f" [{self.evidence_id}]") if idx == 0 else \
               f"({self.condition}; {self.relation}; {self.conclusion}) [{self.evidence_id}]"

    def as_dict(self) -> dict:
        return {
            "condition": self.condition,
            "relation": self.relation,
            "conclusion": self.conclusion,
            "evidence_id": self.evidence_id,
            "source_chunk": self.source_chunk,
        }


# ---------------------------------------------------------------------------
# Gold extraction from structured seed references
# ---------------------------------------------------------------------------

_SENT_SPLIT = re.compile(r"(?<=[.;])\s+")

# one consolidated gold-sentence pattern with named groups; capture classes
# exclude sentence-breaks so spans stay within one sentence
_GOLD_RE = re.compile(
    r"(?P<condition>[^.;]{12,160}?)\s+"
    r"(?P<relation>suggested by|associated with|is diagnostic of|confirms\b"
    r"|indicates|predicts|confirmed by)"
    r"\s+(?P<conclusion>[^.;]{8,110}?)[\s]*[.,;]",
    re.I,
)


def gold_triplets(
    evidence: Evidence,
    evidence_id: str,
    max_triplets: int = 3,
) -> list[Triplet]:
    """Build gold triplets from a seed-reference passage using its canonical
    'Findings / Differential / Investigations' structure.

    Deterministic and offline - used to build extractor training/DPO pairs
    without any large reward model.
    """
    out: list[Triplet] = []
    for sentence in _SENT_SPLIT.split(evidence.text):
        sentence = sentence.strip()
        if len(sentence) < 15 or len(out) >= max_triplets:
            continue
        match = _GOLD_RE.search(sentence)
        if match:
            out.append(
                Triplet(
                    condition=match.group("condition").strip(" -,"),
                    relation=match.group("relation").replace("confirms", "confirmed by"),
                    conclusion=match.group("conclusion").strip(" .,;"),
                    evidence_id=evidence_id,
                    source_chunk=evidence.chunk_id,
                )
            )
    return out[:max_triplets]


# ---------------------------------------------------------------------------
# Rule-based baseline extractor (no model needed)
# ---------------------------------------------------------------------------

_RELATION_CUES = [
    "suggests", "suggested by", "indicates", "is diagnostic of",
    "associated with", "confirms", "confirmed by", "supported by",
    "predicts", "excludes", "unlikely",
]


def rule_based_extract(evidence: Evidence, evidence_id: str) -> list[Triplet]:
    """Baseline 'extractor agent': deterministic cue-based rules.
    The GPT-2 extractor (notebook 03) replaces this after DPO training."""
    triplets: list[Triplet] = []
    ruler = re.compile(
        r"([^.;]{10,160}?)\s+(?P<cue>"
        + "|".join(re.escape(c) for c in _RELATION_CUES)
        + r")\s+([^.;]{8,120}?)\s*[.;]",
        re.I,
    )
    for match in ruler.finditer(evidence.text):
        triplets.append(
            Triplet(
                condition=match.group(1).strip(" ,"),
                relation=match.group("cue").lower().replace(
                    "confirms", "confirmed by"),
                conclusion=match.group(3).strip(" .,"),
                evidence_id=evidence_id,
                source_chunk=evidence.chunk_id,
            )
        )
        if len(triplets) >= 3:
            break
    return triplets


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

EXTRACTOR_PROMPT = """You are a medical reasoning assistant.
Given a medical snippet, extract a structured reasoning triplet in the format: (condition, relation, conclusion)
Definitions:
- Condition: key clinical findings or context
- Relation: medical reasoning or logical connection
- Conclusion: inferred diagnosis or judgment

Snippet: {snippet}
Output format: (condition; relation; conclusion)
"""


def build_extractor_prompt(snippet: str) -> str:
    return EXTRACTOR_PROMPT.format(snippet=snippet)


def parse_triplet_output(text: str, evidence_id: str = "E?") -> Triplet | None:
    """Parse '(condition; relation; conclusion)' from generated text."""
    match = re.search(r"\(([^;()]+);([^;()]+);([^;()]+)\)", text)
    if not match:
        return None
    return Triplet(
        condition=match.group(1).strip(),
        relation=match.group(2).strip(),
        conclusion=match.group(3).strip(),
        evidence_id=evidence_id,
    )


def concatenated_block(triplets: list[Triplet]) -> str:
    """The respondent's evidence context: numbered triplets + provenance.
    Replaces raw snippets in the report prompt (paper's 'M(.)' concat)."""
    lines = [f"{t.render(i + 1)}" for i, t in enumerate(triplets)]
    return "\n".join(f"T{i+1}: {line}" for i, line in enumerate(lines))


def evidence_block_with_links(triplets: list[Triplet]) -> str:
    """Triplets rendered together with the raw snippet titles they cite -
    keeps traceability while shrinking the context window."""
    lines = []
    for t in triplets:
        lines.append(f"[{t.evidence_id}] ({t.render(0)})")
    return "\n".join(lines)
