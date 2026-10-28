"""Conflict-aware evidence handling — a GT-MedRAG extension of ER-MedRAG.

The paper reports degradation under noisy or contradictory retrieval but
provides no mechanism for it. Here, when the fine-grained snippet set
contains triplets that disagree (same condition element, divergent or
negated conclusions), we

  1. merge them into a single flagged conflict triplet, which downstream
     reporting renders in a CONFLICTS / UNCERTAINTY section,
  2. lower the aggregate confidence of the affected condition, and
  3. emit an escalate-to-clinician abstention note when conflict pressure
     is high (no automatic diagnosis before human review).

Conflict signals are deterministic lexical cues (negation, antonymy,
adjacent opposing thresholds) - deliberately conservative so clinicians
see fewer false alarms.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .triplets import Triplet

_NEGATION_RE = re.compile(
    r"\b(not|no|un(?:likely|common)|absent|without|rules? out|excluding|"
    r"rare\b|rarely|negative|improbable|unlikely|de-?[r]?creas)[a-z]*\b", re.I)

_FREQUENCY_RE = re.compile(
    r"\b(rare|common|often|frequent|typical|atypical|most|unusual)\b", re.I)


@dataclass
class ConflictFlag:
    condition_key: str      # shared normalized condition element
    member_evidences: list[str]
    reason: str             # "negation" | "contradictory-conclusions"
    confidence_penalty: float


def _norm_key(text: str) -> str:
    """Very loose condition-key normalizer: content words > 4 chars."""
    stop = {"the", "this", "that", "with", "from", "and", "or", "for", "a",
            "an", "of", "in", "is", "are", "be"}
    words = re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()
    return " ".join(w for w in words if w not in stop and len(w) > 4)[:60]


def _shares_condition(a: Triplet, b: Triplet) -> bool:
    ka, kb = _norm_key(a.condition), _norm_key(b.condition)
    if not ka or not kb:
        return False
    words_a, words_b = set(ka.split()), set(kb.split())
    overlap = words_a & words_b
    # clinically meaningful entity overlap (>=2 content word matches or a
    # single long token match)
    return len(overlap) >= 2 or any(len(w) >= 9 for w in overlap)


def _negation_signature(t: Triplet) -> bool:
    return bool(_NEGATION_RE.search(t.conclusion) or _NEGATION_RE.search(t.relation))


def _frequency_signature(t: Triplet) -> str | None:
    match = _FREQUENCY_RE.search(t.conclusion)
    return match.group(1).lower() if match else None


def detect_conflicts(triplets: list[Triplet]) -> list[ConflictFlag]:
    """Flag conflicting triplet pairs. Returns list of per-condition flags."""
    flags: list[ConflictFlag] = []
    seen_pairs: set[tuple[int, int]] = set()

    for i in range(len(triplets)):
        for j in range(i + 1, len(triplets)):
            a, b = triplets[i], triplets[j]
            if a.evidence_id == b.evidence_id:
                continue  # same source passage: paraphrase, not conflict
            if not _shares_condition(a, b):
                continue
            key = (i, j)
            # signal 1 (most specific): opposite frequency/degree hedging
            fa, fb = _frequency_signature(a), _frequency_signature(b)
            if fa and fb and {fa, fb} in (
                {"rare", "common"}, {"unusual", "typical"}, {"rare", "often"}
            ):
                flags.append(
                    ConflictFlag(
                        condition_key=_norm_key(a.condition),
                        member_evidences=[a.evidence_id, b.evidence_id],
                        reason="contradictory-conclusions",
                        confidence_penalty=0.25,
                    )
                )
                seen_pairs.add(key)
                continue
            # signal 2: one affirmative conclusion, one negated conclusion
            if _negation_signature(a) != _negation_signature(b):
                flags.append(
                    ConflictFlag(
                        condition_key=_norm_key(a.condition) or _norm_key(b.condition),
                        member_evidences=[a.evidence_id, b.evidence_id],
                        reason="negation",
                        confidence_penalty=0.15,
                    )
                )
                seen_pairs.add(key)

    # deduplicate flags sharing the same key and evidence members
    unique: dict[tuple, ConflictFlag] = {}
    for flag in flags:
        k = (flag.condition_key, tuple(sorted(flag.member_evidences)))
        prev = unique.get(k)
        if prev and prev.confidence_penalty >= flag.confidence_penalty:
            continue
        unique[k] = flag
    return list(unique.values())


def aggregate_confidence(
    triplets: list[Triplet], base: float = 1.0
) -> float:
    """System confidence after subtracting conflict penalties."""
    penalty = sum(f.confidence_penalty for f in detect_conflicts(triplets))
    return max(0.0, base - penalty)


def conflict_triplet(flag: ConflictFlag) -> Triplet:
    """Synthesize a flagged triplet that downstream rendering can style."""
    return Triplet(
        condition=flag.condition_key,
        relation="CONFLICT - sources disagree:"
                 + ("negated vs affirmed conclusion" if flag.reason == "negation"
                    else "contradictory hedging on frequency/degree"),
        conclusion="manual review required before any conclusion is drawn "
                   f"(evidence {', '.join(flag.member_evidences)})",
        evidence_id=flag.member_evidences[0],
    )


def system_conflicts_report(triplets: list[Triplet]) -> tuple[str | None, float]:
    """Render the CONFLICTS / UNCERTAINTY section (or None when no conflict)
    and return the aggregate confidence.

    Escalation: SYSTEM NOTE is emitted when confidence is low (< 0.6) OR
    when conflict pressure is high (>= 2 independent flagged pairs) -
    moderate conflicts rarely sum below the aggregate threshold alone.
    """
    flags = detect_conflicts(triplets)
    confidence = aggregate_confidence(triplets)
    if not flags:
        return None, confidence
    lines = [f"{conflict_triplet(f).condition} | "
             f"{conflict_triplet(f).relation} | "
             f"{conflict_triplet(f).conclusion}"
             for f in flags]
    if confidence < 0.6 or len(flags) >= 2:
        lines.append(
            "SYSTEM NOTE: conflict pressure high - output withheld as "
            "decision support; escalate to clinician for adjudication."
        )
    return "CONFLICTS / UNCERTAINTY:\n" + "\n".join(lines), confidence
