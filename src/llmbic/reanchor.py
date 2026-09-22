"""Re-anchoring evidence across a parser change (FR-PROV-008).

When the parser changes, unit ids and offsets move and every stored span
points at the wrong place.  The rule the requirements set is: keep the old
parsed version, and treat re-anchoring as a *separate migration* rather than
as a silent fix-up — because a span that cannot be re-found is a fact worth
seeing, not an error to swallow.

The strategy here is deliberately conservative.  A span is re-anchored only
when its recorded text occurs exactly once in the new parse; anything else is
reported as ambiguous or lost, and the value keeps its old evidence plus a note
saying so.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Iterable, Sequence

from .provenance import FieldArtifact
from .source import DocumentUnit, EvidenceReference, EvidenceSpan, ParsedSource


class AnchorOutcome(str, Enum):
    #: The unit id still exists and the text still matches: nothing to do.
    UNCHANGED = "unchanged"
    #: Found exactly once in the new parse, at a new location.
    REANCHORED = "reanchored"
    #: Found more than once: llmbic will not guess which one.
    AMBIGUOUS = "ambiguous"
    #: Not found at all.
    LOST = "lost"
    #: The span recorded no text, so there is nothing to search for.
    UNVERIFIABLE = "unverifiable"


@dataclass(frozen=True)
class SpanAnchor:
    before: EvidenceSpan
    outcome: AnchorOutcome
    after: EvidenceSpan | None = None
    candidates: int = 0

    def to_canonical(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "before": self.before.to_canonical(),
            "after": self.after.to_canonical() if self.after else None,
            "candidates": self.candidates,
        }


@dataclass
class ReanchorResult:
    evidence: tuple[EvidenceReference, ...]
    anchors: tuple[SpanAnchor, ...] = ()

    @property
    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for a in self.anchors:
            out[a.outcome.value] = out.get(a.outcome.value, 0) + 1
        return dict(sorted(out.items()))

    @property
    def fully_anchored(self) -> bool:
        return all(
            a.outcome in (AnchorOutcome.UNCHANGED, AnchorOutcome.REANCHORED)
            for a in self.anchors
        )


def reanchor_span(span: EvidenceSpan, parsed: ParsedSource) -> SpanAnchor:
    if not span.text:
        return SpanAnchor(span, AnchorOutcome.UNVERIFIABLE)

    unit = parsed.unit(span.unit_id)
    if unit is not None and unit.text[span.start_char : span.end_char] == span.text:
        return SpanAnchor(span, AnchorOutcome.UNCHANGED, span)

    hits: list[tuple[DocumentUnit, int]] = []
    for candidate in parsed.units:
        start = candidate.text.find(span.text)
        while start >= 0:
            hits.append((candidate, start))
            start = candidate.text.find(span.text, start + 1)

    if len(hits) == 1:
        found, start = hits[0]
        return SpanAnchor(
            span,
            AnchorOutcome.REANCHORED,
            EvidenceSpan(found.unit_id, start, start + len(span.text), span.text),
            candidates=1,
        )
    if len(hits) > 1:
        return SpanAnchor(span, AnchorOutcome.AMBIGUOUS, None, candidates=len(hits))
    return SpanAnchor(span, AnchorOutcome.LOST, None, candidates=0)


def reanchor_evidence(
    evidence: Sequence[EvidenceReference], parsed: ParsedSource
) -> ReanchorResult:
    """Move spans onto ``parsed``, keeping the ones that cannot be moved.

    A set that loses a span keeps its remaining spans and the ``unlocated``
    count rises, so a reader can tell "this value lost its support" from "this
    value never had any".
    """

    anchors: list[SpanAnchor] = []
    out: list[EvidenceReference] = []

    for ref in evidence:
        moved: list[EvidenceSpan] = []
        unlocated = ref.unlocated_quotes
        for span in ref.spans:
            anchor = reanchor_span(span, parsed)
            anchors.append(anchor)
            if anchor.after is not None:
                moved.append(anchor.after)
            else:
                unlocated += 1
        out.append(
            replace(
                ref,
                parse_version=parsed.parse_version,
                spans=tuple(moved),
                unlocated_quotes=unlocated,
            )
        )

    return ReanchorResult(tuple(out), tuple(anchors))


def reanchor_artifacts(
    artifacts: Iterable[FieldArtifact], parsed: ParsedSource
) -> tuple[list[FieldArtifact], dict[str, int]]:
    """Return re-anchored copies and a summary of what happened.

    Only artifacts whose evidence actually moved are returned, so a caller can
    write exactly those and leave the rest alone.
    """

    changed: list[FieldArtifact] = []
    totals: dict[str, int] = {}
    for artifact in artifacts:
        if not artifact.evidence:
            continue
        result = reanchor_evidence(artifact.evidence, parsed)
        for key, count in result.counts.items():
            totals[key] = totals.get(key, 0) + count
        if result.evidence == artifact.evidence:
            continue
        changed.append(
            replace(
                artifact,
                evidence=result.evidence,
                provenance=replace(
                    artifact.provenance,
                    parse_version=parsed.parse_version,
                    notes={
                        **artifact.provenance.notes,
                        "reanchored_from": artifact.evidence[0].parse_version,
                        "anchor_outcomes": result.counts,
                    },
                ),
                derived_from=artifact.artifact_id,
            )
        )
    return changed, dict(sorted(totals.items()))


__all__ = [
    "AnchorOutcome",
    "ReanchorResult",
    "SpanAnchor",
    "reanchor_artifacts",
    "reanchor_evidence",
    "reanchor_span",
]
