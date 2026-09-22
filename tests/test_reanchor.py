"""Evidence re-anchoring across a parser change (FR-PROV-008)."""

from __future__ import annotations

import dataclasses


from llmbic import reanchor_artifacts, reanchor_evidence
from llmbic.source import (
    DocumentUnit,
    EvidenceReference,
    EvidenceSpan,
    ParsedSource,
    UnitKind,
    find_span,
)


def _parsed(*texts: str, parse_version: str = "parse@1") -> ParsedSource:
    units = []
    offset = 0
    for i, text in enumerate(texts):
        units.append(
            DocumentUnit(f"u{i}", UnitKind.SENTENCE, text, "methods", offset, i)
        )
        offset += len(text) + 1
    return ParsedSource("s", "v1", parse_version, tuple(units))


def _ref(unit_id: str, start: int, end: int, text: str, parse_version="parse@1"):
    return EvidenceReference(
        source_id="s",
        source_version="v1",
        parse_version=parse_version,
        spans=(EvidenceSpan(unit_id, start, end, text),),
        locator="model_quote",
    )


def test_a_span_that_still_matches_is_left_alone():
    parsed = _parsed("Stimuli were photographs.", "Responses were vocal.")
    result = reanchor_evidence([_ref("u0", 0, 25, "Stimuli were photographs.")], parsed)
    assert result.counts == {"unchanged": 1}
    assert result.fully_anchored


def test_a_span_that_moved_is_found_once_and_re_anchored():
    new = _parsed(
        "Participants consented.", "Stimuli were photographs.", "Responses were vocal."
    )
    ref = _ref("u0", 0, 25, "Stimuli were photographs.")
    result = reanchor_evidence([ref], new)
    assert result.counts == {"reanchored": 1}
    moved = result.evidence[0].spans[0]
    assert moved.unit_id == "u1"
    assert new.unit(moved.unit_id).text[moved.start_char : moved.end_char] == moved.text


def test_a_span_occurring_twice_is_ambiguous_not_guessed():
    # The original unit id is gone and the text now appears twice, so there is
    # no one place to move the span to.
    new = _parsed("Stimuli were photographs.", "Stimuli were photographs.")
    new = dataclasses.replace(
        new,
        units=tuple(
            dataclasses.replace(u, unit_id=f"v2:{i}") for i, u in enumerate(new.units)
        ),
    )
    result = reanchor_evidence([_ref("u0", 0, 25, "Stimuli were photographs.")], new)
    assert result.counts == {"ambiguous": 1}
    assert not result.fully_anchored
    # The claim is dropped rather than relocated, and the loss is counted.
    assert result.evidence[0].spans == ()
    assert result.evidence[0].unlocated_quotes == 1


def test_a_span_that_disappeared_is_lost_and_counted():
    new = _parsed("Nothing like the original.")
    result = reanchor_evidence([_ref("u0", 0, 25, "Stimuli were photographs.")], new)
    assert result.counts == {"lost": 1}
    assert result.evidence[0].unlocated_quotes == 1


def test_a_span_with_no_recorded_text_cannot_be_verified():
    new = _parsed("Anything at all.")
    result = reanchor_evidence([_ref("u0", 0, 5, "")], new)
    assert result.counts == {"unverifiable": 1}


def test_the_new_parse_version_is_stamped_on_the_reference():
    new = _parsed("Stimuli were photographs.", parse_version="parse@2")
    result = reanchor_evidence([_ref("u0", 0, 25, "Stimuli were photographs.")], new)
    assert result.evidence[0].parse_version == "parse@2"


def test_artifacts_are_rewritten_only_where_something_moved(loaded, papers):
    paper = papers[0]
    artifacts = loaded.artifacts(paper.record_id)

    # A re-parse that prepends a sentence shifts every unit id by one.
    shifted = dataclasses.replace(
        paper.parsed,
        parse_version="parse@2",
        units=tuple(
            [
                DocumentUnit("pre:0", UnitKind.SENTENCE, "A new leading sentence.", "title", 0, 0)
            ]
            + [
                dataclasses.replace(u, unit_id=f"shift:{i + 1}", ordinal=i + 1)
                for i, u in enumerate(paper.parsed.units)
            ]
        ),
    )
    changed, totals = reanchor_artifacts(artifacts, shifted)
    assert changed
    assert totals.get("reanchored", 0) > 0
    assert all(a.derived_from for a in changed)
    assert all(a.provenance.parse_version == "parse@2" for a in changed)
    assert all("reanchored_from" in a.provenance.notes for a in changed)

    # And the originals are untouched in the store.
    for artifact in changed:
        original = loaded.store.get_artifact(artifact.derived_from)
        assert original.provenance.parse_version != "parse@2"


def test_the_old_parse_is_retained_beside_the_new_one(loaded, papers):
    """The requirement is to keep the old version, not to overwrite it."""

    paper = papers[0]
    loaded.store.put_parsed(dataclasses.replace(paper.parsed, parse_version="parse@2"))
    assert loaded.store.list_parse_versions(paper.source.source_id, "v1") == [
        "parse@1",
        "parse@2",
    ]
    assert (
        loaded.store.get_parsed(paper.source.source_id, "v1", "parse@1") is not None
    )


def test_is_anchored_reports_a_stale_parse_version():
    parsed = _parsed("Stimuli were photographs.", parse_version="parse@2")
    ref = _ref("u0", 0, 25, "Stimuli were photographs.", parse_version="parse@1")
    assert not ref.is_anchored(parsed)
    assert dataclasses.replace(ref, parse_version="parse@2").is_anchored(parsed)


def test_find_span_refuses_to_locate_an_ambiguous_quote():
    parsed = _parsed("the same phrase", "the same phrase")
    assert find_span(parsed, "the same phrase") is None
    unique = _parsed("only here once", "something else")
    assert find_span(unique, "only here once") is not None
