"""Decomposition, assembly and value status (§8.6, FR-PROV-009)."""

from __future__ import annotations

import pytest

import studybed as sb
from llmbic import ExtractedValueCodec, ValueStatus, assemble, decompose
from llmbic.records import EMPTY, collection_parent
from llmbic.schema.adapters import from_json_schema

SIMPLE = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "design": {"type": "object", "properties": {"kind": {"type": "string"}}},
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "local_id": {"type": "string"},
                    "n": {"type": "integer"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "assessments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "local_id": {"type": "string"},
                                "name": {"type": "string"},
                            },
                        },
                    },
                },
            },
        },
    },
}


@pytest.fixture
def simple_schema():
    return from_json_schema(SIMPLE, name="simple", version="1")


def test_round_trip_preserves_the_record(simple_schema):
    record = {
        "title": "T",
        "design": {"kind": "rct"},
        "groups": [
            {
                "local_id": "g1",
                "n": 10,
                "tags": ["a", "b"],
                "assessments": [{"local_id": "a1", "name": "BDI"}],
            },
            {"local_id": "g2", "n": 20, "tags": [], "assessments": []},
        ],
    }
    d = decompose(record, simple_schema, record_id="r1")
    assert assemble(d.artifacts, d.entities, simple_schema) == record


def test_an_empty_collection_is_not_an_absent_one(simple_schema):
    record = {"title": "T", "groups": [{"local_id": "g1", "assessments": []}]}
    d = decompose(record, simple_schema, record_id="r1")
    assert any(e.local_id == EMPTY for e in d.entities)
    assert assemble(d.artifacts, d.entities, simple_schema)["groups"][0]["assessments"] == []


def test_entity_keys_carry_the_collection_and_the_local_id(simple_schema):
    record = {"groups": [{"local_id": "g1", "assessments": [{"local_id": "a1"}]}]}
    d = decompose(record, simple_schema, record_id="r1")
    keys = {e.entity for e in d.entities}
    assert "groups[]=g1" in keys
    assert "groups[]=g1/groups[].assessments[]=a1" in keys


def test_reordering_a_list_does_not_change_entity_keys(simple_schema):
    a = {"groups": [{"local_id": "g1", "n": 1}, {"local_id": "g2", "n": 2}]}
    b = {"groups": [{"local_id": "g2", "n": 2}, {"local_id": "g1", "n": 1}]}
    da = decompose(a, simple_schema, record_id="r1")
    db = decompose(b, simple_schema, record_id="r1")
    assert {(x.field_id, x.entity, x.value.value) for x in da.artifacts} == {
        (x.field_id, x.entity, x.value.value) for x in db.artifacts
    }


def test_members_without_an_identity_fall_back_to_position():
    schema = from_json_schema(
        {
            "type": "object",
            "properties": {
                "rows": {
                    "type": "array",
                    "items": {"type": "object", "properties": {"v": {"type": "integer"}}},
                }
            },
        },
        name="t",
        version="1",
    )
    d = decompose({"rows": [{"v": 1}, {"v": 2}]}, schema, record_id="r")
    assert {e.local_id for e in d.entities} == {"#0", "#1"}


def test_collection_parent():
    assert collection_parent("groups[]") == ""
    assert collection_parent("groups[].assessments[]") == "groups[]"
    assert collection_parent("design.arms[]") == ""


# ---- value status (FR-PROV-009) -----------------------------------------

def test_plain_codec_distinguishes_absent_from_null(simple_schema):
    d = decompose({"title": None}, simple_schema, record_id="r")
    by_field = {a.field_id: a for a in d.artifacts}
    assert by_field["title"].value.status is ValueStatus.NOT_REPORTED
    assert by_field["design.kind"].value.status is ValueStatus.NOT_EXTRACTED


def test_extracted_value_codec_reads_study_schemas_wrapper():
    schema = sb.normalized("1.0")
    paper = sb.make_paper(1)
    d = decompose(
        paper.record, schema, record_id=paper.record_id, codec=ExtractedValueCodec()
    )
    stimuli = next(a for a in d.artifacts if a.field_id == "tasks[].stimuli")
    assert stimuli.value.status is ValueStatus.PRESENT
    assert stimuli.value.annotations["value_source"] == "reported"
    assert stimuli.evidence and stimuli.evidence[0].spans


def test_undetermined_is_unknown_not_not_reported():
    """The one UnreportedReason that reports on the pass, not on the source."""

    schema = sb.normalized("1.0")
    record = {
        "local_id": "r",
        "title": {
            "extraction_status": "not_reported",
            "unreported_reason": "undetermined",
            "evidence": {"status": "not_applicable"},
        },
    }
    d = decompose(record, schema, record_id="r", codec=ExtractedValueCodec())
    title = next(a for a in d.artifacts if a.field_id == "title")
    assert title.value.status is ValueStatus.UNKNOWN
    assert title.value.reason == "undetermined"


def test_ambiguous_stays_not_reported_with_its_reason():
    schema = sb.normalized("1.0")
    record = {
        "local_id": "r",
        "title": {
            "extraction_status": "not_reported",
            "unreported_reason": "ambiguous",
            "evidence": {"status": "not_applicable"},
        },
    }
    d = decompose(record, schema, record_id="r", codec=ExtractedValueCodec())
    title = next(a for a in d.artifacts if a.field_id == "title")
    assert title.value.status is ValueStatus.NOT_REPORTED
    assert title.value.reason == "ambiguous"


def test_unlocated_quotes_survive_the_round_trip():
    """A fidelity failure must stay distinguishable from a recall failure."""

    schema = sb.normalized("1.0")
    record = {
        "local_id": "r",
        "title": {
            "extraction_status": "extracted",
            "value": "A title",
            "evidence": {"status": "not_found", "unlocated_quotes": 3},
        },
    }
    codec = ExtractedValueCodec()
    d = decompose(record, schema, record_id="r", codec=codec)
    title = next(a for a in d.artifacts if a.field_id == "title")
    assert title.evidence[0].unlocated_quotes == 3
    out = assemble(d.artifacts, d.entities, schema, codec=codec)
    assert out["title"]["evidence"]["unlocated_quotes"] == 3


@pytest.mark.parametrize(
    "status",
    [
        ValueStatus.NOT_EXTRACTED,
        ValueStatus.EXTRACTION_FAILED,
        ValueStatus.UNKNOWN,
        ValueStatus.REVIEW_REQUIRED,
    ],
)
def test_statuses_the_wrapper_cannot_express_are_kept_beside_it(status):
    from llmbic import FieldValue
    from llmbic.schema.normalized import FieldDefinition

    codec = ExtractedValueCodec()
    fdef = FieldDefinition("f", "f", evidence_bearing=True)
    value = FieldValue(status, "x" if status.has_value else None)
    encoded = codec.encode(value, (), fdef)
    assert encoded["llmbic_value_status"] == status.value


def test_a_present_value_of_none_is_not_an_absence():
    from llmbic import FieldValue

    present = FieldValue.present(None)
    absent = FieldValue.absent(ValueStatus.NOT_REPORTED)
    assert present.to_canonical() != absent.to_canonical()
    assert present.status.has_value and not absent.status.has_value


def test_constructing_an_absent_value_with_a_value_bearing_status_is_refused():
    from llmbic import FieldValue

    with pytest.raises(ValueError):
        FieldValue.absent(ValueStatus.PRESENT)


def test_include_absent_writes_every_declared_slot(simple_schema):
    d = decompose({"title": "T"}, simple_schema, record_id="r")
    dense = assemble(d.artifacts, d.entities, simple_schema, include_absent=True)
    assert "kind" in dense["design"]
    sparse = assemble(d.artifacts, d.entities, simple_schema, include_absent=False)
    assert "design" not in sparse
