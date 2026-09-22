"""Structural and semantic validation (§8.9)."""

from __future__ import annotations

import pytest

import studybed as sb
from llmbic import FieldValue, ValueStatus, validate_artifacts
from llmbic.provenance import FieldArtifact, FieldProvenance, Outcome
from llmbic.schema.normalized import BaseType, Constraints, FieldDefinition
from llmbic.validation import (
    ValidationContext,
    run_semantic_validators,
    validate_field_value,
    worst,
)


def _artifact(field_id, value, entity="", evidence=()):
    return FieldArtifact(
        record_id="r",
        field_id=field_id,
        entity=entity,
        value=value,
        evidence=tuple(evidence),
        provenance=FieldProvenance(schema_version="study@1.0"),
    )


# ---- types and cardinality ----------------------------------------------

def test_a_wrong_type_fails():
    fdef = FieldDefinition("n", "n", base_type=BaseType.INTEGER)
    results = validate_field_value(FieldValue.present("twelve"), fdef)
    assert worst(results) is Outcome.FAIL
    assert "expects integer" in results[0].reason


def test_a_boolean_is_not_an_integer():
    fdef = FieldDefinition("n", "n", base_type=BaseType.INTEGER)
    assert worst(validate_field_value(FieldValue.present(True), fdef)) is Outcome.FAIL


def test_a_scalar_in_a_multivalued_slot_fails():
    fdef = FieldDefinition("t", "t", multivalued=True)
    assert worst(validate_field_value(FieldValue.present("a"), fdef)) is Outcome.FAIL


def test_numeric_bounds_are_checked():
    fdef = FieldDefinition("n", "n", base_type=BaseType.INTEGER, constraints=Constraints(minimum=1))
    assert worst(validate_field_value(FieldValue.present(0), fdef)) is Outcome.FAIL
    assert worst(validate_field_value(FieldValue.present(1), fdef)) is Outcome.PASS


def test_a_closed_vocabulary_rejects_an_unknown_value():
    fdef = FieldDefinition("k", "k", constraints=Constraints(enum=("a", "b")))
    assert worst(validate_field_value(FieldValue.present("c"), fdef)) is Outcome.FAIL


def test_an_open_vocabulary_accepts_the_sources_own_wording():
    fdef = FieldDefinition(
        "k", "k", constraints=Constraints(enum=("a", "b"), open_vocabulary=True)
    )
    assert worst(validate_field_value(FieldValue.present("c"), fdef)) is Outcome.PASS


# ---- requiredness is about status, not about None -----------------------

def test_a_required_field_answered_not_reported_is_valid():
    """A studied absence is an answer; an unrun extraction is not."""

    fdef = FieldDefinition("f", "f", required=True)
    reported = validate_field_value(FieldValue.absent(ValueStatus.NOT_REPORTED), fdef)
    assert worst(reported) is Outcome.PASS

    unrun = validate_field_value(FieldValue(ValueStatus.NOT_EXTRACTED), fdef)
    assert worst(unrun) is Outcome.FAIL


def test_a_required_field_that_failed_extraction_is_invalid():
    fdef = FieldDefinition("f", "f", required=True)
    assert worst(validate_field_value(FieldValue.failed("timeout"), fdef)) is Outcome.FAIL


# ---- whole-record validation --------------------------------------------

def test_a_field_the_schema_does_not_define_is_reported():
    schema = sb.normalized("1.0")
    results = validate_artifacts([_artifact("not_a_field", FieldValue.present(1))], schema)
    assert any("does not define" in r.reason for r in results)


def test_a_missing_required_field_is_reported():
    schema = sb.normalized("1.0")
    results = validate_artifacts([_artifact("title", FieldValue.present("t"))], schema)
    assert any(r.details.get("field_id") == "local_id" for r in results)


def test_a_valid_record_produces_no_failures():
    schema = sb.normalized("1.0")
    paper = sb.make_paper(0)
    from llmbic import ExtractedValueCodec, decompose

    d = decompose(
        paper.record, schema, record_id=paper.record_id, codec=ExtractedValueCodec()
    )
    results = validate_artifacts(d.artifacts, schema)
    assert [r for r in results if r.outcome is Outcome.FAIL] == []


# ---- semantic validators -------------------------------------------------

def _ctx(value, evidence=(), fdef=None, parsed=None):
    return ValidationContext(
        record_id="r",
        field_id="f",
        entity="",
        value=value,
        field_def=fdef or FieldDefinition("f", "f"),
        evidence=tuple(evidence),
        parsed=parsed,
    )


def test_require_evidence_escalates_an_unsupported_value():
    results = run_semantic_validators(["require_evidence"], _ctx(FieldValue.present("x")))
    assert results[0].outcome is Outcome.REVIEW


def test_require_evidence_passes_an_abstention():
    results = run_semantic_validators(
        ["require_evidence"], _ctx(FieldValue.absent(ValueStatus.NOT_REPORTED))
    )
    assert results[0].outcome is Outcome.PASS


def test_evidence_supports_value_catches_a_quote_that_does_not_say_so():
    from llmbic.source import EvidenceReference, EvidenceSpan

    ref = EvidenceReference(
        source_id="s",
        source_version="v",
        parse_version="p",
        spans=(EvidenceSpan("u1", 0, 10, "unrelated"),),
    )
    results = run_semantic_validators(
        ["evidence_supports_value"], _ctx(FieldValue.present("visual"), [ref])
    )
    assert results[0].outcome is Outcome.REVIEW

    ok = EvidenceReference(
        source_id="s",
        source_version="v",
        parse_version="p",
        spans=(EvidenceSpan("u1", 0, 10, "the visual task"),),
    )
    results = run_semantic_validators(
        ["evidence_supports_value"], _ctx(FieldValue.present("visual"), [ok])
    )
    assert results[0].outcome is Outcome.PASS


def test_abstention_is_honest_flags_an_absence_that_cites_spans():
    from llmbic.source import EvidenceReference, EvidenceSpan

    ref = EvidenceReference(
        source_id="s",
        source_version="v",
        parse_version="p",
        spans=(EvidenceSpan("u1", 0, 5, "hello"),),
    )
    results = run_semantic_validators(
        ["abstention_is_honest"],
        _ctx(FieldValue.absent(ValueStatus.NOT_REPORTED), [ref]),
    )
    assert results[0].outcome is Outcome.REVIEW


def test_a_validator_may_return_a_bare_bool_or_string():
    from llmbic.functions import VALIDATORS

    @VALIDATORS.register("bool_validator", "1")
    def _b(ctx):
        return False

    @VALIDATORS.register("string_validator", "1")
    def _s(ctx):
        return "that is not right"

    results = run_semantic_validators(
        ["bool_validator@1", "string_validator@1"], _ctx(FieldValue.present(1))
    )
    assert [r.outcome for r in results] == [Outcome.FAIL, Outcome.FAIL]
    assert results[1].reason == "that is not right"


def test_a_registered_function_cannot_be_rebound_to_different_code():
    from llmbic.errors import LlmbicError
    from llmbic.functions import FunctionRegistry

    reg = FunctionRegistry("thing")

    def one(x):
        return x

    def two(x):
        return x + 1

    reg.add("t", one, version="1")
    reg.add("t", one, version="1")  # idempotent
    with pytest.raises(LlmbicError, match="already registered with different code"):
        reg.add("t", two, version="1")


def test_a_failing_validator_stops_a_value_from_being_committed(loaded):
    """FR-LLM-008: outputs are validated before they count as a result."""

    import dataclasses

    from llmbic.models.base import AdapterRegistry
    from llmbic.models.mock import ScriptedAdapter

    recipe = sb.stimulus_recipe("1")
    strict = dataclasses.replace(recipe, validators=("in_vocabulary",))
    loaded.registry._recipes[recipe.ref] = strict

    adapters = AdapterRegistry()
    adapters.register(
        "main",
        ScriptedAdapter(
            default={
                "tasks[].stimulus_modality": {"status": "present", "value": ["visual"]}
            }
        ),
    )
    adapters.register("backup", adapters.get("main"))
    loaded.adapters = adapters
    plan, result = loaded.migrate("study@1.1")
    # `visual` is permissible, so this run succeeds...
    assert result.metrics.model_calls > 0

    for entry in loaded.records():
        for a in loaded.artifacts(entry.record_id):
            if a.field_id == "tasks[].stimulus_modality" and a.value.status.has_value:
                assert a.value.value == ["visual"]
                assert any(v.validator == "in_vocabulary" for v in a.provenance.validations)
                return
    pytest.fail("no validated artifact was written")
