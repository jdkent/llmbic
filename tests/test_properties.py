"""Property-based tests (§15.2).

Six properties the requirements name explicitly, plus the canonicalisation
invariants the whole content-addressing scheme rests on.
"""

from __future__ import annotations

import json

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

import studybed as sb
from llmbic import (
    ExecutionPolicy,
    ExtractedValueCodec,
    FieldValue,
    Project,
    ValueStatus,
    assemble,
    decompose,
)
from llmbic.ids import canonical_json, content_hash
from llmbic.migration.spec import MigrationStep
from llmbic.planner.dependencies import cache_key
from llmbic.schema.normalized import Constraints, FieldDefinition

SLOW = settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)

json_scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-10**6, max_value=10**6),
    st.floats(allow_nan=False, allow_infinity=False, width=32),
    st.text(max_size=40),
)
json_values = st.recursive(
    json_scalars,
    lambda children: st.one_of(
        st.lists(children, max_size=4),
        st.dictionaries(st.text(min_size=1, max_size=8), children, max_size=4),
    ),
    max_leaves=12,
)


# ---- canonicalisation ----------------------------------------------------

@given(json_values)
@settings(max_examples=200, deadline=None)
def test_canonical_json_is_stable_under_key_order(value):
    if not isinstance(value, dict):
        return
    shuffled = dict(reversed(list(value.items())))
    assert canonical_json(value) == canonical_json(shuffled)
    assert content_hash(value) == content_hash(shuffled)


@given(json_values, json_values)
@settings(max_examples=200, deadline=None)
def test_different_values_hash_differently(a, b):
    assume(canonical_json(a) != canonical_json(b))
    assert content_hash(a) != content_hash(b)


@given(json_values)
@settings(max_examples=200, deadline=None)
def test_field_values_round_trip(value):
    fv = FieldValue.present(value)
    assert FieldValue.from_canonical(fv.to_canonical()) == fv


@pytest.mark.parametrize(
    "status",
    [s for s in ValueStatus],
)
def test_every_status_round_trips_distinctly(status):
    value = FieldValue(status, "x" if status.has_value else None, reason="r")
    restored = FieldValue.from_canonical(json.loads(json.dumps(value.to_canonical())))
    assert restored.status is status


# ---- cache keys (§15.2, FR-LLM-006/007) ---------------------------------

@st.composite
def dependency_maps(draw):
    keys = draw(
        st.lists(st.sampled_from(["recipe", "prompt", "source", "parse", "field_def:f"]),
                 min_size=1, max_size=5, unique=True)
    )
    return {k: draw(st.text(min_size=1, max_size=8)) for k in keys}


STEP = MigrationStep(
    id="s",
    kind=sb.StepKind.STRUCTURAL,
    writes=("f",),
    transform="derive_is_healthy@1",
)


@given(dependency_maps())
@settings(max_examples=100, deadline=None)
def test_equivalent_inputs_produce_the_same_cache_key(deps):
    a = cache_key(
        step=STEP, record_id="r", entity="", field_ids=["f"], dependencies=deps
    )
    b = cache_key(
        step=STEP,
        record_id="r",
        entity="",
        field_ids=["f"],
        dependencies=dict(reversed(list(deps.items()))),
    )
    assert a == b


@given(dependency_maps(), st.text(min_size=1, max_size=6))
@settings(max_examples=100, deadline=None)
def test_any_changed_dependency_changes_the_cache_key(deps, salt):
    base = cache_key(
        step=STEP, record_id="r", entity="", field_ids=["f"], dependencies=deps
    )
    for key in deps:
        mutated = {**deps, key: deps[key] + salt}
        assume(mutated != deps)
        assert (
            cache_key(
                step=STEP, record_id="r", entity="", field_ids=["f"], dependencies=mutated
            )
            != base
        )


@given(st.text(min_size=1, max_size=8), st.text(min_size=0, max_size=8))
@settings(max_examples=60, deadline=None)
def test_the_record_and_entity_are_part_of_the_key(record_id, entity):
    base = cache_key(
        step=STEP, record_id="r0", entity="", field_ids=["f"], dependencies={"a": "1"}
    )
    other = cache_key(
        step=STEP,
        record_id=record_id,
        entity=entity,
        field_ids=["f"],
        dependencies={"a": "1"},
    )
    assert (base == other) == (record_id == "r0" and entity == "")


# ---- schema identities ---------------------------------------------------

@given(
    st.text(min_size=1, max_size=20),
    st.text(min_size=1, max_size=20),
    st.text(max_size=60),
)
@settings(max_examples=100, deadline=None)
def test_a_rename_never_changes_the_semantic_key(old_path, new_path, description):
    a = FieldDefinition("fid", old_path, description=description)
    b = FieldDefinition("fid", new_path, description=description)
    assert a.semantic_key() == b.semantic_key()


@given(st.text(max_size=60), st.text(max_size=60))
@settings(max_examples=100, deadline=None)
def test_semantic_keys_differ_exactly_when_the_words_differ(one, two):
    a = FieldDefinition("f", "f", description=one)
    b = FieldDefinition("f", "f", description=two)
    same_words = " ".join(one.split()) == " ".join(two.split())
    assert (a.semantic_key() == b.semantic_key()) == same_words


@given(st.lists(st.text(min_size=1, max_size=6), max_size=5, unique=True))
@settings(max_examples=60, deadline=None)
def test_widening_a_vocabulary_changes_the_semantic_key(extra):
    base = FieldDefinition("f", "f", constraints=Constraints(enum=("a", "b")))
    values = tuple(["a", "b"] + [e for e in extra if e not in ("a", "b")])
    widened = FieldDefinition("f", "f", constraints=Constraints(enum=values))
    assert (base.semantic_key() == widened.semantic_key()) == (values == ("a", "b"))


# ---- record round trips --------------------------------------------------

@st.composite
def simple_records(draw):
    n_groups = draw(st.integers(min_value=0, max_value=3))
    groups = []
    for i in range(n_groups):
        groups.append(
            {
                "local_id": f"g{i}",
                "n": draw(st.integers(min_value=0, max_value=500)),
                "tags": draw(st.lists(st.text(min_size=1, max_size=6), max_size=3)),
            }
        )
    return {
        "title": draw(st.text(min_size=1, max_size=20)),
        "groups": groups,
    }


SIMPLE_SCHEMA = None


def _simple_schema():
    global SIMPLE_SCHEMA
    if SIMPLE_SCHEMA is None:
        from llmbic import from_json_schema

        SIMPLE_SCHEMA = from_json_schema(
            {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "groups": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "local_id": {"type": "string"},
                                "n": {"type": "integer"},
                                "tags": {"type": "array", "items": {"type": "string"}},
                            },
                        },
                    },
                },
            },
            name="simple",
            version="1",
        )
    return SIMPLE_SCHEMA


@given(simple_records())
@settings(max_examples=80, deadline=None)
def test_decompose_and_assemble_are_inverses(record):
    schema = _simple_schema()
    d = decompose(record, schema, record_id="r")
    assert assemble(d.artifacts, d.entities, schema) == record


@given(simple_records())
@settings(max_examples=60, deadline=None)
def test_no_required_field_is_silently_lost(record):
    schema = _simple_schema()
    d = decompose(record, schema, record_id="r")
    produced = {(a.field_id, a.entity) for a in d.artifacts}
    for group in record["groups"]:
        assert ("groups[].n", f"groups[]=g{record['groups'].index(group)}") in produced or any(
            fid == "groups[].n" for fid, _ in produced
        )
    assert ("title", "") in produced


# ---- migration properties ------------------------------------------------

@given(st.integers(min_value=1, max_value=4))
@SLOW
def test_a_deterministic_migration_is_idempotent(tmp_path_factory, n_papers):
    """Running it twice changes nothing the second time."""

    papers = sb.corpus(n_papers)
    tmp = tmp_path_factory.mktemp("idem")
    from llmbic.models.base import AdapterRegistry

    adapters = AdapterRegistry()
    adapters.register("main", sb.stimulus_adapter())
    adapters.register("backup", sb.stimulus_adapter(provider="b", model="r2"))
    project = Project(
        tmp / "s.db",
        registry=sb.build_registry(),
        adapters=adapters,
        codec=ExtractedValueCodec(),
    )
    for paper in papers:
        project.ingest(
            paper.record,
            schema_ref="study@1.0",
            record_id=paper.record_id,
            source=paper.source,
            parsed=paper.parsed,
        )
    project.migrate("study@1.1")
    project.migrate("study@1.2", policy=ExecutionPolicy(allow_lossy=True))

    snapshot = {
        e.record_id: sorted(a.artifact_id for a in project.artifacts(e.record_id))
        for e in project.records()
    }
    plan, result = project.migrate("study@1.2", policy=ExecutionPolicy(allow_lossy=True))
    again = {
        e.record_id: sorted(a.artifact_id for a in project.artifacts(e.record_id))
        for e in project.records()
    }
    assert snapshot == again
    assert result.metrics.model_calls == 0
    project.close()


@given(st.sampled_from(["1.1", "1.2", "1.3", "1.4"]))
@settings(max_examples=4, deadline=None)
def test_path_composition_reaches_the_target(target):
    """Stepping the graph one hop at a time arrives where the planner said."""

    registry = sb.build_registry()
    path = registry.find_path("study@1.0", f"study@{target}")
    current = "study@1.0"
    for migration in path:
        assert migration.from_schema == current
        current = migration.to_schema
    assert current == f"study@{target}"


@given(st.integers(min_value=1, max_value=3))
@SLOW
def test_prior_record_versions_are_immutable(tmp_path_factory, n_papers):
    papers = sb.corpus(n_papers)
    tmp = tmp_path_factory.mktemp("immutable")
    from llmbic.models.base import AdapterRegistry

    adapters = AdapterRegistry()
    adapters.register("main", sb.stimulus_adapter())
    adapters.register("backup", sb.stimulus_adapter(provider="b", model="r2"))
    project = Project(
        tmp / "s.db",
        registry=sb.build_registry(),
        adapters=adapters,
        codec=ExtractedValueCodec(),
    )
    for paper in papers:
        project.ingest(
            paper.record,
            schema_ref="study@1.0",
            record_id=paper.record_id,
            source=paper.source,
            parsed=paper.parsed,
        )
    before = {
        v.version_id: v.to_canonical()
        for p in papers
        for v in project.store.record_versions(p.record_id)
    }
    project.migrate("study@1.1")
    for version_id, snapshot in before.items():
        current = project.store.get_record_version(version_id)
        assert current is not None
        # Only the publication state may move, and only to `superseded`.
        assert current.artifact_ids == tuple(snapshot["artifact_ids"])
        assert current.schema_ref == snapshot["schema_ref"]
        assert current.state.value in (snapshot["state"], "superseded")
    project.close()
