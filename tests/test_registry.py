"""Migration graph resolution, cycle detection and immutability (§8.2, §15.1)."""

from __future__ import annotations

import dataclasses

import pytest

import studybed as sb
from llmbic import Migration, MigrationStep, Ref, Registry, StepKind
from llmbic.errors import ErrorCode, LlmbicError
from llmbic.migration.spec import Fidelity
from llmbic.registry import PathPreference


def _noop_migration(mid: str, src: str, dst: str, **kwargs) -> Migration:
    return Migration(id=mid, from_schema=src, to_schema=dst, steps=(), **kwargs)


# ---- schema immutability (FR-SCH-001) -----------------------------------

def test_registering_the_same_schema_twice_is_fine():
    reg = Registry()
    reg.register_schema(sb.normalized("1.0"))
    reg.register_schema(sb.normalized("1.0"))
    assert len(reg.schemas()) == 1


def test_a_version_identifier_cannot_change_meaning():
    reg = Registry()
    reg.register_schema(sb.normalized("1.0"))
    impostor = dataclasses.replace(sb.normalized("1.1"), version="1.0")
    with pytest.raises(LlmbicError) as exc:
        reg.register_schema(impostor)
    assert exc.value.code is ErrorCode.SCHEMA_IMMUTABLE


def test_a_migration_identity_is_immutable():
    reg = sb.build_registry()
    payload = sb.migration_1_2_to_1_3().to_canonical()
    payload["description"] = "same id, different content"
    payload["steps"] = []
    with pytest.raises(LlmbicError) as exc:
        reg.register_migration(Migration.from_canonical(payload), validate=False)
    assert exc.value.code is ErrorCode.SCHEMA_IMMUTABLE


# ---- graph shape ---------------------------------------------------------

def test_path_spans_every_intermediate_version():
    reg = sb.build_registry()
    path = reg.find_path("study@1.0", "study@1.4")
    assert [m.id for m in path] == [
        "study-1.0-to-1.1",
        "study-1.1-to-1.2",
        "study-1.2-to-1.3",
        "study-1.3-to-1.4",
    ]


def test_path_to_the_current_version_is_empty():
    reg = sb.build_registry()
    assert reg.find_path("study@1.4", "study@1.4") == []


def test_missing_path_is_a_structured_error():
    reg = sb.build_registry()
    with pytest.raises(LlmbicError) as exc:
        reg.find_path("study@1.4", "study@1.0")
    assert exc.value.code is ErrorCode.NO_MIGRATION_PATH


def test_cycles_are_rejected():
    reg = sb.build_registry()
    with pytest.raises(LlmbicError) as exc:
        reg.register_migration(
            _noop_migration("backwards", "study@1.2", "study@1.0"), validate=False
        )
    assert exc.value.code is ErrorCode.MIGRATION_CYCLE
    assert "is_downgrade" in exc.value.message


def test_a_downgrade_is_stored_apart_from_upgrade_planning():
    """FR-MIG-004."""

    reg = sb.build_registry()
    reg.register_migration(
        _noop_migration("study-1.1-to-1.0", "study@1.1", "study@1.0", is_downgrade=True),
        validate=False,
    )
    assert [m.id for m in reg.downgrades_from("study@1.1")] == ["study-1.1-to-1.0"]
    # It must not appear in an upgrade path.
    assert [m.id for m in reg.find_path("study@1.0", "study@1.4")][0] == "study-1.0-to-1.1"


def test_self_loops_are_rejected():
    reg = sb.build_registry()
    with pytest.raises(LlmbicError) as exc:
        reg.register_migration(
            _noop_migration("loop", "study@1.1", "study@1.1"), validate=False
        )
    assert exc.value.code is ErrorCode.MIGRATION_CYCLE


def test_two_approved_production_edges_between_the_same_pair_are_refused():
    """FR-MIG-012."""

    reg = sb.build_registry()
    rival = sb.migration_1_2_to_1_3()
    payload = rival.to_canonical()
    payload["id"] = "study-1.2-to-1.3-rival"
    with pytest.raises(LlmbicError) as exc:
        reg.register_migration(Migration.from_canonical(payload), validate=False)
    assert exc.value.code is ErrorCode.AMBIGUOUS_MIGRATION_PATH


def test_a_branch_may_carry_a_rival_migration():
    reg = sb.build_registry()
    payload = sb.migration_1_2_to_1_3().to_canonical()
    payload["id"] = "study-1.2-to-1.3-experiment"
    payload["branch"] = "experiment"
    reg.register_migration(Migration.from_canonical(payload), validate=False)

    # The production path is unchanged...
    assert [m.id for m in reg.find_path("study@1.2", "study@1.3")] == ["study-1.2-to-1.3"]
    # ...and the branch is reachable only when asked for.
    pref = PathPreference(branches=("main", "experiment"))
    ids = {p[0].id for p in reg.find_paths("study@1.2", "study@1.3", preference=pref)}
    assert ids == {"study-1.2-to-1.3", "study-1.2-to-1.3-experiment"}


def test_unapproved_migrations_are_skipped_by_default():
    reg = sb.build_registry(up_to="1.2")
    reg.register_schema(sb.normalized("1.3"))
    payload = sb.migration_1_2_to_1_3().to_canonical()
    payload["approved"] = False
    reg.register_migration(Migration.from_canonical(payload), validate=False)
    with pytest.raises(LlmbicError):
        reg.find_path("study@1.2", "study@1.3")
    pref = PathPreference(require_approved=False)
    assert reg.find_path("study@1.2", "study@1.3", preference=pref)


def test_path_selection_is_deterministic_across_repeated_calls():
    """FR-MIG-003."""

    reg = sb.build_registry()
    first = [m.id for m in reg.find_path("study@1.0", "study@1.4")]
    for _ in range(5):
        assert [m.id for m in reg.find_path("study@1.0", "study@1.4")] == first


def test_alternative_paths_are_compared_by_declared_policy():
    """FR-PLN-007: a cheaper path must not be taken if it is lossier."""

    reg = sb.build_registry(up_to="1.2")
    # A shortcut that skips 1.1 but is destructive.
    reg.register_schema(sb.normalized("1.3"))
    shortcut = Migration(
        id="shortcut-1.0-to-1.2",
        from_schema="study@1.0",
        to_schema="study@1.2",
        steps=(
            MigrationStep(
                id="wipe",
                kind=StepKind.STRUCTURAL,
                writes=("tasks[].stimulus_modality",),
                transform="rename_response_mode@1",
                fidelity=Fidelity.DESTRUCTIVE,
            ),
        ),
        acknowledged={
            "groups[].is_healthy": "left unset",
            "groups[].other_characteristics": "left unset",
            "tasks[].response_mode": "renamed in place",
        },
    )
    reg.register_migration(shortcut, validate=False)

    by_cost = reg.find_path(
        "study@1.0", "study@1.2", preference=PathPreference(optimise="cost")
    )
    by_fidelity = reg.find_path(
        "study@1.0", "study@1.2", preference=PathPreference(optimise="fidelity")
    )
    assert [m.id for m in by_cost] == ["shortcut-1.0-to-1.2"]
    assert [m.id for m in by_fidelity] == ["study-1.0-to-1.1", "study-1.1-to-1.2"]


# ---- static validation ---------------------------------------------------

def test_every_changed_field_needs_a_disposition():
    """§21.2 — the check that stops half a schema change from shipping."""

    reg = Registry()
    reg.register_schema(sb.normalized("1.0"))
    reg.register_schema(sb.normalized("1.1"))
    bare = Migration(id="bare", from_schema="study@1.0", to_schema="study@1.1")
    problems = reg.validate_migration(bare)
    assert any("tasks[].stimulus_modality" in p for p in problems)


def test_an_acknowledged_change_needs_no_step():
    reg = Registry()
    reg.register_schema(sb.normalized("1.3"))
    reg.register_schema(sb.normalized("1.4"))
    reg.register_recipe(sb.stimulus_recipe("2"))
    assert reg.validate_migration(sb.migration_1_3_to_1_4()) == []


def test_writing_a_field_the_target_does_not_define_is_invalid():
    reg = Registry()
    reg.register_schema(sb.normalized("1.0"))
    reg.register_schema(sb.normalized("1.1"))
    bogus = Migration(
        id="bogus",
        from_schema="study@1.0",
        to_schema="study@1.1",
        steps=(
            MigrationStep(
                id="s",
                kind=StepKind.STRUCTURAL,
                writes=("nope",),
                transform="rename_response_mode@1",
            ),
        ),
    )
    assert any("which study@1.1 does not define" in p for p in reg.validate_migration(bogus))


def test_a_semantic_step_must_name_a_registered_recipe():
    reg = Registry()
    reg.register_schema(sb.normalized("1.0"))
    reg.register_schema(sb.normalized("1.1"))
    m = sb.migration_1_0_to_1_1()
    assert any("unknown recipe" in p for p in reg.validate_migration(m))


def test_a_deterministic_step_must_name_a_transform():
    with pytest.raises(LlmbicError, match="must name a transform"):
        MigrationStep(id="s", kind=StepKind.STRUCTURAL, writes=("a",))


def test_a_semantic_step_must_name_a_recipe():
    with pytest.raises(LlmbicError, match="must name an extraction recipe"):
        MigrationStep(id="s", kind=StepKind.SOURCE_SEMANTIC, writes=("a",))


def test_a_step_must_write_something():
    with pytest.raises(LlmbicError, match="declares no outputs"):
        MigrationStep(id="s", kind=StepKind.STRUCTURAL, transform="t@1")


def test_an_evidence_local_step_may_not_hold_a_full_document_policy():
    from llmbic import ContextPolicy, full_document

    with pytest.raises(LlmbicError, match="evidence-local"):
        MigrationStep(
            id="s",
            kind=StepKind.EVIDENCE_SEMANTIC,
            writes=("a",),
            recipe="r@1",
            context=ContextPolicy(sequence=(full_document(),)),
        )


def test_step_ordering_follows_reads_and_writes():
    m = Migration(
        id="ordered",
        from_schema="study@1.1",
        to_schema="study@1.2",
        steps=(
            MigrationStep(
                id="second",
                kind=StepKind.DERIVED,
                reads=(Ref("field", "groups[].is_healthy"),),
                writes=("groups[].other_characteristics",),
                transform="derive_is_healthy@1",
            ),
            MigrationStep(
                id="first",
                kind=StepKind.DERIVED,
                reads=(Ref("field", "groups[].medical_condition"),),
                writes=("groups[].is_healthy",),
                transform="derive_is_healthy@1",
            ),
        ),
    )
    assert [s.id for s in m.ordered_steps()] == ["first", "second"]


def test_a_cycle_among_steps_is_detected():
    m = Migration(
        id="cyclic",
        from_schema="study@1.1",
        to_schema="study@1.2",
        steps=(
            MigrationStep(
                id="a",
                kind=StepKind.DERIVED,
                reads=(Ref("field", "groups[].other_characteristics"),),
                writes=("groups[].is_healthy",),
                transform="derive_is_healthy@1",
            ),
            MigrationStep(
                id="b",
                kind=StepKind.DERIVED,
                reads=(Ref("field", "groups[].is_healthy"),),
                writes=("groups[].other_characteristics",),
                transform="derive_is_healthy@1",
            ),
        ),
    )
    with pytest.raises(LlmbicError) as exc:
        m.ordered_steps()
    assert exc.value.code is ErrorCode.MIGRATION_CYCLE


def test_fidelity_of_a_migration_is_the_worst_of_its_steps():
    assert sb.migration_1_1_to_1_2().fidelity is Fidelity.LOSSY
    assert sb.migration_1_0_to_1_1().fidelity is Fidelity.LOSSLESS


def test_malformed_references_are_rejected():
    with pytest.raises(LlmbicError, match="malformed reference"):
        Ref.parse("nonsense")


# ---- serialisation -------------------------------------------------------

def test_registry_round_trips_through_canonical_form():
    reg = sb.build_registry()
    clone = Registry.from_canonical(reg.to_canonical())
    assert clone.registry_hash() == reg.registry_hash()
    assert [m.id for m in clone.migrations()] == [m.id for m in reg.migrations()]


def test_registry_hash_changes_when_anything_changes():
    reg = sb.build_registry(up_to="1.3")
    before = reg.registry_hash()
    reg.register_schema(sb.normalized("1.4"))
    assert reg.registry_hash() != before
