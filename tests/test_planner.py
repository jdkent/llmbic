"""Planning, invalidation and dry runs (§8.3, §8.7, §11)."""

from __future__ import annotations

import dataclasses


import studybed as sb
from llmbic import Disposition, ExecutionPolicy, PlannerOptions, RecordFilter
from llmbic.errors import ErrorCode
from llmbic.planner.dependencies import (
    Currency,
    DependencyContext,
    assess_currency,
    cache_key,
    dependency_hashes,
)
from llmbic.provenance import FieldArtifact, FieldProvenance
from llmbic.values import FieldValue

LOSSY = ExecutionPolicy(allow_lossy=True)


def dispositions(plan, *, step_id: str | None = None):
    return [
        s.disposition
        for r in plan.records
        for s in r.steps
        if step_id is None or s.step_id == step_id
    ]


# ---- dependency hashing --------------------------------------------------

def _ctx(version="1.1", registry=None):
    registry = registry or sb.build_registry()
    return DependencyContext(
        schema=sb.normalized(version),
        recipes={r.ref: r for r in registry.recipes()},
        vocabularies={v.ref: v for v in registry.vocabularies()},
    )


def test_dependency_hashes_name_what_changed_not_merely_that_something_did():
    ctx = _ctx()
    fdef = ctx.schema.field("tasks[].stimulus_modality")
    hashes = dependency_hashes(None, fdef, ctx, recipe=None)
    assert set(hashes) >= {"field_def:tasks[].stimulus_modality", "recipe"}


def test_changing_a_prompt_changes_the_recipe_hash_and_nothing_else():
    """Acceptance criterion 5."""

    one, two = sb.stimulus_recipe("1"), sb.stimulus_recipe("2")
    assert one.recipe_hash() != two.recipe_hash()
    assert one.prompt_hash != two.prompt_hash
    assert one.context_policy.policy_hash == two.context_policy.policy_hash


def test_a_validators_code_is_part_of_the_recipe_hash():
    from llmbic.functions import VALIDATORS

    before = sb.stimulus_recipe("1").recipe_hash()

    @VALIDATORS.register("test_only_validator", "1")
    def _v(ctx):
        return True

    recipe = dataclasses.replace(
        sb.stimulus_recipe("1"), validators=("test_only_validator@1",)
    )
    assert recipe.recipe_hash() != before


def test_missing_provenance_is_never_evidence_of_currency():
    """FR-DEP-006."""

    artifact = FieldArtifact(
        record_id="r",
        field_id="f",
        value=FieldValue.present(1),
        provenance=FieldProvenance(schema_version="study@1.0"),
    )
    verdict = assess_currency(artifact, {"field_def:f": "x"})
    assert verdict.currency is Currency.NO_PROVENANCE
    assert not verdict.reusable


def test_a_failed_value_is_not_reusable():
    artifact = FieldArtifact(
        record_id="r",
        field_id="f",
        value=FieldValue.failed("transport"),
        provenance=FieldProvenance(schema_version="s", input_hashes={"a": "1"}),
    )
    assert assess_currency(artifact, {"a": "1"}).currency is Currency.FAILED


def test_an_explicitly_accepted_legacy_value_is_reusable():
    """FR-DEP-007."""

    artifact = FieldArtifact(
        record_id="r",
        field_id="f",
        value=FieldValue.present(1),
        provenance=FieldProvenance(
            schema_version="s", accepted_under="recipe@2", actor_id="curator"
        ),
    )
    verdict = assess_currency(artifact, {"a": "changed"}, required_recipe="recipe@2")
    assert verdict.currency is Currency.ACCEPTED
    assert verdict.reusable


def test_extra_recorded_dependencies_do_not_make_a_value_stale():
    artifact = FieldArtifact(
        record_id="r",
        field_id="f",
        value=FieldValue.present(1),
        provenance=FieldProvenance(
            schema_version="s", input_hashes={"field_def:f": "x", "transform": "y"}
        ),
    )
    assert assess_currency(artifact, {"field_def:f": "x"}).currency is Currency.CURRENT


def test_cache_keys_differ_for_every_output_affecting_input():
    step = sb.migration_1_0_to_1_1().steps[0]
    base = dict(
        step=step,
        record_id="r1",
        entity="tasks[]=t1",
        field_ids=["tasks[].stimulus_modality"],
        dependencies={"recipe": "a"},
        context_hash="ctx-1",
        schema_ref="study@1.1",
    )
    key = cache_key(**base)
    for change in (
        {"record_id": "r2"},
        {"entity": "tasks[]=t2"},
        {"dependencies": {"recipe": "b"}},
        {"context_hash": "ctx-2"},
        {"schema_ref": "study@1.2"},
    ):
        assert cache_key(**{**base, **change}) != key
    assert cache_key(**base) == key


# ---- planning ------------------------------------------------------------

def test_a_dry_run_reports_without_touching_anything(loaded):
    before = {e.record_id: e.version_id for e in loaded.records()}
    plan = loaded.plan("study@1.1")
    after = {e.record_id: e.version_id for e in loaded.records()}
    assert before == after
    assert plan.summary.n_model_calls > 0
    assert plan.summary.est_cost_usd >= 0.0


def test_the_plan_reports_context_access_before_anything_runs(loaded):
    plan = loaded.plan("study@1.1")
    semantic = [s for s in plan.steps() if s.disposition is Disposition.SEMANTIC]
    assert semantic
    for step in semantic:
        assert step.context is not None
        assert step.context.unit_ids
        assert step.context.includes_full_document is False
    assert plan.summary.n_full_document_transmissions == 0


def test_a_rename_that_keeps_its_identity_plans_no_steps_at_all(loaded):
    """Acceptance criterion 2: zero model calls, value and evidence retained."""

    plan = loaded.plan("study@1.1")
    assert not [s for s in plan.steps() if "response" in s.step_id]
    assert all(
        s.step_id in ("extract_stimulus_modality", "validate") for s in plan.steps()
    )


def test_adding_a_field_plans_only_that_fields_recipe(loaded):
    """Acceptance criterion 3."""

    plan = loaded.plan("study@1.1")
    written = {w for s in plan.steps() for w in s.writes}
    assert written == {"tasks[].stimulus_modality"}


def test_unrelated_fields_are_neither_planned_nor_marked_stale(loaded):
    plan = loaded.plan("study@1.1")
    stale = {s.split("@")[0] for r in plan.records for s in r.stale_fields}
    # Only the field whose description was rewritten is stale.
    assert stale == {"tasks[].stimuli"}


def test_replanning_after_a_run_reuses_everything(loaded):
    plan, _ = loaded.migrate("study@1.1")
    again = loaded.plan("study@1.1")
    assert again.summary.n_model_calls == 0
    # The one paper with no Methods section stays escalated: a record that
    # could not be answered the first time is not answered by asking again.
    assert set(dispositions(again)) <= {
        Disposition.REUSE,
        Disposition.VALIDATE,
        Disposition.REVIEW,
    }
    # A record already at the target has no path and therefore no steps at all;
    # reuse shows up for a record that stayed behind and replays a migration
    # whose work is already in the store.
    loaded.migrate("study@1.2", policy=LOSSY)
    onward = loaded.plan("study@1.3", policy=LOSSY)
    assert Disposition.REUSE in dispositions(onward)


def test_a_missing_path_blocks_the_record_with_a_structured_reason(project, papers):
    project.ingest(
        papers[0].record,
        schema_ref="study@1.4",
        record_id=papers[0].record_id,
        source=papers[0].source,
        parsed=papers[0].parsed,
    )
    plan = project.plan("study@1.0")
    rp = plan.record(papers[0].record_id)
    assert rp.blocked
    assert rp.blocked_code == ErrorCode.NO_MIGRATION_PATH.value


def test_a_lossy_migration_is_blocked_without_an_explicit_policy(loaded):
    loaded.migrate("study@1.1")
    plan = loaded.plan("study@1.2")
    blocked = [r for r in plan.records if r.blocked]
    assert blocked
    assert blocked[0].blocked_code == ErrorCode.FIDELITY_NOT_PERMITTED.value
    assert "allow_lossy" in blocked[0].blocked_reason


def test_allowing_lossy_unblocks_it(loaded):
    loaded.migrate("study@1.1")
    plan = loaded.plan("study@1.2", policy=LOSSY)
    assert not [r for r in plan.records if r.blocked]


def test_a_destructive_migration_must_also_be_named(registry, project, papers):
    from llmbic import MigrationStep, StepKind
    from llmbic.migration.spec import Fidelity

    project.registry.register_schema(sb.normalized("1.0"))
    # A destructive step inside an existing migration is enough for the check.
    policy = ExecutionPolicy(allow_lossy=True, allow_destructive=True)
    assert policy.permits(sb.migration_1_1_to_1_2())[0] is True

    step = MigrationStep(
        id="wipe",
        kind=StepKind.STRUCTURAL,
        writes=("groups[].other_characteristics",),
        transform="derive_is_healthy@1",
        fidelity=Fidelity.DESTRUCTIVE,
    )
    wipe = sb.migration_1_1_to_1_2().with_steps([step])
    ok, why = ExecutionPolicy(allow_lossy=True, allow_destructive=True).permits(wipe)
    assert not ok and "approved_migrations" in why
    ok, _ = ExecutionPolicy(
        allow_lossy=True, allow_destructive=True, approved_migrations=("study-1.1-to-1.2",)
    ).permits(wipe)
    assert ok


def test_records_without_required_context_are_escalated_not_invented(loaded, papers):
    """Acceptance criterion 10."""

    plan = loaded.plan("study@1.1")
    no_methods = [p for p in papers if not any(u.section == "methods" for u in p.parsed.units)]
    assert no_methods, "the corpus must contain a paper with no Methods section"
    for paper in no_methods:
        steps = [
            s
            for s in plan.record(paper.record_id).steps
            if s.step_id == "extract_stimulus_modality"
        ]
        assert steps and steps[0].disposition is Disposition.REVIEW
        assert steps[0].blocked_code == ErrorCode.CONTEXT_UNAVAILABLE.value


def test_a_policy_forbidding_full_documents_blocks_a_step_that_needs_one(loaded):
    from llmbic import ContextPolicy, full_document
    from llmbic.context.policy import FallbackMode

    migration = sb.migration_1_0_to_1_1()
    greedy = dataclasses.replace(
        migration.steps[0],
        context=ContextPolicy(
            sequence=(full_document(),), full_document_fallback=FallbackMode.ALLOWED
        ),
    )
    payload = migration.with_steps([greedy]).to_canonical()
    payload["id"] = "greedy-1.0-to-1.1"
    payload["branch"] = "experiment"
    from llmbic import Migration
    from llmbic.registry import PathPreference

    loaded.registry.register_migration(Migration.from_canonical(payload), validate=False)
    plan = loaded.plan(
        "study@1.1",
        options=PlannerOptions(
            preference=PathPreference(branches=("experiment",)),
        ),
    )
    blocked = [s for s in plan.steps() if s.disposition is Disposition.BLOCKED]
    assert blocked
    assert blocked[0].blocked_code == ErrorCode.CONTEXT_POLICY_FORBIDS.value


def test_the_plan_counts_records_fields_and_calls(loaded):
    plan = loaded.plan("study@1.1")
    s = plan.summary
    assert s.n_records == len(list(loaded.records()))
    assert s.n_steps == sum(len(r.steps) for r in plan.records)
    assert s.by_disposition["semantic"] == s.n_model_calls
    assert s.est_input_tokens > 0


def test_a_plan_is_serialisable_and_signed(loaded):
    plan = loaded.plan("study@1.1")
    from llmbic.planner.plan import ExecutionPlan

    clone = ExecutionPlan.from_canonical(plan.to_canonical())
    assert clone.plan_id == plan.plan_id
    assert clone.signature() == plan.signature()


def test_the_plan_id_changes_when_the_registry_moves(loaded):
    before = loaded.plan("study@1.1").signature()
    loaded.registry.register_recipe(sb.stimulus_recipe("3"))
    assert loaded.plan("study@1.1").signature() != before


def test_record_selection_narrows_the_plan(loaded, papers):
    only = papers[0].record_id
    plan = loaded.plan("study@1.1", filt=RecordFilter(record_ids=[only]))
    assert [r.record_id for r in plan.records] == [only]


def test_filtering_by_schema_version_selects_the_right_cohort(loaded):
    loaded.migrate("study@1.1")
    at_1_1 = loaded.plan("study@1.1", filt=RecordFilter(schema_ref="study@1.1"))
    at_1_0 = loaded.plan("study@1.1", filt=RecordFilter(schema_ref="study@1.0"))
    assert at_1_1.records
    # The held record stays at 1.0 and is the only one still needing work.
    assert all(
        s.disposition in (Disposition.REUSE, Disposition.VALIDATE)
        for s in at_1_1.steps()
    )
    assert any(s.disposition is Disposition.REVIEW for s in at_1_0.steps())


def test_on_stale_review_turns_stale_fields_into_queue_items(loaded):
    plan = loaded.plan("study@1.1", options=PlannerOptions(on_stale="review"))
    stale_steps = [s for s in plan.steps() if s.migration_id == "__stale__"]
    assert stale_steps
    assert all(s.disposition is Disposition.REVIEW for s in stale_steps)
    assert all(not r.stale_fields for r in plan.records)


def test_steps_are_ordered_by_what_they_read_and_write(loaded):
    loaded.migrate("study@1.1")
    plan = loaded.plan("study@1.2", policy=LOSSY)
    for rp in plan.records:
        keys = [s.key for s in rp.steps]
        for step in rp.steps:
            for dep in step.depends_on:
                assert keys.index(dep) < keys.index(step.key) or dep not in keys


def test_the_plan_renders(loaded):
    text = loaded.plan("study@1.1").render(verbose=True)
    assert "model calls" in text
    assert "source access" in text
