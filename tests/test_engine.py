"""Corpus execution: caching, resumption, isolation, budgets, atomicity (§8.8)."""

from __future__ import annotations

import dataclasses

import pytest

import studybed as sb
from llmbic import (
    ExecutionPolicy,
    ExecutionStatus,
    Project,
)
from llmbic.errors import ErrorCode, LlmbicError
from llmbic.execution.engine import Engine
from llmbic.execution.state import StepStatus
from llmbic.models.base import AdapterRegistry, Pricing
from llmbic.models.mock import FailingAdapter, HostedMockAdapter, ScriptedAdapter
from llmbic.provenance import RecordState

LOSSY = ExecutionPolicy(allow_lossy=True)


# ---- happy path ----------------------------------------------------------

def test_a_full_chain_runs_through_four_schema_versions(loaded):
    """Acceptance criterion 1."""

    for target in ("study@1.1", "study@1.2", "study@1.3", "study@1.4"):
        plan, result = loaded.migrate(target, policy=LOSSY)
        assert result.status in (ExecutionStatus.SUCCEEDED, ExecutionStatus.PARTIAL)
    versions = {e.schema_ref for e in loaded.records()}
    assert "study@1.4" in versions


def test_a_rename_keeps_its_value_and_its_evidence(loaded, papers):
    """Acceptance criterion 2."""

    record_id = papers[0].record_id
    before = {a.field_id: a for a in loaded.artifacts(record_id)}
    plan, result = loaded.migrate("study@1.1")

    after = {a.field_id: a for a in loaded.artifacts(record_id)}
    old = before["tasks[].response_mode"]
    new = after["tasks[].response_mode"]
    assert old.artifact_id == new.artifact_id  # nothing was rewritten
    assert new.value.value == ["button_press"]

    record = loaded.get_record(record_id)
    assert "response_modality" in record["tasks"][0]
    assert "response_mode" not in record["tasks"][0]


def test_adding_a_field_reuses_every_unrelated_value(loaded, papers, adapters):
    """Acceptance criterion 3."""

    record_id = papers[0].record_id
    before = {(a.field_id, a.entity): a.artifact_id for a in loaded.artifacts(record_id)}
    loaded.migrate("study@1.1")
    after = {(a.field_id, a.entity): a.artifact_id for a in loaded.artifacts(record_id)}

    new_keys = set(after) - set(before)
    assert new_keys == {("tasks[].stimulus_modality", "tasks[]=t1")}
    assert all(after[k] == before[k] for k in before)


def test_a_semantic_value_reads_evidence_first_and_never_the_document(loaded, papers):
    """Acceptance criterion 4."""

    loaded.migrate("study@1.1")
    paper = next(p for p in papers if p.truth == "visual")
    artifact = next(
        a
        for a in loaded.artifacts(paper.record_id)
        if a.field_id == "tasks[].stimulus_modality"
    )
    assert artifact.value.value == ["visual"]
    assert artifact.provenance.context_selector_ref == "prior_evidence@1"
    assert artifact.provenance.context_units
    # Every supplied unit is an evidence window, not the article.
    assert len(artifact.provenance.context_units) < len(paper.parsed.units)


def test_every_new_value_traces_back_to_everything_that_produced_it(loaded, papers):
    """Acceptance criterion 9."""

    loaded.migrate("study@1.1")
    paper = next(p for p in papers if p.truth)
    rows = loaded.provenance(paper.record_id, "tasks[].stimulus_modality", "tasks[]=t1")
    assert rows
    p = rows[-1]["provenance"]
    assert p["schema_version"] == "study@1.1"
    assert p["recipe_ref"] == "stimulus-modality@1"
    assert p["migration_id"] == "study-1.0-to-1.1"
    assert p["execution_id"]
    assert p["context_units"] and p["context_hash"]
    assert p["model_call"]["model"] and p["model_call"]["provider_request_id"]
    assert p["prompt_hash"]
    assert p["validations"]
    assert p["input_hashes"]


def test_a_derived_field_records_the_exact_values_it_came_from(loaded):
    """FR-PROV-005."""

    loaded.migrate("study@1.1")
    loaded.migrate("study@1.2", policy=LOSSY)
    for entry in loaded.records():
        for artifact in loaded.artifacts(entry.record_id):
            if artifact.field_id == "groups[].is_healthy":
                assert artifact.provenance.lineage
                assert "medical_condition" in artifact.provenance.lineage[0]
                assert artifact.provenance.actor.value == "migration"
                return
    pytest.fail("no derived is_healthy artifact was produced")


def test_a_structural_move_carries_the_evidence_with_it(loaded):
    """FR-PROV-003."""

    loaded.migrate("study@1.1")
    loaded.migrate("study@1.2", policy=LOSSY)
    kept = [
        a
        for e in loaded.records()
        for a in loaded.artifacts(e.record_id)
        if a.field_id == "groups[].population_characteristics"
        and a.provenance.step_id == "split_characteristics"
        and a.value.status.has_value
    ]
    assert kept
    assert any(a.evidence for a in kept)


# ---- caching and resumption ---------------------------------------------

def test_rerunning_the_same_plan_buys_nothing_twice(loaded, adapters):
    """Product principle 7 / FR-EXE-004."""

    main = adapters.get("main")
    plan = loaded.plan("study@1.1")
    loaded.run(plan)
    calls_after_first = len(main.calls)

    second = loaded.run(loaded.plan("study@1.1"))
    assert len(main.calls) == calls_after_first
    assert second.metrics.model_calls == 0


def test_re_executing_an_approved_plan_hits_the_cache_instead_of_the_model(
    loaded, adapters
):
    """FR-EXE-004: work is addressed by its cache key, not by the execution."""

    main = adapters.get("main")
    plan = loaded.plan("study@1.1")
    loaded.run(plan, execution_id="exec-a")
    calls = len(main.calls.successes)

    # The same approved plan, run again under a different execution id. The
    # plan still says SEMANTIC, and the engine still pays nothing.
    second = loaded.run(plan, execution_id="exec-b")
    assert second.metrics.cache_hits > 0
    assert second.metrics.model_calls == 0
    assert len(main.calls.successes) == calls


def test_the_planner_reports_a_cache_hit_as_its_own_disposition(loaded):
    """A step whose answer is already in the cache is not a model call."""

    from llmbic.store.base import CacheEntry

    plan = loaded.plan("study@1.1")
    semantic = [s for s in plan.steps() if s.disposition.value == "semantic"]
    assert semantic
    loaded.store.cache_put(
        CacheEntry(
            cache_key=semantic[0].cache_key,
            payload={"output": {"tasks[].stimulus_modality": {"status": "not_reported"}}},
            created_at="",
        )
    )
    replanned = loaded.plan("study@1.1")
    cached = [s for s in replanned.steps() if s.disposition.value == "cached"]
    assert [s.key for s in cached] == [semantic[0].key]
    assert replanned.summary.n_model_calls == len(semantic) - 1
    assert replanned.summary.n_cached_calls == 1


def test_interrupting_and_resuming_creates_no_duplicate_calls(tmp_path, registry, papers):
    """Acceptance criterion 8 / NFR-PERF-004."""

    from llmbic import ExtractedValueCodec

    main = sb.stimulus_adapter()
    adapters = AdapterRegistry()
    adapters.register("main", main)
    adapters.register("backup", sb.stimulus_adapter(provider="b", model="r2"))

    project = Project(
        tmp_path / "s.db", registry=registry, adapters=adapters, codec=ExtractedValueCodec()
    )
    for paper in papers:
        project.ingest(
            paper.record,
            schema_ref="study@1.0",
            record_id=paper.record_id,
            source=paper.source,
            parsed=paper.parsed,
        )

    plan = project.plan("study@1.1")
    half = [r.record_id for r in plan.records][: len(plan.records) // 2]
    project.run(plan, execution_id="exec-interrupted", record_ids=half)
    calls_after_half = len(main.calls.successes)
    assert calls_after_half > 0

    resumed = project.resume("exec-interrupted")
    assert resumed.metrics.steps_skipped > 0
    # Every successful call from the first half was reused, not repeated.
    assert len(main.calls.successes) > calls_after_half
    distinct = main.calls.distinct_requests()
    assert len(distinct) == len(main.calls.successes)

    third = project.resume("exec-interrupted")
    assert third.metrics.model_calls == 0
    project.close()


def test_resuming_a_plan_that_changed_is_refused(loaded):
    plan = loaded.plan("study@1.1")
    loaded.run(plan, execution_id="exec-1", record_ids=[])
    other = loaded.plan("study@1.1", policy=ExecutionPolicy(allow_lossy=True))
    engine = Engine(loaded.registry, loaded.store, adapters=loaded.adapters)
    with pytest.raises(LlmbicError) as exc:
        engine.run(other, execution_id="exec-1", resume=True)
    assert exc.value.code is ErrorCode.PLAN_SIGNATURE_MISMATCH


def test_running_a_stale_plan_is_refused(loaded):
    plan = loaded.plan("study@1.1")
    loaded.registry.register_recipe(sb.stimulus_recipe("9"))
    with pytest.raises(LlmbicError) as exc:
        loaded.run(plan)
    assert exc.value.code is ErrorCode.PLAN_SIGNATURE_MISMATCH


def test_starting_an_existing_execution_without_resume_is_refused(loaded):
    plan = loaded.plan("study@1.1")
    loaded.run(plan, execution_id="exec-x")
    with pytest.raises(LlmbicError):
        loaded.run(plan, execution_id="exec-x")


# ---- failure isolation and atomicity ------------------------------------

def test_one_records_failure_does_not_roll_back_the_others(tmp_path, registry, papers):
    """FR-EXE-002."""

    from llmbic import ExtractedValueCodec

    def answer(request):
        quote = request.context_units[0].text[:40] if request.context_units else ""
        return {
            "tasks[].stimulus_modality": {
                "status": "present",
                "value": ["visual"],
                "evidence": [quote],
            }
        }

    class OneBadApple(ScriptedAdapter):
        def generate(self, request):
            if "pmid:100002" in request.prompt:
                raise LlmbicError("boom", code=ErrorCode.MODEL_TRANSPORT)
            return super().generate(request)

    adapters = AdapterRegistry()
    adapters.register("main", OneBadApple(default=answer))
    adapters.register("backup", FailingAdapter(provider="backup"))
    project = Project(
        tmp_path / "s.db", registry=registry, adapters=adapters, codec=ExtractedValueCodec()
    )
    for paper in papers:
        project.ingest(
            paper.record,
            schema_ref="study@1.0",
            record_id=paper.record_id,
            source=paper.source,
            parsed=paper.parsed,
        )
    plan, result = project.migrate("study@1.1")
    assert result.status is ExecutionStatus.PARTIAL
    assert len(result.published) >= len(papers) - 2
    assert "pmid:100002" not in result.published
    project.close()


def test_a_failed_record_keeps_its_previous_published_version(loaded, papers):
    """Acceptance criterion 11 / FR-EXE-007."""

    record_id = next(
        p.record_id
        for p in papers
        if not any(u.section == "methods" for u in p.parsed.units)
    )
    before = loaded.store.current_version(record_id)
    loaded.migrate("study@1.1")
    after = loaded.store.current_version(record_id)
    assert after.version_id == before.version_id
    assert after.schema_ref == "study@1.0"


def test_a_held_record_is_still_inspectable_as_a_draft(loaded, papers):
    """FR-EXE-008."""

    record_id = next(
        p.record_id
        for p in papers
        if not any(u.section == "methods" for u in p.parsed.units)
    )
    loaded.migrate("study@1.1")
    versions = loaded.store.record_versions(record_id)
    held = [v for v in versions if v.state is RecordState.REVIEW_NEEDED]
    assert held
    assert held[0].schema_ref == "study@1.1"
    assert held[0].notes["held_steps"]


def test_a_published_version_supersedes_the_previous_one(loaded, papers):
    record_id = papers[0].record_id
    loaded.migrate("study@1.1")
    versions = loaded.store.record_versions(record_id)
    states = [v.state for v in versions]
    assert RecordState.SUPERSEDED in states
    assert loaded.store.current_version(record_id).schema_ref == "study@1.1"


def test_prior_artifacts_are_never_mutated(loaded, papers):
    """FR-EXE-006."""

    record_id = papers[0].record_id
    before = {a.artifact_id: a.to_canonical() for a in loaded.artifacts(record_id)}
    loaded.migrate("study@1.1")
    loaded.migrate("study@1.2", policy=LOSSY)
    for artifact_id, snapshot in before.items():
        current = loaded.store.get_artifact(artifact_id)
        assert current is not None
        assert current.to_canonical() == snapshot


# ---- retries, fallbacks, failure kinds ----------------------------------

def _hosted_project(tmp_path, registry, papers, **adapter_kwargs):
    from llmbic import ExtractedValueCodec

    hosted = HostedMockAdapter(
        responder=lambda req: {
            "tasks[].stimulus_modality": {
                "status": "present",
                "value": ["visual"],
                "evidence": [],
            }
        },
        **adapter_kwargs,
    )
    adapters = AdapterRegistry()
    adapters.register("main", hosted)
    adapters.register("backup", sb.stimulus_adapter(provider="b", model="r2"))
    project = Project(
        tmp_path / "s.db", registry=registry, adapters=adapters, codec=ExtractedValueCodec()
    )
    for paper in papers:
        project.ingest(
            paper.record,
            schema_ref="study@1.0",
            record_id=paper.record_id,
            source=paper.source,
            parsed=paper.parsed,
        )
    return project, hosted


def test_transient_failures_are_retried_and_every_attempt_is_recorded(
    tmp_path, registry, papers
):
    """FR-LLM-005."""

    project, hosted = _hosted_project(
        tmp_path, registry, papers, transient_failure_rate=0.5
    )
    plan, result = project.migrate("study@1.1")
    assert result.metrics.retries > 0
    attempts = project.store.get_attempts(result.execution_id)
    assert any(a.get("outcome") == "error" for a in attempts)
    assert any(a.get("outcome") == "ok" for a in attempts)
    # Attempts share one logical step key.
    for a in attempts:
        assert a["step_key"]
    project.close()


def test_a_schema_invalid_answer_is_a_different_failure_from_a_dropped_connection(
    tmp_path, registry, papers
):
    """FR-LLM-004."""

    project, hosted = _hosted_project(tmp_path, registry, papers, schema_failure_rate=1.0)
    plan, result = project.migrate("study@1.1")

    attempts = project.store.get_attempts(result.execution_id)
    codes = {
        (a.get("error") or {}).get("code") for a in attempts if a.get("outcome") == "error"
    }
    assert ErrorCode.MODEL_SCHEMA_INVALID.value in codes
    assert ErrorCode.MODEL_TRANSPORT.value not in codes

    # The declared fallback takes over, which is the point of distinguishing
    # the failure kinds in the first place.
    assert result.published
    project.close()


def test_a_context_too_large_for_the_model_is_unsupported_not_transient(
    tmp_path, registry, papers
):
    project, hosted = _hosted_project(tmp_path, registry, papers, max_input_tokens=1)
    plan, result = project.migrate("study@1.1")
    attempts = project.store.get_attempts(result.execution_id)
    errors = [a for a in attempts if a.get("outcome") == "error"]
    codes = {(a.get("error") or {}).get("code") for a in errors}
    assert codes == {ErrorCode.MODEL_UNSUPPORTED_CONTEXT.value}
    # Unsupported context is not retryable, so each step tried the primary
    # exactly once before falling back.
    per_step = {}
    for a in errors:
        per_step[a["step_key"]] = per_step.get(a["step_key"], 0) + 1
    assert set(per_step.values()) == {1}
    project.close()


def test_a_fallback_model_takes_over_when_the_primary_is_down(
    tmp_path, registry, papers
):
    from llmbic import ExtractedValueCodec

    backup = sb.stimulus_adapter(provider="backup", model="rules-2")
    adapters = AdapterRegistry()
    adapters.register("main", FailingAdapter())
    adapters.register("backup", backup)
    project = Project(
        tmp_path / "s.db", registry=registry, adapters=adapters, codec=ExtractedValueCodec()
    )
    for paper in papers[:3]:
        project.ingest(
            paper.record,
            schema_ref="study@1.0",
            record_id=paper.record_id,
            source=paper.source,
            parsed=paper.parsed,
        )
    plan, result = project.migrate("study@1.1")
    assert result.metrics.model_calls > 0
    assert len(backup.calls.successes) > 0
    project.close()


def test_a_pinned_fingerprint_that_moves_routes_to_review(tmp_path, registry, papers):
    """Decision 18.8: a stable model name is not a stable model."""

    from llmbic import ExtractedValueCodec

    recipe = sb.stimulus_recipe("1")
    pinned = dataclasses.replace(
        recipe,
        model_policy=dataclasses.replace(recipe.model_policy, pin_fingerprint="2025-01-01"),
    )
    registry._recipes[recipe.ref] = pinned  # same ref, pinned policy

    adapters = AdapterRegistry()
    adapters.register("main", sb.stimulus_adapter())
    adapters.register("backup", sb.stimulus_adapter(provider="b", model="r2"))
    project = Project(
        tmp_path / "s.db", registry=registry, adapters=adapters, codec=ExtractedValueCodec()
    )
    project.ingest(
        papers[0].record,
        schema_ref="study@1.0",
        record_id=papers[0].record_id,
        source=papers[0].source,
        parsed=papers[0].parsed,
    )
    plan, result = project.migrate("study@1.1")
    assert result.metrics.steps_review > 0
    project.close()


# ---- budgets, cancellation, concurrency ---------------------------------

def test_a_budget_stops_scheduling_new_billable_calls(loaded, adapters):
    """FR-EXE-010."""

    priced = dataclasses.replace(
        sb.stimulus_recipe("1"),
        model_policy=dataclasses.replace(
            sb.stimulus_recipe("1").model_policy, pricing=Pricing(1_000_000, 1_000_000)
        ),
    )
    loaded.registry._recipes["stimulus-modality@1"] = priced

    plan, result = loaded.migrate(
        "study@1.1", policy=ExecutionPolicy(budget_usd=0.5)
    )
    assert result.metrics.steps_blocked > 0
    assert result.metrics.model_calls >= 1
    blocked = [
        s
        for s in loaded.store.get_step_states(result.execution_id)
        if s.state == StepStatus.BLOCKED.value
    ]
    assert any(
        s.detail.get("code") == ErrorCode.MODEL_BUDGET_EXCEEDED.value for s in blocked
    )


def test_a_call_cap_is_enforced(loaded):
    plan, result = loaded.migrate(
        "study@1.1", policy=ExecutionPolicy(max_model_calls=2)
    )
    assert result.metrics.model_calls == 2
    assert result.metrics.steps_blocked > 0


def test_cancellation_stops_the_run_and_leaves_it_resumable(loaded):
    plan = loaded.plan("study@1.1")
    engine = Engine(
        loaded.registry, loaded.store, adapters=loaded.adapters, max_workers=1
    )
    engine.cancel("exec-cancel")
    result = engine.run(plan, execution_id="exec-cancel")
    assert result.status is ExecutionStatus.CANCELLED
    assert not result.published

    engine2 = Engine(loaded.registry, loaded.store, adapters=loaded.adapters)
    resumed = engine2.run(plan, execution_id="exec-cancel", resume=True)
    assert resumed.published


def test_concurrency_produces_the_same_result_as_serial(tmp_path, registry, papers):
    from llmbic import ExtractedValueCodec

    outputs = []
    for workers in (1, 4):
        adapters = AdapterRegistry()
        adapters.register("main", sb.stimulus_adapter())
        adapters.register("backup", sb.stimulus_adapter(provider="b", model="r2"))
        project = Project(
            tmp_path / f"s{workers}.db",
            registry=sb.build_registry(),
            adapters=adapters,
            codec=ExtractedValueCodec(),
            max_workers=workers,
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
        outputs.append(
            {
                e.record_id: sorted(
                    (a.field_id, a.entity, a.artifact_id)
                    for a in project.artifacts(e.record_id)
                )
                for e in project.records()
            }
        )
        project.close()
    assert outputs[0] == outputs[1]


def test_events_are_emitted_with_identifiers_and_no_prompt_text(
    tmp_path, registry, papers
):
    """NFR-OBS-002/003."""

    from llmbic import ExtractedValueCodec

    seen = []
    adapters = AdapterRegistry()
    adapters.register("main", sb.stimulus_adapter())
    adapters.register("backup", sb.stimulus_adapter(provider="b", model="r2"))
    project = Project(
        tmp_path / "s.db",
        registry=registry,
        adapters=adapters,
        codec=ExtractedValueCodec(),
        on_event=seen.append,
    )
    project.ingest(
        papers[0].record,
        schema_ref="study@1.0",
        record_id=papers[0].record_id,
        source=papers[0].source,
        parsed=papers[0].parsed,
    )
    project.migrate("study@1.1")
    kinds = {e["event"] for e in seen}
    assert {"execution.started", "step.succeeded", "record.published"} <= kinds
    blob = repr(seen)
    assert "Read the supplied context" not in blob
    assert "IAPS photographs" not in blob
    project.close()


def test_deterministic_migrations_need_no_model_runtime(loaded):
    """NFR-PERF-005 / FR-STO-005."""

    loaded.migrate("study@1.1")
    loaded.adapters = AdapterRegistry()  # no adapters at all
    plan, result = loaded.migrate("study@1.2", policy=LOSSY)
    assert result.metrics.model_calls == 0
    assert result.metrics.steps_succeeded > 0
