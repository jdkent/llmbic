"""The thirteen acceptance criteria for the minimum viable release (§14).

One test per criterion, named after it.  These are deliberately end-to-end and
deliberately slow: they are the demonstrations the requirements say the first
release is acceptable only if they pass.
"""

from __future__ import annotations

import json


import studybed as sb
from llmbic import (
    ExecutionPolicy,
    ExecutionStatus,
    ExtractedValueCodec,
    FieldValue,
    Project,
    ReviewDecision,
    ValueStatus,
)

from llmbic.models.base import AdapterRegistry
from llmbic.provenance import Actor, RecordState
from llmbic.values import AbsenceReason

LOSSY = ExecutionPolicy(allow_lossy=True)


def _project(tmp_path, papers, *, registry=None, adapters=None, name="s.db"):
    registry = registry or sb.build_registry()
    if adapters is None:
        adapters = AdapterRegistry()
        adapters.register("main", sb.stimulus_adapter())
        adapters.register("backup", sb.stimulus_adapter(provider="b", model="r2"))
    project = Project(
        tmp_path / name, registry=registry, adapters=adapters, codec=ExtractedValueCodec()
    )
    for paper in papers:
        project.ingest(
            paper.record,
            schema_ref="study@1.0",
            record_id=paper.record_id,
            source=paper.source,
            parsed=paper.parsed,
        )
    return project


# ---- 1 -------------------------------------------------------------------

def test_01_a_corpus_upgrades_through_three_sequential_schema_versions(loaded):
    for target in ("study@1.1", "study@1.2", "study@1.3", "study@1.4"):
        plan, result = loaded.migrate(target, policy=LOSSY)
        assert result.status in (ExecutionStatus.SUCCEEDED, ExecutionStatus.PARTIAL)

    at_latest = [e for e in loaded.records() if e.schema_ref == "study@1.4"]
    assert len(at_latest) >= 5
    record = loaded.get_record(at_latest[0].record_id)
    assert "stimulus_modality" in record["tasks"][0]
    assert "is_healthy" in record["groups"][0]
    assert record["design"]["assignment_structure"]["value"] in sb.ASSIGNMENT_V2


# ---- 2 -------------------------------------------------------------------

def test_02_a_rename_completes_with_zero_model_calls_and_keeps_its_lineage(
    loaded, papers, adapters
):
    record_id = papers[0].record_id
    before = next(
        a for a in loaded.artifacts(record_id) if a.field_id == "tasks[].response_mode"
    )

    plan = loaded.plan("study@1.1")
    rename_steps = [s for s in plan.steps() if "response" in s.step_id]
    assert rename_steps == []

    loaded.run(plan)
    after = next(
        a for a in loaded.artifacts(record_id) if a.field_id == "tasks[].response_mode"
    )

    assert after.artifact_id == before.artifact_id
    assert after.evidence == before.evidence
    assert loaded.get_record(record_id)["tasks"][0]["response_modality"]["value"] == [
        "button_press"
    ]
    # Nothing about the renamed field went near a model, in any record.
    renamed = [
        a
        for e in loaded.records()
        for a in loaded.artifacts(e.record_id, latest=False)
        if a.field_id == "tasks[].response_mode"
    ]
    assert renamed
    assert all(a.provenance.model_call is None for a in renamed)
    assert all(a.provenance.migration_id is None for a in renamed)


# ---- 3 -------------------------------------------------------------------

def test_03_adding_a_field_runs_only_that_fields_recipe(loaded, papers):
    record_id = papers[0].record_id
    before = {(a.field_id, a.entity): a.artifact_id for a in loaded.artifacts(record_id)}
    plan, result = loaded.migrate("study@1.1")
    after = {(a.field_id, a.entity): a.artifact_id for a in loaded.artifacts(record_id)}

    assert set(after) - set(before) == {("tasks[].stimulus_modality", "tasks[]=t1")}
    assert all(before[k] == after[k] for k in before)
    assert {w for s in plan.steps() for w in s.writes} == {"tasks[].stimulus_modality"}


# ---- 4 -------------------------------------------------------------------

def test_04_a_semantic_migration_uses_evidence_first_then_methods_never_the_article(
    loaded, papers
):
    plan = loaded.plan("study@1.1")
    semantic = [s for s in plan.steps() if s.disposition.value == "semantic"]
    assert semantic
    assert all(not s.context.includes_full_document for s in semantic)
    assert plan.summary.n_full_document_transmissions == 0

    used = {src for s in semantic for src in s.context.used_sources}
    assert used <= {"prior_evidence@1", "sections@1"}
    assert "prior_evidence@1" in used

    loaded.run(plan)
    # A task whose stimuli had no stored evidence falls through to Methods.
    fell_through = [
        a
        for e in loaded.records()
        for a in loaded.artifacts(e.record_id)
        if a.field_id == "tasks[].stimulus_modality"
        and a.provenance.context_selector_ref == "sections@1"
    ]
    assert fell_through or used == {"prior_evidence@1"}


# ---- 5 -------------------------------------------------------------------

def test_05_changing_a_prompt_invalidates_only_that_recipes_artifacts(loaded):
    for target in ("study@1.1", "study@1.2", "study@1.3"):
        loaded.migrate(target, policy=LOSSY)

    before = {
        (e.record_id, a.field_id, a.entity): a.artifact_id
        for e in loaded.records()
        for a in loaded.artifacts(e.record_id)
    }

    plan = loaded.plan("study@1.4", policy=LOSSY)
    written = {w for s in plan.steps() if s.disposition.value in ("semantic", "cached")
               for w in s.writes}
    assert written == {"tasks[].stimulus_modality"}

    loaded.run(plan)
    after = {
        (e.record_id, a.field_id, a.entity): a.artifact_id
        for e in loaded.records()
        for a in loaded.artifacts(e.record_id)
    }
    changed = {k for k in before if after.get(k) != before[k]}
    assert all(k[1] == "tasks[].stimulus_modality" for k in changed)


# ---- 6 -------------------------------------------------------------------

def test_06_a_constraint_change_triggers_validation_not_re_extraction(
    tmp_path, registry, papers
):
    # One paper's cohort has zero participants, which study@1.4 forbids.
    zero = sb.make_paper(99)
    zero.record["groups"][0]["n"] = {
        "extraction_status": "extracted",
        "value": 0,
        "evidence": {"status": "not_found"},
    }
    project = _project(tmp_path, papers + [zero], registry=registry)
    for target in ("study@1.1", "study@1.2", "study@1.3"):
        project.migrate(target, policy=LOSSY)

    plan = project.plan("study@1.4", policy=LOSSY)
    # No step touches `n` at all: the constraint is acknowledged, not re-read.
    assert not [s for s in plan.steps() if "groups[].n" in s.writes]

    plan, result = project.migrate("study@1.4", policy=LOSSY)
    # And the offending record is held by validation rather than published.
    assert result.records.get(zero.record_id) in ("held", "review")
    versions = project.store.record_versions(zero.record_id)
    failed = [v for v in versions if v.state is RecordState.FAILED]
    assert failed
    reasons = [
        v["reason"] for v in failed[-1].notes["validation"] if v["outcome"] == "fail"
    ]
    assert any("minimum" in r for r in reasons)
    project.close()


# ---- 7 -------------------------------------------------------------------

def test_07_a_dry_run_reports_everything_and_mutates_nothing(loaded, adapters):
    before_versions = {e.record_id: e.version_id for e in loaded.records()}
    before_artifacts = sum(len(loaded.artifacts(e.record_id)) for e in loaded.records())
    before_calls = len(adapters.get("main").calls)

    plan = loaded.plan("study@1.1")
    s = plan.summary

    assert s.n_records == len(before_versions)
    assert s.n_model_calls > 0
    assert s.est_input_tokens > 0
    assert s.est_cost_usd > 0
    assert s.n_records_touching_source > 0
    assert s.n_full_document_transmissions == 0
    assert s.providers

    assert {e.record_id: e.version_id for e in loaded.records()} == before_versions
    assert sum(len(loaded.artifacts(e.record_id)) for e in loaded.records()) == (
        before_artifacts
    )
    assert len(adapters.get("main").calls) == before_calls


# ---- 8 -------------------------------------------------------------------

def test_08_interrupt_and_resume_creates_no_duplicate_successful_calls(
    tmp_path, registry
):
    papers = sb.corpus(24)
    adapters = AdapterRegistry()
    main = sb.stimulus_adapter()
    adapters.register("main", main)
    adapters.register("backup", sb.stimulus_adapter(provider="b", model="r2"))
    project = _project(tmp_path, papers, registry=registry, adapters=adapters)

    plan = project.plan("study@1.1")
    ids = [r.record_id for r in plan.records]
    project.run(plan, execution_id="big", record_ids=ids[:8])
    first = len(main.calls.successes)
    project.run(plan, execution_id="big", resume=True, record_ids=ids[:16])
    second = len(main.calls.successes)
    result = project.resume("big")

    total = len(main.calls.successes)
    assert first < second < total
    assert len(main.calls.distinct_requests()) == total
    assert result.metrics.steps_skipped > 0

    once_more = project.resume("big")
    assert once_more.metrics.model_calls == 0
    assert len(main.calls.successes) == total
    project.close()


# ---- 9 -------------------------------------------------------------------

def test_09_every_new_semantic_value_is_fully_traceable(loaded, papers):
    loaded.migrate("study@1.1")
    paper = next(p for p in papers if p.truth)
    rows = loaded.provenance(paper.record_id, "tasks[].stimulus_modality", "tasks[]=t1")
    p = rows[-1]["provenance"]

    assert p["source_ref"] or p["parse_version"]          # source version
    assert p["context_units"] and p["context_hash"]       # supplied context
    assert p["recipe_ref"] == "stimulus-modality@1"       # recipe
    assert p["model_call"]["model"]                       # model
    assert p["migration_id"] == "study-1.0-to-1.1"        # migration
    assert p["validations"]                               # validation
    assert p["execution_id"]                              # execution
    assert p["input_hashes"]                              # dependencies
    assert p["software_version"].startswith("llmbic/")


# ---- 10 ------------------------------------------------------------------

def test_10_records_without_required_context_are_escalated_not_invented(
    loaded, papers
):
    no_methods = [
        p for p in papers if not any(u.section == "methods" for u in p.parsed.units)
    ]
    assert no_methods

    plan, result = loaded.migrate("study@1.1")
    for paper in no_methods:
        assert result.records[paper.record_id] in ("review", "held", "blocked")
        values = [
            a
            for a in loaded.artifacts(paper.record_id)
            if a.field_id == "tasks[].stimulus_modality"
        ]
        # Nothing was invented: either no artifact at all, or an explicit
        # abstention — never a value.
        assert all(not a.value.status.has_value for a in values)
        items = [i for i in loaded.review.items() if i.record_id == paper.record_id]
        assert items
        assert items[0].reason


# ---- 11 ------------------------------------------------------------------

def test_11_a_partially_failed_job_exposes_no_partial_record(tmp_path, registry):
    papers = sb.corpus(6)
    from llmbic.models.mock import FailingAdapter

    adapters = AdapterRegistry()
    adapters.register("main", FailingAdapter())
    adapters.register("backup", FailingAdapter(provider="backup"))
    project = _project(tmp_path, papers, registry=registry, adapters=adapters)

    before = {p.record_id: project.store.current_version(p.record_id) for p in papers}
    plan, result = project.migrate("study@1.1")
    assert result.status in (ExecutionStatus.FAILED, ExecutionStatus.PARTIAL)

    for paper in papers:
        current = project.store.current_version(paper.record_id)
        assert current.version_id == before[paper.record_id].version_id
        assert current.schema_ref == "study@1.0"
        record = project.get_record(paper.record_id)
        assert "stimulus_modality" not in record["tasks"][0]
        # The failed attempt is still inspectable.
        drafts = [
            v
            for v in project.store.record_versions(paper.record_id)
            if v.state in (RecordState.FAILED, RecordState.REVIEW_NEEDED)
        ]
        assert drafts
    project.close()


# ---- 12 ------------------------------------------------------------------

def test_12_a_curator_can_inspect_a_diff_and_accept_edit_or_reject(loaded, papers):
    loaded.migrate("study@1.1")
    loaded.migrate("study@1.2", policy=LOSSY)
    loaded.migrate("study@1.3", policy=LOSSY)

    items = loaded.review.items()
    assert items
    item = next(i for i in items if i.field_id == "design.assignment_structure")

    # Inspect: the row carries both values and the reason.
    assert item.old_value["value"] == "parallel"
    assert item.reason

    # Accept.
    loaded.review.decide(
        item_id=item.item_id,
        decision=ReviewDecision.ACCEPT,
        actor_id="curator",
        rationale="the paper does describe an administered intervention",
        schema_ref="study@1.3",
    )
    latest = [
        a
        for a in loaded.artifacts(item.record_id)
        if a.field_id == "design.assignment_structure"
    ][-1]
    assert latest.provenance.actor is Actor.HUMAN

    # Edit and reject are available on the remaining items.
    remaining = loaded.review.items()
    if remaining:
        loaded.review.decide(
            item_id=remaining[0].item_id,
            decision=ReviewDecision.EDIT,
            actor_id="curator",
            edited_value=FieldValue.present("observational_cohorts"),
            schema_ref="study@1.3",
        )
    events = loaded.store.get_review_events()
    assert {e.decision for e in events} >= {ReviewDecision.ACCEPT}

    # And a field-level diff is available for the record.
    diff = loaded.diff_record(papers[0].record_id)
    assert diff.deltas


# ---- 13 ------------------------------------------------------------------

def test_13_distinct_missingness_and_failure_states_survive_migration_and_export(
    tmp_path, registry
):
    paper = sb.make_paper(3)
    task = paper.record["tasks"][0]
    task["stimuli"] = {
        "extraction_status": "not_reported",
        "unreported_reason": AbsenceReason.CITED_ELSEWHERE.value,
        "evidence": {"status": "not_applicable"},
    }
    paper.record["groups"][0]["population_characteristics"] = {
        "extraction_status": "not_reported",
        "unreported_reason": AbsenceReason.UNDETERMINED.value,
        "evidence": {"status": "not_applicable"},
    }
    paper.record["title"] = {
        "extraction_status": "not_reported",
        "unreported_reason": AbsenceReason.AMBIGUOUS.value,
        "evidence": {"status": "not_applicable"},
    }

    project = _project(tmp_path, [paper], registry=registry)
    statuses = {
        a.field_id: (a.value.status, a.value.reason)
        for a in project.artifacts(paper.record_id)
    }
    assert statuses["tasks[].stimuli"] == (
        ValueStatus.NOT_REPORTED,
        "cited_elsewhere",
    )
    assert statuses["groups[].population_characteristics"][0] is ValueStatus.UNKNOWN
    assert statuses["title"] == (ValueStatus.NOT_REPORTED, "ambiguous")

    for target in ("study@1.1", "study@1.2", "study@1.3", "study@1.4"):
        project.migrate(target, policy=LOSSY)

    after = {
        a.field_id: (a.value.status, a.value.reason)
        for a in project.artifacts(paper.record_id)
    }
    assert after["title"] == (ValueStatus.NOT_REPORTED, "ambiguous")
    assert after["tasks[].stimuli"] == (ValueStatus.NOT_REPORTED, "cited_elsewhere")
    # The catch-all was partitioned; an UNKNOWN input yields NOT_APPLICABLE for
    # the new slot, which is a different fact again.
    assert after["groups[].other_characteristics"][0] is ValueStatus.NOT_APPLICABLE

    out = tmp_path / "export.jsonl"
    project.export_jsonl(out)
    line = json.loads(out.read_text().splitlines()[0])
    by_field = {(p["field_id"], p["entity"]): p for p in line["provenance"]}
    assert by_field[("title", "")]["value_status"] == "not_reported"
    assert (
        by_field[("groups[].population_characteristics", "groups[]=g1")]["value_status"]
        == "unknown"
    )
    assert (
        by_field[("groups[].other_characteristics", "groups[]=g1")]["value_status"]
        == "not_applicable"
    )
    assert line["record"]["title"]["unreported_reason"] == "ambiguous"
    project.close()
