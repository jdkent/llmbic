"""Review queue, curator decisions and field diffs (§8.9, FR-VAL-004..007)."""

from __future__ import annotations

import json

import pytest

import studybed as sb
from llmbic import ExecutionPolicy, FieldValue, ReviewDecision, ValueStatus
from llmbic.diffing import DeltaKind, diff_records
from llmbic.provenance import Actor
from llmbic.review import review_report

LOSSY = ExecutionPolicy(allow_lossy=True)


@pytest.fixture
def escalated(loaded):
    """A project with an open review item on the vocabulary remap."""

    loaded.migrate("study@1.1")
    loaded.migrate("study@1.2", policy=LOSSY)
    loaded.migrate("study@1.3", policy=LOSSY)
    return loaded


def test_the_vocabulary_remap_escalates_only_what_it_cannot_settle(escalated):
    """Scenario 8: remap what is settled, escalate the rest."""

    items = escalated.review.items()
    assert items
    assignment = [i for i in items if i.field_id == "design.assignment_structure"]
    assert assignment
    assert "does not settle" in assignment[0].reason

    # And the settled ones were remapped with no human involved.
    remapped = [
        a
        for e in escalated.records()
        for a in escalated.artifacts(e.record_id)
        if a.field_id == "design.assignment_structure"
        and a.value.value == "observational_cohorts"
    ]
    assert remapped
    assert remapped[0].provenance.actor is Actor.MIGRATION
    assert remapped[0].provenance.notes["rule"] == "no arms, not randomized"


def test_replaying_a_migration_does_not_duplicate_the_queue(escalated):
    """A curator gets one row per open question, not one per run."""

    before = {i.item_id for i in escalated.review.items()}
    assert before
    counted = [(i.record_id, i.field_id, i.entity) for i in escalated.review.items()]
    assert len(counted) == len(set(counted))

    escalated.migrate("study@1.3", policy=LOSSY)
    escalated.migrate("study@1.3", policy=LOSSY)
    after = {i.item_id for i in escalated.review.items()}
    assert after == before


def test_a_queue_row_carries_everything_needed_to_decide(escalated):
    """FR-VAL-005."""

    item = next(
        i for i in escalated.review.items() if i.field_id == "design.assignment_structure"
    )
    assert item.old_value["value"] == "parallel"
    assert item.proposed_value is not None
    assert item.migration_id == "study-1.2-to-1.3"
    assert item.step_id == "remap_assignment"
    assert item.reason


def test_export_and_import_round_trip(escalated, tmp_path):
    """Decision 18.6: JSONL in, JSONL out."""

    path = tmp_path / "queue.jsonl"
    n = escalated.review.export_jsonl(path)
    assert n == len(escalated.review.items())

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert all(row["decision"] == "" for row in rows)

    for row in rows:
        row["decision"] = "accept"
        row["rationale"] = "checked against the paper"
    path.write_text("\n".join(json.dumps(r) for r in rows))

    events = escalated.review.import_jsonl(
        path, actor_id="curator@example.org", schema_ref="study@1.3"
    )
    assert len(events) == len(rows)
    assert all(e.actor_id == "curator@example.org" for e in events)
    assert not escalated.review.items(state="open")


def test_a_row_with_no_decision_is_skipped(escalated, tmp_path):
    path = tmp_path / "queue.jsonl"
    escalated.review.export_jsonl(path)
    assert escalated.review.import_jsonl(path, actor_id="c") == []
    assert escalated.review.items(state="open")


def test_accepting_commits_the_proposed_value_as_a_human_artifact(escalated):
    item = next(
        i for i in escalated.review.items() if i.field_id == "design.assignment_structure"
    )
    escalated.review.decide(
        item_id=item.item_id,
        decision=ReviewDecision.ACCEPT,
        actor_id="curator",
        rationale="the paper does describe an administered intervention",
        schema_ref="study@1.3",
    )
    latest = [
        a
        for a in escalated.artifacts(item.record_id)
        if a.field_id == "design.assignment_structure"
    ][-1]
    assert latest.provenance.actor is Actor.HUMAN
    assert latest.provenance.actor_id == "curator"
    assert latest.value.status is ValueStatus.PRESENT


def test_editing_records_the_curators_own_value(escalated):
    item = escalated.review.items()[0]
    escalated.review.decide(
        item_id=item.item_id,
        decision=ReviewDecision.EDIT,
        actor_id="curator",
        edited_value=FieldValue.present("crossover"),
        rationale="the design section says the order was counterbalanced",
        schema_ref="study@1.3",
    )
    events = escalated.store.get_review_events(item.record_id)
    assert events[-1].decision is ReviewDecision.EDIT
    assert events[-1].edited_value.value == "crossover"


def test_editing_without_a_value_is_refused(escalated):
    from llmbic.errors import LlmbicError

    item = escalated.review.items()[0]
    with pytest.raises(LlmbicError, match="carries no edited_value"):
        escalated.review.decide(
            item_id=item.item_id, decision=ReviewDecision.EDIT, actor_id="c"
        )


def test_rejecting_leaves_the_value_alone_but_records_the_decision(escalated):
    item = escalated.review.items()[0]
    before = len(escalated.artifacts(item.record_id, latest=False))
    escalated.review.decide(
        item_id=item.item_id,
        decision=ReviewDecision.REJECT,
        actor_id="curator",
        rationale="not enough in the paper to say",
    )
    after = len(escalated.artifacts(item.record_id, latest=False))
    assert after == before
    assert escalated.store.get_review_events(item.record_id)[-1].decision is (
        ReviewDecision.REJECT
    )


def test_requesting_expanded_context_is_a_distinct_decision(escalated):
    item = escalated.review.items()[0]
    escalated.review.decide(
        item_id=item.item_id,
        decision=ReviewDecision.REQUEST_CONTEXT,
        actor_id="curator",
        rationale="the Methods section alone does not say",
    )
    rows = escalated.store.get_review_items(state="context_requested")
    assert rows


def test_a_human_correction_is_never_indistinguishable_from_model_output(escalated):
    """FR-PROV-006."""

    item = escalated.review.items()[0]
    escalated.review.decide(
        item_id=item.item_id,
        decision=ReviewDecision.EDIT,
        actor_id="curator",
        edited_value=FieldValue.present("factorial"),
        schema_ref="study@1.3",
    )
    actors = {a.provenance.actor for a in escalated.artifacts(item.record_id, latest=False)}
    assert Actor.HUMAN in actors
    assert Actor.MIGRATION in actors or Actor.IMPORT in actors


def test_accepting_a_legacy_value_records_the_decision_without_rewriting_history(
    loaded, papers
):
    """FR-DEP-007."""

    record_id = papers[0].record_id
    accepted = loaded.review.accept_legacy_value(
        record_id=record_id,
        field_id="tasks[].stimuli",
        entity="tasks[]=t1",
        recipe_ref="task@2",
        actor_id="maintainer",
        rationale="the reworded description does not change what we asked for",
        schema_ref="study@1.1",
    )
    assert accepted.provenance.accepted_under == "task@2"
    assert accepted.provenance.actor is Actor.HUMAN
    original = loaded.store.get_artifact(accepted.derived_from)
    assert original is not None and original.provenance.actor is not Actor.HUMAN


def test_a_review_report_summarises_the_queue(escalated):
    report = review_report(escalated.review)
    assert report["total"] > 0
    assert "open" in report["by_state"]


# ---- field-level diffs ---------------------------------------------------

def test_a_field_diff_shows_what_changed_and_what_did_not(loaded, papers):
    """FR-VAL-004 / acceptance criterion 12."""

    record_id = papers[0].record_id
    before = loaded.artifacts(record_id)
    loaded.migrate("study@1.1")
    after = loaded.artifacts(record_id)

    diff = diff_records(
        before,
        after,
        record_id=record_id,
        from_schema=sb.normalized("1.0"),
        to_schema=sb.normalized("1.1"),
    )
    kinds = {d.kind for d in diff.deltas}
    assert DeltaKind.ADDED in kinds
    assert DeltaKind.MOVED in kinds or DeltaKind.UNCHANGED in kinds

    moved = [d for d in diff.deltas if d.kind is DeltaKind.MOVED]
    assert moved
    assert moved[0].before_path == "tasks[].response_mode"
    assert moved[0].after_path == "tasks[].response_modality"
    assert moved[0].detail["evidence_preserved"] is True


def test_diff_between_two_stored_versions(loaded, papers):
    loaded.migrate("study@1.1")
    diff = loaded.diff_record(papers[0].record_id)
    assert diff.from_schema == "study@1.0"
    assert diff.to_schema == "study@1.1"
    assert diff.counts()


def test_diff_renders(loaded, papers):
    loaded.migrate("study@1.1")
    text = loaded.diff_record(papers[0].record_id).render()
    assert "study@1.0 -> study@1.1" in text


def test_a_migration_report_covers_change_rate_and_cost(loaded):
    plan, result = loaded.migrate("study@1.1")
    report = loaded.report(result.execution_id)
    assert report.n_records > 0
    assert report.missingness
    assert 0.0 <= report.change_rate <= 1.0
    assert "model calls" in report.render()
