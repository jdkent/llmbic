"""Evaluation of semantic migrations and rollout gates (§15.5, FR-VAL-008/010)."""

from __future__ import annotations

import pytest

import studybed as sb
from llmbic import (
    GoldCorpus,
    GoldValue,
    RolloutGate,
    ValueStatus,
    evaluate,
    gold_from_artifacts,
)
from llmbic.errors import LlmbicError


@pytest.fixture
def gold(papers):
    """A frozen adjudicated sample: what a curator says each paper's task is."""

    values = []
    for paper in papers:
        values.append(
            GoldValue(
                record_id=paper.record_id,
                field_id="tasks[].stimulus_modality",
                entity="tasks[]=t1",
                status=ValueStatus.PRESENT if paper.truth else ValueStatus.NOT_EXTRACTED,
                value=[paper.truth] if paper.truth else None,
                note="adjudicated from the Methods section",
            )
        )
    return GoldCorpus(name="stimulus-modality-v1", values=tuple(values))


def test_a_gold_corpus_round_trips_through_a_frozen_file(gold, tmp_path):
    path = gold.save(tmp_path / "gold.jsonl")
    loaded = GoldCorpus.load(path)
    assert loaded.name == gold.name
    assert loaded.values == gold.values
    assert loaded.record_ids() == gold.record_ids()


def test_an_empty_gold_file_is_refused(tmp_path):
    (tmp_path / "empty.jsonl").write_text("")
    with pytest.raises(LlmbicError):
        GoldCorpus.load(tmp_path / "empty.jsonl")


def test_evaluation_scores_the_migration_that_will_actually_run(loaded, gold):
    report = evaluate(loaded, gold, "study@1.1")
    score = report.fields["tasks[].stimulus_modality"]

    assert score.n == len(gold.values)
    assert score.precision == 1.0  # the rule-based model never guesses wrong
    assert score.recall == 1.0
    assert score.evidence_support == 1.0
    assert report.model_calls > 0
    assert "precision" in report.render()


def test_abstention_is_measured_not_punished_as_an_error(loaded, gold):
    report = evaluate(loaded, gold, "study@1.1")
    score = report.fields["tasks[].stimulus_modality"]
    # The paper with no Methods section is a true negative, not a miss: gold
    # says there is nothing to find and the pass said so too.
    assert score.true_negative >= 1
    assert score.missed == 0
    assert 0 < score.abstention_rate < 1


def test_a_wrong_answer_is_counted_as_wrong_not_missing(loaded, papers):
    """Gold disagreeing with the model shows up as precision, not recall."""

    wrong = GoldCorpus(
        name="deliberately-wrong",
        values=tuple(
            GoldValue(
                record_id=p.record_id,
                field_id="tasks[].stimulus_modality",
                entity="tasks[]=t1",
                status=ValueStatus.PRESENT,
                value=["olfactory"],
            )
            for p in papers
            if p.truth
        ),
    )
    report = evaluate(loaded, wrong, "study@1.1")
    score = report.fields["tasks[].stimulus_modality"]
    assert score.wrong_value > 0
    assert score.true_positive == 0
    assert score.precision == 0.0


def test_also_acceptable_answers_count_as_correct(loaded, papers):
    paper = next(p for p in papers if p.truth == "visual")
    forgiving = GoldCorpus(
        name="forgiving",
        values=(
            GoldValue(
                record_id=paper.record_id,
                field_id="tasks[].stimulus_modality",
                entity="tasks[]=t1",
                status=ValueStatus.PRESENT,
                value=["audiovisual"],
                also_acceptable=(["visual"],),
            ),
        ),
    )
    report = evaluate(loaded, forgiving, "study@1.1")
    assert report.fields["tasks[].stimulus_modality"].true_positive == 1


def test_the_change_rate_is_measured_against_the_previous_extraction(loaded, gold):
    report = evaluate(loaded, gold, "study@1.1")
    score = report.fields["tasks[].stimulus_modality"]
    # The field is new, so nothing to compare against yet.
    assert score.change_rate in (None, 0.0)


def test_a_gate_blocks_a_rollout_when_precision_is_too_low(loaded, papers):
    wrong = GoldCorpus(
        name="wrong",
        values=tuple(
            GoldValue(
                record_id=p.record_id,
                field_id="tasks[].stimulus_modality",
                entity="tasks[]=t1",
                status=ValueStatus.PRESENT,
                value=["olfactory"],
            )
            for p in papers
            if p.truth
        ),
    )
    report = evaluate(
        loaded,
        wrong,
        "study@1.1",
        gates=[RolloutGate(metric="precision", min_value=0.9)],
    )
    assert not report.passed
    assert "precision" in report.gate_failures[0]
    assert "GATES FAILED" in report.render()


def test_a_gate_passes_when_the_numbers_are_right(loaded, gold):
    report = evaluate(
        loaded,
        gold,
        "study@1.1",
        gates=[
            RolloutGate(metric="precision", min_value=0.9),
            RolloutGate(metric="evidence_support", min_value=0.8),
            RolloutGate(metric="schema_validity", min_value=1.0),
        ],
    )
    assert report.passed
    assert "all gates passed" in report.render()


def test_a_cost_gate_can_stop_an_expensive_rollout(loaded, gold):
    report = evaluate(
        loaded, gold, "study@1.1", gates=[RolloutGate(metric="cost_usd", max_value=-1.0)]
    )
    assert not report.passed
    assert "cost_usd" in report.gate_failures[0]


def test_a_review_rate_gate_sees_escalations(loaded, gold):
    report = evaluate(
        loaded, gold, "study@1.1", gates=[RolloutGate(metric="review_rate", max_value=0.0)]
    )
    # One paper in the corpus has no Methods section, so it escalates.
    assert not report.passed
    assert "review_rate" in report.gate_failures[0]


def test_a_gate_scoped_to_one_field_ignores_the_others(loaded, gold):
    report = evaluate(
        loaded,
        gold,
        "study@1.1",
        gates=[
            RolloutGate(
                metric="precision", field_id="tasks[].stimulus_modality", min_value=0.5
            ),
            RolloutGate(metric="precision", field_id="not_a_field", min_value=2.0),
        ],
    )
    # The gate on an unmeasured field cannot fail; the measured one passes.
    assert report.passed


def test_evaluation_touches_only_the_sampled_records(loaded, gold, papers):
    sample = GoldCorpus(name="one", values=(gold.values[0],))
    report = evaluate(loaded, sample, "study@1.1")
    assert report.n_records == 1
    others = [p for p in papers if p.record_id != sample.values[0].record_id]
    for paper in others:
        assert loaded.store.current_version(paper.record_id).schema_ref == "study@1.0"


def test_a_gold_corpus_can_be_frozen_from_accepted_values(loaded):
    loaded.migrate("study@1.1")
    artifacts = [
        a for e in loaded.records() for a in loaded.artifacts(e.record_id)
    ]
    frozen = gold_from_artifacts("bootstrap", artifacts, ["tasks[].stimulus_modality"])
    assert frozen.field_ids() == ["tasks[].stimulus_modality"]
    assert all(g.note.startswith("from artifact") for g in frozen.values)


def test_evaluating_an_empty_corpus_is_refused(loaded):
    with pytest.raises(LlmbicError):
        evaluate(loaded, GoldCorpus(name="none"), "study@1.1")


# ---- shadow mode (FR-LLM-010) -------------------------------------------

def test_shadow_mode_records_disagreement_without_committing_it(loaded):
    from llmbic.models.base import AdapterRegistry

    loaded.registry.register_recipe(sb.stimulus_recipe("2"))
    loaded.shadow = {"stimulus-modality@1": "stimulus-modality@2"}

    adapters = AdapterRegistry()
    adapters.register("main", sb.stimulus_adapter())
    adapters.register("backup", sb.stimulus_adapter(provider="b", model="r2"))
    loaded.adapters = adapters

    plan, result = loaded.migrate("study@1.1")
    attempts = loaded.store.get_attempts(result.execution_id)
    shadows = [a for a in attempts if a.get("outcome") == "shadow"]
    assert shadows
    assert all(a["shadow_recipe"] == "stimulus-modality@2" for a in shadows)
    assert all("agrees" in a for a in shadows)

    # Nothing the shadow said was committed.
    for entry in loaded.records():
        for a in loaded.artifacts(entry.record_id):
            if a.field_id == "tasks[].stimulus_modality":
                assert a.provenance.recipe_ref == "stimulus-modality@1"


def test_a_shadow_failure_is_recorded_and_does_not_fail_the_step(loaded):
    import dataclasses

    from llmbic.models.base import AdapterRegistry
    from llmbic.models.mock import FailingAdapter

    broken = dataclasses.replace(
        sb.stimulus_recipe("2"),
        version="2-candidate",
        model_policy=dataclasses.replace(
            sb.stimulus_recipe("2").model_policy, adapter="broken", fallbacks=()
        ),
    )
    loaded.registry.register_recipe(broken)
    loaded.shadow = {"stimulus-modality@1": broken.ref}

    adapters = AdapterRegistry()
    adapters.register("main", sb.stimulus_adapter())
    adapters.register("backup", sb.stimulus_adapter(provider="b", model="r2"))
    adapters.register("broken", FailingAdapter())
    loaded.adapters = adapters

    plan, result = loaded.migrate("study@1.1")
    assert result.published
    attempts = loaded.store.get_attempts(result.execution_id)
    assert any(a.get("outcome") == "shadow_error" for a in attempts)
