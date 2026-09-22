"""Asking for more context when the context given is not enough.

A semantic migration may need a second look before it can answer. llmbic lets
a recipe say so, and then walks *the chain the migration already declared* —
one position at a time, never past its end, and never past what the execution
policy permits.
"""

from __future__ import annotations

import dataclasses

import studybed as sb
from llmbic import (
    ContextPolicy,
    ExecutionPolicy,
    Pricing,
    evidence_window,
    full_document,
    prior_evidence,
    sections,
)
from llmbic.context.policy import FallbackMode, OnMissingContext
from llmbic.errors import ErrorCode
from llmbic.models.base import AdapterRegistry, ModelPolicy
from llmbic.models.mock import ScriptedAdapter
from llmbic.recipe import EscalationSpec

ANSWER = {
    "tasks[].stimulus_modality": {
        "status": "present",
        "value": ["visual"],
        "evidence": [],
    }
}
ASKS = {
    "tasks[].stimulus_modality": {
        "status": "unknown",
        "needs_more_context": True,
        "context_request": "the cited span names the set but not the channel",
    }
}


class AsksThenAnswers(ScriptedAdapter):
    """Asks for more context until it is shown a section, then answers.

    A real model does this by reading: a one-clause evidence span may name a
    stimulus set without saying how it was delivered, and the sentence around
    it does.
    """

    def __init__(self, satisfied_by: str = "sections@1", **kwargs):
        super().__init__(**kwargs)
        self.satisfied_by = satisfied_by
        self.seen: list[tuple[str, ...]] = []

    def generate(self, request):
        origins = tuple(u.origin for u in request.context_units)
        self.seen.append(origins)
        self._default = ANSWER if self.satisfied_by in origins else ASKS
        return super().generate(request)


#: A recipe that may ask for more has to be allowed to say so: the answer
#: schema is what constrains the model, so it needs the vocabulary for it.
ESCALATING_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "tasks[].stimulus_modality": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["present", "not_reported", "unknown"],
                },
                "value": {"type": "array", "items": {"type": "string"}},
                "evidence": {"type": "array", "items": {"type": "string"}},
                "needs_more_context": {"type": "boolean"},
                "context_request": {"type": "string"},
            },
            "required": ["status"],
        }
    },
    "required": ["tasks[].stimulus_modality"],
}


def _recipe(
    *,
    enabled: bool = True,
    on_unknown: bool = False,
    max_escalations: int = 2,
    policy: ContextPolicy | None = None,
):
    base = sb.stimulus_recipe("1")
    return dataclasses.replace(
        base,
        version="escalating",
        escalation=EscalationSpec(
            enabled=enabled, on_unknown=on_unknown, max_escalations=max_escalations
        ),
        context_policy=policy or base.context_policy,
        output_schema=ESCALATING_OUTPUT_SCHEMA,
        model_policy=ModelPolicy(adapter="main", pricing=Pricing(3.0, 15.0)),
        validators=(),
    )


def _wire(project, recipe, adapter):
    """Point the 1.0 -> 1.1 migration's step at ``recipe`` and ``adapter``."""

    project.registry.register_recipe(recipe)
    migration = project.registry.migration("study-1.0-to-1.1")
    step = dataclasses.replace(
        migration.steps[0],
        recipe=recipe.ref,
        context=recipe.context_policy,
        validators=(),
    )
    project.registry._migrations["study-1.0-to-1.1"] = migration.with_steps([step])

    adapters = AdapterRegistry()
    adapters.register("main", adapter)
    adapters.register("backup", adapter)
    project.adapters = adapters
    return project


def test_a_step_that_asks_for_more_gets_the_next_declared_source(loaded):
    """Evidence first; when that is not enough, the Methods section."""

    adapter = AsksThenAnswers()
    _wire(loaded, _recipe(), adapter)

    plan, result = loaded.migrate("study@1.1")
    assert result.metrics.model_calls > 0

    # The first request carried evidence spans; a later one carried sections.
    assert any(o == ("prior_evidence@1",) for o in adapter.seen)
    assert any("sections@1" in o for o in adapter.seen)

    answered = [
        a
        for e in loaded.records()
        for a in loaded.artifacts(e.record_id)
        if a.field_id == "tasks[].stimulus_modality" and a.value.status.has_value
    ]
    assert answered
    # The committed value is provenanced to the context that actually produced
    # it, not to the one that was insufficient.
    assert all("sections@1" in a.provenance.context_selector_ref for a in answered)


def test_the_escalation_is_recorded_as_its_own_attempt(loaded):
    _wire(loaded, _recipe(), AsksThenAnswers())
    plan, result = loaded.migrate("study@1.1")

    attempts = loaded.store.get_attempts(result.execution_id)
    requests = [a for a in attempts if a.get("outcome") == "context_requested"]
    assert requests
    assert requests[0]["tried"] == ["prior_evidence@1"]
    assert requests[0]["note"].startswith("the cited span names the set")
    assert requests[0]["exhausted"] is False


def test_the_chain_is_a_ceiling_not_a_suggestion(loaded):
    """A model that is never satisfied does not get the article."""

    never = ScriptedAdapter(default=ASKS)
    _wire(loaded, _recipe(), never)

    plan, result = loaded.migrate("study@1.1")
    assert result.metrics.steps_review > 0

    items = loaded.review.items()
    assert items
    assert any("asked for more context" in i.reason for i in items)

    # Nothing was invented, and nothing was published.
    values = [
        a
        for e in loaded.records()
        for a in loaded.artifacts(e.record_id)
        if a.field_id == "tasks[].stimulus_modality"
    ]
    assert all(not a.value.status.has_value for a in values)


def test_max_escalations_bounds_the_walk(loaded):
    long_chain = ContextPolicy(
        sequence=(
            prior_evidence(),
            evidence_window(150),
            sections("methods"),
            sections("results"),
            sections("abstract"),
        ),
        full_document_fallback=FallbackMode.FORBIDDEN,
        on_missing_context=OnMissingContext.REVIEW,
    )
    never = ScriptedAdapter(default=ASKS)
    _wire(loaded, _recipe(max_escalations=1, policy=long_chain), never)

    plan, result = loaded.migrate("study@1.1")
    attempts = loaded.store.get_attempts(result.execution_id)
    requests = [a for a in attempts if a.get("outcome") == "context_requested"]
    assert requests
    # One escalation, then the ceiling — not all the way down a five-step chain.
    assert max(a["level"] for a in requests) == 1
    assert any(a["exhausted"] for a in requests)


def test_escalation_never_reaches_a_forbidden_full_document(loaded):
    greedy = ContextPolicy(
        sequence=(prior_evidence(), sections("methods"), full_document()),
        full_document_fallback=FallbackMode.ALLOWED,
        on_missing_context=OnMissingContext.REVIEW,
    )
    never = ScriptedAdapter(default=ASKS)
    _wire(loaded, _recipe(max_escalations=5, policy=greedy), never)

    # The policy names the full document; the *execution* policy does not
    # permit it, so the step is blocked rather than escalating into it.
    plan, result = loaded.migrate(
        "study@1.1", policy=ExecutionPolicy(allow_full_document=False)
    )
    states = loaded.store.get_step_states(result.execution_id)
    blocked = [
        s
        for s in states
        if s.detail.get("code") == ErrorCode.CONTEXT_POLICY_FORBIDS.value
    ]
    assert blocked
    assert "full document" in blocked[0].detail["reason"]


def test_permitting_the_full_document_lets_the_last_step_be_taken(loaded):
    greedy = ContextPolicy(
        sequence=(prior_evidence(), sections("methods"), full_document()),
        full_document_fallback=FallbackMode.ALLOWED,
        on_missing_context=OnMissingContext.REVIEW,
    )
    adapter = AsksThenAnswers(satisfied_by="full_document@1")
    _wire(loaded, _recipe(max_escalations=5, policy=greedy), adapter)

    plan, result = loaded.migrate(
        "study@1.1", policy=ExecutionPolicy(allow_full_document=True)
    )
    assert any("full_document@1" in o for o in adapter.seen)
    assert result.published


def test_each_escalation_level_is_cached_separately(loaded):
    adapter = AsksThenAnswers()
    _wire(loaded, _recipe(), adapter)
    plan, result = loaded.migrate("study@1.1")
    calls = len(adapter.calls.successes)
    assert calls > 0

    # Re-running the same plan replays from the cache at every level, so the
    # insufficient first ask is not paid for again either.
    second = loaded.run(plan, execution_id="again")
    assert second.metrics.model_calls == 0
    assert second.metrics.cache_hits > 0
    assert len(adapter.calls.successes) == calls


def test_escalation_is_off_unless_the_recipe_asks_for_it(loaded):
    """A model saying `unknown` is an abstention, not a request, by default."""

    never = ScriptedAdapter(default=ASKS)
    _wire(loaded, _recipe(enabled=False), never)

    plan, result = loaded.migrate("study@1.1")
    attempts = loaded.store.get_attempts(result.execution_id)
    assert not [a for a in attempts if a.get("outcome") == "context_requested"]
    # The abstention is recorded as an abstention.
    values = [
        a
        for e in loaded.records()
        for a in loaded.artifacts(e.record_id)
        if a.field_id == "tasks[].stimulus_modality"
    ]
    assert values
    assert all(not a.value.status.has_value for a in values)


def test_on_unknown_treats_an_abstention_as_a_request(loaded):
    bare_unknown = {"tasks[].stimulus_modality": {"status": "unknown"}}

    class AbstainsThenAnswers(ScriptedAdapter):
        def generate(self, request):
            origins = tuple(u.origin for u in request.context_units)
            self._default = ANSWER if "sections@1" in origins else bare_unknown
            return super().generate(request)

    adapter = AbstainsThenAnswers()
    _wire(loaded, _recipe(on_unknown=True), adapter)

    plan, result = loaded.migrate("study@1.1")
    attempts = loaded.store.get_attempts(result.execution_id)
    assert [a for a in attempts if a.get("outcome") == "context_requested"]
    assert result.published


def test_escalation_is_part_of_the_recipe_hash(loaded):
    base = sb.stimulus_recipe("1")
    escalating = dataclasses.replace(
        base, escalation=EscalationSpec(enabled=True)
    )
    assert base.recipe_hash() != escalating.recipe_hash()
