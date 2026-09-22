"""Context policies: ordered fallbacks, budgets, privacy, cost (§8.4)."""

from __future__ import annotations

import pytest

import studybed as sb
from llmbic import (
    ContextBudget,
    ContextPolicy,
    PrivacyPolicy,
    evidence_window,
    full_document,
    no_context,
    prior_evidence,
    retrieve,
    raw_document,
    sections,
    structured_fields,
    units_of_kind,
)
from llmbic.context.policy import FallbackMode, OnBudget, OnMissingContext, policy_from_spec
from llmbic.context.resolver import (
    LexicalRetriever,
    SelectionInput,
    check_provider,
    register_retriever,
    resolve_context,
)
from llmbic.errors import ErrorCode, LlmbicError
from llmbic.provenance import FieldArtifact, FieldProvenance
from llmbic.source import EvidenceReference, EvidenceSpan
from llmbic.values import FieldValue


@pytest.fixture
def paper():
    return sb.make_paper(1, modality="visual")


@pytest.fixture
def selection(paper):
    stimuli = FieldArtifact(
        record_id=paper.record_id,
        field_id="tasks[].stimuli",
        value=FieldValue.present(sb.MODALITY_CUES["visual"]),
        evidence=(
            EvidenceReference(
                source_id=paper.source.source_id,
                source_version="v1",
                parse_version="parse@1",
                spans=(EvidenceSpan("methods:3", 0, 20, sb.MODALITY_CUES["visual"][:20]),),
                locator="model_quote",
            ),
        ),
        provenance=FieldProvenance(schema_version="study@1.0"),
    )
    return SelectionInput(
        record_id=paper.record_id,
        parsed=paper.parsed,
        source=paper.source,
        prior_evidence=stimuli.evidence,
        prior_fields={"tasks[].stimuli": stimuli},
    )


# ---- ordered fallbacks (FR-CTX-003) --------------------------------------

def test_the_first_satisfying_source_wins(selection):
    policy = ContextPolicy(sequence=(prior_evidence(), sections("methods")))
    resolution = resolve_context(policy, selection)
    assert resolution.satisfied
    assert resolution.used_sources == ("prior_evidence@1",)
    assert all(u.kind == "evidence_span" for u in resolution.units)


def test_the_chain_falls_through_when_a_source_yields_nothing(selection):
    selection.prior_evidence = ()
    policy = ContextPolicy(sequence=(prior_evidence(), sections("methods")))
    resolution = resolve_context(policy, selection)
    assert resolution.satisfied
    assert resolution.used_sources == ("sections@1",)
    assert resolution.attempted == ("prior_evidence@1", "sections@1")


def test_a_forbidden_full_document_is_never_reached(selection):
    """Acceptance criterion 4: evidence first, then Methods, never the article."""

    selection.prior_evidence = ()
    policy = ContextPolicy(
        sequence=(prior_evidence(), sections("discussion")),
        full_document_fallback=FallbackMode.FORBIDDEN,
    )
    resolution = resolve_context(policy, selection)
    assert not resolution.satisfied
    assert not resolution.includes_full_document
    assert resolution.blocked_code == ErrorCode.CONTEXT_UNAVAILABLE.value


def test_an_allowed_full_document_is_appended_to_the_chain(selection):
    selection.prior_evidence = ()
    policy = ContextPolicy(
        sequence=(prior_evidence(), sections("discussion")),
        full_document_fallback=FallbackMode.ALLOWED,
    )
    resolution = resolve_context(policy, selection)
    assert resolution.satisfied
    assert resolution.includes_full_document
    assert len(resolution.units) == len(selection.parsed.units)


def test_accumulate_concatenates_the_whole_chain(selection):
    policy = ContextPolicy(sequence=(prior_evidence(), sections("methods")), accumulate=True)
    resolution = resolve_context(policy, selection)
    kinds = {u.kind for u in resolution.units}
    assert kinds == {"evidence_span", "sentence"}


def test_no_context_satisfies_a_policy_that_wants_none(selection):
    policy = ContextPolicy(sequence=(no_context(),))
    resolution = resolve_context(policy, selection)
    assert resolution.satisfied
    assert resolution.units == ()


def test_an_empty_policy_with_review_on_missing_is_not_satisfied(selection):
    resolution = resolve_context(ContextPolicy(), selection)
    assert not resolution.satisfied


def test_proceed_lets_a_step_run_with_no_context(selection):
    policy = ContextPolicy(on_missing_context=OnMissingContext.PROCEED)
    assert resolve_context(policy, selection).satisfied


# ---- selectors -----------------------------------------------------------

def test_structured_fields_render_values_with_their_status(selection):
    policy = ContextPolicy(sequence=(structured_fields("tasks[].stimuli"),))
    resolution = resolve_context(policy, selection)
    assert resolution.units[0].kind == "structured_field"
    assert "status=present" in resolution.units[0].text


def test_evidence_window_widens_the_span(selection):
    narrow = resolve_context(ContextPolicy(sequence=(prior_evidence(0),)), selection)
    wide = resolve_context(ContextPolicy(sequence=(evidence_window(200),)), selection)
    assert wide.n_chars > narrow.n_chars


def test_unit_kinds_select_by_kind(paper, selection):
    resolution = resolve_context(
        ContextPolicy(sequence=(units_of_kind("sentence"),)), selection
    )
    assert resolution.satisfied
    assert {u.kind for u in resolution.units} == {"sentence"}


def test_lexical_retrieval_is_deterministic_and_ranked(selection):
    policy = ContextPolicy(sequence=(retrieve("IAPS photographs screen", top_k=2),))
    first = resolve_context(policy, selection)
    second = resolve_context(policy, selection)
    assert first.unit_ids == second.unit_ids
    assert len(first.units) <= 2
    assert "IAPS" in first.units[0].text


def test_a_billable_selector_is_skipped_during_a_dry_run(selection):
    class ModelReranker(LexicalRetriever):
        billable = True

    register_retriever("reranker", ModelReranker())
    policy = ContextPolicy(
        sequence=(retrieve("IAPS", top_k=1, retriever="reranker"), sections("methods"))
    )
    dry = resolve_context(policy, selection, allow_model_selectors=False)
    assert dry.used_sources == ("sections@1",)

    live = resolve_context(policy, selection, allow_model_selectors=True)
    assert live.used_sources == ("retrieve@1",)


def test_an_unknown_retriever_is_a_configuration_error(selection):
    policy = ContextPolicy(sequence=(retrieve("x", retriever="nope"),))
    with pytest.raises(LlmbicError) as exc:
        resolve_context(policy, selection, allow_model_selectors=True)
    assert exc.value.code is ErrorCode.CONFIG_INVALID


def test_raw_document_units_carry_the_source_hash_not_the_text(paper, selection):
    policy = ContextPolicy(
        sequence=(raw_document(),), full_document_fallback=FallbackMode.ALLOWED
    )
    resolution = resolve_context(policy, selection)
    assert resolution.units[0].kind == "raw_document"
    assert resolution.units[0].unit_hash == paper.source.content_hash
    assert resolution.units[0].text == ""


# ---- budgets (FR-CTX-006) ------------------------------------------------

def test_a_character_budget_truncates_in_selection_order(selection):
    policy = ContextPolicy(
        sequence=(sections("methods", "results", "abstract"),),
        budget=ContextBudget(max_chars=60),
    )
    resolution = resolve_context(policy, selection)
    assert resolution.n_chars <= 60
    assert resolution.truncated


def test_a_token_budget_is_estimated_from_characters(selection):
    policy = ContextPolicy(
        sequence=(sections("methods"),), budget=ContextBudget(max_input_tokens=5)
    )
    resolution = resolve_context(policy, selection)
    assert resolution.estimate_tokens() <= 5


def test_on_budget_skip_falls_through_to_the_next_source(selection):
    policy = ContextPolicy(
        sequence=(sections("methods"), structured_fields("tasks[].stimuli")),
        budget=ContextBudget(max_chars=200),
        on_budget=OnBudget.SKIP,
    )
    resolution = resolve_context(policy, selection)
    assert resolution.used_sources == ("structured_fields@1",)


def test_on_budget_fail_blocks_with_a_budget_code(selection):
    policy = ContextPolicy(
        sequence=(sections("methods"),),
        budget=ContextBudget(max_chars=10),
        on_budget=OnBudget.FAIL,
    )
    resolution = resolve_context(policy, selection)
    assert not resolution.satisfied
    assert resolution.blocked_code == ErrorCode.CONTEXT_BUDGET_EXCEEDED.value


def test_a_unit_budget_caps_the_number_of_units(selection):
    policy = ContextPolicy(
        sequence=(full_document(),),
        full_document_fallback=FallbackMode.ALLOWED,
        budget=ContextBudget(max_units=2),
    )
    assert len(resolve_context(policy, selection).units) == 2


# ---- privacy (FR-CTX-009, NFR-SEC-003) -----------------------------------

def test_units_carrying_a_forbidden_label_never_enter_the_context(paper):
    from llmbic.source import DocumentUnit, ParsedSource, UnitKind

    units = (
        DocumentUnit("ok:0", UnitKind.SECTION, "public methods", "methods", 0, 0),
        DocumentUnit(
            "phi:1",
            UnitKind.SECTION,
            "participant names",
            "methods",
            20,
            1,
            metadata={"labels": ["phi"]},
        ),
    )
    parsed = ParsedSource("s", "v1", "parse@1", units)
    policy = ContextPolicy(
        sequence=(sections("methods"),),
        privacy=PrivacyPolicy(forbidden_unit_labels=("phi",)),
    )
    resolution = resolve_context(
        policy, SelectionInput(record_id="r", parsed=parsed, source=paper.source)
    )
    assert resolution.unit_ids == ("ok:0",)


def test_a_provider_that_retains_data_is_refused_when_the_policy_forbids_it():
    policy = ContextPolicy(
        privacy=PrivacyPolicy(require_provider_attributes={"retains_data": False})
    )
    with pytest.raises(LlmbicError) as exc:
        check_provider(policy, "hosted", {"retains_data": True})
    assert exc.value.code is ErrorCode.PRIVACY_POLICY_FORBIDS


def test_an_allow_list_excludes_everything_else():
    policy = ContextPolicy(privacy=PrivacyPolicy(allowed_providers=("local",)))
    check_provider(policy, "local", {})
    with pytest.raises(LlmbicError):
        check_provider(policy, "hosted", {})


# ---- identity and serialisation ------------------------------------------

def test_the_context_hash_covers_the_units_supplied(selection):
    a = resolve_context(ContextPolicy(sequence=(sections("methods"),)), selection)
    b = resolve_context(ContextPolicy(sequence=(sections("results"),)), selection)
    assert a.context_hash() != b.context_hash()


def test_changing_a_selector_version_changes_its_identity():
    from llmbic.context.policy import ContextSource

    one = ContextSource("sections", {"names": ["methods"]}, version="1")
    two = ContextSource("sections", {"names": ["methods"]}, version="2")
    assert one.selector_hash() != two.selector_hash()


def test_policy_round_trips_through_its_canonical_form():
    policy = sb.EVIDENCE_FIRST_METHODS
    clone = ContextPolicy.from_canonical(policy.to_canonical())
    assert clone.policy_hash == policy.policy_hash


def test_the_yaml_shape_from_the_requirements_loads():
    policy = policy_from_spec(
        {
            "sequence": [
                "prior_evidence",
                {"sections": ["methods", "supplement"]},
                {"retrieve": {"query": "analysis software package and version", "top_k": 8}},
            ],
            "full_document_fallback": "forbidden",
            "max_input_tokens": 12000,
        }
    )
    assert [s.kind for s in policy.sequence] == ["prior_evidence", "sections", "retrieve"]
    assert policy.sequence[1].params["names"] == ["methods", "supplement"]
    assert policy.sequence[2].params["top_k"] == 8
    assert policy.budget.max_input_tokens == 12000
    assert policy.full_document_fallback is FallbackMode.FORBIDDEN
    assert not policy.permits_full_document()
