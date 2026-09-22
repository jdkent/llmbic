"""A miniature of neurostuff/study_schema, used as the testing bed.

Every schema change below is taken from a real commit in that repository, so
the migrations exercised here are the ones a maintainer actually writes:

``1.0 -> 1.1``  commit 5f282f0, *"Track stimulus modality, and name it
    symmetrically with response"*.  ``response_mode`` is renamed to
    ``response_modality`` (a pure rename, identity preserved), a new
    ``stimulus_modality`` slot is added that must be read out of the source,
    and ``stimuli``'s description is rewritten without its shape changing.

``1.1 -> 1.2``  commits 6b20e30 / 653cd5b, *"Derive is_healthy instead of
    asking"* and *"Set aside the cohort traits no query can filter on"*.
    ``is_healthy`` becomes a derived field computed from
    ``medical_condition``; ``population_characteristics`` is partitioned in
    place, with the unremarkable traits moving to a new deterministic
    ``other_characteristics``.

``1.2 -> 1.3``  commit dc5752d, *"Give the design vocabulary a value for
    cohorts nobody assigned"*.  ``AssignmentStructure`` gains
    ``observational_cohorts`` and ``parallel`` narrows its meaning, so stored
    ``parallel`` values must be remapped — deterministically where the record
    settles it, and by escalation where it does not.

``1.3 -> 1.4``  a prompt revision: ``stimulus-modality@1`` becomes
    ``stimulus-modality@2`` with no schema shape change at all, plus a
    tightened ``n`` constraint that must trigger validation without triggering
    re-extraction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

from llmbic import (
    ContextBudget,
    ContextPolicy,
    ExtractionRecipe,
    FieldValue,
    Migration,
    MigrationStep,
    ModelPolicy,
    ParsedSource,
    Pricing,
    Ref,
    Registry,
    SourceArtifact,
    StepKind,
    TransformResult,
    ValueStatus,
    Vocabulary,
    build_parsed_source,
    find_span,
    from_json_schema,
    prior_evidence,
    sections,
    transform,
)
from llmbic.context.policy import FallbackMode, OnMissingContext, retrieve
from llmbic.migration.spec import Fidelity
from llmbic.models.mock import RuleBasedExtractor, ScriptedAdapter


# =========================================================================
# Schemas
# =========================================================================

RESPONSE_MODALITIES = ["button_press", "verbal", "eye_movement", "none"]
STIMULUS_MODALITIES = [
    "visual",
    "auditory",
    "tactile",
    "olfactory",
    "gustatory",
    "interoceptive",
    "none",
]
ASSIGNMENT_V1 = ["parallel", "crossover", "single_group", "factorial"]
ASSIGNMENT_V2 = ASSIGNMENT_V1 + ["observational_cohorts"]
ALLOCATION = ["randomized", "non_randomized", "not_applicable"]

STIMULI_DESC_V1 = "Visuals or other materials presented to participants."
STIMULI_DESC_V2 = (
    "The materials presented to participants, in the source's own terms -- the "
    "stimulus set, its name, how many items. `stimulus_modality` carries the "
    "sensory channel; this carries what the stimuli actually were."
)

POP_CHAR_DESC_V1 = (
    "The catch-all for what characterizes this cohort and has no field of its own."
)
POP_CHAR_DESC_V2 = (
    "How this cohort deviates from the typical, where no other field covers it. "
    "Deviation is the test: if every plausible cohort could carry the value, leave "
    "it out."
)

PARALLEL_DESC_V1 = "Two or more arms measured in parallel."
PARALLEL_DESC_V2 = (
    "Two or more arms measured in parallel, where something was administered. A "
    "cohort difference the participants already had is observational_cohorts."
)


def _extracted(
    base: dict[str, Any], *, description: str, recipe: str | None = None, **extra: Any
) -> dict[str, Any]:
    out = {**base, "description": description}
    if recipe:
        out["x-llmbic-recipe"] = recipe
    out.update(extra)
    return out


def _enum(values: Sequence[str], *, open_vocabulary: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {"type": "string", "enum": list(values)}
    if open_vocabulary:
        out["x-llmbic-open-vocabulary"] = True
    return out


def schema_v1_0() -> dict[str, Any]:
    return {
        "title": "Study",
        "type": "object",
        "description": "A neuroimaging publication, extracted.",
        "required": ["local_id"],
        "properties": {
            "local_id": {"type": "string", "x-llmbic-deterministic": True},
            "title": _extracted(
                {"type": "string"}, description="The publication title.", recipe="study-front@1"
            ),
            "design": {
                "type": "object",
                "properties": {
                    "assignment_structure": _extracted(
                        {
                            **_enum(ASSIGNMENT_V1),
                            "x-llmbic-vocabulary": "AssignmentStructure@1",
                        },
                        description=(
                            "How participants were assigned to the study's arms. "
                            + PARALLEL_DESC_V1
                        ),
                        recipe="design@1",
                    ),
                    "allocation": _extracted(
                        _enum(ALLOCATION),
                        description="Whether assignment was randomized.",
                        recipe="design@1",
                    ),
                    "n_arms": _extracted(
                        {"type": "integer", "minimum": 0},
                        description="How many arms the design declares.",
                        recipe="design@1",
                    ),
                },
            },
            "groups": {
                "type": "array",
                "description": "Participant cohorts.",
                "items": {
                    "type": "object",
                    "required": ["local_id"],
                    "properties": {
                        "local_id": {"type": "string", "x-llmbic-deterministic": True},
                        "name": _extracted(
                            {"type": "string"},
                            description="The cohort as the source names it.",
                            recipe="group@1",
                        ),
                        "n": _extracted(
                            {"type": "integer", "minimum": 0},
                            description="Number of participants in the cohort.",
                            recipe="group@1",
                        ),
                        "medical_condition": _extracted(
                            {"type": "array", "items": {"type": "string"}},
                            description="Diagnoses the cohort was selected for.",
                            recipe="group@1",
                        ),
                        "population_characteristics": _extracted(
                            {"type": "array", "items": {"type": "string"}},
                            description=POP_CHAR_DESC_V1,
                            recipe="group@1",
                        ),
                    },
                },
            },
            "tasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["local_id"],
                    "properties": {
                        "local_id": {"type": "string", "x-llmbic-deterministic": True},
                        "name": _extracted(
                            {"type": "string"},
                            description="The paradigm as the source names it.",
                            recipe="task@1",
                        ),
                        "stimuli": _extracted(
                            {"type": "string"},
                            description=STIMULI_DESC_V1,
                            recipe="task@1",
                        ),
                        "response_mode": _extracted(
                            {
                                "type": "array",
                                "items": _enum(RESPONSE_MODALITIES, open_vocabulary=True),
                            },
                            description="How a participant answered.",
                            recipe="task@1",
                        ),
                    },
                },
            },
        },
    }


def schema_v1_1() -> dict[str, Any]:
    """5f282f0: rename response_mode, add stimulus_modality, reword stimuli."""

    doc = schema_v1_0()
    task = doc["properties"]["tasks"]["items"]["properties"]
    task["response_modality"] = task.pop("response_mode")
    task["stimuli"]["description"] = STIMULI_DESC_V2
    task["stimulus_modality"] = _extracted(
        {"type": "array", "items": _enum(STIMULUS_MODALITIES, open_vocabulary=True)},
        description=(
            "The sensory channel the stimuli were delivered through. Multivalued "
            "because a paradigm may use more than one. Often not stated in those "
            "words: a source naming IAPS, NimStim or film clips has said `visual` "
            "without using the word, and that cue is what belongs in the evidence."
        ),
        recipe="stimulus-modality@1",
    )
    return doc


def schema_v1_2() -> dict[str, Any]:
    """6b20e30 + 653cd5b: derive is_healthy, partition population_characteristics."""

    doc = schema_v1_1()
    group = doc["properties"]["groups"]["items"]["properties"]
    group["is_healthy"] = _extracted(
        {"type": "boolean"},
        description=(
            "True when the cohort carries no diagnosis. Derived from "
            "`medical_condition` rather than asked, because a description cannot "
            "outvote the source's own wording."
        ),
        **{"x-llmbic-deterministic": True, "x-llmbic-evidence": False},
    )
    group["other_characteristics"] = _extracted(
        {"type": "array", "items": {"type": "string"}},
        description=(
            "Cohort traits that no query filters on. Moved rather than dropped: a "
            "reader auditing a cohort wants to see them; they cannot share a field "
            "with a trait a filter selects on."
        ),
        **{"x-llmbic-deterministic": True, "x-llmbic-evidence": False},
    )
    group["population_characteristics"]["description"] = POP_CHAR_DESC_V2
    return doc


def schema_v1_3() -> dict[str, Any]:
    """dc5752d: AssignmentStructure gains observational_cohorts."""

    doc = schema_v1_2()
    design = doc["properties"]["design"]["properties"]
    design["assignment_structure"] = _extracted(
        {**_enum(ASSIGNMENT_V2), "x-llmbic-vocabulary": "AssignmentStructure@2"},
        description="How participants were assigned to the study's arms. " + PARALLEL_DESC_V2,
        recipe="design@1",
    )
    return doc


def schema_v1_4() -> dict[str, Any]:
    """A prompt revision and a tightened constraint; no shape change."""

    doc = schema_v1_3()
    doc["properties"]["tasks"]["items"]["properties"]["stimulus_modality"][
        "x-llmbic-recipe"
    ] = "stimulus-modality@2"
    doc["properties"]["groups"]["items"]["properties"]["n"]["minimum"] = 1
    return doc


SCHEMA_DOCS = {
    "1.0": schema_v1_0,
    "1.1": schema_v1_1,
    "1.2": schema_v1_2,
    "1.3": schema_v1_3,
    "1.4": schema_v1_4,
}

#: The rename in 1.1 keeps its logical identity, which is what makes it free.
IDENTITY_MAPS = {
    "1.1": {"tasks[].response_modality": "tasks[].response_mode"},
    "1.2": {"tasks[].response_modality": "tasks[].response_mode"},
    "1.3": {"tasks[].response_modality": "tasks[].response_mode"},
    "1.4": {"tasks[].response_modality": "tasks[].response_mode"},
}


def normalized(version: str):
    return from_json_schema(
        SCHEMA_DOCS[version](),
        name="study",
        version=version,
        identity_map=IDENTITY_MAPS.get(version),
    )


# =========================================================================
# Deterministic transforms
# =========================================================================

UNREMARKABLE = {
    "normal weight",
    "healthy weight",
    "typically developing",
    "right-handed",
    "no hearing loss",
    "native speakers",
}


@transform("derive_is_healthy", "1")
def derive_is_healthy(ctx) -> TransformResult:
    """A cohort is healthy when it carries no diagnosis.

    6b20e30's argument: asking the model produced answers a description could
    not police, while the audit already assumed the derivation.  So derive it.
    """

    conditions = ctx.field_value("groups[].medical_condition")
    if conditions.status is ValueStatus.NOT_EXTRACTED:
        return TransformResult(
            {"groups[].is_healthy": FieldValue(ValueStatus.UNKNOWN, reason="no condition field")}
        )
    values = conditions.value or [] if conditions.status.has_value else []
    return TransformResult(
        {"groups[].is_healthy": FieldValue.present(len(values) == 0)},
        # A derived value carries no evidence of its own: its support is the
        # field it was derived from, which `lineage` records.
        evidence={"groups[].is_healthy": ()},
        lineage=(f"groups[].medical_condition:{conditions.value!r}",),
    )


@transform("split_population_characteristics", "1")
def split_population_characteristics(ctx) -> TransformResult:
    """653cd5b: partition the catch-all by whether a query could filter on it."""

    current = ctx.field_value("groups[].population_characteristics")
    if not current.status.has_value:
        return TransformResult(
            {
                "groups[].population_characteristics": current,
                "groups[].other_characteristics": FieldValue(
                    ValueStatus.NOT_APPLICABLE, reason="no characteristics to partition"
                ),
            }
        )
    kept, moved = [], []
    for item in current.value or []:
        (moved if item.strip().lower() in UNREMARKABLE else kept).append(item)
    return TransformResult(
        {
            "groups[].population_characteristics": FieldValue.present(kept),
            "groups[].other_characteristics": FieldValue.present(moved),
        },
        evidence={
            # The surviving entries keep the spans that supported them; the
            # moved ones are deterministic and carry none.
            "groups[].population_characteristics": ctx.evidence(
                "groups[].population_characteristics"
            ),
            "groups[].other_characteristics": (),
        },
        lineage=(f"groups[].population_characteristics:{current.value!r}",),
    )


@transform("remap_assignment_structure", "1")
def remap_assignment_structure(ctx) -> TransformResult:
    """dc5752d: `parallel` on a study with no arms is `observational_cohorts`.

    The deterministic case is the one the record already settles: parallel,
    nothing randomized, and no arm declared.  Everything else keeps its value
    and is left for the escalation step, because guessing here would assert an
    assignment the paper never made.
    """

    current = ctx.field_value("design.assignment_structure")
    if not current.status.has_value or current.value != "parallel":
        return TransformResult({"design.assignment_structure": current})

    allocation = ctx.value("design.allocation")
    n_arms = ctx.value("design.n_arms")
    settled = allocation in ("non_randomized", "not_applicable") and (n_arms or 0) == 0
    if settled:
        return TransformResult(
            {"design.assignment_structure": FieldValue.present("observational_cohorts")},
            evidence={
                "design.assignment_structure": ctx.evidence("design.assignment_structure")
            },
            lineage=(
                f"design.assignment_structure:{current.value!r}",
                f"design.allocation:{allocation!r}",
                f"design.n_arms:{n_arms!r}",
            ),
            notes={"remap": "parallel->observational_cohorts", "rule": "no arms, not randomized"},
        )
    return TransformResult(
        {
            "design.assignment_structure": FieldValue(
                ValueStatus.REVIEW_REQUIRED,
                current.value,
                reason=(
                    "`parallel` now requires something to have been administered; this "
                    f"record declares allocation={allocation!r} and n_arms={n_arms!r}, "
                    "which the rule does not settle"
                ),
            )
        },
        evidence={"design.assignment_structure": ctx.evidence("design.assignment_structure")},
    )


@transform("rename_response_mode", "1")
def rename_response_mode(ctx) -> TransformResult:
    """Only needed when a rename does *not* preserve identity.

    Kept here so the "rename onto a new field id" path has a test, beside the
    identity-preserving rename that needs no step at all.
    """

    value = ctx.field_value("tasks[].response_mode")
    return TransformResult(
        {"tasks[].response_modality": value},
        evidence={"tasks[].response_modality": ctx.evidence("tasks[].response_mode")},
    )


# =========================================================================
# Recipes
# =========================================================================

STIMULUS_PROMPT_V1 = """\
Read the supplied context and answer with the sensory channel(s) the stimuli were
delivered through.

Record: {record_id} {entity}
Field: {field_path}
Definition: {field_description}
Permitted values: {permitted_values}

Answer with JSON: {{"tasks[].stimulus_modality": {{"status": "present"|"not_reported",
"value": [<modality>, ...], "evidence": ["<verbatim quote>"]}}}}
Abstain with status "not_reported" when the context does not license an answer.
"""

STIMULUS_PROMPT_V2 = STIMULUS_PROMPT_V1 + """\

A named stimulus set (IAPS, NimStim, Ekman faces, film clips) licenses `visual`
without the word appearing. Quote the cue, not the whole sentence: an inferred
value stretched past the span that licensed it is worse than an absent one.
"""

STIMULUS_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "tasks[].stimulus_modality": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["present", "not_reported"]},
                "value": {"type": "array", "items": {"type": "string"}},
                "evidence": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["status"],
        }
    },
    "required": ["tasks[].stimulus_modality"],
}

EVIDENCE_FIRST_METHODS = ContextPolicy(
    sequence=(
        prior_evidence(expand_chars=120),
        sections("methods", "supplement"),
    ),
    full_document_fallback=FallbackMode.FORBIDDEN,
    on_missing_context=OnMissingContext.REVIEW,
    budget=ContextBudget(max_input_tokens=3000),
)


def stimulus_recipe(version: str = "1") -> ExtractionRecipe:
    return ExtractionRecipe(
        name="stimulus-modality",
        version=version,
        writes=("tasks[].stimulus_modality",),
        prompt_template=STIMULUS_PROMPT_V1 if version == "1" else STIMULUS_PROMPT_V2,
        system="You extract structured facts from neuroimaging methods sections.",
        output_schema=STIMULUS_OUTPUT_SCHEMA,
        model_policy=ModelPolicy(
            adapter="main",
            fallbacks=("backup",),
            parameters={"temperature": 0.0},
            pricing=Pricing(3.0, 15.0),
            max_concurrency=4,
        ),
        context_policy=EVIDENCE_FIRST_METHODS,
        validators=("require_evidence", "in_vocabulary"),
        reads_fields=("tasks[].stimuli",),
        description="Read the stimulus modality out of the methods section.",
    )


def assignment_escalation_recipe() -> ExtractionRecipe:
    """The ambiguous half of the vocabulary remap, escalated to a model."""

    return ExtractionRecipe(
        name="assignment-remap",
        version="1",
        writes=("design.assignment_structure",),
        prompt_template=(
            "The vocabulary changed: `parallel` now requires that something was "
            "administered, and a study comparing cohorts that differ by something "
            "the participants already had is `observational_cohorts`.\n"
            "Stored value: {design_assignment_structure}\n"
            "Allocation: {design_allocation}\n"
            "Answer with JSON: "
            '{{"design.assignment_structure": {{"status": "present", '
            '"value": "<permissible value>"}}}}\n'
        ),
        output_schema={
            "type": "object",
            "properties": {
                "design.assignment_structure": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                        "value": {"type": "string", "enum": ASSIGNMENT_V2},
                        "evidence": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["status"],
                }
            },
            "required": ["design.assignment_structure"],
        },
        model_policy=ModelPolicy(adapter="main", pricing=Pricing(3.0, 15.0)),
        context_policy=ContextPolicy(
            sequence=(
                sections("methods"),
                retrieve("participants were assigned to groups", top_k=3),
            ),
            full_document_fallback=FallbackMode.FORBIDDEN,
            on_missing_context=OnMissingContext.REVIEW,
        ),
        validators=("in_vocabulary",),
        reads_fields=("design.assignment_structure", "design.allocation"),
        vocabulary_ref="AssignmentStructure@2",
    )


VOCABULARIES = {
    "AssignmentStructure@1": Vocabulary(
        name="AssignmentStructure",
        version="1",
        values=tuple(ASSIGNMENT_V1),
        descriptions={"parallel": PARALLEL_DESC_V1},
    ),
    "AssignmentStructure@2": Vocabulary(
        name="AssignmentStructure",
        version="2",
        values=tuple(ASSIGNMENT_V2),
        reopens=("parallel",),
        descriptions={"parallel": PARALLEL_DESC_V2},
    ),
}


# =========================================================================
# Migrations
# =========================================================================

def migration_1_0_to_1_1() -> Migration:
    return Migration(
        id="study-1.0-to-1.1",
        from_schema="study@1.0",
        to_schema="study@1.1",
        description=(
            "5f282f0. response_mode -> response_modality keeps its identity, so it "
            "needs no step and costs nothing. stimulus_modality is new and has to be "
            "read out of the paper, evidence first and never the whole document."
        ),
        renames={"tasks[].response_mode": "tasks[].response_modality"},
        steps=(
            MigrationStep(
                id="extract_stimulus_modality",
                kind=StepKind.SOURCE_SEMANTIC,
                reads=(
                    Ref("field", "tasks[].stimuli"),
                    Ref("evidence", "tasks[].stimuli"),
                    Ref("source", "parsed"),
                ),
                writes=("tasks[].stimulus_modality",),
                recipe="stimulus-modality@1",
                context=EVIDENCE_FIRST_METHODS,
                on_missing_context=OnMissingContext.REVIEW,
                validators=("require_evidence",),
                entity_scope="tasks[]",
                fidelity=Fidelity.LOSSLESS,
                description="Read the sensory channel out of the methods section.",
            ),
        ),
    )


def migration_1_1_to_1_2() -> Migration:
    return Migration(
        id="study-1.1-to-1.2",
        from_schema="study@1.1",
        to_schema="study@1.2",
        description=(
            "6b20e30 and 653cd5b. is_healthy stops being asked and starts being "
            "derived; the catch-all is partitioned so a filterable trait and an "
            "unremarkable one stop sharing a field."
        ),
        steps=(
            MigrationStep(
                id="derive_is_healthy",
                kind=StepKind.DERIVED,
                reads=(Ref("field", "groups[].medical_condition"),),
                writes=("groups[].is_healthy",),
                transform="derive_is_healthy@1",
                entity_scope="groups[]",
                fidelity=Fidelity.LOSSLESS,
            ),
            MigrationStep(
                id="split_characteristics",
                kind=StepKind.STRUCTURAL,
                reads=(
                    Ref("field", "groups[].population_characteristics"),
                    Ref("evidence", "groups[].population_characteristics"),
                ),
                writes=(
                    "groups[].population_characteristics",
                    "groups[].other_characteristics",
                ),
                transform="split_population_characteristics@1",
                entity_scope="groups[]",
                fidelity=Fidelity.LOSSY,
                description="Moved rather than dropped, but a reader could disagree.",
            ),
        ),
    )


def migration_1_2_to_1_3(*, with_escalation: bool = True) -> Migration:
    steps = [
        MigrationStep(
            id="remap_assignment",
            kind=StepKind.VOCABULARY,
            reads=(
                Ref("field", "design.assignment_structure"),
                Ref("field", "design.allocation"),
                Ref("field", "design.n_arms"),
                Ref("evidence", "design.assignment_structure"),
                Ref("vocab", "AssignmentStructure@2"),
            ),
            writes=("design.assignment_structure",),
            transform="remap_assignment_structure@1",
            fidelity=Fidelity.LOSSY,
            description=(
                "Deterministic where the record settles it; REVIEW_REQUIRED where it "
                "does not, rather than guessing an assignment the paper never made."
            ),
        )
    ]
    return Migration(
        id="study-1.2-to-1.3",
        from_schema="study@1.2",
        to_schema="study@1.3",
        description="dc5752d. A vocabulary gains a value and an old one narrows.",
        steps=tuple(steps),
    )


def migration_1_3_to_1_4() -> Migration:
    return Migration(
        id="study-1.3-to-1.4",
        from_schema="study@1.3",
        to_schema="study@1.4",
        description=(
            "A prompt revision with no shape change. Only artifacts produced by "
            "stimulus-modality@1 are invalidated; `n`'s tightened minimum is a "
            "validation matter, not a re-extraction one."
        ),
        acknowledged={
            "groups[].n": (
                "minimum raised from 0 to 1. A cohort of zero was never a real answer, "
                "so nothing is re-extracted: the constraint is checked at validation "
                "and a record carrying 0 is held rather than silently re-read."
            )
        },
        steps=(
            MigrationStep(
                id="reextract_stimulus_modality",
                kind=StepKind.REEXTRACTION,
                reads=(
                    Ref("field", "tasks[].stimuli"),
                    Ref("evidence", "tasks[].stimuli"),
                    Ref("source", "parsed"),
                ),
                writes=("tasks[].stimulus_modality",),
                recipe="stimulus-modality@2",
                context=EVIDENCE_FIRST_METHODS,
                on_missing_context=OnMissingContext.REVIEW,
                entity_scope="tasks[]",
            ),
        ),
    )


def build_registry(*, up_to: str = "1.4") -> Registry:
    """Register every schema, recipe, vocabulary and migration up to ``up_to``."""

    order = ["1.0", "1.1", "1.2", "1.3", "1.4"]
    wanted = order[: order.index(up_to) + 1]

    reg = Registry()
    for version in wanted:
        reg.register_schema(normalized(version))
    for vocab in VOCABULARIES.values():
        reg.register_vocabulary(vocab)
    reg.register_recipe(stimulus_recipe("1"))
    if "1.4" in wanted:
        reg.register_recipe(stimulus_recipe("2"))
    reg.register_recipe(assignment_escalation_recipe())

    builders = {
        "1.1": migration_1_0_to_1_1,
        "1.2": migration_1_1_to_1_2,
        "1.3": migration_1_2_to_1_3,
        "1.4": migration_1_3_to_1_4,
    }
    for version in wanted[1:]:
        reg.register_migration(builders[version]())
    return reg


# =========================================================================
# Synthetic corpus
# =========================================================================

MODALITY_CUES = {
    "visual": "Stimuli were 60 IAPS photographs presented on a back-projection screen.",
    "auditory": "Participants heard 40 spoken word pairs over MRI-compatible headphones.",
    "tactile": "A piezoelectric device delivered vibrotactile stimulation to the index finger.",
}

CONDITIONS = [["major depressive disorder"], ["schizophrenia"], [], [], ["migraine"]]
CHARACTERISTICS = [
    ["heavy caffeine use", "right-handed"],
    ["musicians", "normal weight"],
    ["veterans"],
    ["typically developing"],
    [],
]


@dataclass
class Paper:
    record_id: str
    source: SourceArtifact
    parsed: ParsedSource
    record: dict[str, Any]
    #: The modality the methods section actually licenses, or None.
    truth: str | None


def make_paper(
    index: int,
    *,
    modality: str | None = None,
    include_methods: bool = True,
    n_groups: int = 2,
    assignment: str = "parallel",
    allocation: str = "non_randomized",
    n_arms: int = 0,
) -> Paper:
    record_id = f"pmid:{100000 + index}"
    if modality is None and include_methods:
        modality = list(MODALITY_CUES)[index % 3]

    cue = MODALITY_CUES.get(modality or "", "")

    # The cohort descriptions go into the Methods text, so the characteristics
    # a record carries are quotable and their spans resolve — which is what
    # makes "a structural move preserves evidence" a real assertion rather
    # than a vacuous one.
    specs = []
    for g in range(n_groups):
        specs.append(
            {
                "local_id": f"g{g + 1}",
                "name": f"cohort {g + 1}",
                "n": 20 + 5 * g + (index % 7),
                "conditions": CONDITIONS[(index + g) % len(CONDITIONS)],
                "characteristics": CHARACTERISTICS[(index + g) % len(CHARACTERISTICS)],
            }
        )

    cohort_sentences = " ".join(
        f"The {s['name']} comprised {s['n']} "
        + (", ".join(s["characteristics"]) + " " if s["characteristics"] else "")
        + ("volunteers" if not s["conditions"] else " and ".join(s["conditions"]) + " patients")
        + "."
        for s in specs
    )
    methods = (
        f"Participants gave written informed consent. {cue} {cohort_sentences} "
        "Responses were recorded with an MRI-compatible button box. "
        f"Imaging was performed on a 3T scanner across {n_groups} cohorts."
    )
    parts = [
        ("title", f"Study {index}: an fMRI investigation"),
        ("abstract", "We investigated cortical responses in a cohort comparison."),
    ]
    if include_methods:
        parts.append(("methods", methods))
    parts.append(("results", "Activation was observed in bilateral insula (p < .001)."))

    source = SourceArtifact(
        source_id=record_id,
        source_version="v1",
        content_hash=f"sha256:paper{index:04d}",
        uri=f"https://example.org/{record_id}",
    )
    parsed = build_parsed_source(source, parts, parse_version="parse@1", split_sentences=True)

    groups = [
        {
            "local_id": s["local_id"],
            "name": _wrap(s["name"], parsed),
            "n": _wrap(s["n"], parsed),
            "medical_condition": _wrap(s["conditions"], parsed),
            "population_characteristics": _wrap(s["characteristics"], parsed),
        }
        for s in specs
    ]

    stimuli_text = cue if cue else ""
    record = {
        "local_id": record_id,
        "title": _wrap(f"Study {index}: an fMRI investigation", parsed),
        "design": {
            "assignment_structure": _wrap(assignment, parsed),
            "allocation": _wrap(allocation, parsed),
            "n_arms": _wrap(n_arms, parsed),
        },
        "groups": groups,
        "tasks": [
            {
                "local_id": "t1",
                "name": _wrap("emotional picture viewing", parsed),
                "stimuli": _wrap(stimuli_text, parsed) if stimuli_text else _absent(),
                "response_mode": _wrap(["button_press"], parsed),
            }
        ],
    }
    return Paper(
        record_id=record_id, source=source, parsed=parsed, record=record, truth=modality
    )


def corpus(
    n: int = 12, *, missing_methods_every: int = 7
) -> list[Paper]:
    """A small corpus with a deliberate spread of hard cases."""

    papers = []
    for i in range(n):
        include_methods = (i % missing_methods_every) != (missing_methods_every - 1)
        papers.append(
            make_paper(
                i,
                include_methods=include_methods,
                n_groups=1 + (i % 2),
                assignment="parallel" if i % 4 else "crossover",
                allocation="randomized" if i % 5 == 0 else "non_randomized",
                n_arms=2 if i % 5 == 0 else 0,
            )
        )
    return papers


# ---- the ExtractedValue wrapper study_schema uses ------------------------

def _wrap(value: Any, parsed: ParsedSource) -> dict[str, Any]:
    """Wrap a value the way study_schema's ExtractedValue does."""

    if value in ([], "", None):
        return {
            "extraction_status": "not_reported",
            "evidence": {"status": "not_applicable"},
        }
    needle = value[0] if isinstance(value, list) else value
    ref = find_span(parsed, str(needle)) if isinstance(needle, str) else None
    out: dict[str, Any] = {
        "extraction_status": "extracted",
        "value": value,
        "value_source": "reported",
        "evidence": {"status": "not_found"},
    }
    if ref is not None:
        out["evidence"] = {
            "status": "present",
            "sets": [
                {
                    "source": ref.locator,
                    "source_id": ref.source_id,
                    "source_version": ref.source_version,
                    "parse_version": ref.parse_version,
                    "spans": [s.to_canonical() for s in ref.spans],
                }
            ],
        }
    return out


def _absent(reason: str | None = None) -> dict[str, Any]:
    out = {"extraction_status": "not_reported", "evidence": {"status": "not_applicable"}}
    if reason:
        out["unreported_reason"] = reason
    return out


# =========================================================================
# Model adapters
# =========================================================================

def _modality_answer(match: re.Match[str], unit) -> dict[str, Any]:
    text = match.group(0)
    if re.search(r"IAPS|photograph|picture|screen|film|face", text, re.I):
        modality = "visual"
    elif re.search(r"heard|spoken|headphones|auditory|tone", text, re.I):
        modality = "auditory"
    else:
        modality = "tactile"
    return {
        "tasks[].stimulus_modality": {
            "status": "present",
            "value": [modality],
            "evidence": [text],
        }
    }


def stimulus_adapter(**kwargs: Any) -> RuleBasedExtractor:
    """A local model that genuinely reads the context it is given.

    When no rule matches it abstains — which is what makes "records with
    unavailable required context become blocked or review-needed rather than
    receiving invented values" testable rather than asserted.
    """

    return RuleBasedExtractor(
        rules=[
            (r"[^.]*\b(IAPS|photographs?|pictures?|film clips?|faces?)\b[^.]*\.", _modality_answer),
            (r"[^.]*\b(heard|spoken|headphones|tones?)\b[^.]*\.", _modality_answer),
            (r"[^.]*\b(vibrotactile|piezoelectric|tactile)\b[^.]*\.", _modality_answer),
        ],
        abstention={
            "tasks[].stimulus_modality": {
                "status": "not_reported",
                "value": [],
                "evidence": [],
            }
        },
        **kwargs,
    )


def assignment_adapter(answer: str = "observational_cohorts") -> ScriptedAdapter:
    return ScriptedAdapter(
        default={"design.assignment_structure": {"status": "present", "value": answer}},
        provider="mock",
        model="assignment-1",
    )


__all__ = [
    "ASSIGNMENT_V1",
    "ASSIGNMENT_V2",
    "EVIDENCE_FIRST_METHODS",
    "MODALITY_CUES",
    "Paper",
    "SCHEMA_DOCS",
    "STIMULUS_MODALITIES",
    "assignment_adapter",
    "assignment_escalation_recipe",
    "build_registry",
    "corpus",
    "make_paper",
    "migration_1_0_to_1_1",
    "migration_1_1_to_1_2",
    "migration_1_2_to_1_3",
    "migration_1_3_to_1_4",
    "normalized",
    "stimulus_adapter",
    "stimulus_recipe",
]
