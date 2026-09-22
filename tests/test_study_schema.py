"""Golden migration tests against the real neurostuff/study_schema (§15.4).

These run the LinkML adapter, the diff and a full migration over the *actual*
schema at two real commits, so what is tested is the tool against the corpus it
was written for rather than a model of it.

The checkout is located with ``LLMBIC_STUDY_SCHEMA``, defaulting to a sibling
clone; the whole module skips when it is not present, which keeps
``NFR-MNT-002`` (core tests run without network) honest.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from llmbic import (
    ChangeKind,
    ContextPolicy,
    ExtractedValueCodec,
    ExtractionRecipe,
    Migration,
    MigrationStep,
    ModelPolicy,
    Pricing,
    Project,
    Ref,
    Registry,
    SourceArtifact,
    StepKind,
    build_parsed_source,
    diff_schemas,
    find_span,
    from_linkml,
    prior_evidence,
    sections,
)
from llmbic.context.policy import FallbackMode, OnMissingContext
from llmbic.models.base import AdapterRegistry
from llmbic.models.mock import RuleBasedExtractor

STUDY_SCHEMA = Path(
    os.environ.get("LLMBIC_STUDY_SCHEMA", "/home/user/study_schema")
).expanduser()

#: 5f282f0 "Track stimulus modality, and name it symmetrically with response".
RENAME_COMMIT = "5f282f0"
#: dc5752d "Give the design vocabulary a value for cohorts nobody assigned".
VOCABULARY_COMMIT = "dc5752d"

ENTRY = "neuroimaging-study-extraction.yaml"

pytestmark = pytest.mark.skipif(
    not (STUDY_SCHEMA / ENTRY).exists(),
    reason=f"no study_schema checkout at {STUDY_SCHEMA}",
)


def _at(revision: str, tmp_path: Path) -> Path:
    """Materialise the whole schema tree at ``revision``."""

    out = tmp_path / revision.replace("^", "_parent")
    out.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(
        ["git", "-C", str(STUDY_SCHEMA), "archive", revision],
        capture_output=True,
        check=True,
    ).stdout
    subprocess.run(["tar", "-x", "-C", str(out)], input=archive, check=True)
    return out / ENTRY


def _schema(revision: str, tmp_path: Path, version: str, **kwargs):
    return from_linkml(_at(revision, tmp_path), name="study", version=version, **kwargs)


# ---- the adapter ---------------------------------------------------------

def test_the_adapter_reads_the_real_schema():
    schema = from_linkml(STUDY_SCHEMA / ENTRY, name="study", version="head")
    assert len(schema.fields) > 150
    assert len(schema.collections) > 15
    assert schema.annotations["root_class"] == "Study"


def test_extracted_value_wrappers_are_unwrapped_to_their_real_types():
    schema = from_linkml(STUDY_SCHEMA / ENTRY, name="study", version="head")
    enrolled = schema.by_path("groups[].enrolled_count")
    assert enrolled is not None and enrolled.base_type.value == "integer"
    assert enrolled.evidence_bearing is True

    age = schema.by_path("groups[].age_mean")
    assert age is not None and age.base_type.value == "number"

    modality = schema.by_path("tasks[].stimulus_modality")
    assert modality.multivalued is True
    assert modality.constraints.open_vocabulary is True
    assert "visual" in modality.constraints.enum
    assert modality.constraints.vocabulary_ref == "StimulusModality"


def test_a_reference_by_local_id_is_a_leaf_not_a_second_copy_of_the_entity():
    """``inlined: false`` means the slot holds an identifier."""

    schema = from_linkml(STUDY_SCHEMA / ENTRY, name="study", version="head")
    ref = schema.by_path("analyses[].groups[].group")
    assert ref is not None
    assert ref.base_type.value == "string"
    assert ref.annotations["references"] == "Group"
    # The cohort's own fields live under groups[], and only there.
    assert schema.by_path("analyses[].groups[].group.enrolled_count") is None
    assert schema.by_path("groups[].enrolled_count") is not None


def test_deterministic_slots_are_marked_as_such():
    schema = from_linkml(STUDY_SCHEMA / ENTRY, name="study", version="head")
    metadata = schema.by_path("extraction_metadata.extractor_model")
    assert metadata.deterministic is True
    assert metadata.evidence_bearing is False


# ---- golden diffs over real commits -------------------------------------

def test_the_rename_commit_diffs_the_way_its_message_describes(tmp_path):
    old = _schema(f"{RENAME_COMMIT}^", tmp_path, "before")
    new = _schema(RENAME_COMMIT, tmp_path, "after")
    diff = diff_schemas(old, new)

    removed = {c.from_path for c in diff.of_kind(ChangeKind.REMOVED)}
    added = {c.to_path for c in diff.of_kind(ChangeKind.ADDED)}
    assert removed == {"tasks[].response_mode"}
    assert "tasks[].stimulus_modality" in added

    # Without an explicit map, the rename is a remove/add pair and only a
    # candidate — never applied.
    assert not diff.of_kind(ChangeKind.RENAMED)
    assert ("tasks[].response_mode", "tasks[].response_modality") in {
        (a, b) for a, b, _ in diff.rename_candidates
    }

    # `stimuli` was reworded without its shape changing.
    reworded = {c.to_path for c in diff.of_kind(ChangeKind.DESCRIPTION_CHANGED)}
    assert "tasks[].stimuli" in reworded

    # Undeclared, the rename reads as a field disappearing, and a disappearing
    # field is a stored-value problem. That is the whole point of FR-SCH-004.
    assert diff.compatibility()["values"] is False


def test_declaring_the_rename_collapses_the_pair(tmp_path):
    old = _schema(f"{RENAME_COMMIT}^", tmp_path, "before")
    new = _schema(
        RENAME_COMMIT,
        tmp_path,
        "after",
        identity_map={"tasks[].response_modality": "tasks[].response_mode"},
    )
    diff = diff_schemas(old, new)
    renamed = diff.of_kind(ChangeKind.RENAMED)
    assert [c.to_path for c in renamed] == ["tasks[].response_modality"]
    assert renamed[0].detail["identity_preserved"] is True
    assert not diff.of_kind(ChangeKind.REMOVED)
    # Declared, the same change no longer threatens a stored value.
    assert diff.compatibility()["values"] is True


def test_the_vocabulary_commit_shows_a_widened_vocabulary(tmp_path):
    old = _schema(f"{VOCABULARY_COMMIT}^", tmp_path, "before")
    new = _schema(VOCABULARY_COMMIT, tmp_path, "after")
    diff = diff_schemas(old, new)

    widened = diff.of_kind(ChangeKind.CONSTRAINT_WIDENED)
    assert any(
        "observational_cohorts" in (c.detail.get("added_values") or ())
        for c in widened
    )
    # Stored values stay valid; what the field asks has changed.
    assert diff.compatibility()["values"] is True
    assert diff.compatibility()["semantics"] is False


def test_the_vocabulary_commit_also_rewrote_the_descriptions_it_narrowed(tmp_path):
    old = _schema(f"{VOCABULARY_COMMIT}^", tmp_path, "before")
    new = _schema(VOCABULARY_COMMIT, tmp_path, "after")
    diff = diff_schemas(old, new)
    reworded = {c.to_path for c in diff.of_kind(ChangeKind.DESCRIPTION_CHANGED)}
    assert any("assignment_structure" in p for p in reworded)


# ---- a golden migration on the real schema ------------------------------

MINIMAL_PROMPT = """\
Read the context and name the sensory channel the stimuli were delivered through.
Record: {record_id} {entity}
Answer with JSON: {{"tasks[].stimulus_modality": {{"status": "present",
"value": [<modality>], "evidence": ["<verbatim quote>"]}}}}
"""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "tasks[].stimulus_modality": {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "value": {"type": "array", "items": {"type": "string"}},
                "evidence": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["status"],
        }
    },
    "required": ["tasks[].stimulus_modality"],
}

CUE = "Stimuli were 60 IAPS photographs presented on a back-projection screen."


def _wrapped(value, parsed=None):
    out = {
        "extraction_status": "extracted",
        "value": value,
        "value_source": "reported",
        "evidence": {"status": "not_found"},
    }
    needle = value[0] if isinstance(value, list) else value
    ref = find_span(parsed, str(needle)) if parsed is not None else None
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


@pytest.fixture
def real_project(tmp_path):
    """A project holding the real schema on both sides of the rename commit."""

    before = _schema(f"{RENAME_COMMIT}^", tmp_path, "before")
    after = _schema(
        RENAME_COMMIT,
        tmp_path,
        "after",
        identity_map={"tasks[].response_modality": "tasks[].response_mode"},
    )

    registry = Registry()
    registry.register_schema(before)
    registry.register_schema(after)
    registry.register_recipe(
        ExtractionRecipe(
            name="stimulus-modality",
            version="1",
            writes=("tasks[].stimulus_modality",),
            prompt_template=MINIMAL_PROMPT,
            output_schema=OUTPUT_SCHEMA,
            model_policy=ModelPolicy(adapter="main", pricing=Pricing(3.0, 15.0)),
            context_policy=ContextPolicy(
                sequence=(prior_evidence(80), sections("methods")),
                full_document_fallback=FallbackMode.FORBIDDEN,
                on_missing_context=OnMissingContext.REVIEW,
            ),
            validators=("require_evidence",),
            reads_fields=("tasks[].stimuli",),
        )
    )
    registry.register_migration(
        Migration(
            id="study-rename-commit",
            from_schema="study@before",
            to_schema="study@after",
            renames={"tasks[].response_mode": "tasks[].response_modality"},
            acknowledged={
                "tasks[].stimuli": "reworded, not redefined; existing values stand",
                "tasks[].response_mode": (
                    "the enum was renamed ResponseMode -> ResponseModality with the same "
                    "permissible values, so stored values are untouched"
                ),
            },
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
                    entity_scope="tasks[]",
                    on_missing_context=OnMissingContext.REVIEW,
                ),
            ),
        ),
        validate=True,
    )

    adapters = AdapterRegistry()
    adapters.register(
        "main",
        RuleBasedExtractor(
            rules=[
                (
                    r"[^.]*\b(IAPS|photographs?)\b[^.]*\.",
                    lambda m, unit: {
                        "tasks[].stimulus_modality": {
                            "status": "present",
                            "value": ["visual"],
                            "evidence": [m.group(0)],
                        }
                    },
                )
            ],
            abstention={
                "tasks[].stimulus_modality": {"status": "not_reported", "value": []}
            },
        ),
    )

    project = Project(
        tmp_path / "real.db",
        registry=registry,
        adapters=adapters,
        codec=ExtractedValueCodec(),
    )
    return project


def _ingest_one(project):
    source = SourceArtifact("pmid:999", "v1", "sha256:real-0001")
    parsed = build_parsed_source(
        source,
        [
            ("title", "A study of emotional picture viewing"),
            (
                "methods",
                f"Participants gave consent. {CUE} Responses were button presses.",
            ),
        ],
        split_sentences=True,
    )
    record = {
        "extraction_metadata": {
            "extractor_model": "gpt-test",
            "extractor_version": "0.0.1",
        },
        "local_id": "pmid:999",
        "tasks": [
            {
                "local_id": "t1",
                "name": _wrapped("emotional picture viewing", parsed),
                "stimuli": _wrapped(CUE, parsed),
                "response_mode": _wrapped(["button_press"], parsed),
            }
        ],
    }
    project.ingest(
        record,
        schema_ref="study@before",
        record_id="pmid:999",
        source=source,
        parsed=parsed,
    )
    return record


def test_a_golden_migration_over_the_real_schema(real_project):
    """§15.4: representative record in, expected record and counts out."""

    _ingest_one(real_project)

    plan = real_project.plan("study@after")
    # Expected effects, declared: one model call, no full documents, one field.
    assert plan.summary.n_model_calls == 1
    assert plan.summary.n_full_document_transmissions == 0
    assert {w for s in plan.steps() for w in s.writes} == {"tasks[].stimulus_modality"}
    assert plan.summary.by_disposition.get("semantic") == 1

    result = real_project.run(plan)
    assert result.published == ["pmid:999"]
    assert result.metrics.model_calls == 1

    record = real_project.get_record("pmid:999")
    task = record["tasks"][0]
    assert task["response_modality"]["value"] == ["button_press"]
    assert "response_mode" not in task
    assert task["stimulus_modality"]["value"] == ["visual"]
    assert task["stimulus_modality"]["evidence"]["status"] == "present"

    # The renamed field's artifact was not rewritten.
    renamed = next(
        a
        for a in real_project.artifacts("pmid:999")
        if a.field_id == "tasks[].response_mode"
    )
    assert renamed.provenance.migration_id is None
    assert renamed.provenance.model_call is None
    real_project.close()


def test_the_golden_migration_is_idempotent(real_project):
    _ingest_one(real_project)
    real_project.migrate("study@after")
    before = sorted(a.artifact_id for a in real_project.artifacts("pmid:999"))

    plan, result = real_project.migrate("study@after")
    after = sorted(a.artifact_id for a in real_project.artifacts("pmid:999"))
    assert before == after
    assert result.metrics.model_calls == 0
    real_project.close()


def test_the_real_migration_validates_statically(real_project):
    problems = real_project.registry.validate_migration(
        real_project.registry.migration("study-rename-commit")
    )
    assert problems == []
    real_project.close()
