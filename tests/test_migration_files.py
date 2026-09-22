"""Migration YAML and scaffolding (§10, FR-MIG-009/010)."""

from __future__ import annotations

import textwrap

import yaml

import studybed as sb
from llmbic import Registry, StepKind, dump_migration, load_migration_file, scaffold_migration
from llmbic.migration.loader import migration_from_dict, migration_to_dict
from llmbic.migration.spec import Fidelity

# The exact document from requirements §10.
SPEC_EXAMPLE = textwrap.dedent(
    """
    id: study-schema-1.2-to-1.3
    from_schema: study@1.2
    to_schema: study@1.3
    steps:
      - id: rename_sample_size
        kind: structural
        reads: [field:sample_size]
        writes: [field:n_total]
        transform: migrations.v1_3.rename_sample_size
        fidelity: lossless

      - id: extract_analysis_software
        kind: source_semantic
        reads:
          - field:analyses
          - evidence:analyses
          - source:parsed_article
        writes: [field:analysis_software]
        recipe: analysis-software@1
        context:
          sequence:
            - prior_evidence
            - sections: [methods, supplement]
            - retrieve:
                query: "analysis software package and version"
                top_k: 8
          full_document_fallback: forbidden
          max_input_tokens: 12000
        on_missing_context: review
        validators:
          - validate_software_evidence
    """
)


def test_the_specification_example_loads_verbatim(tmp_path):
    path = tmp_path / "m.yaml"
    path.write_text(SPEC_EXAMPLE)
    migrations = load_migration_file(path)
    assert len(migrations) == 1
    m = migrations[0]
    assert m.id == "study-schema-1.2-to-1.3"
    assert m.from_schema == "study@1.2"
    assert [s.id for s in m.steps] == ["rename_sample_size", "extract_analysis_software"]

    rename = m.step("rename_sample_size")
    assert rename.kind is StepKind.STRUCTURAL
    assert rename.reads_fields == ("sample_size",)
    assert rename.writes == ("n_total",)
    assert rename.transform == "migrations.v1_3.rename_sample_size"
    assert rename.fidelity is Fidelity.LOSSLESS

    semantic = m.step("extract_analysis_software")
    assert semantic.recipe == "analysis-software@1"
    assert semantic.reads_evidence == ("analyses",)
    assert semantic.reads_source == ("parsed_article",)
    assert [s.kind for s in semantic.context.sequence] == [
        "prior_evidence",
        "sections",
        "retrieve",
    ]
    assert semantic.context.budget.max_input_tokens == 12000
    assert not semantic.context.permits_full_document()
    assert semantic.on_missing_context.value == "review"
    assert semantic.validators == ("validate_software_evidence",)


def test_a_migration_round_trips_through_yaml():
    for builder in (
        sb.migration_1_0_to_1_1,
        sb.migration_1_1_to_1_2,
        sb.migration_1_2_to_1_3,
        sb.migration_1_3_to_1_4,
    ):
        original = builder()
        text = dump_migration(original)
        restored = migration_from_dict(yaml.safe_load(text))
        assert restored.migration_hash() == original.migration_hash(), original.id


def test_several_migrations_in_one_file(tmp_path):
    path = tmp_path / "all.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "migrations": [
                    migration_to_dict(sb.migration_1_0_to_1_1()),
                    migration_to_dict(sb.migration_1_1_to_1_2()),
                ]
            }
        )
    )
    assert [m.id for m in load_migration_file(path)] == [
        "study-1.0-to-1.1",
        "study-1.1-to-1.2",
    ]


def test_multi_document_yaml_is_accepted(tmp_path):
    path = tmp_path / "docs.yaml"
    path.write_text(
        dump_migration(sb.migration_1_0_to_1_1())
        + "\n---\n"
        + dump_migration(sb.migration_1_1_to_1_2())
    )
    assert len(load_migration_file(path)) == 2


# ---- scaffolding ---------------------------------------------------------

def _scaffold(from_v, to_v):
    reg = Registry()
    old = reg.register_schema(sb.normalized(from_v))
    new = reg.register_schema(sb.normalized(to_v))
    diff = reg.diff(old.ref, new.ref)
    return scaffold_migration(diff, old=old, new=new), reg, old, new


def test_a_scaffold_enumerates_every_change_that_needs_a_decision():
    (migration, notes), reg, old, new = _scaffold("1.0", "1.1")
    assert [s.writes[0] for s in migration.steps] == ["tasks[].stimulus_modality"]
    assert migration.renames == {
        "tasks[].response_mode": "tasks[].response_modality"
    }
    assert any("identity preserved" in n for n in notes)
    assert any("no longer semantically current" in n for n in notes)


def test_a_scaffold_does_not_pretend_to_know_the_semantics():
    (migration, _), *_ = _scaffold("1.0", "1.1")
    step = migration.steps[0]
    assert step.recipe.startswith("TODO")
    assert step.description.startswith("TODO")
    assert migration.approved is False


def test_a_scaffold_does_not_validate_until_it_is_filled_in():
    (migration, _), reg, *_ = _scaffold("1.0", "1.1")
    problems = reg.validate_migration(migration)
    assert problems
    assert any("unknown recipe" in p for p in problems)


def test_a_scaffold_for_a_vocabulary_change_proposes_a_vocabulary_step():
    (migration, notes), *_ = _scaffold("1.2", "1.3")
    step = migration.step("design_assignment_structure")
    assert step.kind is StepKind.VOCABULARY
    assert step.fidelity is Fidelity.LOSSY
    # Widening the permissible values additionally gets a note, because stored
    # values stay valid and the question still changed.
    assert any("widened its constraint" in n for n in notes)


def test_a_scaffold_for_a_removal_explains_why_there_is_no_step():
    import dataclasses

    old = sb.normalized("1.1")
    trimmed = dataclasses.replace(
        old,
        version="1.1-trimmed",
        fields=tuple(f for f in old.fields if f.field_id != "tasks[].stimuli"),
    )
    from llmbic import diff_schemas

    (migration, notes) = scaffold_migration(
        diff_schemas(old, trimmed), old=old, new=trimmed
    )
    assert not migration.steps
    assert any("stays in the artifact store" in n for n in notes)


def test_a_scaffold_serialises_to_readable_yaml():
    (migration, _), *_ = _scaffold("1.0", "1.1")
    text = dump_migration(migration)
    assert "TODO" in text
    assert yaml.safe_load(text)["approved"] is False
