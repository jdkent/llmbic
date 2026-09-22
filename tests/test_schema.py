"""Schema normalization, stable field identities and diffing (§15.1)."""

from __future__ import annotations

import pytest

import studybed as sb
from llmbic import BaseType, ChangeKind, diff_schemas, from_json_schema
from llmbic.schema.normalized import Constraints, FieldDefinition, parent_collection


# ---- normalization -------------------------------------------------------

def test_nested_objects_flatten_to_dotted_paths():
    schema = sb.normalized("1.0")
    assert schema.by_path("design.assignment_structure") is not None
    assert schema.by_path("design.assignment_structure").collection_path == ""


def test_arrays_of_objects_become_collections_with_identity_fields():
    schema = sb.normalized("1.0")
    assert {c.path for c in schema.collections} == {"groups[]", "tasks[]"}
    assert schema.collection("groups[]").identity_field == "local_id"
    assert schema.by_path("groups[].n").collection_path == "groups[]"


def test_arrays_of_scalars_are_multivalued_fields_not_collections():
    schema = sb.normalized("1.0")
    field = schema.by_path("groups[].medical_condition")
    assert field.multivalued is True
    assert field.base_type is BaseType.STRING
    assert "groups[].medical_condition[]" not in {c.path for c in schema.collections}


def test_open_vocabulary_is_distinguished_from_a_closed_enum():
    schema = sb.normalized("1.1")
    modality = schema.by_path("tasks[].stimulus_modality")
    assert modality.constraints.open_vocabulary is True
    assert "visual" in modality.constraints.enum

    allocation = schema.by_path("design.allocation")
    assert allocation.constraints.open_vocabulary is False


def test_duplicate_paths_are_rejected():
    with pytest.raises(ValueError, match="duplicate path"):
        sb.normalized("1.0").with_fields(
            [
                FieldDefinition("a", "x"),
                FieldDefinition("b", "x"),
            ]
        )


@pytest.mark.parametrize(
    "path,expected",
    [
        ("title", ""),
        ("design.allocation", ""),
        ("groups[].n", "groups[]"),
        ("tasks[].conditions[].name", "tasks[].conditions[]"),
        ("", ""),
    ],
)
def test_parent_collection(path, expected):
    assert parent_collection(path) == expected


# ---- stable identity (FR-SCH-005) ---------------------------------------

def test_rename_keeps_its_field_identity_when_declared():
    v1_0 = sb.normalized("1.0")
    v1_1 = sb.normalized("1.1")
    old = v1_0.by_path("tasks[].response_mode")
    new = v1_1.by_path("tasks[].response_modality")
    assert old.field_id == new.field_id == "tasks[].response_mode"


def test_without_an_identity_map_a_rename_is_two_different_fields():
    bare = from_json_schema(sb.schema_v1_1(), name="study", version="1.1")
    assert bare.by_path("tasks[].response_modality").field_id == "tasks[].response_modality"
    assert bare.get("tasks[].response_mode") is None


# ---- semantic vs structural identity ------------------------------------

def test_reflowing_a_description_is_not_a_semantic_change():
    a = FieldDefinition("f", "f", description="one   two\n three")
    b = FieldDefinition("f", "f", description="one two three")
    assert a.semantic_key() == b.semantic_key()


def test_rewording_a_description_is_a_semantic_change():
    a = FieldDefinition("f", "f", description="Visuals presented to participants.")
    b = FieldDefinition("f", "f", description="The materials presented, in the source's terms.")
    assert a.semantic_key() != b.semantic_key()


def test_renaming_a_field_does_not_change_its_semantic_key():
    """FR-DEP-003: a rename must not be able to invalidate a value."""

    a = FieldDefinition("f", "old.path", description="same")
    b = FieldDefinition("f", "new.path", description="same")
    assert a.semantic_key() == b.semantic_key()
    assert a.structural_key() != b.structural_key()


def test_numeric_bounds_are_validation_not_semantics():
    """Decision 18.10 lists description, prompt, validators and vocabulary."""

    a = FieldDefinition("n", "n", constraints=Constraints(minimum=0))
    b = FieldDefinition("n", "n", constraints=Constraints(minimum=1))
    assert a.semantic_key() == b.semantic_key()
    assert a.structural_key() != b.structural_key()


def test_vocabulary_changes_are_semantics():
    a = FieldDefinition("k", "k", constraints=Constraints(enum=("a", "b")))
    b = FieldDefinition("k", "k", constraints=Constraints(enum=("a", "b", "c")))
    assert a.semantic_key() != b.semantic_key()


# ---- diffing -------------------------------------------------------------

def test_the_1_1_diff_reads_like_the_commit_that_produced_it():
    diff = diff_schemas(sb.normalized("1.0"), sb.normalized("1.1"))
    kinds = {(c.kind, c.to_path or c.from_path) for c in diff.changes}
    assert (ChangeKind.RENAMED, "tasks[].response_modality") in kinds
    assert (ChangeKind.ADDED, "tasks[].stimulus_modality") in kinds
    assert (ChangeKind.DESCRIPTION_CHANGED, "tasks[].stimuli") in kinds


def test_an_identity_preserving_rename_is_marked_as_such():
    diff = diff_schemas(sb.normalized("1.0"), sb.normalized("1.1"))
    rename = diff.of_kind(ChangeKind.RENAMED)[0]
    assert rename.detail["identity_preserved"] is True


def test_compatibility_is_reported_on_three_independent_axes():
    """FR-SCH-008."""

    diff = diff_schemas(sb.normalized("1.2"), sb.normalized("1.3"))
    compat = diff.compatibility()
    # Adding a permissible value changes nothing about the shape or the stored
    # values, and changes everything about what the field is asking.
    assert compat["shape"] is True
    assert compat["values"] is True
    assert compat["semantics"] is False


def test_narrowing_a_constraint_is_a_value_compatibility_problem():
    diff = diff_schemas(sb.normalized("1.3"), sb.normalized("1.4"))
    assert diff.compatibility()["shape"] is True
    assert diff.compatibility()["values"] is False
    narrowed = diff.of_kind(ChangeKind.CONSTRAINT_NARROWED)
    assert [c.detail["constraint"] for c in narrowed] == ["minimum"]


def test_similar_names_are_reported_as_candidates_but_never_applied():
    """FR-SCH-004: name similarity is a hint for a human, not a migration."""

    old = from_json_schema(sb.schema_v1_0(), name="study", version="1.0")
    new = from_json_schema(sb.schema_v1_1(), name="study", version="1.1")
    diff = diff_schemas(old, new)

    assert not diff.of_kind(ChangeKind.RENAMED)
    assert {c.from_path for c in diff.of_kind(ChangeKind.REMOVED)} == {
        "tasks[].response_mode"
    }
    assert any(
        a == "tasks[].response_mode" and b == "tasks[].response_modality"
        for a, b, _ in diff.rename_candidates
    )


def test_an_explicit_rename_map_turns_the_pair_into_one_change():
    old = from_json_schema(sb.schema_v1_0(), name="study", version="1.0")
    new = from_json_schema(sb.schema_v1_1(), name="study", version="1.1")
    diff = diff_schemas(
        old, new, renames={"tasks[].response_mode": "tasks[].response_modality"}
    )
    assert len(diff.of_kind(ChangeKind.RENAMED)) == 1
    assert not diff.of_kind(ChangeKind.REMOVED)


def test_diff_of_a_schema_with_itself_is_empty():
    schema = sb.normalized("1.2")
    assert diff_schemas(schema, schema).is_empty()
    assert diff_schemas(schema, schema).compatibility() == {
        "shape": True,
        "values": True,
        "semantics": True,
    }


def test_diff_renders_without_exploding():
    text = diff_schemas(sb.normalized("1.0"), sb.normalized("1.1")).render()
    assert "study@1.0 -> study@1.1" in text
    assert "compatible:" in text


# ---- json schema adapter corners ----------------------------------------

def test_refs_into_defs_are_resolved():
    doc = {
        "type": "object",
        "properties": {"a": {"$ref": "#/$defs/Inner"}},
        "$defs": {
            "Inner": {"type": "object", "properties": {"b": {"type": "integer"}}}
        },
    }
    schema = from_json_schema(doc, name="t", version="1")
    assert schema.by_path("a.b").base_type is BaseType.INTEGER


def test_recursive_definitions_terminate():
    doc = {
        "type": "object",
        "properties": {"node": {"$ref": "#/$defs/Node"}},
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "children": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/Node"},
                    },
                },
            }
        },
    }
    schema = from_json_schema(doc, name="t", version="1")
    assert schema.by_path("node.label") is not None
    assert "node.children[]" in {c.path for c in schema.collections}


def test_anyof_enum_plus_string_is_an_open_vocabulary():
    doc = {
        "type": "object",
        "properties": {
            "k": {"anyOf": [{"type": "string", "enum": ["a", "b"]}, {"type": "string"}]}
        },
    }
    schema = from_json_schema(doc, name="t", version="1")
    field = schema.by_path("k")
    assert field.constraints.enum == ("a", "b")
    assert field.constraints.open_vocabulary is True


def test_pydantic_adapter_matches_the_json_schema_adapter():
    pydantic = pytest.importorskip("pydantic")

    class Inner(pydantic.BaseModel):
        label: str

    class Outer(pydantic.BaseModel):
        name: str
        count: int = 0
        items: list[Inner] = []

    from llmbic import from_pydantic

    schema = from_pydantic(Outer, name="t", version="1")
    assert schema.by_path("name").required is True
    assert schema.by_path("count").required is False
    assert "items[]" in {c.path for c in schema.collections}
    assert schema.by_path("items[].label").base_type is BaseType.STRING
