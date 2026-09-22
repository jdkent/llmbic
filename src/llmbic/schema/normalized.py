"""The normalized schema representation.

Everything in llmbic plans, diffs and validates against this structure, never
against a Pydantic model or a raw JSON Schema document.  Adapters
(:mod:`llmbic.schema.adapters`) are the only code that knows about authoring
formats, which is what FR-SCH-002 and NFR-MNT-003 ask for.

A normalized schema is a *flat* list of :class:`FieldDefinition`.  Nesting is
carried in the path (``groups[].age_mean``) rather than in a tree, because the
unit of computation, invalidation and provenance is the field, not the record
(product principle 3).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Iterable, Iterator, Sequence

from ..ids import content_hash


class BaseType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    OBJECT = "object"
    ANY = "any"


@dataclass(frozen=True)
class Constraints:
    """Value constraints that are checkable without a model.

    ``enum`` is the *closed* vocabulary.  ``open_vocabulary`` marks the
    study_schema pattern where a permissible value is preferred but the
    source's own wording is allowed when none fits — a distinction that matters
    because widening an open vocabulary is not a value-invalidating change
    while narrowing a closed one is.
    """

    enum: tuple[str, ...] | None = None
    open_vocabulary: bool = False
    vocabulary_ref: str | None = None  # "AssignmentStructure@2"
    minimum: float | None = None
    maximum: float | None = None
    min_length: int | None = None
    max_length: int | None = None
    pattern: str | None = None
    min_items: int | None = None
    max_items: int | None = None

    def to_canonical(self) -> dict[str, Any]:
        out = {
            "enum": list(self.enum) if self.enum is not None else None,
            "open_vocabulary": self.open_vocabulary,
            "vocabulary_ref": self.vocabulary_ref,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "min_length": self.min_length,
            "max_length": self.max_length,
            "pattern": self.pattern,
            "min_items": self.min_items,
            "max_items": self.max_items,
        }
        return {k: v for k, v in out.items() if v not in (None, False)}

    def is_empty(self) -> bool:
        return not self.to_canonical()

    def vocabulary_part(self) -> dict[str, Any]:
        """The part of the constraints that says what the field *means*.

        Decision 18.10 lists description, prompt, validators and vocabulary as
        the things that can semantically invalidate a value — and not numeric
        or length bounds.  A permissible value appearing or disappearing
        changes the question; ``minimum: 0 -> 1`` does not, it only changes
        which stored answers still validate.
        """

        return {
            "enum": list(self.enum) if self.enum is not None else None,
            "open_vocabulary": self.open_vocabulary,
            "vocabulary_ref": self.vocabulary_ref,
        }

    def bounds_part(self) -> dict[str, Any]:
        """The purely checkable part: what validation, not extraction, uses."""

        out = {
            "minimum": self.minimum,
            "maximum": self.maximum,
            "min_length": self.min_length,
            "max_length": self.max_length,
            "pattern": self.pattern,
            "min_items": self.min_items,
            "max_items": self.max_items,
        }
        return {k: v for k, v in out.items() if v is not None}


@dataclass(frozen=True)
class FieldDefinition:
    """One slot, with an identity that survives renaming.

    ``field_id`` is the stable logical identity (FR-SCH-005).  ``path`` is the
    current location, which a structural migration may change without touching
    ``field_id`` — and FR-DEP-003 then guarantees the value and its evidence are
    not invalidated.
    """

    field_id: str
    path: str
    base_type: BaseType = BaseType.STRING
    multivalued: bool = False
    required: bool = False
    description: str = ""
    constraints: Constraints = field(default_factory=Constraints)
    #: ``name@version`` of the recipe that is *required* to produce this field.
    #: A field whose artifact was produced under a different recipe is not
    #: semantically current (FR-DEP-004).
    recipe_ref: str | None = None
    #: True when the field is filled by deterministic code rather than by a
    #: model (study_schema's ``deterministic`` marker).
    deterministic: bool = False
    #: True when the field carries evidence spans.
    evidence_bearing: bool = True
    #: Arbitrary semantic annotations from the authoring format (FR-SCH-006).
    annotations: dict[str, Any] = field(default_factory=dict)

    # ---- derived ---------------------------------------------------------
    @property
    def collection_path(self) -> str:
        """The path of the collection this field lives in, ``""`` at the root.

        ``groups[].age_mean`` -> ``groups[]``;  ``description`` -> ``""``.
        """

        return parent_collection(self.path)

    @property
    def leaf(self) -> str:
        return self.path.rsplit(".", 1)[-1]

    @property
    def depth(self) -> int:
        return self.path.count(".")

    def semantic_key(self) -> str:
        """Hash of everything that changes what the field *means*.

        FR-SCH-007: a description change with no structural change must be
        detectable, because the description is the extraction instruction.
        """

        return content_hash(
            {
                "description": _normalize_ws(self.description),
                "vocabulary": self.constraints.vocabulary_part(),
                "base_type": self.base_type.value,
                "multivalued": self.multivalued,
                "recipe_ref": self.recipe_ref,
                "annotations": {
                    k: v for k, v in sorted(self.annotations.items()) if k != "source_path"
                },
            }
        )

    def structural_key(self) -> str:
        """Hash of everything that changes the record's *shape*."""

        return content_hash(
            {
                "path": self.path,
                "base_type": self.base_type.value,
                "multivalued": self.multivalued,
                "required": self.required,
                "constraints": self.constraints.to_canonical(),
            }
        )

    def to_canonical(self) -> dict[str, Any]:
        return {
            "field_id": self.field_id,
            "path": self.path,
            "base_type": self.base_type.value,
            "multivalued": self.multivalued,
            "required": self.required,
            "description": _normalize_ws(self.description),
            "constraints": self.constraints.to_canonical(),
            "recipe_ref": self.recipe_ref,
            "deterministic": self.deterministic,
            "evidence_bearing": self.evidence_bearing,
            "annotations": dict(sorted(self.annotations.items())),
        }

    @classmethod
    def from_canonical(cls, data: dict[str, Any]) -> "FieldDefinition":
        return cls(
            field_id=data["field_id"],
            path=data["path"],
            base_type=BaseType(data.get("base_type", "string")),
            multivalued=bool(data.get("multivalued", False)),
            required=bool(data.get("required", False)),
            description=data.get("description", ""),
            constraints=Constraints(
                enum=tuple(data["constraints"]["enum"])
                if data.get("constraints", {}).get("enum") is not None
                else None,
                open_vocabulary=data.get("constraints", {}).get("open_vocabulary", False),
                vocabulary_ref=data.get("constraints", {}).get("vocabulary_ref"),
                minimum=data.get("constraints", {}).get("minimum"),
                maximum=data.get("constraints", {}).get("maximum"),
                min_length=data.get("constraints", {}).get("min_length"),
                max_length=data.get("constraints", {}).get("max_length"),
                pattern=data.get("constraints", {}).get("pattern"),
                min_items=data.get("constraints", {}).get("min_items"),
                max_items=data.get("constraints", {}).get("max_items"),
            ),
            recipe_ref=data.get("recipe_ref"),
            deterministic=bool(data.get("deterministic", False)),
            evidence_bearing=bool(data.get("evidence_bearing", True)),
            annotations=dict(data.get("annotations") or {}),
        )


@dataclass(frozen=True)
class CollectionDefinition:
    """A repeated nested entity, e.g. ``groups[]``.

    Members have persistent logical identities (FR-DEP-008).  ``identity_field``
    names the slot inside the entity that carries the source-local id; llmbic
    uses it when importing records so that reordering a list does not rewrite
    every artifact.
    """

    path: str
    identity_field: str | None = "local_id"
    description: str = ""

    @property
    def parent_path(self) -> str:
        segs = path_segments(self.path)
        return ".".join(segs[:-1]) if len(segs) > 1 else ""

    def to_canonical(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "identity_field": self.identity_field,
            "description": _normalize_ws(self.description),
        }


@dataclass(frozen=True)
class NormalizedSchema:
    """An immutable snapshot of one schema version."""

    name: str
    version: str
    fields: tuple[FieldDefinition, ...] = ()
    collections: tuple[CollectionDefinition, ...] = ()
    title: str = ""
    description: str = ""
    annotations: dict[str, Any] = field(default_factory=dict)

    # ---- lookup ----------------------------------------------------------
    def __post_init__(self) -> None:
        by_id: dict[str, FieldDefinition] = {}
        by_path: dict[str, FieldDefinition] = {}
        for f in self.fields:
            if f.field_id in by_id:
                raise ValueError(f"duplicate field_id {f.field_id!r} in {self.name}@{self.version}")
            if f.path in by_path:
                raise ValueError(f"duplicate path {f.path!r} in {self.name}@{self.version}")
            by_id[f.field_id] = f
            by_path[f.path] = f
        object.__setattr__(self, "_by_id", by_id)
        object.__setattr__(self, "_by_path", by_path)

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.version}"

    def field(self, field_id: str) -> FieldDefinition:
        return self._by_id[field_id]  # type: ignore[attr-defined]

    def get(self, field_id: str) -> FieldDefinition | None:
        return self._by_id.get(field_id)  # type: ignore[attr-defined]

    def by_path(self, path: str) -> FieldDefinition | None:
        return self._by_path.get(path)  # type: ignore[attr-defined]

    def field_ids(self) -> set[str]:
        return set(self._by_id)  # type: ignore[attr-defined]

    def collection(self, path: str) -> CollectionDefinition | None:
        for c in self.collections:
            if c.path == path:
                return c
        return None

    def fields_in(self, collection_path: str) -> list[FieldDefinition]:
        return [f for f in self.fields if f.collection_path == collection_path]

    def __iter__(self) -> Iterator[FieldDefinition]:
        return iter(self.fields)

    def __len__(self) -> int:
        return len(self.fields)

    # ---- transformation --------------------------------------------------
    def with_fields(self, fields: Iterable[FieldDefinition]) -> "NormalizedSchema":
        return replace(self, fields=tuple(fields))

    def rekey(self, identity_map: dict[str, str]) -> "NormalizedSchema":
        """Return a copy where ``path -> field_id`` overrides are applied.

        This is how a rename keeps its identity: registering ``study@1.3`` with
        ``identity_map={"tasks[].response_modality": "<id of response_mode>"}``
        makes the new schema's field the *same* logical field, so FR-DEP-003
        applies and nothing is invalidated.
        """

        out = []
        for f in self.fields:
            if f.path in identity_map:
                out.append(replace(f, field_id=identity_map[f.path]))
            else:
                out.append(f)
        return replace(self, fields=tuple(out))

    # ---- identity --------------------------------------------------------
    def structural_hash(self) -> str:
        return content_hash(
            {
                "name": self.name,
                "fields": sorted((f.structural_key() for f in self.fields)),
                "collections": [c.to_canonical() for c in sorted(self.collections, key=lambda c: c.path)],
            }
        )

    def semantic_hash(self) -> str:
        return content_hash(
            {
                "name": self.name,
                "fields": sorted(f"{f.field_id}:{f.semantic_key()}" for f in self.fields),
            }
        )

    def schema_hash(self) -> str:
        return content_hash([self.structural_hash(), self.semantic_hash(), self.version])

    def to_canonical(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "title": self.title,
            "description": _normalize_ws(self.description),
            "fields": [f.to_canonical() for f in self.fields],
            "collections": [c.to_canonical() for c in self.collections],
            "annotations": dict(sorted(self.annotations.items())),
        }

    @classmethod
    def from_canonical(cls, data: dict[str, Any]) -> "NormalizedSchema":
        return cls(
            name=data["name"],
            version=data["version"],
            title=data.get("title", ""),
            description=data.get("description", ""),
            fields=tuple(FieldDefinition.from_canonical(f) for f in data.get("fields", [])),
            collections=tuple(
                CollectionDefinition(
                    path=c["path"],
                    identity_field=c.get("identity_field"),
                    description=c.get("description", ""),
                )
                for c in data.get("collections", [])
            ),
            annotations=dict(data.get("annotations") or {}),
        )


_WS_RE = re.compile(r"\s+")


def _normalize_ws(text: str) -> str:
    """Collapse whitespace so reflowing a description is not a semantic change.

    Rewrapping a YAML block should not invalidate a corpus.  Changing the words
    should.
    """

    return _WS_RE.sub(" ", (text or "")).strip()


def path_segments(path: str) -> Sequence[str]:
    """``groups[].assessments[].name`` -> ``["groups[]", "assessments[]", "name"]``."""

    return path.split(".") if path else []


def parent_collection(path: str) -> str:
    """The enclosing collection path of a field path, ``""`` at the root."""

    segs = path_segments(path)
    for i in range(len(segs) - 1, -1, -1):
        if segs[i].endswith("[]"):
            return ".".join(segs[: i + 1])
    return ""
