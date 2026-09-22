"""Structural and semantic comparison of two schema versions.

FR-SCH-003/004/007/008.  Three things are reported *independently*:

``shape``
    Can a record of the old shape be parsed as the new one?
``values``
    Do the stored values remain valid under the new constraints?
``semantics``
    Does the new schema still ask for the same thing?

A diff never authorises data movement.  A field that disappeared from one path
and appeared at another is reported as ``removed`` + ``added`` unless the
caller explicitly declared the rename (FR-SCH-004): name similarity is a hint
for a human, never a migration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum
from typing import Any, Iterable

from .normalized import FieldDefinition, NormalizedSchema, _normalize_ws


class ChangeKind(str, Enum):
    ADDED = "added"
    REMOVED = "removed"
    RENAMED = "renamed"  # only when explicitly mapped
    TYPE_CHANGED = "type_changed"
    CARDINALITY_CHANGED = "cardinality_changed"
    REQUIREDNESS_CHANGED = "requiredness_changed"
    NESTING_CHANGED = "nesting_changed"
    CONSTRAINT_WIDENED = "constraint_widened"
    CONSTRAINT_NARROWED = "constraint_narrowed"
    VOCABULARY_CHANGED = "vocabulary_changed"
    DESCRIPTION_CHANGED = "description_changed"
    RECIPE_CHANGED = "recipe_changed"
    COLLECTION_ADDED = "collection_added"
    COLLECTION_REMOVED = "collection_removed"


#: Changes that alter the record's shape.
SHAPE_KINDS = frozenset(
    {
        ChangeKind.ADDED,
        ChangeKind.REMOVED,
        ChangeKind.RENAMED,
        ChangeKind.TYPE_CHANGED,
        ChangeKind.CARDINALITY_CHANGED,
        ChangeKind.REQUIREDNESS_CHANGED,
        ChangeKind.NESTING_CHANGED,
        ChangeKind.COLLECTION_ADDED,
        ChangeKind.COLLECTION_REMOVED,
    }
)

#: Changes that can make an already-stored value invalid.
VALUE_KINDS = frozenset(
    {
        ChangeKind.TYPE_CHANGED,
        ChangeKind.CARDINALITY_CHANGED,
        ChangeKind.CONSTRAINT_NARROWED,
        ChangeKind.REMOVED,
    }
)

#: Changes that alter what the field *means* and therefore whether a value
#: extracted under the old definition is still the right answer.
SEMANTIC_KINDS = frozenset(
    {
        ChangeKind.DESCRIPTION_CHANGED,
        ChangeKind.RECIPE_CHANGED,
        ChangeKind.VOCABULARY_CHANGED,
        ChangeKind.ADDED,
        ChangeKind.CONSTRAINT_NARROWED,
        ChangeKind.CONSTRAINT_WIDENED,
    }
)


@dataclass(frozen=True)
class FieldChange:
    kind: ChangeKind
    field_id: str
    from_path: str | None = None
    to_path: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_canonical(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "field_id": self.field_id,
            "from_path": self.from_path,
            "to_path": self.to_path,
            "detail": self.detail,
        }

    def describe(self) -> str:
        where = self.to_path or self.from_path or self.field_id
        extra = ", ".join(f"{k}={v!r}" for k, v in sorted(self.detail.items()) if k != "similar_to")
        return f"{self.kind.value:<22} {where}" + (f"  ({extra})" if extra else "")


@dataclass(frozen=True)
class SchemaDiff:
    from_schema: str
    to_schema: str
    changes: tuple[FieldChange, ...] = ()
    #: Unmapped add/remove pairs whose names look alike.  Advisory only —
    #: FR-SCH-004 forbids acting on them without explicit confirmation.
    rename_candidates: tuple[tuple[str, str, float], ...] = ()

    def of_kind(self, *kinds: ChangeKind) -> list[FieldChange]:
        want = set(kinds)
        return [c for c in self.changes if c.kind in want]

    def for_field(self, field_id: str) -> list[FieldChange]:
        return [c for c in self.changes if c.field_id == field_id]

    # ---- FR-SCH-008: three independent compatibility verdicts -------------
    @property
    def shape_compatible(self) -> bool:
        return not any(c.kind in SHAPE_KINDS for c in self.changes)

    @property
    def values_compatible(self) -> bool:
        return not any(c.kind in VALUE_KINDS for c in self.changes)

    @property
    def semantics_compatible(self) -> bool:
        return not any(c.kind in SEMANTIC_KINDS for c in self.changes)

    def compatibility(self) -> dict[str, bool]:
        return {
            "shape": self.shape_compatible,
            "values": self.values_compatible,
            "semantics": self.semantics_compatible,
        }

    def is_empty(self) -> bool:
        return not self.changes

    def to_canonical(self) -> dict[str, Any]:
        return {
            "from_schema": self.from_schema,
            "to_schema": self.to_schema,
            "compatibility": self.compatibility(),
            "changes": [c.to_canonical() for c in self.changes],
            "rename_candidates": [
                {"removed": a, "added": b, "similarity": round(s, 3)}
                for a, b, s in self.rename_candidates
            ],
        }

    def render(self) -> str:
        lines = [f"{self.from_schema} -> {self.to_schema}"]
        compat = self.compatibility()
        lines.append(
            "  compatible: "
            + ", ".join(f"{k}={'yes' if v else 'no'}" for k, v in compat.items())
        )
        if not self.changes:
            lines.append("  (no changes)")
        for c in self.changes:
            lines.append("  " + c.describe())
        if self.rename_candidates:
            lines.append("  possible renames (NOT applied - declare them explicitly):")
            for a, b, s in self.rename_candidates:
                lines.append(f"    {a} ~ {b}  ({s:.0%})")
        return "\n".join(lines)


def diff_schemas(
    old: NormalizedSchema,
    new: NormalizedSchema,
    *,
    renames: dict[str, str] | None = None,
    rename_candidate_threshold: float = 0.6,
) -> SchemaDiff:
    """Compare two normalized schemas.

    ``renames`` maps *old path* to *new path* and is the only thing that turns
    a remove/add pair into a :data:`ChangeKind.RENAMED`.  When the two schemas
    already share a ``field_id`` for the two paths (because the new version was
    registered with an identity map), the rename is detected without needing
    ``renames`` — the identity is the confirmation.
    """

    renames = dict(renames or {})
    changes: list[FieldChange] = []

    old_by_id = {f.field_id: f for f in old.fields}
    new_by_id = {f.field_id: f for f in new.fields}
    old_by_path = {f.path: f for f in old.fields}
    new_by_path = {f.path: f for f in new.fields}

    # Fold explicit path renames into an identity correspondence.
    pairs: list[tuple[FieldDefinition, FieldDefinition]] = []
    matched_old: set[str] = set()
    matched_new: set[str] = set()

    for old_path, new_path in sorted(renames.items()):
        of = old_by_path.get(old_path)
        nf = new_by_path.get(new_path)
        if of is None or nf is None:
            continue
        pairs.append((of, nf))
        matched_old.add(of.field_id)
        matched_new.add(nf.field_id)

    for fid, of in sorted(old_by_id.items()):
        if fid in matched_old:
            continue
        nf = new_by_id.get(fid)
        if nf is not None and nf.field_id not in matched_new:
            pairs.append((of, nf))
            matched_old.add(fid)
            matched_new.add(nf.field_id)

    for of, nf in sorted(pairs, key=lambda p: p[1].path):
        changes.extend(_compare_field(of, nf))

    removed = [f for f in old.fields if f.field_id not in matched_old]
    added = [f for f in new.fields if f.field_id not in matched_new]

    for f in sorted(removed, key=lambda f: f.path):
        changes.append(FieldChange(ChangeKind.REMOVED, f.field_id, from_path=f.path))
    for f in sorted(added, key=lambda f: f.path):
        changes.append(
            FieldChange(
                ChangeKind.ADDED,
                f.field_id,
                to_path=f.path,
                detail={"required": f.required, "recipe_ref": f.recipe_ref},
            )
        )

    old_colls = {c.path for c in old.collections}
    new_colls = {c.path for c in new.collections}
    for path in sorted(new_colls - old_colls):
        changes.append(FieldChange(ChangeKind.COLLECTION_ADDED, path, to_path=path))
    for path in sorted(old_colls - new_colls):
        changes.append(FieldChange(ChangeKind.COLLECTION_REMOVED, path, from_path=path))

    candidates = _rename_candidates(removed, added, rename_candidate_threshold)

    return SchemaDiff(
        from_schema=old.ref,
        to_schema=new.ref,
        changes=tuple(changes),
        rename_candidates=tuple(candidates),
    )


def _compare_field(old: FieldDefinition, new: FieldDefinition) -> Iterable[FieldChange]:
    fid = new.field_id
    if old.path != new.path:
        kind = (
            ChangeKind.NESTING_CHANGED
            if old.collection_path != new.collection_path
            else ChangeKind.RENAMED
        )
        # A rename that keeps the logical identity moves no data: artifacts are
        # keyed by field_id, so the new path is a display change and FR-DEP-003
        # applies.  A rename onto a *new* identity is a real data movement and
        # needs a step.
        yield FieldChange(
            kind,
            fid,
            from_path=old.path,
            to_path=new.path,
            detail={"identity_preserved": old.field_id == new.field_id},
        )

    if old.base_type != new.base_type:
        yield FieldChange(
            ChangeKind.TYPE_CHANGED,
            fid,
            from_path=old.path,
            to_path=new.path,
            detail={"from": old.base_type.value, "to": new.base_type.value},
        )

    if old.multivalued != new.multivalued:
        yield FieldChange(
            ChangeKind.CARDINALITY_CHANGED,
            fid,
            from_path=old.path,
            to_path=new.path,
            detail={"from_multivalued": old.multivalued, "to_multivalued": new.multivalued},
        )

    if old.required != new.required:
        yield FieldChange(
            ChangeKind.REQUIREDNESS_CHANGED,
            fid,
            from_path=old.path,
            to_path=new.path,
            detail={"from_required": old.required, "to_required": new.required},
        )

    yield from _compare_constraints(old, new)

    if _normalize_ws(old.description) != _normalize_ws(new.description):
        yield FieldChange(
            ChangeKind.DESCRIPTION_CHANGED,
            fid,
            from_path=old.path,
            to_path=new.path,
            detail={
                "from_hash": old.semantic_key(),
                "to_hash": new.semantic_key(),
                "from_chars": len(_normalize_ws(old.description)),
                "to_chars": len(_normalize_ws(new.description)),
            },
        )

    if old.recipe_ref != new.recipe_ref:
        yield FieldChange(
            ChangeKind.RECIPE_CHANGED,
            fid,
            from_path=old.path,
            to_path=new.path,
            detail={"from": old.recipe_ref, "to": new.recipe_ref},
        )


def _compare_constraints(old: FieldDefinition, new: FieldDefinition) -> Iterable[FieldChange]:
    oc, nc = old.constraints, new.constraints
    fid = new.field_id

    if oc.vocabulary_ref != nc.vocabulary_ref:
        yield FieldChange(
            ChangeKind.VOCABULARY_CHANGED,
            fid,
            from_path=old.path,
            to_path=new.path,
            detail={"from": oc.vocabulary_ref, "to": nc.vocabulary_ref},
        )

    old_enum = set(oc.enum or ())
    new_enum = set(nc.enum or ())
    if old_enum != new_enum:
        gone = sorted(old_enum - new_enum)
        fresh = sorted(new_enum - old_enum)
        # Removing a permissible value can invalidate stored values; adding one
        # cannot, but it does change what the field means (an extraction that
        # never had `observational_cohorts` available may have answered
        # `parallel` because nothing better existed).
        if gone:
            yield FieldChange(
                ChangeKind.CONSTRAINT_NARROWED,
                fid,
                from_path=old.path,
                to_path=new.path,
                detail={"removed_values": gone, "added_values": fresh},
            )
        elif fresh:
            yield FieldChange(
                ChangeKind.CONSTRAINT_WIDENED,
                fid,
                from_path=old.path,
                to_path=new.path,
                detail={"added_values": fresh},
            )

    if oc.open_vocabulary != nc.open_vocabulary:
        kind = (
            ChangeKind.CONSTRAINT_NARROWED
            if oc.open_vocabulary and not nc.open_vocabulary
            else ChangeKind.CONSTRAINT_WIDENED
        )
        yield FieldChange(
            kind,
            fid,
            from_path=old.path,
            to_path=new.path,
            detail={"open_vocabulary": nc.open_vocabulary},
        )

    numeric = [
        ("minimum", oc.minimum, nc.minimum, "raise"),
        ("maximum", oc.maximum, nc.maximum, "lower"),
        ("min_length", oc.min_length, nc.min_length, "raise"),
        ("max_length", oc.max_length, nc.max_length, "lower"),
        ("min_items", oc.min_items, nc.min_items, "raise"),
        ("max_items", oc.max_items, nc.max_items, "lower"),
    ]
    for name, o, n, tighten in numeric:
        if o == n:
            continue
        if o is None:
            narrowed = True
        elif n is None:
            narrowed = False
        else:
            narrowed = n > o if tighten == "raise" else n < o
        yield FieldChange(
            ChangeKind.CONSTRAINT_NARROWED if narrowed else ChangeKind.CONSTRAINT_WIDENED,
            fid,
            from_path=old.path,
            to_path=new.path,
            detail={"constraint": name, "from": o, "to": n},
        )

    if oc.pattern != nc.pattern:
        yield FieldChange(
            ChangeKind.CONSTRAINT_NARROWED if nc.pattern else ChangeKind.CONSTRAINT_WIDENED,
            fid,
            from_path=old.path,
            to_path=new.path,
            detail={"constraint": "pattern", "from": oc.pattern, "to": nc.pattern},
        )


def _rename_candidates(
    removed: list[FieldDefinition],
    added: list[FieldDefinition],
    threshold: float,
) -> list[tuple[str, str, float]]:
    """Advisory only.  Never consumed by the planner."""

    out: list[tuple[str, str, float]] = []
    for r in removed:
        for a in added:
            if r.collection_path != a.collection_path:
                continue
            score = SequenceMatcher(None, r.leaf, a.leaf).ratio()
            if score >= threshold:
                out.append((r.path, a.path, score))
    out.sort(key=lambda t: (-t[2], t[0], t[1]))
    return out
