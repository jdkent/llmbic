"""Scaffolding a migration from a schema diff (FR-MIG-010).

The scaffold is deliberately *incomplete*.  It enumerates every change that
needs a disposition and writes a step stub for each, marked ``TODO``, with the
kind the change most likely needs — and it refuses to guess semantics: a
scaffolded migration does not validate until a human has filled in the
transforms and recipes, which is exactly the behaviour requirement §4 asks for
when it says the tool must not "pretend to infer semantic behavior".
"""

from __future__ import annotations

from typing import Any

from ..schema.diff import ChangeKind, SchemaDiff
from ..schema.normalized import NormalizedSchema
from .spec import Fidelity, Migration, MigrationStep, Ref, StepKind

TODO = "TODO"

_KIND_HINT: dict[ChangeKind, tuple[StepKind, Fidelity, str]] = {
    ChangeKind.ADDED: (
        StepKind.SOURCE_SEMANTIC,
        Fidelity.LOSSLESS,
        "New field: decide whether it can be derived from existing values, read "
        "from stored evidence, or needs fresh source context.",
    ),
    ChangeKind.RENAMED: (
        StepKind.STRUCTURAL,
        Fidelity.LOSSLESS,
        "Rename onto a new logical identity. If the identity is preserved, delete "
        "this step: the value moves with the field id and nothing runs.",
    ),
    ChangeKind.NESTING_CHANGED: (
        StepKind.STRUCTURAL,
        Fidelity.LOSSY,
        "The field moved between nesting levels, so one value may become many or "
        "many may become one. Say which.",
    ),
    ChangeKind.TYPE_CHANGED: (
        StepKind.STRUCTURAL,
        Fidelity.LOSSY,
        "Type change: state how existing values are converted and what happens to "
        "the ones that cannot be.",
    ),
    ChangeKind.CARDINALITY_CHANGED: (
        StepKind.STRUCTURAL,
        Fidelity.LOSSY,
        "Cardinality change: collapsing a list to a scalar is lossy unless the "
        "transform proves otherwise.",
    ),
    ChangeKind.CONSTRAINT_NARROWED: (
        StepKind.VOCABULARY,
        Fidelity.LOSSY,
        "Values outside the narrowed constraint need a remap, an abstention or a "
        "review escalation.",
    ),
    ChangeKind.VOCABULARY_CHANGED: (
        StepKind.VOCABULARY,
        Fidelity.LOSSY,
        "Vocabulary change: map the values you can and escalate the ambiguous ones.",
    ),
    ChangeKind.REQUIREDNESS_CHANGED: (
        StepKind.DERIVED,
        Fidelity.LOSSLESS,
        "A field became required: decide what fills it where it is absent.",
    ),
}

#: Changes that do not need a step, but that the planner will act on.
_NO_STEP = {
    ChangeKind.REMOVED,
    ChangeKind.DESCRIPTION_CHANGED,
    ChangeKind.RECIPE_CHANGED,
    ChangeKind.CONSTRAINT_WIDENED,
    ChangeKind.COLLECTION_ADDED,
    ChangeKind.COLLECTION_REMOVED,
}


def scaffold_migration(
    diff: SchemaDiff,
    *,
    migration_id: str | None = None,
    old: NormalizedSchema | None = None,
    new: NormalizedSchema | None = None,
) -> tuple[Migration, list[str]]:
    """Return ``(migration, notes)``.

    ``notes`` explains what the author still has to decide — including the
    changes that got *no* step, so nothing disappears silently.
    """

    migration_id = migration_id or _default_id(diff)
    steps: list[MigrationStep] = []
    notes: list[str] = []
    renames: dict[str, str] = {}
    seen: set[str] = set()

    for change in diff.changes:
        if change.kind in _NO_STEP:
            notes.append(_note_for_no_step(change))
            continue
        if change.kind is ChangeKind.RENAMED and change.detail.get("identity_preserved"):
            renames[change.from_path or ""] = change.to_path or ""
            notes.append(
                f"{change.from_path} -> {change.to_path}: identity preserved, so no step "
                "is needed and the value and its evidence are untouched."
            )
            continue

        kind, fidelity, hint = _KIND_HINT.get(
            change.kind, (StepKind.MANUAL_REVIEW, Fidelity.LOSSY, "Decide what to do here.")
        )
        step_id = _step_id(change.field_id, seen)
        reads: list[Ref] = []
        if change.from_path is not None and old is not None:
            prior = old.by_path(change.from_path)
            if prior is not None:
                reads.append(Ref("field", prior.field_id))
                reads.append(Ref("evidence", prior.field_id))
        if kind in (StepKind.SOURCE_SEMANTIC, StepKind.REEXTRACTION):
            reads.append(Ref("source", "parsed"))

        steps.append(
            MigrationStep(
                id=step_id,
                kind=kind,
                writes=(change.field_id,),
                reads=tuple(reads),
                transform=f"{TODO}.{step_id}" if kind.is_deterministic else None,
                recipe=f"{TODO}-{step_id}@1" if kind.is_semantic else None,
                fidelity=fidelity,
                description=f"{TODO}: {hint}",
                entity_scope=_scope_for(change.field_id, new),
            )
        )

    if diff.rename_candidates:
        notes.append(
            "Possible renames were detected but NOT applied. Declare them in `renames:` "
            "if they are real: "
            + "; ".join(f"{a} ~ {b}" for a, b, _ in diff.rename_candidates)
        )

    migration = Migration(
        id=migration_id,
        from_schema=diff.from_schema,
        to_schema=diff.to_schema,
        steps=tuple(steps),
        renames=renames,
        approved=False,
        description=(
            f"{TODO}: scaffolded from the {diff.from_schema} -> {diff.to_schema} diff. "
            "Every step below needs a real transform or recipe before this migration "
            "can be approved."
        ),
        metadata={"scaffolded": True},
    )
    return migration, notes


def _note_for_no_step(change: Any) -> str:
    if change.kind is ChangeKind.REMOVED:
        return (
            f"{change.from_path} was removed. No step is scaffolded: the value stays in "
            "the artifact store and simply stops being assembled. Add a destructive "
            "step only if you mean to purge it."
        )
    if change.kind is ChangeKind.DESCRIPTION_CHANGED:
        return (
            f"{change.to_path} changed its description, so every value extracted under "
            "the old wording is no longer semantically current. Either bump the field's "
            "recipe and add a reextraction step, or accept the legacy values explicitly."
        )
    if change.kind is ChangeKind.RECIPE_CHANGED:
        return (
            f"{change.to_path} now requires {change.detail.get('to')!r}. Artifacts "
            "produced under the old recipe will plan as stale."
        )
    if change.kind is ChangeKind.CONSTRAINT_WIDENED:
        return (
            f"{change.to_path} widened its constraint (added {change.detail.get('added_values')}). "
            "Stored values stay valid, but an extraction that never had the new value "
            "available may have answered something else — decide whether that needs a pass."
        )
    return f"{change.kind.value} on {change.to_path or change.from_path}: no step scaffolded."


def _default_id(diff: SchemaDiff) -> str:
    return f"{diff.from_schema}-to-{diff.to_schema}".replace("@", "").replace(".", "_")


def _step_id(field_id: str, seen: set[str]) -> str:
    base = field_id.replace("[]", "").replace(".", "_")
    candidate = base
    n = 2
    while candidate in seen:
        candidate = f"{base}_{n}"
        n += 1
    seen.add(candidate)
    return candidate


def _scope_for(field_id: str, schema: NormalizedSchema | None) -> str:
    if schema is None:
        return ""
    fdef = schema.get(field_id)
    return fdef.collection_path if fdef else ""


__all__ = ["scaffold_migration"]
