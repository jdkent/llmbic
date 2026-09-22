"""YAML projection of a migration (requirements §10).

The internal representation is Python; this is the serialisation that makes a
migration inspectable and reviewable as an ordinary file in the repository
(FR-MIG-009).  The two directions are exact inverses for everything the
dataclasses carry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from ..context.policy import ContextPolicy, OnMissingContext, policy_from_spec
from ..errors import ErrorCode, RegistryError
from .spec import Fidelity, Migration, MigrationStep, Ref, StepKind


def migration_from_dict(data: Mapping[str, Any]) -> Migration:
    try:
        steps = [_step_from_dict(s) for s in data.get("steps") or ()]
    except KeyError as exc:
        raise RegistryError(
            f"migration {data.get('id')!r} has a step missing {exc.args[0]!r}",
            code=ErrorCode.MIGRATION_INVALID,
        ) from exc

    return Migration(
        id=str(data["id"]),
        from_schema=str(data["from_schema"] if "from_schema" in data else data["from"]),
        to_schema=str(data["to_schema"] if "to_schema" in data else data["to"]),
        steps=tuple(steps),
        description=data.get("description", ""),
        renames=dict(data.get("renames") or {}),
        acknowledged=dict(data.get("acknowledged") or {}),
        downgrade=data.get("downgrade"),
        is_downgrade=bool(data.get("is_downgrade", False)),
        branch=data.get("branch", "main"),
        approved=bool(data.get("approved", True)),
        metadata=dict(data.get("metadata") or {}),
    )


def _step_from_dict(data: Mapping[str, Any]) -> MigrationStep:
    context = data.get("context")
    policy: ContextPolicy | None = None
    if context:
        spec = dict(context)
        if "on_missing_context" not in spec and data.get("on_missing_context"):
            spec["on_missing_context"] = data["on_missing_context"]
        policy = policy_from_spec(spec)

    return MigrationStep(
        id=str(data["id"]),
        kind=StepKind(data["kind"]),
        writes=tuple(_names(data.get("writes") or ())),
        reads=tuple(Ref.coerce(r) for r in data.get("reads") or ()),
        transform=data.get("transform"),
        recipe=data.get("recipe"),
        context=policy,
        on_missing_context=OnMissingContext(data["on_missing_context"])
        if data.get("on_missing_context")
        else None,
        validators=tuple(data.get("validators") or ()),
        fidelity=Fidelity(data.get("fidelity", "lossless")),
        entity_scope=data.get("entity_scope", ""),
        after=tuple(data.get("after") or ()),
        params=dict(data.get("params") or {}),
        description=data.get("description", ""),
        reusable=bool(data.get("reusable", True)),
    )


def _names(values: Iterable[Any]) -> list[str]:
    out = []
    for v in values:
        text = str(v)
        out.append(text.split(":", 1)[1] if text.startswith("field:") else text)
    return out


def migration_to_dict(migration: Migration) -> dict[str, Any]:
    """The YAML shape, dropping defaults so the file stays readable."""

    out: dict[str, Any] = {
        "id": migration.id,
        "from_schema": migration.from_schema,
        "to_schema": migration.to_schema,
    }
    if migration.description:
        out["description"] = migration.description
    if migration.renames:
        out["renames"] = dict(sorted(migration.renames.items()))
    if migration.acknowledged:
        out["acknowledged"] = dict(sorted(migration.acknowledged.items()))
    if migration.branch != "main":
        out["branch"] = migration.branch
    if not migration.approved:
        out["approved"] = False
    if migration.is_downgrade:
        out["is_downgrade"] = True
    if migration.downgrade:
        out["downgrade"] = migration.downgrade

    steps: list[dict[str, Any]] = []
    for s in migration.steps:
        step: dict[str, Any] = {"id": s.id, "kind": s.kind.value}
        if s.description:
            step["description"] = s.description
        if s.reads:
            step["reads"] = [str(r) for r in s.reads]
        if s.writes:
            step["writes"] = [f"field:{w}" for w in s.writes]
        if s.transform:
            step["transform"] = s.transform
        if s.recipe:
            step["recipe"] = s.recipe
        if s.context is not None:
            step["context"] = _context_to_spec(s.context)
        if s.on_missing_context is not None:
            step["on_missing_context"] = s.on_missing_context.value
        if s.validators:
            step["validators"] = list(s.validators)
        if s.fidelity is not Fidelity.LOSSLESS:
            step["fidelity"] = s.fidelity.value
        if s.entity_scope:
            step["entity_scope"] = s.entity_scope
        if s.after:
            step["after"] = list(s.after)
        if s.params:
            step["params"] = dict(sorted(s.params.items()))
        if not s.reusable:
            step["reusable"] = False
        steps.append(step)
    out["steps"] = steps
    if migration.metadata:
        out["metadata"] = dict(sorted(migration.metadata.items()))
    return out


def _context_to_spec(policy: ContextPolicy) -> dict[str, Any]:
    sequence: list[Any] = []
    for src in policy.sequence:
        if not src.params:
            sequence.append(src.kind)
        else:
            sequence.append({src.kind: dict(sorted(src.params.items()))})
    spec: dict[str, Any] = {"sequence": sequence}
    spec["full_document_fallback"] = policy.full_document_fallback.value
    budget = policy.budget.to_canonical()
    for key in ("max_input_tokens", "max_chars", "max_units", "max_sections", "max_cost_usd"):
        if budget.get(key) is not None:
            spec[key] = budget[key]
    if policy.accumulate:
        spec["accumulate"] = True
    if policy.on_missing_context is not OnMissingContext.REVIEW:
        spec["on_missing_context"] = policy.on_missing_context.value
    privacy = policy.privacy.to_canonical()
    if any(privacy.values()):
        spec["privacy"] = {k: v for k, v in privacy.items() if v}
    return spec


def load_migration_file(path: str | Path) -> list[Migration]:
    """Read one or many migrations from a YAML file."""

    with Path(path).open("r", encoding="utf-8") as fh:
        docs = [d for d in yaml.safe_load_all(fh) if d]
    out: list[Migration] = []
    for doc in docs:
        if isinstance(doc, list):
            out.extend(migration_from_dict(d) for d in doc)
        elif "migrations" in doc:
            out.extend(migration_from_dict(d) for d in doc["migrations"])
        else:
            out.append(migration_from_dict(doc))
    return out


def dump_migration(migration: Migration) -> str:
    return yaml.safe_dump(
        migration_to_dict(migration), sort_keys=False, default_flow_style=False, width=100
    )


def save_migration(migration: Migration, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(dump_migration(migration), encoding="utf-8")
    return p


__all__ = [
    "dump_migration",
    "load_migration_file",
    "migration_from_dict",
    "migration_to_dict",
    "save_migration",
]
