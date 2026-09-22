"""Migration declarations.

Requirements §10 fixes the shape a migration must be able to express.  This
module is that shape in Python, with a YAML projection in
:mod:`llmbic.migration.loader`, and it is deliberately *declarative*: a step
says what it reads, what it writes, whether it is deterministic or
model-assisted, what context it may touch and how faithful it is.  The planner
reads those declarations; it never inspects a transform's body.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from ..context.policy import ContextPolicy, OnMissingContext, policy_from_spec
from ..errors import ErrorCode, RegistryError
from ..ids import content_hash


class StepKind(str, Enum):
    """FR-MIG-007's seven kinds, plus the two the engine adds itself."""

    STRUCTURAL = "structural"
    DERIVED = "derived"
    VOCABULARY = "vocabulary"
    #: Semantic extraction restricted to stored evidence spans.
    EVIDENCE_SEMANTIC = "evidence_semantic"
    #: Semantic extraction allowed to select fresh source context.
    SOURCE_SEMANTIC = "source_semantic"
    #: Re-run the recipe for a field or field group from scratch.
    REEXTRACTION = "reextraction"
    MANUAL_REVIEW = "manual_review"
    VALIDATION = "validation"
    ASSEMBLY = "assembly"

    @property
    def is_semantic(self) -> bool:
        return self in (
            StepKind.EVIDENCE_SEMANTIC,
            StepKind.SOURCE_SEMANTIC,
            StepKind.REEXTRACTION,
        )

    @property
    def is_deterministic(self) -> bool:
        return self in (StepKind.STRUCTURAL, StepKind.DERIVED, StepKind.VOCABULARY)


class Fidelity(str, Enum):
    LOSSLESS = "lossless"
    LOSSY = "potentially_lossy"
    DESTRUCTIVE = "destructive"

    @property
    def rank(self) -> int:
        return {"lossless": 0, "potentially_lossy": 1, "destructive": 2}[self.value]


_REF_RE = re.compile(r"^(?P<kind>field|evidence|source|vocab|recipe|status|entity):(?P<name>.+)$")

_REF_KINDS = {"field", "evidence", "source", "vocab", "recipe", "status", "entity"}


@dataclass(frozen=True)
class Ref:
    """A typed dependency reference, e.g. ``field:sample_size``."""

    kind: str
    name: str

    def __str__(self) -> str:
        return f"{self.kind}:{self.name}"

    @classmethod
    def parse(cls, text: str) -> "Ref":
        m = _REF_RE.match(text.strip())
        if not m:
            raise RegistryError(
                f"malformed reference {text!r}; expected one of "
                f"{sorted(_REF_KINDS)} followed by ':name'",
                code=ErrorCode.MIGRATION_INVALID,
            )
        return cls(m.group("kind"), m.group("name"))

    @classmethod
    def coerce(cls, value: "Ref | str") -> "Ref":
        return value if isinstance(value, Ref) else cls.parse(value)


def field_ref(name: str) -> Ref:
    return Ref("field", name)


@dataclass(frozen=True)
class MigrationStep:
    """One operation with an explicit contract.

    FR-MIG-008: a migration may mix deterministic and semantic work, but each
    step is separate in the plan, so a dry run can say "3 renames, 1 model
    call" rather than "1 migration".
    """

    id: str
    kind: StepKind
    writes: tuple[str, ...] = ()
    reads: tuple[Ref, ...] = ()
    transform: str | None = None
    recipe: str | None = None
    context: ContextPolicy | None = None
    on_missing_context: OnMissingContext | None = None
    validators: tuple[str, ...] = ()
    fidelity: Fidelity = Fidelity.LOSSLESS
    #: Collection path this step runs inside; ``""`` means once per record,
    #: ``"groups[]"`` means once per group.
    entity_scope: str = ""
    #: Step ids within the same migration that must run first.  Ordering is
    #: otherwise derived from reads/writes.
    after: tuple[str, ...] = ()
    params: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    #: When true, the step may be skipped for a record whose declared
    #: dependencies are unchanged; the default for everything but reextraction.
    reusable: bool = True

    def __post_init__(self) -> None:
        if not self.writes and self.kind not in (StepKind.VALIDATION, StepKind.ASSEMBLY):
            raise RegistryError(
                f"step {self.id!r} declares no outputs; every step but validation "
                "and assembly must write something (FR-MIG-001)",
                code=ErrorCode.MIGRATION_INVALID,
            )
        if self.kind.is_deterministic and not self.transform:
            raise RegistryError(
                f"deterministic step {self.id!r} must name a transform",
                code=ErrorCode.MIGRATION_INVALID,
            )
        if self.kind.is_semantic and not self.recipe:
            raise RegistryError(
                f"semantic step {self.id!r} must name an extraction recipe",
                code=ErrorCode.MIGRATION_INVALID,
            )
        if self.kind is StepKind.EVIDENCE_SEMANTIC and self.context is not None:
            if self.context.permits_full_document():
                raise RegistryError(
                    f"step {self.id!r} is evidence-local but its context policy "
                    "permits the full document; declare it source_semantic instead",
                    code=ErrorCode.MIGRATION_INVALID,
                )

    # ---- dependency views -----------------------------------------------
    def reads_of(self, kind: str) -> tuple[str, ...]:
        return tuple(r.name for r in self.reads if r.kind == kind)

    @property
    def reads_fields(self) -> tuple[str, ...]:
        return self.reads_of("field")

    @property
    def reads_evidence(self) -> tuple[str, ...]:
        return self.reads_of("evidence")

    @property
    def reads_source(self) -> tuple[str, ...]:
        return self.reads_of("source")

    @property
    def reads_vocabularies(self) -> tuple[str, ...]:
        return self.reads_of("vocab")

    @property
    def touches_source(self) -> bool:
        return bool(self.reads_source) or self.kind in (
            StepKind.SOURCE_SEMANTIC,
            StepKind.REEXTRACTION,
        )

    def step_hash(self) -> str:
        return content_hash(self.to_canonical())

    def to_canonical(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "writes": list(self.writes),
            "reads": [str(r) for r in self.reads],
            "transform": self.transform,
            "recipe": self.recipe,
            "context": self.context.to_canonical() if self.context else None,
            "on_missing_context": self.on_missing_context.value
            if self.on_missing_context
            else None,
            "validators": list(self.validators),
            "fidelity": self.fidelity.value,
            "entity_scope": self.entity_scope,
            "after": list(self.after),
            "params": dict(sorted(self.params.items())),
            "description": self.description,
            "reusable": self.reusable,
        }

    @classmethod
    def from_canonical(cls, data: Mapping[str, Any]) -> "MigrationStep":
        ctx = data.get("context")
        return cls(
            id=data["id"],
            kind=StepKind(data["kind"]),
            writes=tuple(_as_names(data.get("writes") or ())),
            reads=tuple(Ref.coerce(r) for r in data.get("reads") or ()),
            transform=data.get("transform"),
            recipe=data.get("recipe"),
            context=(
                ContextPolicy.from_canonical(ctx)
                if isinstance(ctx, Mapping) and "sequence" in ctx and _looks_canonical(ctx)
                else policy_from_spec(ctx)
                if ctx
                else None
            ),
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


def _looks_canonical(ctx: Mapping[str, Any]) -> bool:
    seq = ctx.get("sequence") or []
    return all(isinstance(s, Mapping) and "kind" in s for s in seq)


def _as_names(values: Iterable[Any]) -> list[str]:
    """Accept ``field:x`` or a bare field id in ``writes``."""

    out = []
    for v in values:
        text = str(v)
        if ":" in text:
            ref = Ref.parse(text)
            if ref.kind != "field":
                raise RegistryError(
                    f"a step can only write fields, not {ref.kind!r}",
                    code=ErrorCode.MIGRATION_INVALID,
                )
            out.append(ref.name)
        else:
            out.append(text)
    return out


@dataclass(frozen=True)
class Migration:
    """A directed change from one schema version to another."""

    id: str
    from_schema: str
    to_schema: str
    steps: tuple[MigrationStep, ...] = ()
    description: str = ""
    #: Explicit path renames, ``old_path -> new_path``.  Required for the diff
    #: to report a rename rather than a remove/add pair (FR-SCH-004).
    renames: dict[str, str] = field(default_factory=dict)
    #: ``field_id -> rationale`` for changes the author deliberately handles
    #: with no step.  §21.2 wants every changed field to have an *explicit
    #: disposition*, which is not the same as wanting every changed field to
    #: be recomputed: tightening ``minimum`` is a validation matter, and
    #: saying so here is the disposition.
    acknowledged: dict[str, str] = field(default_factory=dict)
    #: A deterministic reverse transform, when one exists (FR-MIG-011).
    downgrade: str | None = None
    #: A downgrade edge is stored separately from upgrade planning (FR-MIG-004).
    is_downgrade: bool = False
    #: Development branches are registered but never planned through unless
    #: the planner is told to (FR-MIG-012).
    branch: str = "main"
    approved: bool = True
    #: Free-form, e.g. the source-control revision this was authored at.
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for step in self.steps:
            if step.id in seen:
                raise RegistryError(
                    f"migration {self.id!r} has two steps called {step.id!r}",
                    code=ErrorCode.MIGRATION_INVALID,
                )
            seen.add(step.id)
        for step in self.steps:
            for dep in step.after:
                if dep not in seen:
                    raise RegistryError(
                        f"step {step.id!r} waits on unknown step {dep!r}",
                        code=ErrorCode.MIGRATION_INVALID,
                    )

    @property
    def fidelity(self) -> Fidelity:
        """The worst fidelity any step declares (FR-MIG-005)."""

        if not self.steps:
            return Fidelity.LOSSLESS
        return max((s.fidelity for s in self.steps), key=lambda f: f.rank)

    @property
    def writes(self) -> tuple[str, ...]:
        out: list[str] = []
        for s in self.steps:
            out.extend(s.writes)
        return tuple(dict.fromkeys(out))

    def step(self, step_id: str) -> MigrationStep:
        for s in self.steps:
            if s.id == step_id:
                return s
        raise KeyError(step_id)

    def semantic_steps(self) -> tuple[MigrationStep, ...]:
        return tuple(s for s in self.steps if s.kind.is_semantic)

    def touches_source(self) -> bool:
        return any(s.touches_source for s in self.steps)

    def ordered_steps(self) -> tuple[MigrationStep, ...]:
        """Steps in dependency order, deterministic for a given declaration.

        Ordering comes from ``after`` plus the read/write relation between
        steps: a step reading a field another step writes runs later.
        """

        writers: dict[str, list[str]] = {}
        for s in self.steps:
            for w in s.writes:
                writers.setdefault(w, []).append(s.id)

        deps: dict[str, set[str]] = {s.id: set(s.after) for s in self.steps}
        for s in self.steps:
            for r in s.reads_fields:
                for producer in writers.get(r, ()):
                    if producer != s.id:
                        deps[s.id].add(producer)

        by_id = {s.id: s for s in self.steps}
        ordered: list[MigrationStep] = []
        done: set[str] = set()
        remaining = sorted(by_id)
        while remaining:
            ready = [sid for sid in remaining if deps[sid] <= done]
            if not ready:
                raise RegistryError(
                    f"migration {self.id!r} has a cycle among steps {sorted(remaining)}",
                    code=ErrorCode.MIGRATION_CYCLE,
                    details={"steps": sorted(remaining)},
                )
            for sid in ready:
                ordered.append(by_id[sid])
                done.add(sid)
                remaining.remove(sid)
        return tuple(ordered)

    def migration_hash(self) -> str:
        """Immutable migration identity (FR-MIG-001, NFR-REP-003)."""

        return content_hash(
            {
                "id": self.id,
                "from": self.from_schema,
                "to": self.to_schema,
                "steps": [s.to_canonical() for s in self.steps],
                "renames": dict(sorted(self.renames.items())),
                "acknowledged": dict(sorted(self.acknowledged.items())),
                "downgrade": self.downgrade,
            }
        )

    def with_steps(self, steps: Sequence[MigrationStep]) -> "Migration":
        return replace(self, steps=tuple(steps))

    def to_canonical(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "from_schema": self.from_schema,
            "to_schema": self.to_schema,
            "description": self.description,
            "steps": [s.to_canonical() for s in self.steps],
            "renames": dict(sorted(self.renames.items())),
            "acknowledged": dict(sorted(self.acknowledged.items())),
            "downgrade": self.downgrade,
            "is_downgrade": self.is_downgrade,
            "branch": self.branch,
            "approved": self.approved,
            "fidelity": self.fidelity.value,
            "metadata": dict(sorted(self.metadata.items())),
        }

    @classmethod
    def from_canonical(cls, data: Mapping[str, Any]) -> "Migration":
        return cls(
            id=data["id"],
            from_schema=data["from_schema"],
            to_schema=data["to_schema"],
            description=data.get("description", ""),
            steps=tuple(MigrationStep.from_canonical(s) for s in data.get("steps") or ()),
            renames=dict(data.get("renames") or {}),
            acknowledged=dict(data.get("acknowledged") or {}),
            downgrade=data.get("downgrade"),
            is_downgrade=bool(data.get("is_downgrade", False)),
            branch=data.get("branch", "main"),
            approved=bool(data.get("approved", True)),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class ExecutionPolicy:
    """What an operator permits this run to do (FR-MIG-006).

    A potentially lossy or destructive migration does not run unless the
    operator said so, by fidelity and — for destructive changes — by naming the
    migration.  "I said yes once" is not a standing permission.
    """

    allow_lossy: bool = False
    allow_destructive: bool = False
    #: Migration ids explicitly approved for a destructive run.
    approved_migrations: tuple[str, ...] = ()
    #: Permit sending whole documents to a provider.
    allow_full_document: bool = False
    #: Permit selectors that themselves call a billable model.
    allow_model_selectors: bool = False
    budget_usd: float | None = None
    max_model_calls: int | None = None
    actor_id: str = "unknown"

    def permits(self, migration: Migration) -> tuple[bool, str]:
        fidelity = migration.fidelity
        if fidelity is Fidelity.LOSSLESS:
            return True, ""
        if fidelity is Fidelity.LOSSY and not self.allow_lossy:
            return False, (
                f"migration {migration.id!r} is potentially lossy; "
                "re-run with allow_lossy=True to permit it"
            )
        if fidelity is Fidelity.DESTRUCTIVE:
            if not self.allow_destructive:
                return False, (
                    f"migration {migration.id!r} is destructive; "
                    "re-run with allow_destructive=True to permit it"
                )
            if migration.id not in self.approved_migrations:
                return False, (
                    f"migration {migration.id!r} is destructive and must be named in "
                    "approved_migrations"
                )
        return True, ""

    def to_canonical(self) -> dict[str, Any]:
        return {
            "allow_lossy": self.allow_lossy,
            "allow_destructive": self.allow_destructive,
            "approved_migrations": list(self.approved_migrations),
            "allow_full_document": self.allow_full_document,
            "allow_model_selectors": self.allow_model_selectors,
            "budget_usd": self.budget_usd,
            "max_model_calls": self.max_model_calls,
            "actor_id": self.actor_id,
        }

    @classmethod
    def from_canonical(cls, data: Mapping[str, Any]) -> "ExecutionPolicy":
        return cls(
            allow_lossy=bool(data.get("allow_lossy", False)),
            allow_destructive=bool(data.get("allow_destructive", False)),
            approved_migrations=tuple(data.get("approved_migrations") or ()),
            allow_full_document=bool(data.get("allow_full_document", False)),
            allow_model_selectors=bool(data.get("allow_model_selectors", False)),
            budget_usd=data.get("budget_usd"),
            max_model_calls=data.get("max_model_calls"),
            actor_id=data.get("actor_id", "unknown"),
        )


__all__ = [
    "ExecutionPolicy",
    "Fidelity",
    "Migration",
    "MigrationStep",
    "Ref",
    "StepKind",
    "field_ref",
]
