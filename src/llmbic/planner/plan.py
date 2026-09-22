"""The execution plan — frozen, inspectable, signable.

FR-PLN-001..008.  A plan is produced before anything runs and says, for every
selected record, exactly which outputs will be reused, transformed,
regenerated, sent to review or blocked; which source units would be
transmitted and to whom; and what it is expected to cost.

It is serialisable and carries a :meth:`ExecutionPlan.signature` over its
contents plus the registry state, so an execution can be tied to an approved
plan and refuse to run when either has moved underneath it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from ..ids import content_hash, now
from ..migration.spec import ExecutionPolicy, Fidelity, StepKind


class Disposition(str, Enum):
    #: The stored value's dependencies are unchanged; nothing runs.
    REUSE = "reuse"
    #: A deterministic function will produce the value.
    DETERMINISTIC = "deterministic"
    #: A model will be called.
    SEMANTIC = "semantic"
    #: A model *would* be called, but the answer is already cached.
    CACHED = "cached"
    #: Escalated to a curator.
    REVIEW = "review"
    #: Cannot proceed: missing context, forbidden policy, absent dependency.
    BLOCKED = "blocked"
    #: Structural/semantic validation only.
    VALIDATE = "validate"

    @property
    def is_billable(self) -> bool:
        return self is Disposition.SEMANTIC

    @property
    def runs(self) -> bool:
        return self in (
            Disposition.DETERMINISTIC,
            Disposition.SEMANTIC,
            Disposition.CACHED,
            Disposition.VALIDATE,
        )


@dataclass(frozen=True)
class ContextPreview:
    """What the context policy resolved to, without calling anything."""

    satisfied: bool = False
    used_sources: tuple[str, ...] = ()
    attempted: tuple[str, ...] = ()
    unit_ids: tuple[str, ...] = ()
    n_chars: int = 0
    est_input_tokens: int = 0
    includes_full_document: bool = False
    truncated: bool = False
    blocked_reason: str | None = None
    context_hash: str | None = None

    def to_canonical(self) -> dict[str, Any]:
        return {
            "satisfied": self.satisfied,
            "used_sources": list(self.used_sources),
            "attempted": list(self.attempted),
            "unit_ids": list(self.unit_ids),
            "n_chars": self.n_chars,
            "est_input_tokens": self.est_input_tokens,
            "includes_full_document": self.includes_full_document,
            "truncated": self.truncated,
            "blocked_reason": self.blocked_reason,
            "context_hash": self.context_hash,
        }

    @classmethod
    def from_canonical(cls, data: Mapping[str, Any]) -> "ContextPreview":
        return cls(
            satisfied=bool(data.get("satisfied", False)),
            used_sources=tuple(data.get("used_sources") or ()),
            attempted=tuple(data.get("attempted") or ()),
            unit_ids=tuple(data.get("unit_ids") or ()),
            n_chars=int(data.get("n_chars", 0)),
            est_input_tokens=int(data.get("est_input_tokens", 0)),
            includes_full_document=bool(data.get("includes_full_document", False)),
            truncated=bool(data.get("truncated", False)),
            blocked_reason=data.get("blocked_reason"),
            context_hash=data.get("context_hash"),
        )


@dataclass(frozen=True)
class PlannedStep:
    """One unit of work, or one decision not to do it."""

    key: str
    record_id: str
    entity: str
    migration_id: str
    step_id: str
    kind: StepKind
    disposition: Disposition
    writes: tuple[str, ...] = ()
    reason: str = ""
    fidelity: Fidelity = Fidelity.LOSSLESS
    cache_key: str | None = None
    dependencies: dict[str, str] = field(default_factory=dict)
    depends_on: tuple[str, ...] = ()
    context: ContextPreview | None = None
    recipe_ref: str | None = None
    transform_ref: str | None = None
    provider: str | None = None
    est_input_tokens: int = 0
    est_output_tokens: int = 0
    est_cost_usd: float = 0.0
    blocked_code: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_canonical(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "record_id": self.record_id,
            "entity": self.entity,
            "migration_id": self.migration_id,
            "step_id": self.step_id,
            "kind": self.kind.value,
            "disposition": self.disposition.value,
            "writes": list(self.writes),
            "reason": self.reason,
            "fidelity": self.fidelity.value,
            "cache_key": self.cache_key,
            "dependencies": dict(sorted(self.dependencies.items())),
            "depends_on": list(self.depends_on),
            "context": self.context.to_canonical() if self.context else None,
            "recipe_ref": self.recipe_ref,
            "transform_ref": self.transform_ref,
            "provider": self.provider,
            "est_input_tokens": self.est_input_tokens,
            "est_output_tokens": self.est_output_tokens,
            "est_cost_usd": round(self.est_cost_usd, 8),
            "blocked_code": self.blocked_code,
            "detail": self.detail,
        }

    @classmethod
    def from_canonical(cls, data: Mapping[str, Any]) -> "PlannedStep":
        return cls(
            key=data["key"],
            record_id=data["record_id"],
            entity=data.get("entity", ""),
            migration_id=data["migration_id"],
            step_id=data["step_id"],
            kind=StepKind(data["kind"]),
            disposition=Disposition(data["disposition"]),
            writes=tuple(data.get("writes") or ()),
            reason=data.get("reason", ""),
            fidelity=Fidelity(data.get("fidelity", "lossless")),
            cache_key=data.get("cache_key"),
            dependencies=dict(data.get("dependencies") or {}),
            depends_on=tuple(data.get("depends_on") or ()),
            context=ContextPreview.from_canonical(data["context"]) if data.get("context") else None,
            recipe_ref=data.get("recipe_ref"),
            transform_ref=data.get("transform_ref"),
            provider=data.get("provider"),
            est_input_tokens=int(data.get("est_input_tokens", 0)),
            est_output_tokens=int(data.get("est_output_tokens", 0)),
            est_cost_usd=float(data.get("est_cost_usd", 0.0)),
            blocked_code=data.get("blocked_code"),
            detail=dict(data.get("detail") or {}),
        )


@dataclass(frozen=True)
class RecordPlan:
    record_id: str
    from_schema: str
    to_schema: str
    path: tuple[str, ...] = ()
    steps: tuple[PlannedStep, ...] = ()
    blocked: bool = False
    blocked_reason: str | None = None
    blocked_code: str | None = None
    #: Fields that are not semantically current and that no step refreshes.
    stale_fields: tuple[str, ...] = ()
    source_ref: str | None = None

    @property
    def n_model_calls(self) -> int:
        return sum(1 for s in self.steps if s.disposition.is_billable)

    @property
    def est_cost_usd(self) -> float:
        return sum(s.est_cost_usd for s in self.steps if s.disposition.is_billable)

    def to_canonical(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "from_schema": self.from_schema,
            "to_schema": self.to_schema,
            "path": list(self.path),
            "steps": [s.to_canonical() for s in self.steps],
            "blocked": self.blocked,
            "blocked_reason": self.blocked_reason,
            "blocked_code": self.blocked_code,
            "stale_fields": list(self.stale_fields),
            "source_ref": self.source_ref,
        }

    @classmethod
    def from_canonical(cls, data: Mapping[str, Any]) -> "RecordPlan":
        return cls(
            record_id=data["record_id"],
            from_schema=data["from_schema"],
            to_schema=data["to_schema"],
            path=tuple(data.get("path") or ()),
            steps=tuple(PlannedStep.from_canonical(s) for s in data.get("steps") or ()),
            blocked=bool(data.get("blocked", False)),
            blocked_reason=data.get("blocked_reason"),
            blocked_code=data.get("blocked_code"),
            stale_fields=tuple(data.get("stale_fields") or ()),
            source_ref=data.get("source_ref"),
        )


@dataclass(frozen=True)
class PlanSummary:
    """FR-PLN-002/003/004 in numbers."""

    n_records: int = 0
    n_blocked_records: int = 0
    n_steps: int = 0
    by_disposition: dict[str, int] = field(default_factory=dict)
    fields_by_disposition: dict[str, int] = field(default_factory=dict)
    n_model_calls: int = 0
    n_cached_calls: int = 0
    est_input_tokens: int = 0
    est_output_tokens: int = 0
    est_cost_usd: float = 0.0
    n_records_touching_source: int = 0
    n_full_document_transmissions: int = 0
    providers: dict[str, int] = field(default_factory=dict)
    n_stale_unaddressed: int = 0
    worst_fidelity: str = Fidelity.LOSSLESS.value

    def to_canonical(self) -> dict[str, Any]:
        return {
            "n_records": self.n_records,
            "n_blocked_records": self.n_blocked_records,
            "n_steps": self.n_steps,
            "by_disposition": dict(sorted(self.by_disposition.items())),
            "fields_by_disposition": dict(sorted(self.fields_by_disposition.items())),
            "n_model_calls": self.n_model_calls,
            "n_cached_calls": self.n_cached_calls,
            "est_input_tokens": self.est_input_tokens,
            "est_output_tokens": self.est_output_tokens,
            "est_cost_usd": round(self.est_cost_usd, 6),
            "n_records_touching_source": self.n_records_touching_source,
            "n_full_document_transmissions": self.n_full_document_transmissions,
            "providers": dict(sorted(self.providers.items())),
            "n_stale_unaddressed": self.n_stale_unaddressed,
            "worst_fidelity": self.worst_fidelity,
        }

    @classmethod
    def from_canonical(cls, data: Mapping[str, Any]) -> "PlanSummary":
        return cls(
            n_records=int(data.get("n_records", 0)),
            n_blocked_records=int(data.get("n_blocked_records", 0)),
            n_steps=int(data.get("n_steps", 0)),
            by_disposition=dict(data.get("by_disposition") or {}),
            fields_by_disposition=dict(data.get("fields_by_disposition") or {}),
            n_model_calls=int(data.get("n_model_calls", 0)),
            n_cached_calls=int(data.get("n_cached_calls", 0)),
            est_input_tokens=int(data.get("est_input_tokens", 0)),
            est_output_tokens=int(data.get("est_output_tokens", 0)),
            est_cost_usd=float(data.get("est_cost_usd", 0.0)),
            n_records_touching_source=int(data.get("n_records_touching_source", 0)),
            n_full_document_transmissions=int(data.get("n_full_document_transmissions", 0)),
            providers=dict(data.get("providers") or {}),
            n_stale_unaddressed=int(data.get("n_stale_unaddressed", 0)),
            worst_fidelity=data.get("worst_fidelity", Fidelity.LOSSLESS.value),
        )


def summarise(records: Sequence[RecordPlan]) -> PlanSummary:
    by_disp: Counter[str] = Counter()
    by_field: Counter[str] = Counter()
    providers: Counter[str] = Counter()
    n_steps = in_tok = out_tok = 0
    cost = 0.0
    calls = cached = full_doc = touching = 0
    worst = Fidelity.LOSSLESS

    for rp in records:
        if rp.blocked:
            continue
        touched = False
        for s in rp.steps:
            n_steps += 1
            by_disp[s.disposition.value] += 1
            for w in s.writes:
                by_field[s.disposition.value] += 1
            if s.fidelity.rank > worst.rank:
                worst = s.fidelity
            if s.disposition.is_billable:
                calls += 1
                in_tok += s.est_input_tokens
                out_tok += s.est_output_tokens
                cost += s.est_cost_usd
                if s.provider:
                    providers[s.provider] += 1
            if s.disposition is Disposition.CACHED:
                cached += 1
            if s.context is not None:
                if s.context.unit_ids:
                    touched = True
                if s.context.includes_full_document:
                    full_doc += 1
        if touched:
            touching += 1

    return PlanSummary(
        n_records=len(records),
        n_blocked_records=sum(1 for r in records if r.blocked),
        n_steps=n_steps,
        by_disposition=dict(by_disp),
        fields_by_disposition=dict(by_field),
        n_model_calls=calls,
        n_cached_calls=cached,
        est_input_tokens=in_tok,
        est_output_tokens=out_tok,
        est_cost_usd=cost,
        n_records_touching_source=touching,
        n_full_document_transmissions=full_doc,
        providers=dict(providers),
        n_stale_unaddressed=sum(len(r.stale_fields) for r in records),
        worst_fidelity=worst.value,
    )


@dataclass(frozen=True)
class ExecutionPlan:
    target_schema: str
    records: tuple[RecordPlan, ...] = ()
    summary: PlanSummary = field(default_factory=PlanSummary)
    policy: ExecutionPolicy = field(default_factory=ExecutionPolicy)
    registry_hash: str = ""
    created_at: str = field(default_factory=now)
    software_version: str = ""
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def plan_id(self) -> str:
        return content_hash(
            {
                "target_schema": self.target_schema,
                "records": [r.to_canonical() for r in self.records],
                "policy": self.policy.to_canonical(),
                "registry_hash": self.registry_hash,
            }
        )

    def signature(self) -> str:
        """Ties an execution to an approved plan (FR-PLN-008)."""

        return content_hash({"plan_id": self.plan_id, "registry_hash": self.registry_hash})

    def steps(self) -> Iterable[PlannedStep]:
        for r in self.records:
            yield from r.steps

    def runnable_steps(self) -> list[PlannedStep]:
        return [s for s in self.steps() if s.disposition.runs]

    def record(self, record_id: str) -> RecordPlan | None:
        for r in self.records:
            if r.record_id == record_id:
                return r
        return None

    def to_canonical(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "signature": self.signature(),
            "target_schema": self.target_schema,
            "created_at": self.created_at,
            "software_version": self.software_version,
            "registry_hash": self.registry_hash,
            "policy": self.policy.to_canonical(),
            "summary": self.summary.to_canonical(),
            "records": [r.to_canonical() for r in self.records],
            "notes": dict(sorted(self.notes.items())),
        }

    @classmethod
    def from_canonical(cls, data: Mapping[str, Any]) -> "ExecutionPlan":
        return cls(
            target_schema=data["target_schema"],
            records=tuple(RecordPlan.from_canonical(r) for r in data.get("records") or ()),
            summary=PlanSummary.from_canonical(data.get("summary") or {}),
            policy=ExecutionPolicy.from_canonical(data.get("policy") or {}),
            registry_hash=data.get("registry_hash", ""),
            created_at=data.get("created_at", ""),
            software_version=data.get("software_version", ""),
            notes=dict(data.get("notes") or {}),
        )

    # ---- human-readable ---------------------------------------------------
    def render(self, *, verbose: bool = False, max_records: int = 20) -> str:
        s = self.summary
        lines = [
            f"plan {self.plan_id}  ->  {self.target_schema}",
            f"  records:          {s.n_records}"
            + (f"  ({s.n_blocked_records} blocked)" if s.n_blocked_records else ""),
            f"  steps:            {s.n_steps}",
        ]
        for name in (
            Disposition.REUSE,
            Disposition.DETERMINISTIC,
            Disposition.CACHED,
            Disposition.SEMANTIC,
            Disposition.VALIDATE,
            Disposition.REVIEW,
            Disposition.BLOCKED,
        ):
            count = s.by_disposition.get(name.value, 0)
            if count:
                lines.append(f"    {name.value:<15} {count}")
        lines.append(f"  model calls:      {s.n_model_calls} ({s.n_cached_calls} already cached)")
        lines.append(
            f"  est. tokens:      {s.est_input_tokens} in / {s.est_output_tokens} out"
        )
        lines.append(f"  est. cost:        ${s.est_cost_usd:.4f}")
        lines.append(
            f"  source access:    {s.n_records_touching_source} records; "
            f"full documents transmitted: {s.n_full_document_transmissions}"
        )
        if s.providers:
            lines.append(
                "  providers:        "
                + ", ".join(f"{k}={v}" for k, v in sorted(s.providers.items()))
            )
        lines.append(f"  worst fidelity:   {s.worst_fidelity}")
        if s.n_stale_unaddressed:
            lines.append(
                f"  stale fields with no step: {s.n_stale_unaddressed} "
                "(structurally valid, not semantically current)"
            )

        if verbose:
            for rp in self.records[:max_records]:
                head = f"\n  {rp.record_id}: {rp.from_schema} -> {rp.to_schema}"
                if rp.path:
                    head += "  via " + " > ".join(rp.path)
                lines.append(head)
                if rp.blocked:
                    lines.append(f"    BLOCKED: {rp.blocked_reason}")
                for st in rp.steps:
                    where = f"[{st.entity}] " if st.entity else ""
                    lines.append(
                        f"    {st.disposition.value:<14} {where}{st.step_id} "
                        f"-> {', '.join(st.writes) or '-'}   {st.reason}"
                    )
            if len(self.records) > max_records:
                lines.append(f"  ... and {len(self.records) - max_records} more records")
        return "\n".join(lines)


__all__ = [
    "ContextPreview",
    "Disposition",
    "ExecutionPlan",
    "PlanSummary",
    "PlannedStep",
    "RecordPlan",
    "summarise",
]
