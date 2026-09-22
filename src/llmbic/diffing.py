"""Field-level record diffs (FR-VAL-004) and migration reports (FR-VAL-009).

A curator comparing a record before and after a migration wants to see the
values that moved, the values that changed and — just as importantly — the
values that did not, because "unchanged" is the claim selective migration
makes and the thing worth auditing.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from .provenance import FieldArtifact, latest_per_key
from .schema.normalized import NormalizedSchema
from .values import FieldValue, ValueStatus


class DeltaKind(str, Enum):
    UNCHANGED = "unchanged"
    #: Same value, new path — the rename case.
    MOVED = "moved"
    VALUE_CHANGED = "value_changed"
    STATUS_CHANGED = "status_changed"
    ADDED = "added"
    REMOVED = "removed"
    EVIDENCE_CHANGED = "evidence_changed"
    PROVENANCE_CHANGED = "provenance_changed"


@dataclass(frozen=True)
class FieldDelta:
    field_id: str
    entity: str
    kind: DeltaKind
    before: FieldValue | None = None
    after: FieldValue | None = None
    before_path: str | None = None
    after_path: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_canonical(self) -> dict[str, Any]:
        return {
            "field_id": self.field_id,
            "entity": self.entity,
            "kind": self.kind.value,
            "before": self.before.to_canonical() if self.before else None,
            "after": self.after.to_canonical() if self.after else None,
            "before_path": self.before_path,
            "after_path": self.after_path,
            "detail": self.detail,
        }

    def describe(self) -> str:
        where = f"{self.after_path or self.before_path or self.field_id}"
        if self.entity:
            where = f"{where} [{self.entity}]"
        if self.kind is DeltaKind.MOVED:
            return f"moved           {self.before_path} -> {self.after_path}"
        if self.kind is DeltaKind.VALUE_CHANGED:
            return (
                f"value_changed   {where}: {_short(self.before)} -> {_short(self.after)}"
            )
        if self.kind is DeltaKind.STATUS_CHANGED:
            return (
                f"status_changed  {where}: {self.before.status.value if self.before else '-'}"
                f" -> {self.after.status.value if self.after else '-'}"
            )
        if self.kind is DeltaKind.ADDED:
            return f"added           {where}: {_short(self.after)}"
        if self.kind is DeltaKind.REMOVED:
            return f"removed         {where}: was {_short(self.before)}"
        return f"{self.kind.value:<15} {where}"


@dataclass(frozen=True)
class RecordDiff:
    record_id: str
    from_schema: str
    to_schema: str
    deltas: tuple[FieldDelta, ...] = ()

    def changed(self) -> list[FieldDelta]:
        return [d for d in self.deltas if d.kind is not DeltaKind.UNCHANGED]

    def counts(self) -> dict[str, int]:
        return dict(sorted(Counter(d.kind.value for d in self.deltas).items()))

    def to_canonical(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "from_schema": self.from_schema,
            "to_schema": self.to_schema,
            "counts": self.counts(),
            "deltas": [d.to_canonical() for d in self.deltas],
        }

    def render(self, *, include_unchanged: bool = False) -> str:
        lines = [f"{self.record_id}: {self.from_schema} -> {self.to_schema}"]
        counts = self.counts()
        lines.append("  " + ", ".join(f"{k}={v}" for k, v in counts.items()))
        for d in self.deltas:
            if d.kind is DeltaKind.UNCHANGED and not include_unchanged:
                continue
            lines.append("  " + d.describe())
        return "\n".join(lines)


def diff_records(
    before: Iterable[FieldArtifact],
    after: Iterable[FieldArtifact],
    *,
    record_id: str = "",
    from_schema: NormalizedSchema | None = None,
    to_schema: NormalizedSchema | None = None,
    include_provenance: bool = False,
) -> RecordDiff:
    """Compare two sets of artifacts for the same record."""

    b = {(a.field_id, a.entity): a for a in latest_per_key(list(before))}
    a_ = {(a.field_id, a.entity): a for a in latest_per_key(list(after))}
    record_id = record_id or next(
        (x.record_id for x in list(b.values()) + list(a_.values())), ""
    )

    deltas: list[FieldDelta] = []
    for key in sorted(set(b) | set(a_)):
        fid, entity = key
        old = b.get(key)
        new = a_.get(key)
        before_path = from_schema.get(fid).path if from_schema and from_schema.get(fid) else None
        after_path = to_schema.get(fid).path if to_schema and to_schema.get(fid) else None

        if old is None and new is not None:
            deltas.append(
                FieldDelta(fid, entity, DeltaKind.ADDED, None, new.value, None, after_path)
            )
            continue
        if new is None and old is not None:
            deltas.append(
                FieldDelta(fid, entity, DeltaKind.REMOVED, old.value, None, before_path, None)
            )
            continue
        assert old is not None and new is not None

        if old.value.status is not new.value.status:
            deltas.append(
                FieldDelta(
                    fid, entity, DeltaKind.STATUS_CHANGED, old.value, new.value,
                    before_path, after_path,
                )
            )
        elif old.value.value != new.value.value:
            deltas.append(
                FieldDelta(
                    fid, entity, DeltaKind.VALUE_CHANGED, old.value, new.value,
                    before_path, after_path,
                )
            )
        elif before_path and after_path and before_path != after_path:
            deltas.append(
                FieldDelta(
                    fid, entity, DeltaKind.MOVED, old.value, new.value, before_path, after_path,
                    detail={"evidence_preserved": old.evidence == new.evidence},
                )
            )
        elif old.evidence != new.evidence:
            deltas.append(
                FieldDelta(
                    fid, entity, DeltaKind.EVIDENCE_CHANGED, old.value, new.value,
                    before_path, after_path,
                    detail={
                        "before_spans": sum(len(e.spans) for e in old.evidence),
                        "after_spans": sum(len(e.spans) for e in new.evidence),
                    },
                )
            )
        elif include_provenance and old.provenance.recipe_ref != new.provenance.recipe_ref:
            deltas.append(
                FieldDelta(
                    fid, entity, DeltaKind.PROVENANCE_CHANGED, old.value, new.value,
                    before_path, after_path,
                    detail={
                        "before_recipe": old.provenance.recipe_ref,
                        "after_recipe": new.provenance.recipe_ref,
                    },
                )
            )
        else:
            deltas.append(
                FieldDelta(
                    fid, entity, DeltaKind.UNCHANGED, old.value, new.value,
                    before_path, after_path,
                )
            )

    return RecordDiff(
        record_id=record_id,
        from_schema=from_schema.ref if from_schema else "",
        to_schema=to_schema.ref if to_schema else "",
        deltas=tuple(deltas),
    )


@dataclass
class MigrationReport:
    """FR-VAL-009: coverage, missingness, change rate, failures, review, cost."""

    n_records: int = 0
    n_fields: int = 0
    coverage: dict[str, float] = field(default_factory=dict)
    missingness: dict[str, int] = field(default_factory=dict)
    change_rate: float = 0.0
    changed_by_field: dict[str, int] = field(default_factory=dict)
    validation_failure_rate: float = 0.0
    review_rate: float = 0.0
    cost_usd: float = 0.0
    model_calls: int = 0

    def to_canonical(self) -> dict[str, Any]:
        return {
            "n_records": self.n_records,
            "n_fields": self.n_fields,
            "coverage": {k: round(v, 4) for k, v in sorted(self.coverage.items())},
            "missingness": dict(sorted(self.missingness.items())),
            "change_rate": round(self.change_rate, 4),
            "changed_by_field": dict(
                sorted(self.changed_by_field.items(), key=lambda kv: (-kv[1], kv[0]))
            ),
            "validation_failure_rate": round(self.validation_failure_rate, 4),
            "review_rate": round(self.review_rate, 4),
            "cost_usd": round(self.cost_usd, 6),
            "model_calls": self.model_calls,
        }

    def render(self) -> str:
        lines = [
            f"records: {self.n_records}   fields: {self.n_fields}",
            f"change rate: {self.change_rate:.1%}   "
            f"validation failures: {self.validation_failure_rate:.1%}   "
            f"review: {self.review_rate:.1%}",
            f"model calls: {self.model_calls}   cost: ${self.cost_usd:.4f}",
        ]
        if self.changed_by_field:
            lines.append("most-changed fields:")
            for fid, n in list(self.changed_by_field.items())[:10]:
                lines.append(f"  {n:>6}  {fid}")
        if self.missingness:
            lines.append("value statuses:")
            for status, n in self.missingness.items():
                lines.append(f"  {n:>6}  {status}")
        return "\n".join(lines)


def build_report(
    diffs: Sequence[RecordDiff],
    artifacts_after: Sequence[FieldArtifact],
    schema: NormalizedSchema,
    *,
    metrics: Mapping[str, Any] | None = None,
) -> MigrationReport:
    report = MigrationReport(n_records=len(diffs))
    changed = Counter()
    total_deltas = 0
    changed_deltas = 0
    for d in diffs:
        for delta in d.deltas:
            total_deltas += 1
            if delta.kind not in (DeltaKind.UNCHANGED, DeltaKind.MOVED):
                changed_deltas += 1
                changed[delta.field_id] += 1
    report.change_rate = changed_deltas / total_deltas if total_deltas else 0.0
    report.changed_by_field = dict(changed)

    status_counts = Counter(a.value.status.value for a in artifacts_after)
    report.missingness = dict(sorted(status_counts.items()))
    report.n_fields = len(artifacts_after)

    per_field_present = Counter(
        a.field_id for a in artifacts_after if a.value.status is ValueStatus.PRESENT
    )
    per_field_total = Counter(a.field_id for a in artifacts_after)
    report.coverage = {
        fid: per_field_present.get(fid, 0) / total
        for fid, total in sorted(per_field_total.items())
        if total
    }

    if metrics:
        total_steps = max(1, int(metrics.get("steps_total", 0)))
        report.validation_failure_rate = int(metrics.get("validation_failures", 0)) / total_steps
        report.review_rate = int(metrics.get("steps_review", 0)) / total_steps
        report.model_calls = int(metrics.get("model_calls", 0))
        report.cost_usd = float((metrics.get("usage") or {}).get("cost_usd", 0.0))
    return report


def _short(value: FieldValue | None, limit: int = 60) -> str:
    if value is None:
        return "-"
    if not value.status.has_value:
        return f"<{value.status.value}>"
    text = repr(value.value)
    return text if len(text) <= limit else text[:limit] + "..."


__all__ = [
    "DeltaKind",
    "FieldDelta",
    "MigrationReport",
    "RecordDiff",
    "build_report",
    "diff_records",
]
