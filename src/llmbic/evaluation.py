"""Evaluating a semantic migration before a corpus-wide rollout.

§15.5 asks for evaluation on a frozen adjudicated sample, reporting field-level
precision, recall where measurable, evidence support, null/abstention
behaviour, schema validity, change rate relative to the previous extraction,
and cost.  §21.7 asks the same migration to be tried on a canary subset before
it is turned loose, and FR-VAL-010 asks for configurable gates that block a
rollout when the numbers are wrong.

The gold corpus is a frozen file — the adjudication is data, not code — and the
evaluation runs the ordinary planner and engine against it, so what is measured
is the migration that will actually run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .errors import ErrorCode, LlmbicError
from .ids import now
from .provenance import FieldArtifact
from .values import FieldValue, ValueStatus


@dataclass(frozen=True)
class GoldValue:
    """One adjudicated answer."""

    record_id: str
    field_id: str
    entity: str = ""
    status: ValueStatus = ValueStatus.PRESENT
    value: Any = None
    #: Alternatives an adjudicator judged equally correct.
    also_acceptable: tuple[Any, ...] = ()
    note: str = ""

    def matches(self, actual: FieldValue) -> bool:
        if actual.status is not self.status:
            return False
        if not self.status.has_value:
            return True
        return _eq(actual.value, self.value) or any(
            _eq(actual.value, alt) for alt in self.also_acceptable
        )

    def to_canonical(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "field_id": self.field_id,
            "entity": self.entity,
            "status": self.status.value,
            "value": self.value,
            "also_acceptable": list(self.also_acceptable),
            "note": self.note,
        }

    @classmethod
    def from_canonical(cls, data: Mapping[str, Any]) -> "GoldValue":
        return cls(
            record_id=data["record_id"],
            field_id=data["field_id"],
            entity=data.get("entity", ""),
            status=ValueStatus(data.get("status", "present")),
            value=data.get("value"),
            also_acceptable=tuple(data.get("also_acceptable") or ()),
            note=data.get("note", ""),
        )


def _eq(a: Any, b: Any) -> bool:
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return sorted(map(str, a)) == sorted(map(str, b))
    return a == b


@dataclass
class GoldCorpus:
    """A frozen adjudicated sample (FR-VAL-008)."""

    name: str
    values: tuple[GoldValue, ...] = ()
    frozen_at: str = field(default_factory=now)
    note: str = ""

    def record_ids(self) -> list[str]:
        return sorted({g.record_id for g in self.values})

    def field_ids(self) -> list[str]:
        return sorted({g.field_id for g in self.values})

    def for_field(self, field_id: str) -> list[GoldValue]:
        return [g for g in self.values if g.field_id == field_id]

    def lookup(self, record_id: str, field_id: str, entity: str = "") -> GoldValue | None:
        for g in self.values:
            if (g.record_id, g.field_id, g.entity) == (record_id, field_id, entity):
                return g
        return None

    # ---- io -------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            fh.write(
                json.dumps({"name": self.name, "frozen_at": self.frozen_at, "note": self.note})
                + "\n"
            )
            for g in self.values:
                fh.write(json.dumps(g.to_canonical(), ensure_ascii=False) + "\n")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "GoldCorpus":
        lines = [l for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
        if not lines:
            raise LlmbicError(f"{path} is empty", code=ErrorCode.CONFIG_INVALID)
        header = json.loads(lines[0])
        return cls(
            name=header.get("name", str(path)),
            frozen_at=header.get("frozen_at", ""),
            note=header.get("note", ""),
            values=tuple(GoldValue.from_canonical(json.loads(l)) for l in lines[1:]),
        )


@dataclass
class FieldScore:
    """Per-field evaluation of one semantic migration."""

    field_id: str
    n: int = 0
    #: Gold says a value is there, we produced the same one.
    true_positive: int = 0
    #: Gold says a value is there, we produced a different one.
    wrong_value: int = 0
    #: Gold says a value is there, we abstained.
    missed: int = 0
    #: Gold says nothing is there, we produced something.
    spurious: int = 0
    #: Gold says nothing is there, we abstained.
    true_negative: int = 0
    #: Produced a value with at least one resolvable span.
    supported: int = 0
    with_value: int = 0
    schema_valid: int = 0
    changed_from_previous: int = 0
    compared_with_previous: int = 0

    @property
    def precision(self) -> float | None:
        produced = self.true_positive + self.wrong_value + self.spurious
        return self.true_positive / produced if produced else None

    @property
    def recall(self) -> float | None:
        expected = self.true_positive + self.wrong_value + self.missed
        return self.true_positive / expected if expected else None

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if p is None or r is None or (p + r) == 0:
            return None
        return 2 * p * r / (p + r)

    @property
    def abstention_rate(self) -> float:
        return (self.n - self.with_value) / self.n if self.n else 0.0

    @property
    def evidence_support(self) -> float | None:
        return self.supported / self.with_value if self.with_value else None

    @property
    def schema_validity(self) -> float:
        return self.schema_valid / self.n if self.n else 0.0

    @property
    def change_rate(self) -> float | None:
        return (
            self.changed_from_previous / self.compared_with_previous
            if self.compared_with_previous
            else None
        )

    def to_canonical(self) -> dict[str, Any]:
        return {
            "field_id": self.field_id,
            "n": self.n,
            "true_positive": self.true_positive,
            "wrong_value": self.wrong_value,
            "missed": self.missed,
            "spurious": self.spurious,
            "true_negative": self.true_negative,
            "precision": _round(self.precision),
            "recall": _round(self.recall),
            "f1": _round(self.f1),
            "abstention_rate": _round(self.abstention_rate),
            "evidence_support": _round(self.evidence_support),
            "schema_validity": _round(self.schema_validity),
            "change_rate": _round(self.change_rate),
        }


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


@dataclass
class EvaluationReport:
    corpus: str
    target_schema: str
    fields: dict[str, FieldScore] = field(default_factory=dict)
    n_records: int = 0
    model_calls: int = 0
    cost_usd: float = 0.0
    blocked: int = 0
    review: int = 0
    gate_failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.gate_failures

    def to_canonical(self) -> dict[str, Any]:
        return {
            "corpus": self.corpus,
            "target_schema": self.target_schema,
            "n_records": self.n_records,
            "model_calls": self.model_calls,
            "cost_usd": round(self.cost_usd, 6),
            "blocked": self.blocked,
            "review": self.review,
            "fields": {k: v.to_canonical() for k, v in sorted(self.fields.items())},
            "gate_failures": list(self.gate_failures),
            "passed": self.passed,
        }

    def render(self) -> str:
        lines = [
            f"evaluation of {self.target_schema} against {self.corpus}",
            f"  records: {self.n_records}   model calls: {self.model_calls}   "
            f"cost: ${self.cost_usd:.4f}   blocked: {self.blocked}   review: {self.review}",
        ]
        for score in self.fields.values():
            lines.append(f"  {score.field_id}")
            lines.append(
                "    precision={p}  recall={r}  f1={f}  abstention={a}  "
                "evidence={e}  schema_valid={v}  change_rate={c}".format(
                    p=_fmt(score.precision),
                    r=_fmt(score.recall),
                    f=_fmt(score.f1),
                    a=_fmt(score.abstention_rate),
                    e=_fmt(score.evidence_support),
                    v=_fmt(score.schema_validity),
                    c=_fmt(score.change_rate),
                )
            )
        if self.gate_failures:
            lines.append("  GATES FAILED:")
            lines.extend(f"    - {g}" for g in self.gate_failures)
        else:
            lines.append("  all gates passed")
        return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


@dataclass(frozen=True)
class RolloutGate:
    """A threshold that blocks a rollout (FR-VAL-010)."""

    metric: str  # precision | recall | f1 | change_rate | abstention_rate |
    # evidence_support | schema_validity | review_rate | blocked_rate | cost_usd
    field_id: str | None = None
    min_value: float | None = None
    max_value: float | None = None

    def describe(self) -> str:
        where = f" on {self.field_id}" if self.field_id else ""
        bounds = []
        if self.min_value is not None:
            bounds.append(f">= {self.min_value}")
        if self.max_value is not None:
            bounds.append(f"<= {self.max_value}")
        return f"{self.metric}{where} must be {' and '.join(bounds)}"

    def check(self, report: EvaluationReport) -> str | None:
        value = _metric_value(report, self)
        if value is None:
            return None
        if self.min_value is not None and value < self.min_value:
            return f"{self.describe()}, got {value:.4f}"
        if self.max_value is not None and value > self.max_value:
            return f"{self.describe()}, got {value:.4f}"
        return None


def _metric_value(report: EvaluationReport, gate: RolloutGate) -> float | None:
    corpus_metrics = {
        "cost_usd": report.cost_usd,
        "review_rate": report.review / report.n_records if report.n_records else 0.0,
        "blocked_rate": report.blocked / report.n_records if report.n_records else 0.0,
    }
    if gate.metric in corpus_metrics:
        return corpus_metrics[gate.metric]
    if gate.field_id is not None:
        # A gate naming a field the sample does not measure cannot fail: there
        # is no number, and inventing one from the other fields would make the
        # gate mean something its author did not write.
        score = report.fields.get(gate.field_id)
        scores = [score] if score is not None else []
    else:
        scores = list(report.fields.values())
    if not scores:
        return None
    values = [getattr(s, gate.metric, None) for s in scores]
    values = [v for v in values if v is not None]
    return min(values) if values else None


def evaluate(
    project: Any,
    gold: GoldCorpus,
    target_schema: str,
    *,
    gates: Sequence[RolloutGate] = (),
    policy: Any | None = None,
    baseline: Mapping[tuple[str, str, str], FieldValue] | None = None,
) -> EvaluationReport:
    """Run ``target_schema``'s migration over the gold sample and score it.

    ``baseline`` is the previous extraction's values, used for the change rate
    §15.5 asks for; when omitted it is read from the records as they stand
    before the migration runs, which is the same thing for an ordinary rollout.
    """

    from .store.base import RecordFilter

    record_ids = gold.record_ids()
    if not record_ids:
        raise LlmbicError("gold corpus is empty", code=ErrorCode.CONFIG_INVALID)

    if baseline is None:
        baseline = {
            (a.record_id, a.field_id, a.entity): a.value
            for rid in record_ids
            for a in project.artifacts(rid)
        }

    plan = project.plan(
        target_schema, filt=RecordFilter(record_ids=record_ids), policy=policy
    )
    result = project.run(plan)

    report = EvaluationReport(
        corpus=gold.name,
        target_schema=target_schema,
        n_records=len(record_ids),
        model_calls=result.metrics.model_calls,
        cost_usd=result.metrics.usage.cost_usd,
        blocked=result.metrics.steps_blocked,
        review=result.metrics.steps_review,
    )

    produced: dict[tuple[str, str, str], FieldArtifact] = {}
    for rid in record_ids:
        for a in project.artifacts(rid):
            produced[(a.record_id, a.field_id, a.entity)] = a

    schema = project.registry.schema(target_schema)
    from .validation import validate_field_value, worst
    from .provenance import Outcome

    for gold_value in gold.values:
        score = report.fields.setdefault(
            gold_value.field_id, FieldScore(field_id=gold_value.field_id)
        )
        score.n += 1
        key = (gold_value.record_id, gold_value.field_id, gold_value.entity)
        artifact = produced.get(key)
        actual = artifact.value if artifact else FieldValue(ValueStatus.NOT_EXTRACTED)

        if actual.status.has_value:
            score.with_value += 1
            if artifact and any(ref.spans for ref in artifact.evidence):
                score.supported += 1

        fdef = schema.get(gold_value.field_id)
        if fdef is not None:
            checks = validate_field_value(actual, fdef)
            if worst(checks) is not Outcome.FAIL:
                score.schema_valid += 1

        if gold_value.status.has_value:
            if not actual.status.has_value:
                score.missed += 1
            elif gold_value.matches(actual):
                score.true_positive += 1
            else:
                score.wrong_value += 1
        else:
            if actual.status.has_value:
                score.spurious += 1
            else:
                score.true_negative += 1

        prior = baseline.get(key)
        if prior is not None:
            score.compared_with_previous += 1
            if prior.to_canonical() != actual.to_canonical():
                score.changed_from_previous += 1

    for gate in gates:
        failure = gate.check(report)
        if failure:
            report.gate_failures.append(failure)

    return report


def gold_from_artifacts(
    name: str, artifacts: Iterable[FieldArtifact], field_ids: Sequence[str]
) -> GoldCorpus:
    """Freeze a sample from values a curator has already accepted.

    Useful for building the first gold corpus out of a reviewed subset; the
    result should still be read by a human before it is trusted.
    """

    wanted = set(field_ids)
    values = [
        GoldValue(
            record_id=a.record_id,
            field_id=a.field_id,
            entity=a.entity,
            status=a.value.status,
            value=a.value.value,
            note=f"from artifact {a.artifact_id}",
        )
        for a in artifacts
        if a.field_id in wanted
    ]
    return GoldCorpus(name=name, values=tuple(values))


__all__ = [
    "EvaluationReport",
    "FieldScore",
    "GoldCorpus",
    "GoldValue",
    "RolloutGate",
    "evaluate",
    "gold_from_artifacts",
]
