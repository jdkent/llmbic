"""Execution states, counters and events."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping

from ..ids import now
from ..models.base import Usage


class StepStatus(str, Enum):
    """NFR-REL-002's state set, at step granularity."""

    PLANNED = "planned"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"
    REVIEW_NEEDED = "review_needed"
    SKIPPED = "skipped"

    @property
    def terminal(self) -> bool:
        return self in (
            StepStatus.SUCCEEDED,
            StepStatus.FAILED,
            StepStatus.BLOCKED,
            StepStatus.CANCELLED,
            StepStatus.REVIEW_NEEDED,
            StepStatus.SKIPPED,
        )

    @property
    def blocks_publication(self) -> bool:
        return self in (
            StepStatus.FAILED,
            StepStatus.BLOCKED,
            StepStatus.REVIEW_NEEDED,
            StepStatus.CANCELLED,
        )


class ExecutionStatus(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BUDGET_EXHAUSTED = "budget_exhausted"


@dataclass
class Metrics:
    """NFR-OBS-001's metric set."""

    steps_total: int = 0
    steps_succeeded: int = 0
    steps_failed: int = 0
    steps_blocked: int = 0
    steps_review: int = 0
    steps_reused: int = 0
    steps_skipped: int = 0
    model_calls: int = 0
    cache_hits: int = 0
    cache_writes: int = 0
    retries: int = 0
    validation_failures: int = 0
    records_published: int = 0
    records_held: int = 0
    usage: Usage = field(default_factory=Usage)
    elapsed_s: float = 0.0

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.model_calls
        return self.cache_hits / total if total else 0.0

    def to_canonical(self) -> dict[str, Any]:
        return {
            "steps_total": self.steps_total,
            "steps_succeeded": self.steps_succeeded,
            "steps_failed": self.steps_failed,
            "steps_blocked": self.steps_blocked,
            "steps_review": self.steps_review,
            "steps_reused": self.steps_reused,
            "steps_skipped": self.steps_skipped,
            "model_calls": self.model_calls,
            "cache_hits": self.cache_hits,
            "cache_writes": self.cache_writes,
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "retries": self.retries,
            "validation_failures": self.validation_failures,
            "records_published": self.records_published,
            "records_held": self.records_held,
            "usage": self.usage.to_canonical(),
            "elapsed_s": round(self.elapsed_s, 3),
        }


@dataclass
class ExecutionResult:
    execution_id: str
    plan_id: str
    status: ExecutionStatus
    metrics: Metrics = field(default_factory=Metrics)
    #: ``record_id -> one of published | held | blocked | failed | unchanged``
    records: dict[str, str] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)
    review_items: list[str] = field(default_factory=list)
    started_at: str = field(default_factory=now)
    finished_at: str = ""

    @property
    def published(self) -> list[str]:
        return sorted(r for r, s in self.records.items() if s == "published")

    @property
    def held(self) -> list[str]:
        return sorted(r for r, s in self.records.items() if s != "published")

    def to_canonical(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "plan_id": self.plan_id,
            "status": self.status.value,
            "metrics": self.metrics.to_canonical(),
            "records": dict(sorted(self.records.items())),
            "errors": self.errors,
            "review_items": sorted(self.review_items),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


EventHook = Callable[[Mapping[str, Any]], None]


def emit(hook: EventHook | None, kind: str, **payload: Any) -> None:
    """Structured events for orchestration and monitoring (FR-API-006).

    Deliberately carries identifiers, never prompts or source text
    (NFR-OBS-003).
    """

    if hook is None:
        return
    hook({"event": kind, "at": now(), **payload})


__all__ = [
    "EventHook",
    "ExecutionResult",
    "ExecutionStatus",
    "Metrics",
    "StepStatus",
    "emit",
]
