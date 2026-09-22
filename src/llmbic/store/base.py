"""Storage interfaces.

FR-STO-001 lists what the core must be able to store: schema registry,
migration registry, source documents, parsed document units, records, field
artifacts, evidence, execution state and review events.  They are one protocol
here rather than nine, because every backend that has been useful in practice
implements them over one transaction boundary — but the protocol is narrow
enough that a different backend is a few hundred lines (FR-STO-003).

Two invariants every implementation must keep:

* a logical execution/cache key is unique (FR-STO-004);
* a field artifact and a record version, once written, are never mutated
  (FR-EXE-006).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Mapping, Protocol, Sequence, runtime_checkable

from ..provenance import FieldArtifact, RecordVersion, ReviewEvent
from ..source import ParsedSource, SourceArtifact


@dataclass(frozen=True)
class RecordIndexEntry:
    """The cheap, always-loaded summary of a record.

    NFR-PERF-001: planning 100,000 records must not load their source text.
    The planner works from these rows plus field artifacts, and only touches
    :class:`ParsedSource` when a step actually needs context.
    """

    record_id: str
    schema_ref: str
    source_id: str | None = None
    source_version: str | None = None
    parse_version: str | None = None
    state: str = "published"
    version_id: str | None = None
    updated_at: str = ""
    labels: tuple[str, ...] = ()


@dataclass
class RecordFilter:
    """Corpus selection (FR-PLN-006)."""

    record_ids: Sequence[str] | None = None
    schema_ref: str | None = None
    source_version: str | None = None
    state: str | None = None
    #: Restrict to records that have a failed or review-needed step in a
    #: previous execution.
    failed_in_execution: str | None = None
    limit: int | None = None
    offset: int = 0

    def is_empty(self) -> bool:
        return not any(
            (
                self.record_ids,
                self.schema_ref,
                self.source_version,
                self.state,
                self.failed_in_execution,
            )
        )


@dataclass(frozen=True)
class CacheEntry:
    cache_key: str
    payload: dict[str, Any]
    created_at: str
    execution_id: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepState:
    execution_id: str
    step_key: str
    state: str
    record_id: str = ""
    artifact_id: str | None = None
    attempts: int = 0
    detail: dict[str, Any] = field(default_factory=dict)
    updated_at: str = ""


@runtime_checkable
class Store(Protocol):
    # ---- registry -------------------------------------------------------
    def save_registry(self, payload: Mapping[str, Any]) -> None: ...
    def load_registry(self) -> dict[str, Any]: ...

    # ---- sources --------------------------------------------------------
    def put_source(self, source: SourceArtifact) -> None: ...
    def get_source(self, source_id: str, source_version: str) -> SourceArtifact | None: ...
    def put_parsed(self, parsed: ParsedSource) -> None: ...
    def get_parsed(
        self, source_id: str, source_version: str, parse_version: str | None = None
    ) -> ParsedSource | None: ...

    # ---- records --------------------------------------------------------
    def put_artifacts(self, artifacts: Sequence[FieldArtifact]) -> list[str]: ...
    def get_artifacts(self, record_id: str) -> list[FieldArtifact]: ...
    def get_artifact(self, artifact_id: str) -> FieldArtifact | None: ...
    def put_record_version(self, version: RecordVersion) -> str: ...
    def get_record_version(self, version_id: str) -> RecordVersion | None: ...
    def publish(self, version: RecordVersion) -> None: ...
    def current_version(self, record_id: str) -> RecordVersion | None: ...
    def index(self, filt: RecordFilter | None = None) -> Iterator[RecordIndexEntry]: ...
    def upsert_index(self, entry: RecordIndexEntry) -> None: ...

    # ---- execution ------------------------------------------------------
    def create_execution(self, execution_id: str, payload: Mapping[str, Any]) -> None: ...
    def update_execution(self, execution_id: str, **changes: Any) -> None: ...
    def get_execution(self, execution_id: str) -> dict[str, Any] | None: ...
    def list_executions(self, limit: int = 50) -> list[dict[str, Any]]: ...
    def put_step_state(self, state: StepState) -> None: ...
    def get_step_states(self, execution_id: str) -> list[StepState]: ...
    def get_step_state(self, execution_id: str, step_key: str) -> StepState | None: ...
    def put_attempt(self, execution_id: str, step_key: str, payload: Mapping[str, Any]) -> None: ...
    def get_attempts(self, execution_id: str, step_key: str | None = None) -> list[dict[str, Any]]: ...

    # ---- cache ----------------------------------------------------------
    def cache_get(self, cache_key: str) -> CacheEntry | None: ...
    def cache_put(self, entry: CacheEntry) -> bool: ...

    # ---- review ---------------------------------------------------------
    def put_review_item(self, item: Mapping[str, Any]) -> None: ...
    def get_review_items(
        self, execution_id: str | None = None, state: str | None = None
    ) -> list[dict[str, Any]]: ...
    def update_review_item(self, item_id: str, **changes: Any) -> None: ...
    def put_review_event(self, event: ReviewEvent) -> None: ...
    def get_review_events(self, record_id: str | None = None) -> list[ReviewEvent]: ...

    # ---- plans ----------------------------------------------------------
    def put_plan(self, plan_id: str, payload: Mapping[str, Any]) -> None: ...
    def get_plan(self, plan_id: str) -> dict[str, Any] | None: ...
    def list_plans(self, limit: int = 50) -> list[dict[str, Any]]: ...

    def close(self) -> None: ...


def batched(items: Iterable[Any], size: int) -> Iterator[list[Any]]:
    """Page corpus operations rather than materialising them (NFR-PERF-002)."""

    buf: list[Any] = []
    for item in items:
        buf.append(item)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf
