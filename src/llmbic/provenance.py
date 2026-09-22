"""Field artifacts, provenance and record versions.

Product principle 3: the record is the assembled *view*; the field is the unit
of computation and provenance.  A :class:`FieldArtifact` is immutable
(FR-EXE-006) and content-addressed, so re-running a migration with identical
inputs produces the identical artifact id and nothing is duplicated
(FR-PRIN-007, FR-EXE-004).

A :class:`RecordVersion` names the exact set of artifacts that make up one
record at one schema version.  Publication swaps a pointer to a complete
record version, which is what makes FR-EXE-007's atomicity possible without
locking the artifact store.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from .ids import content_hash, now
from .source import EvidenceReference
from .values import FieldValue, ValueStatus

#: Entity key for a field at the root of the record.
ROOT = ""


def entity_key(*segments: tuple[str, str]) -> str:
    """Build an entity key from ``(collection_path, local_id)`` pairs.

    ``entity_key(("groups[]", "g1"))`` -> ``"groups[]=g1"``;
    ``entity_key(("tasks[]", "t1"), ("tasks[].conditions[]", "c2"))``
    -> ``"tasks[]=t1/tasks[].conditions[]=c2"``.

    The collection path is part of the key so that two collections that happen
    to use the same local ids never collide.
    """

    return "/".join(f"{path}={local_id}" for path, local_id in segments)


def parse_entity_key(key: str) -> list[tuple[str, str]]:
    if not key:
        return []
    out: list[tuple[str, str]] = []
    for part in key.split("/"):
        path, _, local_id = part.partition("=")
        out.append((path, local_id))
    return out


@dataclass(frozen=True)
class ValidationResult:
    """Structured outcome from one validator (FR-VAL-003)."""

    class Outcome(str, Enum):
        PASS = "pass"
        FAIL = "fail"
        REVIEW = "review"

    validator: str
    outcome: "ValidationResult.Outcome"
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.outcome is ValidationResult.Outcome.PASS

    def to_canonical(self) -> dict[str, Any]:
        return {
            "validator": self.validator,
            "outcome": self.outcome.value,
            "reason": self.reason,
            "details": self.details,
        }

    @classmethod
    def from_canonical(cls, data: Mapping[str, Any]) -> "ValidationResult":
        return cls(
            validator=data["validator"],
            outcome=cls.Outcome(data["outcome"]),
            reason=data.get("reason", ""),
            details=dict(data.get("details") or {}),
        )


Outcome = ValidationResult.Outcome


@dataclass(frozen=True)
class ModelCall:
    """What a provider was asked and what it answered (FR-LLM-011)."""

    provider: str
    model: str
    parameters: dict[str, Any] = field(default_factory=dict)
    prompt_hash: str = ""
    #: Provider-reported identity of the weights behind a stable model name
    #: (decision 18.8).  When it changes, a recipe pinned to a fingerprint
    #: stops matching and its artifacts stop being semantically current.
    model_fingerprint: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    provider_request_id: str | None = None
    attempts: int = 1

    def identity(self) -> dict[str, Any]:
        """The part of a call that determines the answer.

        A provider request id, a latency and a token count describe *this*
        invocation, not the question — so they are recorded but kept out of the
        artifact's content address, which is what lets a re-run recognise its
        own previous result instead of duplicating it.
        """

        return {
            "provider": self.provider,
            "model": self.model,
            "parameters": dict(sorted(self.parameters.items())),
            "prompt_hash": self.prompt_hash,
            "model_fingerprint": self.model_fingerprint,
        }

    def to_canonical(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "parameters": dict(sorted(self.parameters.items())),
            "prompt_hash": self.prompt_hash,
            "model_fingerprint": self.model_fingerprint,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            "provider_request_id": self.provider_request_id,
            "attempts": self.attempts,
        }

    @classmethod
    def from_canonical(cls, data: Mapping[str, Any]) -> "ModelCall":
        return cls(
            provider=data["provider"],
            model=data["model"],
            parameters=dict(data.get("parameters") or {}),
            prompt_hash=data.get("prompt_hash", ""),
            model_fingerprint=data.get("model_fingerprint"),
            input_tokens=int(data.get("input_tokens", 0)),
            output_tokens=int(data.get("output_tokens", 0)),
            latency_ms=float(data.get("latency_ms", 0.0)),
            cost_usd=float(data.get("cost_usd", 0.0)),
            provider_request_id=data.get("provider_request_id"),
            attempts=int(data.get("attempts", 1)),
        )


class Actor(str, Enum):
    """Who produced a value.  FR-PROV-006: never indistinguishable."""

    EXTRACTOR = "extractor"
    MIGRATION = "migration"
    MODEL = "model"
    HUMAN = "human"
    IMPORT = "import"


@dataclass(frozen=True)
class FieldProvenance:
    """Everything FR-PROV-004 requires, in one immutable structure."""

    schema_version: str
    recipe_ref: str | None = None
    source_ref: str | None = None
    parse_version: str | None = None
    migration_id: str | None = None
    step_id: str | None = None
    execution_id: str | None = None
    context_selector_ref: str | None = None
    #: Identifiers and hashes of every context unit given to the model
    #: (FR-CTX-004).
    context_units: tuple[str, ...] = ()
    context_hash: str | None = None
    model_call: ModelCall | None = None
    prompt_hash: str | None = None
    #: ``dependency -> hash`` for every output-affecting input (FR-LLM-006).
    input_hashes: dict[str, str] = field(default_factory=dict)
    #: Exact prior values a derived field was computed from (FR-PROV-005).
    lineage: tuple[str, ...] = ()
    validations: tuple[ValidationResult, ...] = ()
    actor: Actor = Actor.EXTRACTOR
    actor_id: str | None = None
    created_at: str = field(default_factory=now)
    software_version: str = ""
    #: Set when a human accepted a legacy value under a new recipe
    #: (FR-DEP-007): the value is not rewritten, the decision is recorded.
    accepted_under: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    #: Keys that describe *this* run rather than what it produced.  They are
    #: retained in the record and excluded from the artifact's content address.
    VOLATILE_KEYS = ("created_at", "execution_id", "notes", "model_call")

    def identity_part(self) -> dict[str, Any]:
        out = {
            k: v for k, v in self.to_canonical().items() if k not in self.VOLATILE_KEYS
        }
        out["model_call"] = self.model_call.identity() if self.model_call else None
        return out

    def to_canonical(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "recipe_ref": self.recipe_ref,
            "source_ref": self.source_ref,
            "parse_version": self.parse_version,
            "migration_id": self.migration_id,
            "step_id": self.step_id,
            "execution_id": self.execution_id,
            "context_selector_ref": self.context_selector_ref,
            "context_units": list(self.context_units),
            "context_hash": self.context_hash,
            "model_call": self.model_call.to_canonical() if self.model_call else None,
            "prompt_hash": self.prompt_hash,
            "input_hashes": dict(sorted(self.input_hashes.items())),
            "lineage": list(self.lineage),
            "validations": [v.to_canonical() for v in self.validations],
            "actor": self.actor.value,
            "actor_id": self.actor_id,
            "created_at": self.created_at,
            "software_version": self.software_version,
            "accepted_under": self.accepted_under,
            "notes": dict(sorted(self.notes.items())),
        }

    @classmethod
    def from_canonical(cls, data: Mapping[str, Any]) -> "FieldProvenance":
        return cls(
            schema_version=data["schema_version"],
            recipe_ref=data.get("recipe_ref"),
            source_ref=data.get("source_ref"),
            parse_version=data.get("parse_version"),
            migration_id=data.get("migration_id"),
            step_id=data.get("step_id"),
            execution_id=data.get("execution_id"),
            context_selector_ref=data.get("context_selector_ref"),
            context_units=tuple(data.get("context_units") or ()),
            context_hash=data.get("context_hash"),
            model_call=ModelCall.from_canonical(data["model_call"])
            if data.get("model_call")
            else None,
            prompt_hash=data.get("prompt_hash"),
            input_hashes=dict(data.get("input_hashes") or {}),
            lineage=tuple(data.get("lineage") or ()),
            validations=tuple(
                ValidationResult.from_canonical(v) for v in data.get("validations") or ()
            ),
            actor=Actor(data.get("actor", "extractor")),
            actor_id=data.get("actor_id"),
            created_at=data.get("created_at", ""),
            software_version=data.get("software_version", ""),
            accepted_under=data.get("accepted_under"),
            notes=dict(data.get("notes") or {}),
        )


@dataclass(frozen=True)
class FieldArtifact:
    """An immutable value-with-provenance for one field of one record."""

    record_id: str
    field_id: str
    value: FieldValue
    provenance: FieldProvenance
    entity: str = ROOT
    evidence: tuple[EvidenceReference, ...] = ()
    #: Set when the value came from a cache entry or a reused prior artifact.
    derived_from: str | None = None

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.record_id, self.field_id, self.entity)

    @property
    def artifact_id(self) -> str:
        """Content address.  Two identical computations produce one artifact."""

        return content_hash(
            {
                "record_id": self.record_id,
                "field_id": self.field_id,
                "entity": self.entity,
                "value": self.value.to_canonical(),
                "evidence": [e.to_canonical() for e in self.evidence],
                # When a value was computed, by which execution, and what the
                # provider called the request do not change *what* it is.
                "provenance": self.provenance.identity_part(),
            }
        )

    def value_hash(self) -> str:
        """Hash of the value alone — what a *dependent* field depends on."""

        return content_hash(self.value.to_canonical())

    def with_value(self, value: FieldValue) -> "FieldArtifact":
        return replace(self, value=value)

    def to_canonical(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "record_id": self.record_id,
            "field_id": self.field_id,
            "entity": self.entity,
            "value": self.value.to_canonical(),
            "evidence": [e.to_canonical() for e in self.evidence],
            "provenance": self.provenance.to_canonical(),
            "derived_from": self.derived_from,
        }

    @classmethod
    def from_canonical(cls, data: Mapping[str, Any]) -> "FieldArtifact":
        return cls(
            record_id=data["record_id"],
            field_id=data["field_id"],
            entity=data.get("entity", ROOT),
            value=FieldValue.from_canonical(data["value"]),
            evidence=tuple(EvidenceReference.from_canonical(e) for e in data.get("evidence", [])),
            provenance=FieldProvenance.from_canonical(data["provenance"]),
            derived_from=data.get("derived_from"),
        )


@dataclass(frozen=True)
class RecordEntity:
    """A member of a nested collection, with a persistent logical identity."""

    entity: str
    collection_path: str
    local_id: str
    position: int
    parent: str = ROOT

    def to_canonical(self) -> dict[str, Any]:
        return {
            "entity": self.entity,
            "collection_path": self.collection_path,
            "local_id": self.local_id,
            "position": self.position,
            "parent": self.parent,
        }

    @classmethod
    def from_canonical(cls, data: Mapping[str, Any]) -> "RecordEntity":
        return cls(
            entity=data["entity"],
            collection_path=data["collection_path"],
            local_id=data["local_id"],
            position=int(data["position"]),
            parent=data.get("parent", ROOT),
        )


class RecordState(str, Enum):
    DRAFT = "draft"
    PUBLISHED = "published"
    SUPERSEDED = "superseded"
    FAILED = "failed"
    REVIEW_NEEDED = "review_needed"


@dataclass(frozen=True)
class RecordVersion:
    """An immutable assembled record for one schema version."""

    record_id: str
    schema_ref: str
    artifact_ids: tuple[str, ...] = ()
    entities: tuple[RecordEntity, ...] = ()
    state: RecordState = RecordState.DRAFT
    parent_version_id: str | None = None
    execution_id: str | None = None
    source_ref: str | None = None
    created_at: str = field(default_factory=now)
    #: Field ids that are semantically current under the target recipes.
    #: Conformance to the shape is *not* this (terminology §6).
    current_field_ids: tuple[str, ...] = ()
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def version_id(self) -> str:
        return content_hash(
            {
                "record_id": self.record_id,
                "schema_ref": self.schema_ref,
                "artifact_ids": sorted(self.artifact_ids),
                "entities": [e.to_canonical() for e in self.entities],
                "parent": self.parent_version_id,
            }
        )

    def to_canonical(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "record_id": self.record_id,
            "schema_ref": self.schema_ref,
            "artifact_ids": list(self.artifact_ids),
            "entities": [e.to_canonical() for e in self.entities],
            "state": self.state.value,
            "parent_version_id": self.parent_version_id,
            "execution_id": self.execution_id,
            "source_ref": self.source_ref,
            "created_at": self.created_at,
            "current_field_ids": list(self.current_field_ids),
            "notes": dict(sorted(self.notes.items())),
        }

    @classmethod
    def from_canonical(cls, data: Mapping[str, Any]) -> "RecordVersion":
        return cls(
            record_id=data["record_id"],
            schema_ref=data["schema_ref"],
            artifact_ids=tuple(data.get("artifact_ids") or ()),
            entities=tuple(RecordEntity.from_canonical(e) for e in data.get("entities") or ()),
            state=RecordState(data.get("state", "draft")),
            parent_version_id=data.get("parent_version_id"),
            execution_id=data.get("execution_id"),
            source_ref=data.get("source_ref"),
            created_at=data.get("created_at", ""),
            current_field_ids=tuple(data.get("current_field_ids") or ()),
            notes=dict(data.get("notes") or {}),
        )


class ReviewDecision(str, Enum):
    ACCEPT = "accept"
    EDIT = "edit"
    REJECT = "reject"
    DEFER = "defer"
    REQUEST_CONTEXT = "request_expanded_context"


@dataclass(frozen=True)
class ReviewEvent:
    """A durable provenance event carrying a human decision (FR-VAL-007)."""

    record_id: str
    field_id: str
    decision: ReviewDecision
    actor_id: str
    entity: str = ROOT
    rationale: str = ""
    edited_value: FieldValue | None = None
    migration_id: str | None = None
    execution_id: str | None = None
    created_at: str = field(default_factory=now)
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def event_id(self) -> str:
        return content_hash(self.to_canonical())

    def to_canonical(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "field_id": self.field_id,
            "entity": self.entity,
            "decision": self.decision.value,
            "actor_id": self.actor_id,
            "rationale": self.rationale,
            "edited_value": self.edited_value.to_canonical() if self.edited_value else None,
            "migration_id": self.migration_id,
            "execution_id": self.execution_id,
            "created_at": self.created_at,
            "payload": dict(sorted(self.payload.items())),
        }

    @classmethod
    def from_canonical(cls, data: Mapping[str, Any]) -> "ReviewEvent":
        return cls(
            record_id=data["record_id"],
            field_id=data["field_id"],
            entity=data.get("entity", ROOT),
            decision=ReviewDecision(data["decision"]),
            actor_id=data["actor_id"],
            rationale=data.get("rationale", ""),
            edited_value=FieldValue.from_canonical(data["edited_value"])
            if data.get("edited_value")
            else None,
            migration_id=data.get("migration_id"),
            execution_id=data.get("execution_id"),
            created_at=data.get("created_at", now()),
            payload=dict(data.get("payload") or {}),
        )


def artifacts_by_key(
    artifacts: Iterable[FieldArtifact],
) -> dict[tuple[str, str, str], FieldArtifact]:
    return {a.key: a for a in artifacts}


def latest_per_key(artifacts: Sequence[FieldArtifact]) -> list[FieldArtifact]:
    """Keep one artifact per ``(record, field, entity)``, the newest wins.

    Prior artifacts stay in the store; this is only the assembled view.
    """

    best: dict[tuple[str, str, str], FieldArtifact] = {}
    for a in artifacts:
        prior = best.get(a.key)
        if prior is None or a.provenance.created_at >= prior.provenance.created_at:
            best[a.key] = a
    return list(best.values())


__all__ = [
    "Actor",
    "FieldArtifact",
    "FieldProvenance",
    "FieldValue",
    "ModelCall",
    "Outcome",
    "ROOT",
    "RecordEntity",
    "RecordState",
    "RecordVersion",
    "ReviewDecision",
    "ReviewEvent",
    "ValidationResult",
    "ValueStatus",
    "artifacts_by_key",
    "entity_key",
    "latest_per_key",
    "parse_entity_key",
]
