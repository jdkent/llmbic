"""Review queue and curator decisions.

Decision 18.6 is answered with a JSONL round trip: it needs no service, it
diffs in review, and it is trivially adapted to an annotation platform.  The
queue row carries everything FR-VAL-005 lists — old value, proposed value,
evidence, context, migration identity, validator results and the reason for
escalation — so a curator never has to go looking.

A decision is a durable provenance event (FR-VAL-007) *and*, when it changes a
value, a new artifact attributed to a human (FR-PROV-006).  Nothing overwrites
history: the model's answer stays in the store beside the correction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .errors import ErrorCode, LlmbicError
from .ids import now
from .provenance import (
    Actor,
    FieldArtifact,
    FieldProvenance,
    ReviewDecision,
    ReviewEvent,
    ValidationResult,
)
from .source import EvidenceReference
from .store.base import Store
from .values import FieldValue, ValueStatus


@dataclass
class ReviewItem:
    item_id: str
    record_id: str
    field_id: str
    entity: str = ""
    execution_id: str | None = None
    migration_id: str | None = None
    step_id: str | None = None
    state: str = "open"
    reason: str = ""
    old_value: dict[str, Any] | None = None
    proposed_value: dict[str, Any] | None = None
    evidence: list[dict[str, Any]] = field(default_factory=list)
    context: dict[str, Any] | None = None
    validations: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=now)

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "ReviewItem":
        return cls(
            item_id=row["item_id"],
            record_id=row["record_id"],
            field_id=row["field_id"],
            entity=row.get("entity", ""),
            execution_id=row.get("execution_id"),
            migration_id=row.get("migration_id"),
            step_id=row.get("step_id"),
            state=row.get("state", "open"),
            reason=row.get("reason", ""),
            old_value=row.get("old_value"),
            proposed_value=row.get("proposed_value"),
            evidence=list(row.get("evidence") or []),
            context=row.get("context"),
            validations=list(row.get("validations") or []),
            created_at=row.get("created_at", now()),
        )

    def to_row(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "record_id": self.record_id,
            "field_id": self.field_id,
            "entity": self.entity,
            "execution_id": self.execution_id,
            "migration_id": self.migration_id,
            "step_id": self.step_id,
            "state": self.state,
            "reason": self.reason,
            "old_value": self.old_value,
            "proposed_value": self.proposed_value,
            "evidence": self.evidence,
            "context": self.context,
            "validations": self.validations,
            "created_at": self.created_at,
        }

    def to_export(self) -> dict[str, Any]:
        """The row a curator edits.  ``decision`` and ``rationale`` start blank."""

        row = self.to_row()
        row["decision"] = ""
        row["rationale"] = ""
        row["edited_value"] = None
        return row


class ReviewQueue:
    def __init__(self, store: Store) -> None:
        self.store = store

    # ---- reading ---------------------------------------------------------
    def items(
        self, *, execution_id: str | None = None, state: str | None = "open"
    ) -> list[ReviewItem]:
        return [
            ReviewItem.from_row(r)
            for r in self.store.get_review_items(execution_id=execution_id, state=state)
        ]

    def open_count(self, execution_id: str | None = None) -> int:
        return len(self.items(execution_id=execution_id, state="open"))

    # ---- export / import -------------------------------------------------
    def export_jsonl(
        self, path: str | Path, *, execution_id: str | None = None, state: str = "open"
    ) -> int:
        items = self.items(execution_id=execution_id, state=state)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            for item in items:
                fh.write(json.dumps(item.to_export(), ensure_ascii=False) + "\n")
        return len(items)

    def import_jsonl(
        self,
        path: str | Path,
        *,
        actor_id: str,
        schema_ref: str = "",
        apply_values: bool = True,
    ) -> list[ReviewEvent]:
        events: list[ReviewEvent] = []
        with Path(path).open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise LlmbicError(
                        f"{path}:{line_no} is not valid JSON: {exc}",
                        code=ErrorCode.CONFIG_INVALID,
                    ) from exc
                decision = (row.get("decision") or "").strip()
                if not decision:
                    continue
                events.append(
                    self.decide(
                        item_id=row["item_id"],
                        decision=ReviewDecision(decision),
                        actor_id=row.get("actor_id") or actor_id,
                        rationale=row.get("rationale", ""),
                        edited_value=_value_from(row.get("edited_value")),
                        schema_ref=schema_ref,
                        apply_values=apply_values,
                    )
                )
        return events

    # ---- deciding --------------------------------------------------------
    def decide(
        self,
        *,
        item_id: str,
        decision: ReviewDecision,
        actor_id: str,
        rationale: str = "",
        edited_value: FieldValue | None = None,
        schema_ref: str = "",
        apply_values: bool = True,
    ) -> ReviewEvent:
        rows = self.store.get_review_items(state=None)
        row = next((r for r in rows if r["item_id"] == item_id), None)
        if row is None:
            raise LlmbicError(
                f"unknown review item {item_id!r}", code=ErrorCode.RECORD_NOT_FOUND
            )
        item = ReviewItem.from_row(row)

        if decision is ReviewDecision.EDIT and edited_value is None:
            raise LlmbicError(
                f"review item {item_id!r} was edited but carries no edited_value",
                code=ErrorCode.CONFIG_INVALID,
            )

        event = ReviewEvent(
            record_id=item.record_id,
            field_id=item.field_id,
            entity=item.entity,
            decision=decision,
            actor_id=actor_id,
            rationale=rationale,
            edited_value=edited_value,
            migration_id=item.migration_id,
            execution_id=item.execution_id,
            payload={"item_id": item_id, "reason": item.reason},
        )
        self.store.put_review_event(event)

        new_state = {
            ReviewDecision.ACCEPT: "accepted",
            ReviewDecision.EDIT: "edited",
            ReviewDecision.REJECT: "rejected",
            ReviewDecision.DEFER: "deferred",
            ReviewDecision.REQUEST_CONTEXT: "context_requested",
        }[decision]
        self.store.update_review_item(item_id, state=new_state, decided_by=actor_id)

        if apply_values and decision in (ReviewDecision.ACCEPT, ReviewDecision.EDIT):
            artifact = self._artifact_for(item, decision, edited_value, actor_id, rationale, schema_ref)
            if artifact is not None:
                self.store.put_artifacts([artifact])
        return event

    def _artifact_for(
        self,
        item: ReviewItem,
        decision: ReviewDecision,
        edited_value: FieldValue | None,
        actor_id: str,
        rationale: str,
        schema_ref: str,
    ) -> FieldArtifact | None:
        if decision is ReviewDecision.EDIT:
            value = edited_value
        else:
            source = item.proposed_value or item.old_value
            if source is None:
                return None
            value = FieldValue.from_canonical(source)
            if value.status is ValueStatus.REVIEW_REQUIRED:
                value = value.with_status(ValueStatus.PRESENT, reason=None)
        if value is None:
            return None

        evidence = tuple(EvidenceReference.from_canonical(e) for e in item.evidence)
        return FieldArtifact(
            record_id=item.record_id,
            field_id=item.field_id,
            entity=item.entity,
            value=value,
            evidence=evidence,
            provenance=FieldProvenance(
                schema_version=schema_ref,
                migration_id=item.migration_id,
                step_id=item.step_id,
                execution_id=item.execution_id,
                actor=Actor.HUMAN,
                actor_id=actor_id,
                notes={"review_item": item.item_id, "rationale": rationale},
                validations=tuple(
                    ValidationResult.from_canonical(v) for v in item.validations
                ),
            ),
        )

    # ---- FR-DEP-007 ------------------------------------------------------
    def accept_legacy_value(
        self,
        *,
        record_id: str,
        field_id: str,
        recipe_ref: str,
        actor_id: str,
        entity: str = "",
        rationale: str = "",
        schema_ref: str = "",
    ) -> FieldArtifact:
        """Declare an old value acceptable under a new recipe.

        History is not rewritten: a *new* artifact is written that records who
        accepted the value and under which recipe, and the original stays.
        """

        prior = None
        for artifact in self.store.get_artifacts(record_id):
            if artifact.field_id == field_id and artifact.entity == entity:
                prior = artifact
        if prior is None:
            raise LlmbicError(
                f"{record_id}/{field_id} has no stored value to accept",
                code=ErrorCode.RECORD_NOT_FOUND,
            )

        accepted = FieldArtifact(
            record_id=record_id,
            field_id=field_id,
            entity=entity,
            value=prior.value,
            evidence=prior.evidence,
            provenance=FieldProvenance(
                schema_version=schema_ref or prior.provenance.schema_version,
                recipe_ref=recipe_ref,
                source_ref=prior.provenance.source_ref,
                parse_version=prior.provenance.parse_version,
                model_call=prior.provenance.model_call,
                input_hashes=dict(prior.provenance.input_hashes),
                actor=Actor.HUMAN,
                actor_id=actor_id,
                accepted_under=recipe_ref,
                notes={"rationale": rationale, "accepts_artifact": prior.artifact_id},
            ),
            derived_from=prior.artifact_id,
        )
        self.store.put_artifacts([accepted])
        self.store.put_review_event(
            ReviewEvent(
                record_id=record_id,
                field_id=field_id,
                entity=entity,
                decision=ReviewDecision.ACCEPT,
                actor_id=actor_id,
                rationale=rationale or f"legacy value accepted under {recipe_ref}",
                payload={"accepts_artifact": prior.artifact_id, "recipe_ref": recipe_ref},
            )
        )
        return accepted


def review_report(queue: ReviewQueue, execution_id: str | None = None) -> dict[str, Any]:
    rows = queue.store.get_review_items(execution_id=execution_id, state=None)
    by_state: dict[str, int] = {}
    by_reason: dict[str, int] = {}
    for r in rows:
        by_state[r.get("state", "open")] = by_state.get(r.get("state", "open"), 0) + 1
        reason = (r.get("reason") or "").split(";")[0][:80]
        by_reason[reason] = by_reason.get(reason, 0) + 1
    return {
        "total": len(rows),
        "by_state": dict(sorted(by_state.items())),
        "by_reason": dict(sorted(by_reason.items(), key=lambda kv: -kv[1])[:20]),
    }


def _value_from(raw: Any) -> FieldValue | None:
    if raw is None:
        return None
    if isinstance(raw, Mapping) and "status" in raw:
        return FieldValue.from_canonical(dict(raw))
    return FieldValue.present(raw)


__all__ = ["ReviewItem", "ReviewQueue", "review_report"]
