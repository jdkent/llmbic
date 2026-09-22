"""Value status — the state a slot is in, separate from the value it holds.

FR-PROV-009 forbids collapsing "not reported", "not applicable", "not
extracted", "extraction failed" and "unknown" into an undifferentiated null.
Those five are distinct facts with distinct consequences and they must survive
migration and export.  :class:`ValueStatus` is the only place in llmbic where
absence is encoded, and :class:`FieldValue` always carries one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ValueStatus(str, Enum):
    #: A usable value is present.
    PRESENT = "present"
    #: The attribute was examined; the source reports nothing.
    NOT_REPORTED = "not_reported"
    #: The attribute does not apply to this record (e.g. a resting-state study's
    #: stimulus set).  A positive assertion, not silence.
    NOT_APPLICABLE = "not_applicable"
    #: Examined, and the pass could not settle it.  Distinct from NOT_REPORTED:
    #: it reports on the extraction, not on the source.
    UNKNOWN = "unknown"
    #: No pass has produced a value yet.  The state a newly added field starts in.
    NOT_EXTRACTED = "not_extracted"
    #: A pass ran and failed (transport, schema, validator).
    EXTRACTION_FAILED = "extraction_failed"
    #: A value exists but is not publishable until a human decides.
    REVIEW_REQUIRED = "review_required"

    @property
    def has_value(self) -> bool:
        return self in (ValueStatus.PRESENT, ValueStatus.REVIEW_REQUIRED)

    @property
    def is_terminal_absence(self) -> bool:
        """True when absence is a finding about the source rather than a defect."""

        return self in (ValueStatus.NOT_REPORTED, ValueStatus.NOT_APPLICABLE)


#: Free-text qualifier vocabulary for NOT_REPORTED, mirroring study_schema's
#: ``UnreportedReason``.  llmbic does not constrain it; a schema may.
class AbsenceReason(str, Enum):
    AMBIGUOUS = "ambiguous"
    OUTSIDE_TEXT = "outside_text"
    CITED_ELSEWHERE = "cited_elsewhere"
    UNDETERMINED = "undetermined"


@dataclass(frozen=True)
class FieldValue:
    """A value and its status, inseparable.

    ``value`` is only meaningful when ``status.has_value``.  Constructing a
    PRESENT value of ``None`` is allowed (a schema may have a nullable field)
    but it is a different fact from NOT_REPORTED and stays different.
    """

    status: ValueStatus = ValueStatus.NOT_EXTRACTED
    value: Any = None
    reason: str | None = None
    #: Free-form notes a postprocessor or validator attached to the value.
    annotations: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def present(cls, value: Any, **annotations: Any) -> "FieldValue":
        return cls(ValueStatus.PRESENT, value, annotations=dict(annotations))

    @classmethod
    def absent(
        cls,
        status: ValueStatus = ValueStatus.NOT_REPORTED,
        reason: str | None = None,
    ) -> "FieldValue":
        if status.has_value:
            raise ValueError(f"{status} carries a value; use FieldValue.present")
        return cls(status, None, reason=reason)

    @classmethod
    def failed(cls, reason: str) -> "FieldValue":
        return cls(ValueStatus.EXTRACTION_FAILED, None, reason=reason)

    def with_status(self, status: ValueStatus, reason: str | None = None) -> "FieldValue":
        return FieldValue(
            status,
            self.value if status.has_value else None,
            reason if reason is not None else self.reason,
            dict(self.annotations),
        )

    def to_canonical(self) -> dict[str, Any]:
        out: dict[str, Any] = {"status": self.status.value}
        if self.status.has_value:
            out["value"] = self.value
        if self.reason:
            out["reason"] = self.reason
        if self.annotations:
            out["annotations"] = self.annotations
        return out

    @classmethod
    def from_canonical(cls, data: dict[str, Any]) -> "FieldValue":
        return cls(
            ValueStatus(data["status"]),
            data.get("value"),
            data.get("reason"),
            dict(data.get("annotations") or {}),
        )
