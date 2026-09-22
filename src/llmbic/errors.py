"""Structured error codes and exceptions.

NFR-REL-003 requires failures to carry structured error codes and sanitized
diagnostics.  Every exception raised by llmbic carries an :class:`ErrorCode` and
an optional ``details`` mapping that is safe to serialise into an execution
record (callers are responsible for not putting secrets into ``details``; see
:mod:`llmbic.redaction`).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping


class ErrorCode(str, Enum):
    # registry / schema
    SCHEMA_NOT_FOUND = "schema_not_found"
    SCHEMA_IMMUTABLE = "schema_immutable"
    SCHEMA_ADAPTER = "schema_adapter"
    UNMAPPED_RENAME = "unmapped_rename"

    # migration registry
    MIGRATION_NOT_FOUND = "migration_not_found"
    MIGRATION_CYCLE = "migration_cycle"
    NO_MIGRATION_PATH = "no_migration_path"
    AMBIGUOUS_MIGRATION_PATH = "ambiguous_migration_path"
    MIGRATION_INVALID = "migration_invalid"
    FIDELITY_NOT_PERMITTED = "fidelity_not_permitted"

    # planning
    PLAN_INVALID = "plan_invalid"
    PLAN_SIGNATURE_MISMATCH = "plan_signature_mismatch"
    MISSING_DEPENDENCY = "missing_dependency"
    MISSING_PROVENANCE = "missing_provenance"

    # context
    CONTEXT_UNAVAILABLE = "context_unavailable"
    CONTEXT_BUDGET_EXCEEDED = "context_budget_exceeded"
    CONTEXT_POLICY_FORBIDS = "context_policy_forbids"
    PRIVACY_POLICY_FORBIDS = "privacy_policy_forbids"

    # model execution
    MODEL_TRANSPORT = "model_transport"
    MODEL_RATE_LIMITED = "model_rate_limited"
    MODEL_SCHEMA_INVALID = "model_schema_invalid"
    MODEL_UNSUPPORTED_CONTEXT = "model_unsupported_context"
    MODEL_SEMANTIC_INVALID = "model_semantic_invalid"
    MODEL_BUDGET_EXCEEDED = "model_budget_exceeded"
    NO_MODEL_AVAILABLE = "no_model_available"

    # execution engine
    EXECUTION_CANCELLED = "execution_cancelled"
    EXECUTION_NOT_FOUND = "execution_not_found"
    TRANSFORM_FAILED = "transform_failed"
    TRANSFORM_NOT_FOUND = "transform_not_found"

    # validation / review
    VALIDATION_FAILED = "validation_failed"
    REVIEW_REQUIRED = "review_required"

    # storage
    STORAGE_CONFLICT = "storage_conflict"
    RECORD_NOT_FOUND = "record_not_found"

    CONFIG_INVALID = "config_invalid"
    INTERNAL = "internal"


class LlmbicError(Exception):
    """Base class for every error raised by llmbic."""

    code: ErrorCode = ErrorCode.INTERNAL

    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        self.details: dict[str, Any] = dict(details or {})

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code.value, "message": self.message, "details": self.details}

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"[{self.code.value}] {self.message}"


class SchemaError(LlmbicError):
    code = ErrorCode.SCHEMA_ADAPTER


class RegistryError(LlmbicError):
    code = ErrorCode.MIGRATION_INVALID


class PathError(RegistryError):
    code = ErrorCode.NO_MIGRATION_PATH


class PlanError(LlmbicError):
    code = ErrorCode.PLAN_INVALID


class ContextError(LlmbicError):
    code = ErrorCode.CONTEXT_UNAVAILABLE


class PolicyError(LlmbicError):
    code = ErrorCode.CONTEXT_POLICY_FORBIDS


class ModelError(LlmbicError):
    code = ErrorCode.MODEL_TRANSPORT

    #: Whether another attempt with the same request could plausibly succeed.
    retryable = False


class TransportError(ModelError):
    code = ErrorCode.MODEL_TRANSPORT
    retryable = True


class RateLimitedError(ModelError):
    code = ErrorCode.MODEL_RATE_LIMITED
    retryable = True


class SchemaValidationError(ModelError):
    """Model returned something that does not validate against the target schema."""

    code = ErrorCode.MODEL_SCHEMA_INVALID
    retryable = True


class UnsupportedContextError(ModelError):
    """The adapter cannot accept the context units the policy selected."""

    code = ErrorCode.MODEL_UNSUPPORTED_CONTEXT
    retryable = False


class SemanticValidationError(ModelError):
    """Output was schema-valid but failed a semantic validator."""

    code = ErrorCode.MODEL_SEMANTIC_INVALID
    retryable = True


class BudgetExceededError(LlmbicError):
    code = ErrorCode.MODEL_BUDGET_EXCEEDED


class CancelledError(LlmbicError):
    code = ErrorCode.EXECUTION_CANCELLED


class StorageError(LlmbicError):
    code = ErrorCode.STORAGE_CONFLICT


class ValidationFailed(LlmbicError):
    code = ErrorCode.VALIDATION_FAILED
