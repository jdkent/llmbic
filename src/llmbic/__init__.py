"""llmbic — semantic schema migration for LLM-extracted records.

An Alembic-like migration layer for schema-driven information extraction.  The
defining capability is *selective* migration: a schema change does not cause
every document, field and extraction step to run again.  For each stored value
llmbic decides whether it remains valid, can be transformed deterministically,
can be derived from structured data or stored evidence, or genuinely needs
another look at the source — and it records enough to explain the decision
afterwards.

Quick start::

    from llmbic import Project, from_json_schema

    project = Project("study.db")
    project.register_schema(from_json_schema(doc_v1, name="study", version="1.0"))
    project.ingest(record, schema_ref="study@1.0", record_id="pmid:123")

    plan = project.plan("study@1.1")
    print(plan.render(verbose=True))      # nothing has run yet
    plan, result = project.migrate("study@1.1")

The layers are separable and importable on their own: :mod:`llmbic.registry`,
:mod:`llmbic.planner`, :mod:`llmbic.execution`, :mod:`llmbic.validation`,
:mod:`llmbic.review` and :mod:`llmbic.store`.
"""

from .context import (
    ContextBudget,
    ContextPolicy,
    ContextSource,
    FallbackMode,
    OnMissingContext,
    PrivacyPolicy,
    evidence_window,
    full_document,
    no_context,
    prior_evidence,
    raw_document,
    retrieve,
    sections,
    structured_fields,
    units_of_kind,
)
from .diffing import DeltaKind, FieldDelta, RecordDiff, diff_records
from .errors import ErrorCode, LlmbicError
from .evaluation import (
    EvaluationReport,
    FieldScore,
    GoldCorpus,
    GoldValue,
    RolloutGate,
    evaluate,
    gold_from_artifacts,
)
from .execution import Engine, ExecutionResult, ExecutionStatus, TransformContext, TransformResult
from .functions import (
    POSTPROCESSORS,
    TRANSFORMS,
    VALIDATORS,
    postprocessor,
    prompt_builder,
    transform,
    validator,
)
from .migration import (
    ExecutionPolicy,
    Fidelity,
    Migration,
    MigrationStep,
    Ref,
    StepKind,
    dump_migration,
    load_migration_file,
    migration_from_dict,
    scaffold_migration,
)
from .models import (
    AdapterRegistry,
    DataPolicyAttributes,
    ModelIdentity,
    ModelPolicy,
    ModelRequest,
    ModelResponse,
    Pricing,
    RetryPolicy,
)
from .planner import (
    Currency,
    Disposition,
    ExecutionPlan,
    PlannedStep,
    Planner,
    PlannerOptions,
    RecordPlan,
)
from .project import Project
from .provenance import (
    Actor,
    FieldArtifact,
    FieldProvenance,
    RecordVersion,
    ReviewDecision,
    ReviewEvent,
    ValidationResult,
    entity_key,
)
from .reanchor import AnchorOutcome, reanchor_artifacts, reanchor_evidence
from .recipe import ConfidenceSpec, ExtractionRecipe, Vocabulary
from .records import ExtractedValueCodec, PlainCodec, assemble, decompose
from .registry import PathPreference, Registry, SchemaFamily
from .review import ReviewQueue
from .schema import (
    BaseType,
    ChangeKind,
    Constraints,
    FieldDefinition,
    NormalizedSchema,
    SchemaDiff,
    diff_schemas,
)
from .schema.adapters import from_json_schema, from_linkml, from_pydantic
from .source import (
    DocumentUnit,
    EvidenceReference,
    EvidenceSpan,
    ParsedSource,
    SourceArtifact,
    UnitKind,
    build_parsed_source,
    find_span,
)
from .store import RecordFilter, SqliteStore
from .validation import ValidationContext, validate_artifacts
from .values import AbsenceReason, FieldValue, ValueStatus

__version__ = "0.1.0"

__all__ = [
    "AbsenceReason",
    "Actor",
    "AdapterRegistry",
    "BaseType",
    "ChangeKind",
    "ConfidenceSpec",
    "Constraints",
    "ContextBudget",
    "ContextPolicy",
    "ContextSource",
    "Currency",
    "DataPolicyAttributes",
    "DeltaKind",
    "Disposition",
    "DocumentUnit",
    "Engine",
    "EvaluationReport",
    "ErrorCode",
    "EvaluationReport",
    "FieldScore",
    "GoldCorpus",
    "GoldValue",
    "RolloutGate",
    "AnchorOutcome",
    "EvidenceReference",
    "EvidenceSpan",
    "ExecutionPlan",
    "ExecutionPolicy",
    "ExecutionResult",
    "ExecutionStatus",
    "ExtractedValueCodec",
    "ExtractionRecipe",
    "FallbackMode",
    "Fidelity",
    "FieldArtifact",
    "FieldDefinition",
    "FieldDelta",
    "FieldProvenance",
    "FieldValue",
    "LlmbicError",
    "Migration",
    "MigrationStep",
    "ModelIdentity",
    "ModelPolicy",
    "ModelRequest",
    "ModelResponse",
    "NormalizedSchema",
    "OnMissingContext",
    "POSTPROCESSORS",
    "ParsedSource",
    "PathPreference",
    "PlainCodec",
    "PlannedStep",
    "Planner",
    "PlannerOptions",
    "Pricing",
    "PrivacyPolicy",
    "Project",
    "RecordDiff",
    "RecordFilter",
    "RecordPlan",
    "RecordVersion",
    "Ref",
    "Registry",
    "RetryPolicy",
    "ReviewDecision",
    "ReviewEvent",
    "ReviewQueue",
    "SchemaDiff",
    "SchemaFamily",
    "SourceArtifact",
    "SqliteStore",
    "StepKind",
    "TRANSFORMS",
    "TransformContext",
    "TransformResult",
    "UnitKind",
    "VALIDATORS",
    "ValidationContext",
    "ValidationResult",
    "ValueStatus",
    "Vocabulary",
    "__version__",
    "assemble",
    "build_parsed_source",
    "decompose",
    "diff_records",
    "diff_schemas",
    "dump_migration",
    "entity_key",
    "evaluate",
    "evidence_window",
    "find_span",
    "from_json_schema",
    "from_linkml",
    "from_pydantic",
    "full_document",
    "load_migration_file",
    "migration_from_dict",
    "no_context",
    "postprocessor",
    "prior_evidence",
    "gold_from_artifacts",
    "prompt_builder",
    "raw_document",
    "reanchor_artifacts",
    "reanchor_evidence",
    "retrieve",
    "scaffold_migration",
    "sections",
    "structured_fields",
    "transform",
    "units_of_kind",
    "validate_artifacts",
    "validator",
]
