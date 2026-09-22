"""The execution engine — layer 4.

Runs an approved :class:`~llmbic.planner.plan.ExecutionPlan`: deterministic
functions, model adapters with retries and fallbacks, dependency-complete
caching, durable checkpoints, bounded concurrency and atomic publication.

The invariants that shape the code:

* **Resumability** (FR-EXE-001/005).  Every step's terminal state is written to
  the store as it happens, and a resumed run skips steps that already
  succeeded.  Combined with the cache, restarting a 1,000-record job performs
  no duplicate successful model calls (NFR-PERF-004).
* **Failure isolation** (FR-EXE-002).  Records run independently; one record's
  exception never rolls another back.
* **Atomic publication** (FR-EXE-007/008).  A record is published only when
  every one of its steps reached a non-blocking terminal state and its
  assembled form validates.  Otherwise the prior record version stays current
  and the failed attempt is inspectable as a draft.
* **Idempotence** (principle 7).  Work is addressed by the plan's cache key, so
  re-running the same plan writes the same artifacts and buys nothing twice.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from ..context.policy import OnMissingContext
from ..context.resolver import (
    ContextResolution,
    SelectionInput,
    check_provider,
    resolve_context,
)
from ..errors import (
    BudgetExceededError,
    CancelledError,
    ErrorCode,
    LlmbicError,
    ModelError,
    PlanError,
    PolicyError,
    SemanticValidationError,
    TransportError,
)
from ..functions import POSTPROCESSORS, TRANSFORMS
from ..ids import content_hash, now
from ..migration.spec import ExecutionPolicy, MigrationStep
from ..models.base import (
    AdapterRegistry,
    ModelPolicy,
    ModelRequest,
    ModelResponse,
    RateLimiter,
)
from ..provenance import (
    Actor,
    FieldArtifact,
    FieldProvenance,
    ModelCall,
    Outcome,
    RecordEntity,
    RecordState,
    RecordVersion,
    ValidationResult,
    latest_per_key,
)
from ..planner.dependencies import cache_key
from ..planner.plan import Disposition, ExecutionPlan, PlannedStep, RecordPlan
from ..recipe import ExtractionRecipe
from ..registry import Registry
from ..schema.normalized import NormalizedSchema
from ..source import EvidenceReference, ParsedSource, SourceArtifact
from ..store.base import CacheEntry, RecordIndexEntry, StepState, Store
from ..validation import (
    ValidationContext,
    run_semantic_validators,
    validate_artifacts,
    worst,
)
from ..values import FieldValue, ValueStatus
from .state import (
    EventHook,
    ExecutionResult,
    ExecutionStatus,
    Metrics,
    StepStatus,
    emit,
)

SOFTWARE_VERSION = "llmbic/0.1.0"


@dataclass
class TransformContext:
    """What a deterministic transform is given.

    Deliberately narrow: the values it declared it reads, the entity it is
    running for, and its own parameters.  A transform that wants more must
    declare it, which is what keeps the dependency hashes honest.
    """

    record_id: str
    entity: str
    step: MigrationStep
    reads: dict[str, FieldArtifact]
    params: dict[str, Any]
    source_schema: NormalizedSchema
    target_schema: NormalizedSchema
    parsed: ParsedSource | None = None
    vocabularies: Mapping[str, Any] = field(default_factory=dict)

    def value(self, field_id: str, default: Any = None) -> Any:
        artifact = self.reads.get(field_id)
        if artifact is None or not artifact.value.status.has_value:
            return default
        return artifact.value.value

    def field_value(self, field_id: str) -> FieldValue:
        artifact = self.reads.get(field_id)
        return artifact.value if artifact else FieldValue(ValueStatus.NOT_EXTRACTED)

    def evidence(self, field_id: str) -> tuple[EvidenceReference, ...]:
        artifact = self.reads.get(field_id)
        return artifact.evidence if artifact else ()


@dataclass
class TransformResult:
    """What a transform returns.

    ``evidence`` defaults to the evidence of the single field the step read,
    which is FR-PROV-003 in practice: moving a value must not lose the spans
    that support it.
    """

    values: dict[str, FieldValue]
    evidence: dict[str, tuple[EvidenceReference, ...]] = field(default_factory=dict)
    lineage: tuple[str, ...] = ()
    notes: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def coerce(cls, raw: Any, step: MigrationStep) -> "TransformResult":
        if isinstance(raw, TransformResult):
            return raw
        if isinstance(raw, FieldValue):
            if len(step.writes) != 1:
                raise LlmbicError(
                    f"transform for {step.id!r} returned one value but the step writes "
                    f"{len(step.writes)} fields",
                    code=ErrorCode.TRANSFORM_FAILED,
                )
            return cls({step.writes[0]: raw})
        if isinstance(raw, Mapping):
            values: dict[str, FieldValue] = {}
            for key, val in raw.items():
                values[key] = val if isinstance(val, FieldValue) else FieldValue.present(val)
            return cls(values)
        if len(step.writes) == 1:
            return cls({step.writes[0]: FieldValue.present(raw)})
        raise LlmbicError(
            f"transform for {step.id!r} returned {type(raw).__name__}, which cannot be "
            "mapped onto its declared outputs",
            code=ErrorCode.TRANSFORM_FAILED,
        )


class Engine:
    def __init__(
        self,
        registry: Registry,
        store: Store,
        *,
        adapters: AdapterRegistry | None = None,
        max_workers: int = 4,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        on_event: EventHook | None = None,
        codec: Any | None = None,
        shadow: Mapping[str, str] | None = None,
    ) -> None:
        self.registry = registry
        self.store = store
        #: ``recipe_ref -> alternative recipe_ref`` (FR-LLM-010).  The
        #: alternative runs on the same context, its answer is recorded beside
        #: the committed one, and it is never published.
        self.shadow = dict(shadow or {})
        self.adapters = adapters or AdapterRegistry()
        self.max_workers = max(1, max_workers)
        self._sleep = sleep
        self._clock = clock
        self.on_event = on_event
        self.codec = codec
        self._cancelled: dict[str, threading.Event] = {}
        self._limiters: dict[str, RateLimiter] = {}
        self._budget_lock = threading.Lock()
        self._spent = 0.0
        self._calls = 0
        self._metrics_lock = threading.Lock()

    # ---- public API ------------------------------------------------------
    def run(
        self,
        plan: ExecutionPlan,
        *,
        execution_id: str | None = None,
        resume: bool = False,
        record_ids: Sequence[str] | None = None,
    ) -> ExecutionResult:
        if plan.registry_hash and plan.registry_hash != self.registry.registry_hash():
            raise PlanError(
                "the registry has changed since this plan was produced; re-plan before "
                "running",
                code=ErrorCode.PLAN_SIGNATURE_MISMATCH,
                details={
                    "plan_registry_hash": plan.registry_hash,
                    "current_registry_hash": self.registry.registry_hash(),
                },
            )

        execution_id = execution_id or f"exec-{content_hash([plan.plan_id, now()])[7:19]}"
        cancel = self._cancelled.setdefault(execution_id, threading.Event())

        existing = self.store.get_execution(execution_id)
        if existing is None:
            self.store.put_plan(plan.plan_id, plan.to_canonical())
            self.store.create_execution(
                execution_id,
                {
                    "execution_id": execution_id,
                    "plan_id": plan.plan_id,
                    "plan_signature": plan.signature(),
                    "target_schema": plan.target_schema,
                    "state": ExecutionStatus.RUNNING.value,
                    "policy": plan.policy.to_canonical(),
                    "software_version": SOFTWARE_VERSION,
                },
            )
        elif not resume:
            raise PlanError(
                f"execution {execution_id!r} already exists; pass resume=True to continue it",
                code=ErrorCode.STORAGE_CONFLICT,
            )
        else:
            if existing.get("plan_signature") not in (None, plan.signature()):
                raise PlanError(
                    "cannot resume: this execution was started from a different plan",
                    code=ErrorCode.PLAN_SIGNATURE_MISMATCH,
                )
            self.store.update_execution(execution_id, state=ExecutionStatus.RUNNING.value)

        with self._budget_lock:
            self._spent = float(
                (self.store.get_execution(execution_id) or {}).get("spent_usd", 0.0)
            )
            self._calls = int((self.store.get_execution(execution_id) or {}).get("model_calls", 0))

        metrics = Metrics()
        result = ExecutionResult(
            execution_id=execution_id, plan_id=plan.plan_id, status=ExecutionStatus.RUNNING,
            metrics=metrics,
        )
        emit(self.on_event, "execution.started", execution_id=execution_id, plan_id=plan.plan_id)

        selected = [
            rp
            for rp in plan.records
            if record_ids is None or rp.record_id in set(record_ids)
        ]
        started = self._clock()

        target = self.registry.schema(plan.target_schema)
        futures: list[Future] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for rp in selected:
                futures.append(
                    pool.submit(
                        self._run_record_guarded,
                        rp,
                        plan,
                        target,
                        execution_id,
                        cancel,
                        result,
                    )
                )
            for fut in futures:
                fut.result()

        metrics.elapsed_s = self._clock() - started
        result.finished_at = now()
        result.status = self._final_status(result, cancel)
        self.store.update_execution(
            execution_id,
            state=result.status.value,
            metrics=metrics.to_canonical(),
            records=result.records,
            spent_usd=self._spent,
            model_calls=self._calls,
            finished_at=result.finished_at,
        )
        emit(
            self.on_event,
            "execution.finished",
            execution_id=execution_id,
            status=result.status.value,
            **metrics.to_canonical(),
        )
        return result

    def resume(self, execution_id: str) -> ExecutionResult:
        """Continue an interrupted execution from its checkpoints."""

        record = self.store.get_execution(execution_id)
        if record is None:
            raise PlanError(
                f"unknown execution {execution_id!r}", code=ErrorCode.EXECUTION_NOT_FOUND
            )
        plan_payload = self.store.get_plan(record["plan_id"])
        if plan_payload is None:
            raise PlanError(
                f"execution {execution_id!r} refers to plan {record['plan_id']!r}, which is "
                "not stored",
                code=ErrorCode.PLAN_INVALID,
            )
        plan = ExecutionPlan.from_canonical(plan_payload)
        self._cancelled.pop(execution_id, None)
        return self.run(plan, execution_id=execution_id, resume=True)

    def cancel(self, execution_id: str) -> None:
        self._cancelled.setdefault(execution_id, threading.Event()).set()
        emit(self.on_event, "execution.cancelled", execution_id=execution_id)

    # ---- per-record ------------------------------------------------------
    def _run_record_guarded(
        self,
        rp: RecordPlan,
        plan: ExecutionPlan,
        target: NormalizedSchema,
        execution_id: str,
        cancel: threading.Event,
        result: ExecutionResult,
    ) -> None:
        try:
            self._run_record(rp, plan, target, execution_id, cancel, result)
        except CancelledError:
            self._set_record(result, rp.record_id, "cancelled")
        except Exception as exc:  # FR-EXE-002: isolate the failure
            self._set_record(result, rp.record_id, "failed")
            with self._metrics_lock:
                result.errors.append(
                    {
                        "record_id": rp.record_id,
                        "code": exc.code.value if isinstance(exc, LlmbicError) else "internal",
                        "message": str(exc),
                    }
                )
            emit(
                self.on_event,
                "record.failed",
                execution_id=execution_id,
                record_id=rp.record_id,
                error=str(exc),
            )

    def _run_record(
        self,
        rp: RecordPlan,
        plan: ExecutionPlan,
        target: NormalizedSchema,
        execution_id: str,
        cancel: threading.Event,
        result: ExecutionResult,
    ) -> None:
        if rp.blocked:
            self._checkpoint(
                execution_id,
                f"{rp.record_id}|__record__",
                rp.record_id,
                StepStatus.BLOCKED,
                detail={"reason": rp.blocked_reason, "code": rp.blocked_code},
            )
            self._set_record(result, rp.record_id, "blocked")
            self._bump(result, steps_blocked=1)
            return

        artifacts = {
            (a.field_id, a.entity): a
            for a in latest_per_key(self.store.get_artifacts(rp.record_id))
        }
        entry = self._index_entry(rp, target)
        source = (
            self.store.get_source(entry.source_id, entry.source_version)
            if entry.source_id and entry.source_version
            else None
        )
        parsed = (
            self.store.get_parsed(entry.source_id, entry.source_version, entry.parse_version)
            if entry.source_id and entry.source_version
            else None
        )

        produced: dict[tuple[str, str], FieldArtifact] = {}
        statuses: dict[str, StepStatus] = {}
        review_keys: list[str] = []

        for step in _ordered(rp.steps):
            if cancel.is_set():
                raise CancelledError(
                    f"execution {execution_id} cancelled", code=ErrorCode.EXECUTION_CANCELLED
                )

            prior = self.store.get_step_state(execution_id, step.key)
            if prior is not None and StepStatus(prior.state) is StepStatus.SUCCEEDED:
                statuses[step.key] = StepStatus.SUCCEEDED
                self._bump(result, steps_skipped=1, steps_total=1)
                for fid in step.writes:
                    art = artifacts.get((fid, step.entity))
                    if prior.artifact_id:
                        stored = self.store.get_artifact(prior.artifact_id)
                        if stored is not None:
                            art = stored
                    if art is not None:
                        produced[(fid, step.entity)] = art
                continue

            status = self._run_step(
                step,
                rp,
                plan,
                target,
                execution_id,
                artifacts | produced,
                produced,
                source,
                parsed,
                result,
            )
            statuses[step.key] = status
            if status is StepStatus.REVIEW_NEEDED:
                review_keys.append(step.key)

        self._finish_record(
            rp, plan, target, execution_id, artifacts, produced, statuses, review_keys, result
        )

    def _finish_record(
        self,
        rp: RecordPlan,
        plan: ExecutionPlan,
        target: NormalizedSchema,
        execution_id: str,
        artifacts: dict[tuple[str, str], FieldArtifact],
        produced: dict[tuple[str, str], FieldArtifact],
        statuses: Mapping[str, StepStatus],
        review_keys: Sequence[str],
        result: ExecutionResult,
    ) -> None:
        merged = {**artifacts, **produced}
        entities = self._entities_for(rp.record_id, merged)
        chosen = [a for (fid, _), a in sorted(merged.items()) if target.get(fid) is not None]

        held = [k for k, s in statuses.items() if s.blocks_publication]
        structural = validate_artifacts(chosen, target)
        failures = [r for r in structural if r.outcome is Outcome.FAIL]
        if failures:
            self._bump(result, validation_failures=len(failures))

        prior_version = self.store.current_version(rp.record_id)
        version = RecordVersion(
            record_id=rp.record_id,
            schema_ref=target.ref,
            artifact_ids=tuple(sorted(a.artifact_id for a in chosen)),
            entities=tuple(entities),
            state=RecordState.DRAFT,
            parent_version_id=prior_version.version_id if prior_version else None,
            execution_id=execution_id,
            source_ref=rp.source_ref,
            current_field_ids=tuple(
                sorted({a.field_id for a in chosen if a.value.status.has_value})
            ),
            notes={
                "validation": [r.to_canonical() for r in structural],
                "held_steps": list(held),
            },
        )

        if held or failures:
            # FR-EXE-008: inspectable, but not current.
            state = (
                RecordState.REVIEW_NEEDED
                if review_keys and not failures
                else RecordState.FAILED
            )
            self.store.put_record_version(
                RecordVersion(**{**_version_kwargs(version), "state": state})
            )
            self._set_record(
                result, rp.record_id, "review" if state is RecordState.REVIEW_NEEDED else "held"
            )
            self._bump(result, records_held=1)
            emit(
                self.on_event,
                "record.held",
                execution_id=execution_id,
                record_id=rp.record_id,
                reason="validation" if failures else "review",
                held_steps=len(held),
            )
            return

        self.store.publish(version)
        self._set_record(result, rp.record_id, "published")
        self._bump(result, records_published=1)
        emit(
            self.on_event,
            "record.published",
            execution_id=execution_id,
            record_id=rp.record_id,
            version_id=version.version_id,
            schema=target.ref,
        )

    # ---- per-step --------------------------------------------------------
    def _run_step(
        self,
        step: PlannedStep,
        rp: RecordPlan,
        plan: ExecutionPlan,
        target: NormalizedSchema,
        execution_id: str,
        available: Mapping[tuple[str, str], FieldArtifact],
        produced: dict[tuple[str, str], FieldArtifact],
        source: SourceArtifact | None,
        parsed: ParsedSource | None,
        result: ExecutionResult,
    ) -> StepStatus:
        self._bump(result, steps_total=1)
        emit(
            self.on_event,
            "step.started",
            execution_id=execution_id,
            record_id=step.record_id,
            step=step.key,
            disposition=step.disposition.value,
        )

        if step.disposition is Disposition.REUSE:
            for fid in step.writes:
                art = available.get((fid, step.entity))
                if art is not None:
                    produced[(fid, step.entity)] = art
            self._checkpoint(
                execution_id, step.key, step.record_id, StepStatus.SUCCEEDED,
                detail={"disposition": "reuse", "reason": step.reason},
            )
            self._bump(result, steps_reused=1, steps_succeeded=1)
            return StepStatus.SUCCEEDED

        if step.disposition is Disposition.BLOCKED:
            self._checkpoint(
                execution_id, step.key, step.record_id, StepStatus.BLOCKED,
                detail={"reason": step.reason, "code": step.blocked_code},
            )
            self._bump(result, steps_blocked=1)
            return StepStatus.BLOCKED

        if step.disposition is Disposition.REVIEW:
            self._open_review(step, rp, execution_id, available, result)
            self._checkpoint(
                execution_id, step.key, step.record_id, StepStatus.REVIEW_NEEDED,
                detail={"reason": step.reason},
            )
            self._bump(result, steps_review=1)
            return StepStatus.REVIEW_NEEDED

        if step.disposition is Disposition.VALIDATE:
            self._checkpoint(
                execution_id, step.key, step.record_id, StepStatus.SUCCEEDED,
                detail={"reason": step.reason},
            )
            self._bump(result, steps_succeeded=1)
            return StepStatus.SUCCEEDED

        migration = self.registry.migration(step.migration_id)
        mstep = migration.step(step.step_id)
        # The artifact belongs to the schema version the *step* targets, not to
        # the destination of the whole path — so replaying a pending
        # intermediate step produces the identical artifact rather than a
        # near-duplicate stamped with a later version.
        step_schema = (
            self.registry.schema(migration.to_schema)
            if self.registry.has_schema(migration.to_schema)
            else target
        )

        try:
            if step.kind.is_deterministic:
                artifacts = self._run_deterministic(
                    step, mstep, migration, step_schema, execution_id, available, parsed
                )
            else:
                artifacts = self._run_semantic(
                    step, mstep, migration, step_schema, execution_id, available,
                    source, parsed, plan.policy, result,
                )
        except BudgetExceededError as exc:
            self._checkpoint(
                execution_id, step.key, step.record_id, StepStatus.BLOCKED,
                detail={"reason": exc.message, "code": exc.code.value},
            )
            self._bump(result, steps_blocked=1)
            return StepStatus.BLOCKED
        except PolicyError as exc:
            self._checkpoint(
                execution_id, step.key, step.record_id, StepStatus.BLOCKED,
                detail={"reason": exc.message, "code": exc.code.value},
            )
            self._bump(result, steps_blocked=1)
            return StepStatus.BLOCKED
        except _ReviewNeeded as exc:
            self._open_review(step, rp, execution_id, available, result, reason=exc.reason)
            self._checkpoint(
                execution_id, step.key, step.record_id, StepStatus.REVIEW_NEEDED,
                detail={"reason": exc.reason},
            )
            self._bump(result, steps_review=1)
            return StepStatus.REVIEW_NEEDED
        except LlmbicError as exc:
            self._checkpoint(
                execution_id, step.key, step.record_id, StepStatus.FAILED,
                detail=exc.to_dict(),
            )
            self._bump(result, steps_failed=1)
            with self._metrics_lock:
                result.errors.append(
                    {"record_id": step.record_id, "step": step.key, **exc.to_dict()}
                )
            emit(
                self.on_event,
                "step.failed",
                execution_id=execution_id,
                record_id=step.record_id,
                step=step.key,
                code=exc.code.value,
            )
            return StepStatus.FAILED

        self.store.put_artifacts(artifacts)
        for art in artifacts:
            produced[(art.field_id, art.entity)] = art

        # A value a validator or a transform marked REVIEW_REQUIRED is stored
        # (so it is inspectable) but never published: the record is held and a
        # curator gets a queue row carrying both values.
        needs_review = [a for a in artifacts if a.value.status is ValueStatus.REVIEW_REQUIRED]
        if needs_review:
            reason = "; ".join(
                f"{a.field_id}: {a.value.reason or 'flagged for review'}" for a in needs_review
            )
            self._open_review(
                step, rp, execution_id, available, result, reason=reason, proposed=artifacts
            )
            self._checkpoint(
                execution_id,
                step.key,
                step.record_id,
                StepStatus.REVIEW_NEEDED,
                artifact_id=artifacts[0].artifact_id,
                detail={"reason": reason},
            )
            self._bump(result, steps_review=1)
            emit(
                self.on_event,
                "step.review_needed",
                execution_id=execution_id,
                record_id=step.record_id,
                step=step.key,
                fields=[a.field_id for a in needs_review],
            )
            return StepStatus.REVIEW_NEEDED

        self._checkpoint(
            execution_id,
            step.key,
            step.record_id,
            StepStatus.SUCCEEDED,
            artifact_id=artifacts[0].artifact_id if artifacts else None,
            detail={"writes": [a.field_id for a in artifacts]},
        )
        self._bump(result, steps_succeeded=1)
        emit(
            self.on_event,
            "step.succeeded",
            execution_id=execution_id,
            record_id=step.record_id,
            step=step.key,
            writes=[a.field_id for a in artifacts],
        )
        return StepStatus.SUCCEEDED

    # ---- deterministic ---------------------------------------------------
    def _run_deterministic(
        self,
        step: PlannedStep,
        mstep: MigrationStep,
        migration: Any,
        step_schema: NormalizedSchema,
        execution_id: str,
        available: Mapping[tuple[str, str], FieldArtifact],
        parsed: ParsedSource | None,
    ) -> list[FieldArtifact]:
        reads = _reads_for(available, mstep.reads_fields, step.entity)
        source_schema = self.registry.schema(migration.from_schema)
        ctx = TransformContext(
            record_id=step.record_id,
            entity=step.entity,
            step=mstep,
            reads=reads,
            params=dict(mstep.params),
            source_schema=source_schema,
            target_schema=step_schema,
            parsed=parsed,
            vocabularies={v.ref: v for v in self.registry.vocabularies()},
        )
        fn = TRANSFORMS.get(mstep.transform or "")
        try:
            raw = fn(ctx)
        except LlmbicError:
            raise
        except Exception as exc:
            raise LlmbicError(
                f"transform {mstep.transform!r} raised {type(exc).__name__}: {exc}",
                code=ErrorCode.TRANSFORM_FAILED,
                details={"step": step.key},
            ) from exc

        outcome = TransformResult.coerce(raw, mstep)
        inherited = _single_evidence(reads)

        # A field the step both reads and writes is an in-place edit.  Its
        # prior value is a genuine input, but putting it in the content address
        # would make a replay of the step produce a near-duplicate artifact —
        # so it is recorded beside the address instead of inside it.
        in_place = {k: v for k, v in reads.items() if k in set(mstep.writes)}
        default_lineage = tuple(
            f"{k}:{v.value_hash()}" for k, v in sorted(reads.items()) if k not in in_place
        )
        notes = dict(outcome.notes)
        if in_place:
            notes["in_place_inputs"] = {
                k: v.value_hash() for k, v in sorted(in_place.items())
            }

        artifacts: list[FieldArtifact] = []
        for fid in mstep.writes:
            if fid not in outcome.values:
                continue
            fdef = step_schema.field(fid)
            evidence = outcome.evidence.get(fid, inherited)
            provenance = FieldProvenance(
                schema_version=step_schema.ref,
                recipe_ref=fdef.recipe_ref,
                migration_id=migration.id,
                step_id=mstep.id,
                execution_id=execution_id,
                input_hashes=dict(step.dependencies),
                lineage=outcome.lineage or default_lineage,
                actor=Actor.MIGRATION,
                actor_id=mstep.transform,
                software_version=SOFTWARE_VERSION,
                notes=notes,
            )
            artifacts.append(
                FieldArtifact(
                    record_id=step.record_id,
                    field_id=fid,
                    entity=step.entity,
                    value=outcome.values[fid],
                    evidence=evidence,
                    provenance=provenance,
                )
            )
        return artifacts

    # ---- semantic --------------------------------------------------------
    def _run_semantic(
        self,
        step: PlannedStep,
        mstep: MigrationStep,
        migration: Any,
        step_schema: NormalizedSchema,
        execution_id: str,
        available: Mapping[tuple[str, str], FieldArtifact],
        source: SourceArtifact | None,
        parsed: ParsedSource | None,
        policy: ExecutionPolicy,
        result: ExecutionResult,
    ) -> list[FieldArtifact]:
        recipe = self.registry.recipe(mstep.recipe or "")
        ctx_policy = mstep.context or recipe.context_policy
        on_missing = mstep.on_missing_context or ctx_policy.on_missing_context

        reads = _reads_for(available, mstep.reads_fields, step.entity)
        evidence_in = _evidence_for(available, mstep.reads_evidence, step.entity)
        resolution = resolve_context(
            ctx_policy,
            SelectionInput(
                record_id=step.record_id,
                parsed=parsed,
                source=source,
                prior_evidence=evidence_in,
                prior_fields=reads,
                field_id=mstep.writes[0] if mstep.writes else None,
                entity=step.entity,
            ),
            allow_model_selectors=policy.allow_model_selectors,
        )

        if not resolution.satisfied:
            if on_missing is OnMissingContext.REVIEW:
                raise _ReviewNeeded(resolution.blocked_reason or "required context unavailable")
            raise LlmbicError(
                resolution.blocked_reason or "required context unavailable",
                code=ErrorCode(resolution.blocked_code or ErrorCode.CONTEXT_UNAVAILABLE.value),
                details={"attempted": list(resolution.attempted)},
            )

        if resolution.includes_full_document and not policy.allow_full_document:
            raise PolicyError(
                "this step would transmit the full document, which the execution policy "
                "does not permit",
                code=ErrorCode.CONTEXT_POLICY_FORBIDS,
            )

        key = cache_key(
            step=mstep,
            record_id=step.record_id,
            entity=step.entity,
            field_ids=mstep.writes,
            dependencies=step.dependencies,
            context_hash=resolution.context_hash(),
            schema_ref=step_schema.ref,
        )

        selection = SelectionInput(
            record_id=step.record_id,
            parsed=parsed,
            source=source,
            prior_evidence=evidence_in,
            prior_fields=reads,
            field_id=mstep.writes[0] if mstep.writes else None,
            entity=step.entity,
        )
        output, call, resolution, from_cache = self._ask(
            recipe, mstep, step, execution_id, reads, resolution, selection,
            ctx_policy, step_schema, policy, result, key,
        )

        artifacts = self._artifacts_from_output(
            step, mstep, migration, step_schema, execution_id, recipe, output, resolution,
            reads, call, parsed, available, cached=from_cache,
        )

        if recipe.ref in self.shadow:
            self._run_shadow(
                recipe, mstep, step, execution_id, reads, resolution, step_schema,
                output, policy, result,
            )
        return artifacts

    def _run_shadow(
        self,
        recipe: ExtractionRecipe,
        mstep: MigrationStep,
        step: PlannedStep,
        execution_id: str,
        reads: Mapping[str, FieldArtifact],
        resolution: ContextResolution,
        step_schema: NormalizedSchema,
        committed: Any,
        policy: ExecutionPolicy,
        result: ExecutionResult,
    ) -> None:
        """Run an alternative recipe on the same context, and record the gap.

        Nothing the shadow says is committed.  It exists so a maintainer can
        see, on real records and before a rollout, how often the candidate and
        the incumbent disagree — which is the number §15.5 wants and the one a
        dry run cannot produce.
        """

        alternative = self.registry.maybe_recipe(self.shadow[recipe.ref])
        if alternative is None:
            return
        try:
            request = self._build_request(
                alternative, mstep, step, reads, resolution, step_schema
            )
            response, call = self._call_model(
                alternative, request, step, execution_id, policy, result, resolution
            )
        except LlmbicError as exc:
            self.store.put_attempt(
                execution_id,
                step.key,
                {
                    "step_key": step.key,
                    "record_id": step.record_id,
                    "outcome": "shadow_error",
                    "shadow_recipe": alternative.ref,
                    "error": exc.to_dict(),
                },
            )
            return

        self.store.put_attempt(
            execution_id,
            step.key,
            {
                "step_key": step.key,
                "record_id": step.record_id,
                "outcome": "shadow",
                "shadow_recipe": alternative.ref,
                "committed_hash": content_hash(committed),
                "shadow_hash": content_hash(response.output),
                "agrees": content_hash(committed) == content_hash(response.output),
                "shadow_output": response.output,
                "model_call": call.to_canonical(),
            },
        )
        emit(
            self.on_event,
            "step.shadowed",
            execution_id=execution_id,
            record_id=step.record_id,
            step=step.key,
            shadow_recipe=alternative.ref,
            agrees=content_hash(committed) == content_hash(response.output),
        )

    def _ask(
        self,
        recipe: ExtractionRecipe,
        mstep: MigrationStep,
        step: PlannedStep,
        execution_id: str,
        reads: Mapping[str, FieldArtifact],
        resolution: ContextResolution,
        selection: SelectionInput,
        ctx_policy: Any,
        step_schema: NormalizedSchema,
        policy: ExecutionPolicy,
        result: ExecutionResult,
        key: str,
    ) -> tuple[Any, ModelCall | None, ContextResolution, bool]:
        """Ask the model, and let it ask for more context if it needs to.

        This method owns the cache for semantic steps, read and write, because
        an escalation makes *each level* a separate question: different context
        means a different cache key.  Caching every level is what keeps a
        resumed run from paying again for the insufficient first ask on its way
        back to the answer.

        The walk is bounded three ways: by the recipe's ``max_escalations``, by
        the end of the chain the migration declared, and by the execution
        policy, which must still permit the full document if the chain ends
        there.  When the walk runs out, the record goes to review carrying the
        list of what was tried — it never receives an invented value.
        """

        escalation = recipe.escalation
        tried: list[str] = list(resolution.used_sources)
        level = 0
        call: ModelCall | None = None

        while True:
            from_cache = False
            cached = self.store.cache_get(key)
            if cached is not None:
                self._bump(result, cache_hits=1)
                output = cached.payload["output"]
                if cached.payload.get("model_call"):
                    call = ModelCall.from_canonical(cached.payload["model_call"])
                from_cache = True
            else:
                request = self._build_request(
                    recipe, mstep, step, reads, resolution, step_schema
                )
                response, call = self._call_model(
                    recipe, request, step, execution_id, policy, result, resolution
                )
                output = response.output
                if recipe.postprocessor:
                    output = POSTPROCESSORS.get(recipe.postprocessor)(output, step, reads)
                if self.store.cache_put(
                    CacheEntry(
                        cache_key=key,
                        payload={
                            "output": output,
                            "model_call": call.to_canonical() if call else None,
                        },
                        created_at=now(),
                        execution_id=execution_id,
                        usage=response.usage.to_canonical(),
                    )
                ):
                    self._bump(result, cache_writes=1)

            if not escalation.wants_more(output):
                return output, call, resolution, from_cache

            note = escalation.reason_from(output)
            next_at = (resolution.used_index or 0) + 1
            exhausted = (
                level >= escalation.max_escalations
                or next_at >= len(ctx_policy.effective_sequence)
            )
            self.store.put_attempt(
                execution_id,
                step.key,
                {
                    "step_key": step.key,
                    "record_id": step.record_id,
                    "outcome": "context_requested",
                    "level": level,
                    "tried": list(tried),
                    "note": note,
                    "from_cache": from_cache,
                    "exhausted": exhausted,
                },
            )
            emit(
                self.on_event,
                "step.context_requested",
                execution_id=execution_id,
                record_id=step.record_id,
                step=step.key,
                level=level,
                tried=list(tried),
                exhausted=exhausted,
            )
            if exhausted:
                raise _ReviewNeeded(
                    f"{recipe.ref} asked for more context than its policy provides; "
                    f"tried {', '.join(tried) or 'nothing'}"
                    + (f" ({note})" if note else "")
                )

            wider = resolve_context(
                ctx_policy,
                selection,
                allow_model_selectors=policy.allow_model_selectors,
                start_at=next_at,
            )
            if not wider.satisfied or not wider.units:
                raise _ReviewNeeded(
                    f"{recipe.ref} asked for more context and the chain has none left; "
                    f"tried {', '.join(tried) or 'nothing'}"
                )
            if wider.includes_full_document and not policy.allow_full_document:
                raise PolicyError(
                    "the step asked for more context and the next source is the full "
                    "document, which the execution policy does not permit",
                    code=ErrorCode.CONTEXT_POLICY_FORBIDS,
                )

            resolution = wider
            tried.extend(wider.used_sources)
            level += 1
            key = cache_key(
                step=mstep,
                record_id=step.record_id,
                entity=step.entity,
                field_ids=mstep.writes,
                dependencies=step.dependencies,
                context_hash=resolution.context_hash(),
                schema_ref=step_schema.ref,
            )

    def _build_request(
        self,
        recipe: ExtractionRecipe,
        mstep: MigrationStep,
        step: PlannedStep,
        reads: Mapping[str, FieldArtifact],
        resolution: ContextResolution,
        step_schema: NormalizedSchema,
    ) -> ModelRequest:
        variables: dict[str, Any] = {
            "record_id": step.record_id,
            "entity": step.entity,
            "context": resolution.render(),
        }
        for fid, artifact in reads.items():
            variables[_var(fid)] = artifact.value.value
            variables[_var(fid) + "_status"] = artifact.value.status.value
        for fid in mstep.writes:
            fdef = step_schema.get(fid)
            if fdef is not None:
                variables.setdefault("field_description", fdef.description)
                variables.setdefault("field_path", fdef.path)
                if fdef.constraints.enum:
                    variables.setdefault("permitted_values", list(fdef.constraints.enum))
        variables.update(mstep.params.get("prompt_variables") or {})

        return ModelRequest(
            prompt=recipe.render_prompt(variables),
            output_schema=recipe.output_schema,
            parameters=dict(recipe.model_policy.parameters) if recipe.model_policy else {},
            context_units=resolution.units,
            system=recipe.system,
        )

    def _call_model(
        self,
        recipe: ExtractionRecipe,
        request: ModelRequest,
        step: PlannedStep,
        execution_id: str,
        policy: ExecutionPolicy,
        result: ExecutionResult,
        resolution: ContextResolution,
    ) -> tuple[ModelResponse, ModelCall]:
        mp = recipe.model_policy
        if mp is None:
            raise LlmbicError(
                f"recipe {recipe.ref} has no model policy",
                code=ErrorCode.CONFIG_INVALID,
            )
        self._reserve_budget(mp, policy, step)

        last: Exception | None = None
        attempt_no = 0
        for adapter_name in mp.candidates():
            adapter = self.adapters.get(adapter_name)
            if adapter is None:
                last = LlmbicError(
                    f"model adapter {adapter_name!r} is not registered",
                    code=ErrorCode.NO_MODEL_AVAILABLE,
                )
                continue

            ctx_policy = recipe.context_policy
            check_provider(ctx_policy, adapter.identity.provider, adapter.data_policy.as_dict())

            limiter = self._limiters.setdefault(adapter_name, RateLimiter(mp.rate_limit_rps))
            for local_attempt in range(1, mp.retry.max_attempts + 1):
                attempt_no += 1
                limiter.acquire(sleep=self._sleep)
                started = self._clock()
                try:
                    response = adapter.generate(request)
                except ModelError as exc:
                    last = exc
                    self._record_attempt(
                        execution_id, step, attempt_no, adapter_name, error=exc,
                        elapsed=self._clock() - started,
                    )
                    self._bump(result, retries=1)
                    retryable = exc.retryable and (
                        exc.code is not ErrorCode.MODEL_SCHEMA_INVALID
                        or mp.retry.retry_schema_failures
                    )
                    if not retryable or local_attempt == mp.retry.max_attempts:
                        break
                    self._sleep(mp.retry.backoff_for(local_attempt))
                    continue
                except Exception as exc:  # unexpected adapter bug
                    last = TransportError(f"adapter {adapter_name!r} raised {exc!r}")
                    self._record_attempt(
                        execution_id, step, attempt_no, adapter_name, error=last,
                        elapsed=self._clock() - started,
                    )
                    break

                elapsed_ms = (self._clock() - started) * 1000.0
                # The policy's price list is authoritative when it has one: an
                # adapter reports tokens, an operator decides what they cost.
                priced = mp.pricing.cost(
                    response.usage.input_tokens, response.usage.output_tokens
                )
                usage = (
                    response.usage
                    if not priced
                    else dataclasses.replace(response.usage, cost_usd=priced)
                )
                response = dataclasses.replace(response, usage=usage)
                call = ModelCall(
                    provider=response.identity.provider,
                    model=response.identity.model,
                    parameters=dict(request.parameters),
                    prompt_hash=recipe.prompt_hash,
                    model_fingerprint=response.identity.fingerprint,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    latency_ms=response.latency_ms or elapsed_ms,
                    cost_usd=response.usage.cost_usd,
                    provider_request_id=response.provider_request_id,
                    attempts=attempt_no,
                )
                self._settle_budget(response.usage.cost_usd)
                self._record_attempt(
                    execution_id, step, attempt_no, adapter_name, call=call,
                    elapsed=elapsed_ms / 1000.0,
                )
                self._bump(result, model_calls=1)
                with self._metrics_lock:
                    result.metrics.usage = result.metrics.usage + response.usage
                if mp.pin_fingerprint and response.identity.fingerprint != mp.pin_fingerprint:
                    raise _ReviewNeeded(
                        f"{response.identity.ref} answered with fingerprint "
                        f"{response.identity.fingerprint!r}, but the recipe pins "
                        f"{mp.pin_fingerprint!r}"
                    )
                return response, call

        if isinstance(last, LlmbicError):
            raise last
        raise LlmbicError(
            f"every model candidate failed for {step.key}: {last!r}",
            code=ErrorCode.NO_MODEL_AVAILABLE,
        )

    def _artifacts_from_output(
        self,
        step: PlannedStep,
        mstep: MigrationStep,
        migration: Any,
        step_schema: NormalizedSchema,
        execution_id: str,
        recipe: ExtractionRecipe,
        output: Any,
        resolution: ContextResolution,
        reads: Mapping[str, FieldArtifact],
        call: ModelCall | None,
        parsed: ParsedSource | None,
        available: Mapping[tuple[str, str], FieldArtifact],
        *,
        cached: bool,
    ) -> list[FieldArtifact]:
        values, evidence = _decode_output(output, mstep, resolution, parsed)

        artifacts: list[FieldArtifact] = []
        for fid in mstep.writes:
            fdef = step_schema.field(fid)
            value = values.get(fid, FieldValue(ValueStatus.NOT_EXTRACTED))
            refs = evidence.get(fid, ())

            vctx = ValidationContext(
                record_id=step.record_id,
                field_id=fid,
                entity=step.entity,
                value=value,
                field_def=fdef,
                evidence=refs,
                context_units=resolution.units,
                parsed=parsed,
                recipe=recipe,
                artifacts=available,
                schema=step_schema,
            )
            checks: list[ValidationResult] = list(
                run_semantic_validators(tuple(recipe.validators) + tuple(mstep.validators), vctx)
            )
            from ..validation import validate_field_value

            checks.extend(validate_field_value(value, fdef))
            verdict = worst(checks)

            if verdict is Outcome.FAIL:
                raise SemanticValidationError(
                    "; ".join(c.reason for c in checks if c.outcome is Outcome.FAIL),
                    details={"field_id": fid, "validations": [c.to_canonical() for c in checks]},
                )
            if verdict is Outcome.REVIEW:
                value = value.with_status(
                    ValueStatus.REVIEW_REQUIRED
                    if value.status.has_value
                    else value.status,
                    reason="; ".join(c.reason for c in checks if c.outcome is Outcome.REVIEW),
                )

            artifacts.append(
                FieldArtifact(
                    record_id=step.record_id,
                    field_id=fid,
                    entity=step.entity,
                    value=value,
                    evidence=refs,
                    provenance=FieldProvenance(
                        schema_version=step_schema.ref,
                        recipe_ref=recipe.ref,
                        source_ref=resolution.units[0].metadata.get("source_ref")
                        if resolution.units
                        else None,
                        parse_version=parsed.parse_version if parsed else None,
                        migration_id=migration.id,
                        step_id=mstep.id,
                        execution_id=execution_id,
                        context_selector_ref=",".join(resolution.used_sources),
                        context_units=resolution.unit_ids,
                        context_hash=resolution.context_hash(),
                        model_call=call,
                        prompt_hash=recipe.prompt_hash,
                        input_hashes=dict(step.dependencies),
                        lineage=tuple(f"{k}:{v.value_hash()}" for k, v in sorted(reads.items())),
                        validations=tuple(checks),
                        actor=Actor.MODEL,
                        actor_id=call.model if call else recipe.ref,
                        software_version=SOFTWARE_VERSION,
                        notes={"from_cache": True} if cached else {},
                    ),
                    derived_from="cache" if cached else None,
                )
            )
        return artifacts

    # ---- budget, checkpoints, bookkeeping --------------------------------
    def _reserve_budget(
        self, mp: ModelPolicy, policy: ExecutionPolicy, step: PlannedStep
    ) -> None:
        """FR-EXE-010: stop scheduling new billable calls once the cap is hit."""

        with self._budget_lock:
            limits = [x for x in (policy.budget_usd, mp.budget_usd) if x is not None]
            if limits and self._spent >= min(limits):
                raise BudgetExceededError(
                    f"budget of ${min(limits):.4f} is spent (${self._spent:.4f} used); "
                    "no further billable calls will be scheduled",
                    code=ErrorCode.MODEL_BUDGET_EXCEEDED,
                )
            if policy.max_model_calls is not None and self._calls >= policy.max_model_calls:
                raise BudgetExceededError(
                    f"model-call cap of {policy.max_model_calls} reached",
                    code=ErrorCode.MODEL_BUDGET_EXCEEDED,
                )
            self._calls += 1

    def _settle_budget(self, cost: float) -> None:
        with self._budget_lock:
            self._spent += cost

    def _record_attempt(
        self,
        execution_id: str,
        step: PlannedStep,
        attempt_no: int,
        adapter_name: str,
        *,
        call: ModelCall | None = None,
        error: Exception | None = None,
        elapsed: float = 0.0,
    ) -> None:
        """FR-LLM-005: every attempt is recorded under one logical identity."""

        payload: dict[str, Any] = {
            "step_key": step.key,
            "record_id": step.record_id,
            "attempt": attempt_no,
            "adapter": adapter_name,
            "elapsed_s": round(elapsed, 4),
            "cache_key": step.cache_key,
        }
        if call is not None:
            payload["model_call"] = call.to_canonical()
            payload["outcome"] = "ok"
        if error is not None:
            payload["outcome"] = "error"
            payload["error"] = (
                error.to_dict() if isinstance(error, LlmbicError) else {"message": str(error)}
            )
        self.store.put_attempt(execution_id, step.key, payload)

    def _checkpoint(
        self,
        execution_id: str,
        step_key: str,
        record_id: str,
        status: StepStatus,
        *,
        artifact_id: str | None = None,
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        prior = self.store.get_step_state(execution_id, step_key)
        self.store.put_step_state(
            StepState(
                execution_id=execution_id,
                step_key=step_key,
                state=status.value,
                record_id=record_id,
                artifact_id=artifact_id,
                attempts=(prior.attempts if prior else 0) + 1,
                detail=dict(detail or {}),
                updated_at=now(),
            )
        )

    def _open_review(
        self,
        step: PlannedStep,
        rp: RecordPlan,
        execution_id: str,
        available: Mapping[tuple[str, str], FieldArtifact],
        result: ExecutionResult,
        *,
        reason: str | None = None,
        proposed: Sequence[FieldArtifact] = (),
    ) -> None:
        """FR-VAL-005: everything a curator needs to decide, in one row."""

        by_field = {a.field_id: a for a in proposed}
        for fid in step.writes or ("",):
            prior = available.get((fid, step.entity))
            candidate = by_field.get(fid)
            # The item's identity is the *question* — this field, on this
            # record, raised by this step — not the run that raised it.  A
            # migration replayed for a record that is still catching up must
            # not hand the curator the same decision five times.
            item_id = content_hash(
                [step.record_id, step.entity, step.migration_id, step.step_id, fid]
            )
            self.store.put_review_item(
                {
                    "item_id": item_id,
                    "execution_id": execution_id,
                    "record_id": step.record_id,
                    "field_id": fid,
                    "entity": step.entity,
                    "state": "open",
                    "created_at": now(),
                    "migration_id": step.migration_id,
                    "step_id": step.step_id,
                    "reason": reason or step.reason,
                    "old_value": prior.value.to_canonical() if prior else None,
                    "proposed_value": candidate.value.to_canonical() if candidate else None,
                    "evidence": [
                        e.to_canonical()
                        for e in (
                            candidate.evidence
                            if candidate
                            else (prior.evidence if prior else ())
                        )
                    ],
                    "context": step.context.to_canonical() if step.context else None,
                    "validations": [
                        v.to_canonical()
                        for v in (candidate.provenance.validations if candidate else ())
                    ],
                }
            )
            with self._metrics_lock:
                result.review_items.append(item_id)

    def _entities_for(
        self, record_id: str, artifacts: Mapping[tuple[str, str], FieldArtifact]
    ) -> list[RecordEntity]:
        version = self.store.current_version(record_id)
        known = {e.entity: e for e in (version.entities if version else ())}
        for _, entity in artifacts:
            if entity and entity not in known:
                collection = entity.rsplit("/", 1)[-1].split("=", 1)[0]
                local_id = entity.rsplit("=", 1)[-1]
                parent = entity.rsplit("/", 1)[0] if "/" in entity else ""
                known[entity] = RecordEntity(
                    entity=entity,
                    collection_path=collection,
                    local_id=local_id,
                    position=len(known),
                    parent=parent,
                )
        return sorted(known.values(), key=lambda e: (e.parent, e.position, e.entity))

    def _index_entry(self, rp: RecordPlan, target: NormalizedSchema) -> RecordIndexEntry:
        for entry in self.store.index():
            if entry.record_id == rp.record_id:
                return entry
        return RecordIndexEntry(record_id=rp.record_id, schema_ref=rp.from_schema)

    def _set_record(self, result: ExecutionResult, record_id: str, state: str) -> None:
        with self._metrics_lock:
            result.records[record_id] = state

    def _bump(self, result: ExecutionResult, **counters: int) -> None:
        with self._metrics_lock:
            for name, value in counters.items():
                setattr(result.metrics, name, getattr(result.metrics, name) + value)

    def _final_status(self, result: ExecutionResult, cancel: threading.Event) -> ExecutionStatus:
        if cancel.is_set():
            return ExecutionStatus.CANCELLED
        states = set(result.records.values())
        if not states or states == {"published"}:
            return ExecutionStatus.SUCCEEDED
        if "published" in states:
            return ExecutionStatus.PARTIAL
        if states <= {"blocked", "review", "held"}:
            return ExecutionStatus.PARTIAL
        return ExecutionStatus.FAILED


class _ReviewNeeded(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


# ---- helpers -------------------------------------------------------------

def _ordered(steps: Sequence[PlannedStep]) -> list[PlannedStep]:
    """Topological order within one record, deterministic on ties."""

    by_key = {s.key: s for s in steps}
    deps = {s.key: {d for d in s.depends_on if d in by_key} for s in steps}
    out: list[PlannedStep] = []
    done: set[str] = set()
    remaining = sorted(by_key)
    while remaining:
        ready = [k for k in remaining if deps[k] <= done]
        if not ready:
            # A cycle among planned steps should be impossible; fall back to
            # declaration order rather than deadlocking the record.
            ready = remaining[:1]
        for k in ready:
            out.append(by_key[k])
            done.add(k)
            remaining.remove(k)
    return out


def _reads_for(
    available: Mapping[tuple[str, str], FieldArtifact],
    field_ids: Sequence[str],
    entity: str,
) -> dict[str, FieldArtifact]:
    out: dict[str, FieldArtifact] = {}
    for fid in field_ids:
        art = available.get((fid, entity)) or available.get((fid, ""))
        if art is not None:
            out[fid] = art
    return out


def _evidence_for(
    available: Mapping[tuple[str, str], FieldArtifact],
    field_ids: Sequence[str],
    entity: str,
) -> tuple[EvidenceReference, ...]:
    out: list[EvidenceReference] = []
    for fid in field_ids:
        art = available.get((fid, entity)) or available.get((fid, ""))
        if art is not None:
            out.extend(art.evidence)
    return tuple(out)


def _single_evidence(reads: Mapping[str, FieldArtifact]) -> tuple[EvidenceReference, ...]:
    """Carry evidence through a one-in/one-out structural move."""

    if len(reads) == 1:
        return next(iter(reads.values())).evidence
    return ()


def _decode_output(
    output: Any,
    mstep: MigrationStep,
    resolution: ContextResolution,
    parsed: ParsedSource | None,
) -> tuple[dict[str, FieldValue], dict[str, tuple[EvidenceReference, ...]]]:
    """Map a structured answer onto the fields the step declared it writes.

    Accepted shapes, in order:

    * ``{"<field_id>": {"status": ..., "value": ..., "evidence": [...]}, ...}``
    * ``{"<field_id>": <value>, ...}``
    * a bare value, when the step writes exactly one field.

    An answer that omits a declared field is an *abstention*, recorded as
    NOT_REPORTED rather than as a null.
    """

    values: dict[str, FieldValue] = {}
    evidence: dict[str, tuple[EvidenceReference, ...]] = {}

    if not isinstance(output, Mapping):
        if len(mstep.writes) == 1:
            output = {mstep.writes[0]: output}
        else:
            raise SemanticValidationError(
                f"model returned {type(output).__name__} but the step writes "
                f"{len(mstep.writes)} fields"
            )

    for fid in mstep.writes:
        if fid not in output:
            values[fid] = FieldValue(ValueStatus.NOT_REPORTED, reason="model did not answer")
            continue
        raw = output[fid]
        if isinstance(raw, Mapping) and ("status" in raw or "value" in raw or "evidence" in raw):
            status = ValueStatus(raw.get("status", "present"))
            values[fid] = FieldValue(
                status,
                raw.get("value") if status.has_value else None,
                reason=raw.get("reason"),
                annotations={
                    k: v
                    for k, v in raw.items()
                    if k in ("confidence", "value_source", "note")
                },
            )
            evidence[fid] = _evidence_from_quotes(raw.get("evidence"), resolution, parsed)
        else:
            values[fid] = (
                FieldValue.present(raw)
                if raw is not None
                else FieldValue(ValueStatus.NOT_REPORTED)
            )
            evidence[fid] = _evidence_from_quotes(None, resolution, parsed)

    return values, evidence


def _evidence_from_quotes(
    quotes: Any, resolution: ContextResolution, parsed: ParsedSource | None
) -> tuple[EvidenceReference, ...]:
    """Resolve model-supplied quotes against the context that was shown.

    A quote that cannot be placed character-for-character is *counted*, not
    invented: ``unlocated_quotes`` records the fidelity failure so it stays
    distinguishable from a recall failure (the study_schema distinction).
    """

    if not quotes:
        return ()
    if isinstance(quotes, str):
        quotes = [quotes]
    if not isinstance(quotes, Sequence):
        return ()

    from ..source import EvidenceSpan

    spans: list[EvidenceSpan] = []
    unlocated = 0
    for quote in quotes:
        text = quote if isinstance(quote, str) else (quote or {}).get("text", "")
        if not text:
            continue
        placed = False
        for unit in resolution.units:
            idx = unit.text.find(text)
            if idx >= 0:
                spans.append(EvidenceSpan(unit.unit_id, idx, idx + len(text), text))
                placed = True
                break
        if not placed:
            unlocated += 1

    if not spans and not unlocated:
        return ()
    return (
        EvidenceReference(
            source_id=parsed.source_id if parsed else "",
            source_version=parsed.source_version if parsed else "",
            parse_version=parsed.parse_version if parsed else "",
            spans=tuple(spans),
            locator="model_quote",
            unlocated_quotes=unlocated,
        ),
    )


def _var(field_id: str) -> str:
    return field_id.replace("[]", "").replace(".", "_")


def _version_kwargs(version: RecordVersion) -> dict[str, Any]:
    return {
        "record_id": version.record_id,
        "schema_ref": version.schema_ref,
        "artifact_ids": version.artifact_ids,
        "entities": version.entities,
        "state": version.state,
        "parent_version_id": version.parent_version_id,
        "execution_id": version.execution_id,
        "source_ref": version.source_ref,
        "created_at": version.created_at,
        "current_field_ids": version.current_field_ids,
        "notes": version.notes,
    }


__all__ = ["Engine", "TransformContext", "TransformResult", "SOFTWARE_VERSION"]
