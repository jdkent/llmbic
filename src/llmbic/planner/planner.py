"""The planner — layer 3.

§11 step by step.  For every selected record it identifies the versions the
record is at, resolves the approved path to the target, expands each migration
into field-level or field-group steps, matches required outputs against
existing valid artifacts and cache entries, and marks each output as reuse,
deterministic transform, semantic execution, review or blocked.

Two rules are load-bearing and both are negative:

* the planner never uses a schema diff alone to decide a value is reusable —
  only the dependency hashes decide that;
* an ordinary dry run resolves context policies without invoking a billable
  model, and never writes record state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from ..context.policy import OnMissingContext
from ..context.resolver import SelectionInput, check_provider, resolve_context
from ..errors import ErrorCode, LlmbicError, PathError, PolicyError
from ..migration.spec import ExecutionPolicy, Migration, MigrationStep, StepKind
from ..models.base import AdapterRegistry
from ..provenance import FieldArtifact, latest_per_key
from ..recipe import ExtractionRecipe
from ..registry import PathPreference, Registry
from ..schema.normalized import NormalizedSchema

from ..store.base import RecordFilter, RecordIndexEntry, Store

from .dependencies import (
    CurrencyVerdict,
    DependencyContext,
    assess_currency,
    cache_key,
    dependency_hashes,
    entities_for_scope,
    step_key,
)
from .plan import (
    ContextPreview,
    Disposition,
    ExecutionPlan,
    PlannedStep,
    RecordPlan,
    summarise,
)

#: What to do about a field that is not semantically current and that no
#: migration step addresses (decision 18.10 is the operator's, not ours).
STALE_REPORT = "report"
STALE_REVIEW = "review"
STALE_REFRESH = "refresh"


@dataclass
class PlannerOptions:
    preference: PathPreference = field(default_factory=PathPreference)
    on_stale: str = STALE_REPORT
    #: Permit context selectors that themselves call a billable model.  False
    #: during ordinary dry runs (FR-PLN-005).
    allow_model_selectors: bool = False
    check_cache: bool = True
    chars_per_token: float = 4.0
    #: Used for cost estimation when the recipe does not say.
    assumed_output_tokens: int = 250
    #: Emit a validation step per record.
    validate_records: bool = True
    software_version: str = ""


class Planner:
    def __init__(
        self,
        registry: Registry,
        store: Store,
        *,
        options: PlannerOptions | None = None,
        adapters: AdapterRegistry | None = None,
    ) -> None:
        self.registry = registry
        self.store = store
        self.options = options or PlannerOptions()
        self.adapters = adapters

    # ---- entry points ----------------------------------------------------
    def plan(
        self,
        target_schema: str,
        *,
        filt: RecordFilter | None = None,
        policy: ExecutionPolicy | None = None,
    ) -> ExecutionPlan:
        policy = policy or ExecutionPolicy()
        target = self.registry.schema(target_schema)
        record_plans: list[RecordPlan] = []
        for entry in self.store.index(filt):
            record_plans.append(self.plan_record(entry, target, policy))
        record_plans.sort(key=lambda r: r.record_id)
        return ExecutionPlan(
            target_schema=target.ref,
            records=tuple(record_plans),
            summary=summarise(record_plans),
            policy=policy,
            registry_hash=self.registry.registry_hash(),
            software_version=self.options.software_version,
        )

    def plan_record(
        self,
        entry: RecordIndexEntry,
        target: NormalizedSchema,
        policy: ExecutionPolicy,
    ) -> RecordPlan:
        from_ref = entry.schema_ref

        try:
            path = self.registry.find_path(from_ref, target.ref, preference=self.options.preference)
        except (PathError, LlmbicError) as exc:
            return RecordPlan(
                record_id=entry.record_id,
                from_schema=from_ref,
                to_schema=target.ref,
                blocked=True,
                blocked_reason=exc.message if isinstance(exc, LlmbicError) else str(exc),
                blocked_code=(exc.code.value if isinstance(exc, LlmbicError) else ErrorCode.INTERNAL.value),
            )

        for migration in path:
            ok, why = policy.permits(migration)
            if not ok:
                return RecordPlan(
                    record_id=entry.record_id,
                    from_schema=from_ref,
                    to_schema=target.ref,
                    path=tuple(m.id for m in path),
                    blocked=True,
                    blocked_reason=why,
                    blocked_code=ErrorCode.FIDELITY_NOT_PERMITTED.value,
                )

        ctx = self._context_for(entry, target)
        entity_ids = self._entity_ids(entry, ctx)
        steps: list[PlannedStep] = []
        written: set[str] = set()

        for migration in path:
            for step in migration.ordered_steps():
                for entity in entities_for_scope(entity_ids, step.entity_scope):
                    planned = self._plan_step(
                        entry, migration, step, entity, ctx, policy, target
                    )
                    if planned is not None:
                        steps.append(planned)
                        written.update(planned.writes)

        stale = self._stale_fields(ctx, entity_ids, written)
        if self.options.on_stale in (STALE_REVIEW, STALE_REFRESH) and stale:
            steps.extend(self._steps_for_stale(entry, ctx, stale, target, policy))
            written.update(f.split("@", 1)[0] for f in stale)
            stale = ()

        if self.options.validate_records:
            steps.append(self._validation_step(entry, target, steps))

        steps = _link_dependencies(steps)

        return RecordPlan(
            record_id=entry.record_id,
            from_schema=from_ref,
            to_schema=target.ref,
            path=tuple(m.id for m in path),
            steps=tuple(steps),
            stale_fields=tuple(stale),
            source_ref=f"{entry.source_id}@{entry.source_version}"
            if entry.source_id
            else None,
        )

    # ---- per-step planning ----------------------------------------------
    def _plan_step(
        self,
        entry: RecordIndexEntry,
        migration: Migration,
        step: MigrationStep,
        entity: str,
        ctx: DependencyContext,
        policy: ExecutionPolicy,
        target: NormalizedSchema,
    ) -> PlannedStep | None:
        # A step is answerable to the schema *it* targets, not to the final
        # destination of the path: the 1.0 -> 1.1 step satisfies 1.1's
        # definition of the field, and if 1.4 redefines it, that is 1.4's step's
        # problem.  Stamping the final target here would let an intermediate
        # step claim currency for a definition it never saw.
        step_schema = (
            self.registry.schema(migration.to_schema)
            if self.registry.has_schema(migration.to_schema)
            else target
        )
        fdefs = [step_schema.get(fid) for fid in step.writes]
        missing = [fid for fid, fd in zip(step.writes, fdefs) if fd is None]
        key = step_key(entry.record_id, entity, migration.id, step.id)
        if missing:
            return PlannedStep(
                key=key,
                record_id=entry.record_id,
                entity=entity,
                migration_id=migration.id,
                step_id=step.id,
                kind=step.kind,
                disposition=Disposition.BLOCKED,
                writes=tuple(step.writes),
                reason=f"target schema does not define {missing}",
                blocked_code=ErrorCode.MIGRATION_INVALID.value,
                fidelity=step.fidelity,
            )
        resolved_fdefs = [fd for fd in fdefs if fd is not None]

        recipe = self.registry.maybe_recipe(step.recipe)
        deps = dependency_hashes(step, resolved_fdefs, ctx, entity=entity, recipe=recipe)

        # --- reuse? -------------------------------------------------------
        verdicts = {
            fid: assess_currency(
                ctx.artifact(fid, entity), deps, required_recipe=step.recipe or fd.recipe_ref
            )
            for fid, fd in zip(step.writes, resolved_fdefs)
        }
        if step.reusable and verdicts and all(v.reusable for v in verdicts.values()):
            return PlannedStep(
                key=key,
                record_id=entry.record_id,
                entity=entity,
                migration_id=migration.id,
                step_id=step.id,
                kind=step.kind,
                disposition=Disposition.REUSE,
                writes=tuple(step.writes),
                reason=next(iter(verdicts.values())).reason(),
                fidelity=step.fidelity,
                dependencies=deps,
                recipe_ref=step.recipe,
                transform_ref=step.transform,
            )

        change_reason = _why_not_reusable(verdicts)

        if step.kind is StepKind.MANUAL_REVIEW:
            return PlannedStep(
                key=key,
                record_id=entry.record_id,
                entity=entity,
                migration_id=migration.id,
                step_id=step.id,
                kind=step.kind,
                disposition=Disposition.REVIEW,
                writes=tuple(step.writes),
                reason=step.description or "migration requires a human decision",
                fidelity=step.fidelity,
                dependencies=deps,
            )

        # --- required reads present? -------------------------------------
        required = tuple(step.params.get("require_reads") or ())
        absent = [
            fid
            for fid in required
            if (ctx.artifact(fid, entity) or ctx.artifact(fid, "")) is None
            or not (ctx.artifact(fid, entity) or ctx.artifact(fid, "")).value.status.has_value
        ]
        if absent:
            return PlannedStep(
                key=key,
                record_id=entry.record_id,
                entity=entity,
                migration_id=migration.id,
                step_id=step.id,
                kind=step.kind,
                disposition=Disposition.BLOCKED,
                writes=tuple(step.writes),
                reason=f"required input(s) absent: {', '.join(absent)}",
                blocked_code=ErrorCode.MISSING_DEPENDENCY.value,
                fidelity=step.fidelity,
                dependencies=deps,
            )

        if step.kind.is_deterministic:
            return PlannedStep(
                key=key,
                record_id=entry.record_id,
                entity=entity,
                migration_id=migration.id,
                step_id=step.id,
                kind=step.kind,
                disposition=Disposition.DETERMINISTIC,
                writes=tuple(step.writes),
                reason=change_reason,
                fidelity=step.fidelity,
                dependencies=deps,
                transform_ref=step.transform,
                cache_key=cache_key(
                    step=step,
                    record_id=entry.record_id,
                    entity=entity,
                    field_ids=step.writes,
                    dependencies=deps,
                    schema_ref=step_schema.ref,
                ),
            )

        return self._plan_semantic_step(
            entry, migration, step, entity, ctx, policy, step_schema, recipe, deps,
            change_reason,
        )

    def _plan_semantic_step(
        self,
        entry: RecordIndexEntry,
        migration: Migration,
        step: MigrationStep,
        entity: str,
        ctx: DependencyContext,
        policy: ExecutionPolicy,
        step_schema: NormalizedSchema,
        recipe: ExtractionRecipe | None,
        deps: dict[str, str],
        change_reason: str,
    ) -> PlannedStep:
        key = step_key(entry.record_id, entity, migration.id, step.id)
        if recipe is None:
            return PlannedStep(
                key=key,
                record_id=entry.record_id,
                entity=entity,
                migration_id=migration.id,
                step_id=step.id,
                kind=step.kind,
                disposition=Disposition.BLOCKED,
                writes=tuple(step.writes),
                reason=f"recipe {step.recipe!r} is not registered",
                blocked_code=ErrorCode.MIGRATION_NOT_FOUND.value,
                fidelity=step.fidelity,
            )

        ctx_policy = step.context or recipe.context_policy
        on_missing = step.on_missing_context or ctx_policy.on_missing_context

        selection = SelectionInput(
            record_id=entry.record_id,
            parsed=ctx.parsed,
            source=ctx.source,
            prior_evidence=_evidence_for(ctx, step.reads_evidence, entity),
            prior_fields=_fields_for(ctx, step.reads_fields, entity),
            field_id=step.writes[0] if step.writes else None,
            entity=entity,
        )
        resolution = resolve_context(
            ctx_policy, selection, allow_model_selectors=self.options.allow_model_selectors
        )
        preview = ContextPreview(
            satisfied=resolution.satisfied,
            used_sources=resolution.used_sources,
            attempted=resolution.attempted,
            unit_ids=resolution.unit_ids,
            n_chars=resolution.n_chars,
            est_input_tokens=resolution.estimate_tokens(self.options.chars_per_token),
            includes_full_document=resolution.includes_full_document,
            truncated=resolution.truncated,
            blocked_reason=resolution.blocked_reason,
            context_hash=resolution.context_hash(),
        )

        base = dict(
            key=key,
            record_id=entry.record_id,
            entity=entity,
            migration_id=migration.id,
            step_id=step.id,
            kind=step.kind,
            writes=tuple(step.writes),
            fidelity=step.fidelity,
            dependencies=deps,
            recipe_ref=recipe.ref,
            context=preview,
        )

        if not resolution.satisfied:
            if on_missing is OnMissingContext.REVIEW:
                return PlannedStep(
                    **base,
                    disposition=Disposition.REVIEW,
                    reason=resolution.blocked_reason or "required context unavailable",
                    blocked_code=resolution.blocked_code,
                )
            return PlannedStep(
                **base,
                disposition=Disposition.BLOCKED,
                reason=resolution.blocked_reason or "required context unavailable",
                blocked_code=resolution.blocked_code or ErrorCode.CONTEXT_UNAVAILABLE.value,
            )

        if preview.includes_full_document and not policy.allow_full_document:
            return PlannedStep(
                **base,
                disposition=Disposition.BLOCKED,
                reason=(
                    "this step would transmit the full document; re-run with "
                    "allow_full_document=True to permit it"
                ),
                blocked_code=ErrorCode.CONTEXT_POLICY_FORBIDS.value,
            )

        model_policy = recipe.model_policy
        provider = None
        if model_policy is not None:
            adapter = self.adapters.get(model_policy.adapter) if self.adapters else None
            if adapter is not None:
                provider = adapter.identity.provider
                try:
                    check_provider(ctx_policy, provider, adapter.data_policy.as_dict())
                except PolicyError as exc:
                    return PlannedStep(
                        **base,
                        disposition=Disposition.BLOCKED,
                        reason=exc.message,
                        blocked_code=exc.code.value,
                        provider=provider,
                    )
            else:
                provider = model_policy.adapter

        ck = cache_key(
            step=step,
            record_id=entry.record_id,
            entity=entity,
            field_ids=step.writes,
            dependencies=deps,
            context_hash=preview.context_hash,
            schema_ref=step_schema.ref,
        )

        if self.options.check_cache and self.store.cache_get(ck) is not None:
            return PlannedStep(
                **base,
                disposition=Disposition.CACHED,
                reason="a previous run already answered this exact question",
                cache_key=ck,
                provider=provider,
            )

        in_tok = preview.est_input_tokens + _prompt_tokens(recipe, self.options.chars_per_token)
        out_tok = int(step.params.get("expected_output_tokens", self.options.assumed_output_tokens))
        cost = (
            model_policy.pricing.cost(in_tok, out_tok) if model_policy is not None else 0.0
        )

        return PlannedStep(
            **base,
            disposition=Disposition.SEMANTIC,
            reason=change_reason,
            cache_key=ck,
            provider=provider,
            est_input_tokens=in_tok,
            est_output_tokens=out_tok,
            est_cost_usd=cost,
        )

    # ---- stale fields ----------------------------------------------------
    def _stale_fields(
        self, ctx: DependencyContext, entity_ids: Sequence[str], written: set[str]
    ) -> tuple[str, ...]:
        """Target fields that are not current and that no step refreshes.

        These are exactly the values that would look current if llmbic only
        checked the shape.  They are reported, never silently accepted.
        """

        out: list[str] = []
        for fdef in ctx.schema.fields:
            if fdef.field_id in written or fdef.deterministic:
                continue
            for entity in entities_for_scope(entity_ids, fdef.collection_path):
                recipe = ctx.recipes.get(fdef.recipe_ref) if fdef.recipe_ref else None
                expected = dependency_hashes(None, fdef, ctx, entity=entity, recipe=recipe)
                verdict = assess_currency(
                    ctx.artifact(fdef.field_id, entity),
                    expected,
                    required_recipe=fdef.recipe_ref,
                )
                if not verdict.reusable:
                    out.append(f"{fdef.field_id}@{entity}|{verdict.currency.value}")
        return tuple(out)

    def _steps_for_stale(
        self,
        entry: RecordIndexEntry,
        ctx: DependencyContext,
        stale: Sequence[str],
        target: NormalizedSchema,
        policy: ExecutionPolicy,
    ) -> list[PlannedStep]:
        steps: list[PlannedStep] = []
        for item in stale:
            fid_entity, _, currency = item.partition("|")
            fid, _, entity = fid_entity.partition("@")
            steps.append(
                PlannedStep(
                    key=step_key(entry.record_id, entity, "__stale__", fid),
                    record_id=entry.record_id,
                    entity=entity,
                    migration_id="__stale__",
                    step_id=fid,
                    kind=StepKind.MANUAL_REVIEW,
                    disposition=Disposition.REVIEW,
                    writes=(fid,),
                    reason=f"field is {currency} and no migration step addresses it",
                    detail={"currency": currency},
                )
            )
        return steps

    # ---- helpers ---------------------------------------------------------
    def _validation_step(
        self, entry: RecordIndexEntry, target: NormalizedSchema, steps: Sequence[PlannedStep]
    ) -> PlannedStep:
        return PlannedStep(
            key=step_key(entry.record_id, "", "__validate__", target.ref),
            record_id=entry.record_id,
            entity="",
            migration_id="__validate__",
            step_id="validate",
            kind=StepKind.VALIDATION,
            disposition=Disposition.VALIDATE,
            writes=(),
            reason=f"structural validation against {target.ref}",
            depends_on=tuple(s.key for s in steps if s.disposition.runs),
        )

    def _context_for(
        self, entry: RecordIndexEntry, target: NormalizedSchema
    ) -> DependencyContext:
        artifacts = latest_per_key(self.store.get_artifacts(entry.record_id))
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
        return DependencyContext(
            schema=target,
            source=source,
            parsed=parsed,
            recipes={r.ref: r for r in self.registry.recipes()},
            vocabularies={v.ref: v for v in self.registry.vocabularies()},
            artifacts={(a.field_id, a.entity): a for a in artifacts},
        )

    def _entity_ids(self, entry: RecordIndexEntry, ctx: DependencyContext) -> list[str]:
        version = self.store.current_version(entry.record_id)
        if version is not None and version.entities:
            ids = [e.entity for e in version.entities]
        else:
            ids = sorted({e for (_, e) in ctx.artifacts if e})
        return [""] + sorted(set(ids))


def _why_not_reusable(verdicts: Mapping[str, CurrencyVerdict]) -> str:
    if not verdicts:
        return "no prior value"
    parts: list[str] = []
    for fid, verdict in sorted(verdicts.items()):
        if verdict.reusable:
            continue
        parts.append(f"{fid}: {verdict.reason()}")
    return "; ".join(parts) or "dependencies changed"


def _prompt_tokens(recipe: ExtractionRecipe, chars_per_token: float) -> int:
    n = len(recipe.prompt_template) + len(recipe.system)
    return int(n / chars_per_token + 0.999)


def _evidence_for(ctx: DependencyContext, field_ids: Sequence[str], entity: str):
    out = []
    for fid in field_ids:
        artifact = ctx.artifact(fid, entity) or ctx.artifact(fid, "")
        if artifact is not None:
            out.extend(artifact.evidence)
    return tuple(out)


def _fields_for(
    ctx: DependencyContext, field_ids: Sequence[str], entity: str
) -> dict[str, FieldArtifact]:
    out: dict[str, FieldArtifact] = {}
    for fid in field_ids:
        artifact = ctx.artifact(fid, entity) or ctx.artifact(fid, "")
        if artifact is not None:
            out[fid] = artifact
    return out


def _link_dependencies(steps: Sequence[PlannedStep]) -> list[PlannedStep]:
    """Order steps within a record by what they read and write."""

    producers: dict[tuple[str, str], list[str]] = {}
    for s in steps:
        for w in s.writes:
            producers.setdefault((w, s.entity), []).append(s.key)

    out: list[PlannedStep] = []
    for s in steps:
        deps = set(s.depends_on)
        for dep_key in s.dependencies:
            if not dep_key.startswith("dep:"):
                continue
            dep_field = dep_key[4:]
            for candidate in producers.get((dep_field, s.entity), []) + producers.get(
                (dep_field, ""), []
            ):
                if candidate != s.key:
                    deps.add(candidate)
        out.append(
            PlannedStep(
                **{
                    **{k: getattr(s, k) for k in s.__dataclass_fields__},
                    "depends_on": tuple(sorted(deps)),
                }
            )
        )
    return out


__all__ = ["Planner", "PlannerOptions", "STALE_REFRESH", "STALE_REPORT", "STALE_REVIEW"]
