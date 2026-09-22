"""Dependency hashing, cache keys and semantic currency.

§8.3 in one module.  Two questions are answered here and nowhere else:

1. *What does this output depend on?*  :func:`dependency_hashes` enumerates
   every output-affecting input — field definition, recipe, prompt, model
   policy, context policy, vocabulary, source, parse, transform code and the
   exact prior values read — and hashes each separately so a plan can say
   *which* dependency changed, not merely that something did.

2. *Is this stored value still current?*  :func:`assess_currency` compares an
   artifact's recorded ``input_hashes`` against the hashes that hold now.
   Equal means reuse (FR-DEP-002); different means the plan says which key
   differs; absent means missing provenance, which is conservatively treated
   as not current (FR-DEP-006).

The distinction the requirements insist on runs through both: conformance to a
shape is not semantic currency, and a schema diff alone never authorises reuse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from ..functions import TRANSFORMS, VALIDATORS
from ..ids import content_hash
from ..migration.spec import MigrationStep
from ..provenance import FieldArtifact
from ..recipe import ExtractionRecipe, Vocabulary
from ..schema.normalized import FieldDefinition, NormalizedSchema
from ..source import ParsedSource, SourceArtifact
from ..values import ValueStatus


class Currency(str, Enum):
    """Why a stored value may or may not be used as-is."""

    CURRENT = "current"
    #: No artifact at all for this field/entity.
    MISSING = "missing"
    #: An artifact exists but records no dependency hashes.
    NO_PROVENANCE = "no_provenance"
    #: Dependencies changed.  ``changed_keys`` says which.
    STALE = "stale"
    #: The value is there and a human accepted it under the new recipe.
    ACCEPTED = "accepted"
    #: The artifact records a failure, so there is nothing to reuse.
    FAILED = "failed"
    #: The value is waiting on a curator.
    REVIEW = "review"


@dataclass(frozen=True)
class CurrencyVerdict:
    currency: Currency
    changed_keys: tuple[str, ...] = ()
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def reusable(self) -> bool:
        return self.currency in (Currency.CURRENT, Currency.ACCEPTED)

    def reason(self) -> str:
        if self.currency is Currency.CURRENT:
            return "all declared dependencies unchanged"
        if self.currency is Currency.ACCEPTED:
            return f"legacy value accepted under {self.detail.get('accepted_under')!r}"
        if self.currency is Currency.MISSING:
            return "no stored value"
        if self.currency is Currency.NO_PROVENANCE:
            return "stored value carries no dependency provenance"
        if self.currency is Currency.FAILED:
            return f"stored value is {self.detail.get('status')}"
        if self.currency is Currency.REVIEW:
            return "stored value is awaiting review"
        return "changed: " + ", ".join(self.changed_keys)


@dataclass
class DependencyContext:
    """Everything the current state of the world says about one record."""

    schema: NormalizedSchema
    source: SourceArtifact | None = None
    parsed: ParsedSource | None = None
    recipes: Mapping[str, ExtractionRecipe] = field(default_factory=dict)
    vocabularies: Mapping[str, Vocabulary] = field(default_factory=dict)
    #: ``(field_id, entity) -> artifact`` for the record being planned.
    artifacts: Mapping[tuple[str, str], FieldArtifact] = field(default_factory=dict)

    def artifact(self, field_id: str, entity: str) -> FieldArtifact | None:
        return self.artifacts.get((field_id, entity))


def field_definition_hash(fdef: FieldDefinition) -> str:
    """What the field *asks for*.

    FR-DEP-003: the path is deliberately excluded, so renaming a field does not
    invalidate its value.  FR-SCH-007: the description is included, so
    rewriting the extraction instruction does.
    """

    return fdef.semantic_key()


def dependency_hashes(
    step: MigrationStep | None,
    fields: FieldDefinition | Sequence[FieldDefinition],
    ctx: DependencyContext,
    *,
    entity: str = "",
    recipe: ExtractionRecipe | None = None,
) -> dict[str, str]:
    """Every output-affecting input for a field or field group, keyed by what it is."""

    fdefs = [fields] if isinstance(fields, FieldDefinition) else list(fields)
    if not fdefs:
        raise ValueError("dependency_hashes needs at least one field definition")
    fdef = fdefs[0]
    out: dict[str, str] = {
        f"field_def:{f.field_id}": field_definition_hash(f) for f in fdefs
    }

    if recipe is not None:
        out["recipe"] = recipe.recipe_hash()
        out["prompt"] = recipe.prompt_hash
        out["context_policy"] = recipe.context_policy.policy_hash
        if recipe.model_policy is not None:
            out["model_policy"] = recipe.model_policy.policy_hash()
            if recipe.model_policy.pin_fingerprint:
                out["model_fingerprint"] = recipe.model_policy.pin_fingerprint
        for name in recipe.validators:
            entry = VALIDATORS.maybe(name)
            if entry:
                out[f"validator:{name}"] = entry.code_hash
        if recipe.vocabulary_ref:
            vocab = ctx.vocabularies.get(recipe.vocabulary_ref)
            if vocab is not None:
                out[f"vocab:{vocab.name}"] = vocab.vocabulary_hash()
    elif fdef.recipe_ref:
        known = ctx.recipes.get(fdef.recipe_ref)
        out["recipe"] = known.recipe_hash() if known else f"declared:{fdef.recipe_ref}"

    for f in fdefs:
        if f.constraints.vocabulary_ref:
            vocab = _find_vocabulary(ctx, f.constraints.vocabulary_ref)
            if vocab is not None:
                out[f"vocab:{vocab.name}"] = vocab.vocabulary_hash()

    if step is not None:
        out["step"] = step.step_hash()
        if step.transform:
            entry = TRANSFORMS.maybe(step.transform)
            out["transform"] = entry.code_hash if entry else f"declared:{step.transform}"
        if step.context is not None:
            out["context_policy"] = step.context.policy_hash
        for name in step.validators:
            entry = VALIDATORS.maybe(name)
            if entry:
                out[f"validator:{name}"] = entry.code_hash
        for vocab_ref in step.reads_vocabularies:
            vocab = _find_vocabulary(ctx, vocab_ref)
            if vocab is not None:
                out[f"vocab:{vocab.name}"] = vocab.vocabulary_hash()

        needs_source = step.touches_source or bool(step.reads_evidence)
        if needs_source:
            if ctx.source is not None:
                out["source"] = ctx.source.content_hash
            if ctx.parsed is not None:
                out["parse"] = ctx.parsed.parse_version

        # FR-PROV-005 / FR-LLM-006: the *exact prior values* used.
        #
        # A field the step both reads and writes is excluded: hashing the
        # current value of an in-place transform would make the step
        # permanently stale against its own output, and re-running it forever
        # is exactly what principle 7 (idempotence) forbids.  Such a step's
        # identity rests on its other dependencies plus its own step hash.
        written = set(step.writes)
        for dep_field in sorted(set(step.reads_fields) - written):
            dep = ctx.artifact(dep_field, entity) or ctx.artifact(dep_field, "")
            out[f"dep:{dep_field}"] = dep.value_hash() if dep is not None else "absent"

    return out


def _find_vocabulary(ctx: DependencyContext, ref: str) -> Vocabulary | None:
    if ref in ctx.vocabularies:
        return ctx.vocabularies[ref]
    # A bare name: accept it when exactly one version is registered.
    matches = [v for v in ctx.vocabularies.values() if v.name == ref]
    return matches[0] if len(matches) == 1 else None


def assess_currency(
    artifact: FieldArtifact | None,
    expected: Mapping[str, str],
    *,
    required_recipe: str | None = None,
) -> CurrencyVerdict:
    """Decide whether ``artifact`` may be reused."""

    if artifact is None:
        return CurrencyVerdict(Currency.MISSING)

    status = artifact.value.status
    if status is ValueStatus.EXTRACTION_FAILED:
        return CurrencyVerdict(Currency.FAILED, detail={"status": status.value})
    if status is ValueStatus.REVIEW_REQUIRED:
        return CurrencyVerdict(Currency.REVIEW, detail={"status": status.value})

    prov = artifact.provenance
    if prov.accepted_under and (
        required_recipe is None or prov.accepted_under == required_recipe
    ):
        return CurrencyVerdict(
            Currency.ACCEPTED,
            detail={"accepted_under": prov.accepted_under, "actor": prov.actor_id},
        )

    recorded = prov.input_hashes
    if not recorded:
        # FR-DEP-006: absence of provenance is never evidence of currency.
        return CurrencyVerdict(Currency.NO_PROVENANCE)

    # Only the keys ``expected`` declares are compared.  A producer may have
    # tracked *more* than the current question asks — a migration step records
    # its transform hash and the prior values it read, while the field-level
    # currency question asks only about the field definition and its recipe —
    # and having tracked more is never a reason to call a value stale.  A key
    # that is expected and absent from the record is a mismatch, which is what
    # keeps a legacy import from passing a step's check.
    changed = [key for key in sorted(expected) if recorded.get(key) != expected[key]]
    if changed:
        return CurrencyVerdict(
            Currency.STALE,
            tuple(changed),
            detail={
                "recorded": {k: recorded.get(k) for k in changed},
                "expected": {k: expected.get(k) for k in changed},
            },
        )

    if required_recipe and prov.recipe_ref != required_recipe:
        return CurrencyVerdict(
            Currency.STALE,
            ("recipe_ref",),
            detail={"recorded": prov.recipe_ref, "expected": required_recipe},
        )

    return CurrencyVerdict(Currency.CURRENT)


def cache_key(
    *,
    step: MigrationStep,
    record_id: str,
    entity: str,
    field_ids: Sequence[str],
    dependencies: Mapping[str, str],
    context_hash: str | None = None,
    schema_ref: str = "",
) -> str:
    """The logical identity of one unit of work (FR-EXE-004, FR-LLM-006).

    Everything that can change the output is in here; nothing that cannot is.
    The record id is included because the same question about two papers has
    two answers; the execution id is *not*, which is what makes a resumed job
    reuse the previous job's results rather than paying again.
    """

    return content_hash(
        {
            "record_id": record_id,
            "entity": entity,
            "schema_ref": schema_ref,
            "step": step.step_hash(),
            "writes": sorted(field_ids),
            "dependencies": dict(sorted(dependencies.items())),
            "context": context_hash,
        }
    )


def step_key(record_id: str, entity: str, migration_id: str, step_id: str) -> str:
    """Durable checkpoint key for one planned step (FR-EXE-001)."""

    return f"{record_id}|{entity}|{migration_id}|{step_id}"


def semantic_currency_of_record(
    ctx: DependencyContext,
    entities: Sequence[str] = ("",),
) -> dict[str, CurrencyVerdict]:
    """Per-field currency for the whole record, ignoring migrations.

    This is what ``llmbic status`` reports and what decides
    :attr:`RecordVersion.current_field_ids`.  It answers the §6 question
    directly: a record can validate against the latest schema and still have no
    semantically current fields at all.
    """

    out: dict[str, CurrencyVerdict] = {}
    for fdef in ctx.schema.fields:
        scope = fdef.collection_path
        for entity in entities:
            if not _entity_matches(entity, scope):
                continue
            recipe = ctx.recipes.get(fdef.recipe_ref) if fdef.recipe_ref else None
            expected = dependency_hashes(None, fdef, ctx, entity=entity, recipe=recipe)
            artifact = ctx.artifact(fdef.field_id, entity)
            verdict = assess_currency(artifact, expected, required_recipe=fdef.recipe_ref)
            out[f"{fdef.field_id}@{entity}"] = verdict
    return out


def _entity_matches(entity: str, collection_path: str) -> bool:
    if not collection_path:
        return entity == ""
    return entity.rsplit("/", 1)[-1].split("=", 1)[0] == collection_path


def entities_for_scope(entity_ids: Sequence[str], collection_path: str) -> list[str]:
    """Which entity keys a step with ``entity_scope`` runs against."""

    if not collection_path:
        return [""]
    return [e for e in entity_ids if _entity_matches(e, collection_path)]


__all__ = [
    "Currency",
    "CurrencyVerdict",
    "DependencyContext",
    "assess_currency",
    "cache_key",
    "dependency_hashes",
    "entities_for_scope",
    "field_definition_hash",
    "semantic_currency_of_record",
    "step_key",
]
