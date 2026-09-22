"""Structural and semantic validation.

FR-VAL-001..003.  Structural validation runs against the normalized schema —
types, cardinality, requiredness, enums, numeric and length bounds — and is
what every intermediate and final record is checked against.  Semantic
validators are ordinary registered functions that see the value, its evidence,
the context that produced it and the rest of the record, and return pass, fail
or review-needed with a structured reason.

Requiredness is checked against *value status*, not against Python ``None``: a
required field answered NOT_REPORTED with evidence is a legitimate record,
while a required field in state NOT_EXTRACTED is not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .context.resolver import ContextUnit
from .functions import VALIDATORS
from .provenance import FieldArtifact, Outcome, ValidationResult
from .recipe import ExtractionRecipe
from .schema.normalized import BaseType, FieldDefinition, NormalizedSchema
from .source import EvidenceReference, ParsedSource
from .values import FieldValue, ValueStatus

_PY_TYPES: dict[BaseType, tuple[type, ...]] = {
    BaseType.STRING: (str,),
    BaseType.INTEGER: (int,),
    BaseType.NUMBER: (int, float),
    BaseType.BOOLEAN: (bool,),
    BaseType.OBJECT: (dict,),
    BaseType.ANY: (object,),
}


@dataclass
class ValidationContext:
    """What a semantic validator is given."""

    record_id: str
    field_id: str
    entity: str
    value: FieldValue
    field_def: FieldDefinition
    evidence: tuple[EvidenceReference, ...] = ()
    context_units: tuple[ContextUnit, ...] = ()
    parsed: ParsedSource | None = None
    recipe: ExtractionRecipe | None = None
    #: ``(field_id, entity) -> artifact`` for the rest of the record.
    artifacts: Mapping[tuple[str, str], FieldArtifact] = field(default_factory=dict)
    schema: NormalizedSchema | None = None

    def other(self, field_id: str) -> FieldArtifact | None:
        return self.artifacts.get((field_id, self.entity)) or self.artifacts.get((field_id, ""))

    def evidence_text(self) -> list[str]:
        if self.parsed is None:
            return [s.text for ref in self.evidence for s in ref.spans]
        out: list[str] = []
        for ref in self.evidence:
            out.extend(ref.resolve(self.parsed))
        return out


def validate_field_value(
    value: FieldValue, fdef: FieldDefinition, *, validator_name: str = "structural"
) -> list[ValidationResult]:
    """Type, cardinality and constraint checks for one slot."""

    results: list[ValidationResult] = []

    if not value.status.has_value:
        if fdef.required and value.status in (
            ValueStatus.NOT_EXTRACTED,
            ValueStatus.EXTRACTION_FAILED,
        ):
            results.append(
                ValidationResult(
                    validator_name,
                    Outcome.FAIL,
                    f"{fdef.path} is required but is {value.status.value}",
                    {"field_id": fdef.field_id, "status": value.status.value},
                )
            )
        return results

    raw = value.value
    items = raw if fdef.multivalued else [raw]

    if fdef.multivalued and not isinstance(raw, (list, tuple)):
        results.append(
            ValidationResult(
                validator_name,
                Outcome.FAIL,
                f"{fdef.path} is multivalued but holds {type(raw).__name__}",
                {"field_id": fdef.field_id},
            )
        )
        return results

    c = fdef.constraints
    if fdef.multivalued:
        if c.min_items is not None and len(items) < c.min_items:
            results.append(
                ValidationResult(
                    validator_name,
                    Outcome.FAIL,
                    f"{fdef.path} has {len(items)} items, minimum is {c.min_items}",
                    {"field_id": fdef.field_id},
                )
            )
        if c.max_items is not None and len(items) > c.max_items:
            results.append(
                ValidationResult(
                    validator_name,
                    Outcome.FAIL,
                    f"{fdef.path} has {len(items)} items, maximum is {c.max_items}",
                    {"field_id": fdef.field_id},
                )
            )

    expected = _PY_TYPES.get(fdef.base_type, (object,))
    for item in items:
        if item is None:
            continue
        if fdef.base_type is not BaseType.ANY:
            # bool is an int in Python; an integer slot must not accept True.
            if fdef.base_type in (BaseType.INTEGER, BaseType.NUMBER) and isinstance(item, bool):
                results.append(
                    ValidationResult(
                        validator_name,
                        Outcome.FAIL,
                        f"{fdef.path} expects {fdef.base_type.value}, got boolean",
                        {"field_id": fdef.field_id},
                    )
                )
                continue
            if not isinstance(item, expected):
                results.append(
                    ValidationResult(
                        validator_name,
                        Outcome.FAIL,
                        f"{fdef.path} expects {fdef.base_type.value}, got "
                        f"{type(item).__name__}",
                        {"field_id": fdef.field_id, "value": _clip(item)},
                    )
                )
                continue

        if c.enum and isinstance(item, str) and item not in c.enum:
            if c.open_vocabulary:
                results.append(
                    ValidationResult(
                        validator_name,
                        Outcome.PASS,
                        f"{fdef.path} carries the source's own wording {item!r}, which the "
                        "open vocabulary permits",
                        {"field_id": fdef.field_id, "value": _clip(item)},
                    )
                )
            else:
                results.append(
                    ValidationResult(
                        validator_name,
                        Outcome.FAIL,
                        f"{fdef.path} value {item!r} is not in the closed vocabulary",
                        {
                            "field_id": fdef.field_id,
                            "value": _clip(item),
                            "permitted": list(c.enum)[:20],
                        },
                    )
                )

        if isinstance(item, (int, float)) and not isinstance(item, bool):
            if c.minimum is not None and item < c.minimum:
                results.append(
                    ValidationResult(
                        validator_name,
                        Outcome.FAIL,
                        f"{fdef.path} value {item} is below the minimum {c.minimum}",
                        {"field_id": fdef.field_id},
                    )
                )
            if c.maximum is not None and item > c.maximum:
                results.append(
                    ValidationResult(
                        validator_name,
                        Outcome.FAIL,
                        f"{fdef.path} value {item} is above the maximum {c.maximum}",
                        {"field_id": fdef.field_id},
                    )
                )

        if isinstance(item, str):
            if c.min_length is not None and len(item) < c.min_length:
                results.append(
                    ValidationResult(
                        validator_name,
                        Outcome.FAIL,
                        f"{fdef.path} is shorter than {c.min_length} characters",
                        {"field_id": fdef.field_id},
                    )
                )
            if c.max_length is not None and len(item) > c.max_length:
                results.append(
                    ValidationResult(
                        validator_name,
                        Outcome.FAIL,
                        f"{fdef.path} is longer than {c.max_length} characters",
                        {"field_id": fdef.field_id},
                    )
                )
            if c.pattern and not re.search(c.pattern, item):
                results.append(
                    ValidationResult(
                        validator_name,
                        Outcome.FAIL,
                        f"{fdef.path} does not match {c.pattern!r}",
                        {"field_id": fdef.field_id, "value": _clip(item)},
                    )
                )

    return results


def validate_artifacts(
    artifacts: Iterable[FieldArtifact], schema: NormalizedSchema
) -> list[ValidationResult]:
    """Validate a whole record, artifact by artifact (FR-VAL-001)."""

    results: list[ValidationResult] = []
    seen: set[tuple[str, str]] = set()
    by_entity: dict[str, set[str]] = {}

    for artifact in artifacts:
        fdef = schema.get(artifact.field_id)
        if fdef is None:
            results.append(
                ValidationResult(
                    "structural",
                    Outcome.FAIL,
                    f"record carries {artifact.field_id!r}, which {schema.ref} does not define",
                    {"field_id": artifact.field_id, "entity": artifact.entity},
                )
            )
            continue
        seen.add((artifact.field_id, artifact.entity))
        by_entity.setdefault(artifact.entity, set()).add(artifact.field_id)
        for r in validate_field_value(artifact.value, fdef):
            if r.outcome is not Outcome.PASS:
                results.append(
                    ValidationResult(
                        r.validator,
                        r.outcome,
                        r.reason,
                        {**r.details, "entity": artifact.entity},
                    )
                )

    for fdef in schema.fields:
        if not fdef.required:
            continue
        scope = fdef.collection_path
        entities = [e for e in by_entity if _in_scope(e, scope)]
        for entity in entities or ([""] if not scope else []):
            if fdef.field_id not in by_entity.get(entity, set()):
                results.append(
                    ValidationResult(
                        "structural",
                        Outcome.FAIL,
                        f"{fdef.path} is required but no value is present",
                        {"field_id": fdef.field_id, "entity": entity},
                    )
                )

    return results


def run_semantic_validators(
    names: Sequence[str], ctx: ValidationContext
) -> list[ValidationResult]:
    """Run registered validators, normalising whatever they return."""

    out: list[ValidationResult] = []
    for name in names:
        fn = VALIDATORS.get(name)
        raw = fn(ctx)
        out.append(_normalise(name, raw))
    return out


def _normalise(name: str, raw: Any) -> ValidationResult:
    if isinstance(raw, ValidationResult):
        return raw
    if raw is True or raw is None:
        return ValidationResult(name, Outcome.PASS)
    if raw is False:
        return ValidationResult(name, Outcome.FAIL, f"{name} returned False")
    if isinstance(raw, str):
        return ValidationResult(name, Outcome.FAIL, raw)
    if isinstance(raw, tuple) and len(raw) == 2:
        outcome, reason = raw
        return ValidationResult(name, Outcome(outcome), str(reason))
    return ValidationResult(name, Outcome.FAIL, f"{name} returned {raw!r}")


def worst(results: Sequence[ValidationResult]) -> Outcome:
    if any(r.outcome is Outcome.FAIL for r in results):
        return Outcome.FAIL
    if any(r.outcome is Outcome.REVIEW for r in results):
        return Outcome.REVIEW
    return Outcome.PASS


def _in_scope(entity: str, collection_path: str) -> bool:
    if not collection_path:
        return entity == ""
    return entity.rsplit("/", 1)[-1].split("=", 1)[0] == collection_path


def _clip(value: Any, limit: int = 120) -> Any:
    text = value if isinstance(value, str) else repr(value)
    return text if len(text) <= limit else text[:limit] + "..."


# ---- validators shipped with llmbic --------------------------------------

@VALIDATORS.register("require_evidence")
def _require_evidence(ctx: ValidationContext) -> ValidationResult:
    """A value that claims to be reported must cite something.

    Product principle 4 in validator form: a generated value with no span is
    not necessarily wrong, but it is not supported either, and the record
    should say which it is rather than looking the same as a quoted one.
    """

    if not ctx.value.status.has_value:
        return ValidationResult("require_evidence", Outcome.PASS, "no value to support")
    if any(ref.spans for ref in ctx.evidence):
        return ValidationResult("require_evidence", Outcome.PASS)
    return ValidationResult(
        "require_evidence",
        Outcome.REVIEW,
        f"{ctx.field_id} has a value but no supporting span",
        {"field_id": ctx.field_id, "entity": ctx.entity},
    )


@VALIDATORS.register("evidence_supports_value")
def _evidence_supports_value(ctx: ValidationContext) -> ValidationResult:
    """The cited span must actually contain the value, for string values.

    Catches the failure mode §19 calls "unsupported values": a model that
    answers correctly but cites a sentence that does not say so.
    """

    if not ctx.value.status.has_value or not isinstance(ctx.value.value, str):
        return ValidationResult("evidence_supports_value", Outcome.PASS)
    texts = [t.lower() for t in ctx.evidence_text() if t]
    if not texts:
        return ValidationResult(
            "evidence_supports_value",
            Outcome.REVIEW,
            "no resolvable evidence text to check the value against",
        )
    needle = str(ctx.value.value).lower()
    if any(needle in t for t in texts):
        return ValidationResult("evidence_supports_value", Outcome.PASS)
    return ValidationResult(
        "evidence_supports_value",
        Outcome.REVIEW,
        f"none of the cited spans contain {ctx.value.value!r}",
        {"field_id": ctx.field_id, "spans": len(texts)},
    )


@VALIDATORS.register("in_vocabulary")
def _in_vocabulary(ctx: ValidationContext) -> ValidationResult:
    """Value must be a permissible value, unless the vocabulary is open."""

    c = ctx.field_def.constraints
    if not c.enum or not ctx.value.status.has_value:
        return ValidationResult("in_vocabulary", Outcome.PASS)
    values = ctx.value.value if ctx.field_def.multivalued else [ctx.value.value]
    unknown = [v for v in values if isinstance(v, str) and v not in c.enum]
    if not unknown:
        return ValidationResult("in_vocabulary", Outcome.PASS)
    if c.open_vocabulary:
        return ValidationResult(
            "in_vocabulary",
            Outcome.PASS,
            f"open vocabulary permits the source's own wording: {unknown}",
        )
    return ValidationResult(
        "in_vocabulary",
        Outcome.FAIL,
        f"{unknown} not permitted by {c.vocabulary_ref or ctx.field_def.path}",
        {"unknown": unknown, "permitted": list(c.enum)[:20]},
    )


@VALIDATORS.register("abstention_is_honest")
def _abstention_is_honest(ctx: ValidationContext) -> ValidationResult:
    """An abstention must not carry evidence claiming a value was found."""

    if ctx.value.status.has_value:
        return ValidationResult("abstention_is_honest", Outcome.PASS)
    if any(ref.spans for ref in ctx.evidence):
        return ValidationResult(
            "abstention_is_honest",
            Outcome.REVIEW,
            f"{ctx.field_id} is {ctx.value.status.value} but cites supporting spans",
        )
    return ValidationResult("abstention_is_honest", Outcome.PASS)


__all__ = [
    "ValidationContext",
    "run_semantic_validators",
    "validate_artifacts",
    "validate_field_value",
    "worst",
]
