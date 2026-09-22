"""Extraction recipes — the versioned logic that produces a field.

Terminology §6 keeps the *extraction recipe version* separate from the schema
version, and FR-DEP-004 makes each of a recipe's parts independently capable of
invalidating a value: the prompt, the model policy, the context policy, the
validators, the postprocessor and the vocabulary.  :meth:`ExtractionRecipe.recipe_hash`
covers exactly those, which is why changing a prompt invalidates only the
artifacts produced by that recipe (acceptance criterion 5) and reflowing a
comment invalidates nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from .context.policy import ContextPolicy, policy_from_spec
from .functions import POSTPROCESSORS, PROMPTS, VALIDATORS
from .ids import content_hash, text_hash
from .models.base import ModelPolicy


@dataclass(frozen=True)
class ConfidenceSpec:
    """How a recipe's confidence number is defined (FR-LLM-013).

    An uncalibrated model self-rating is recorded as exactly that.  Nothing in
    llmbic presents it as a probability unless ``calibration`` names a method
    and ``calibrated_on`` names the sample it was fitted to.
    """

    emitted: bool = False
    definition: str = ""
    calibration: str | None = None
    calibrated_on: str | None = None

    @property
    def is_calibrated(self) -> bool:
        return bool(self.calibration and self.calibrated_on)

    def to_canonical(self) -> dict[str, Any]:
        return {
            "emitted": self.emitted,
            "definition": self.definition,
            "calibration": self.calibration,
            "calibrated_on": self.calibrated_on,
        }

    @classmethod
    def from_canonical(cls, data: Mapping[str, Any]) -> "ConfidenceSpec":
        return cls(
            emitted=bool(data.get("emitted", False)),
            definition=data.get("definition", ""),
            calibration=data.get("calibration"),
            calibrated_on=data.get("calibrated_on"),
        )


@dataclass(frozen=True)
class ExtractionRecipe:
    """Prompt + model + context + validation + postprocessing, versioned."""

    name: str
    version: str
    #: Fields this recipe produces.  A recipe producing several fields at once
    #: is a *field group*: the group is the unit of computation, which is how
    #: interdependent slots (a value and its unit, say) stay consistent.
    writes: tuple[str, ...] = ()
    prompt_template: str = ""
    system: str = ""
    #: JSON Schema the model's answer must satisfy (FR-LLM-001).
    output_schema: dict[str, Any] = field(default_factory=dict)
    model_policy: ModelPolicy | None = None
    context_policy: ContextPolicy = field(default_factory=ContextPolicy)
    validators: tuple[str, ...] = ()
    postprocessor: str | None = None
    prompt_builder: str | None = None
    vocabulary_ref: str | None = None
    confidence: ConfidenceSpec = field(default_factory=ConfidenceSpec)
    #: Structured field values the prompt interpolates, by field id.
    reads_fields: tuple[str, ...] = ()
    description: str = ""

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.version}"

    # ---- identity --------------------------------------------------------
    @property
    def prompt_hash(self) -> str:
        return text_hash(self.prompt_template + "\x00" + self.system)

    def recipe_hash(self) -> str:
        """Everything that can change the answer.

        Deliberately excludes ``description`` and ``writes`` ordering: neither
        reaches the model.
        """

        return content_hash(
            {
                "name": self.name,
                "version": self.version,
                "prompt": self.prompt_hash,
                "output_schema": self.output_schema,
                "model_policy": self.model_policy.to_canonical() if self.model_policy else None,
                "context_policy": self.context_policy.to_canonical(),
                "validators": sorted(self.validators),
                "validator_code": _code_hashes(VALIDATORS, self.validators),
                "postprocessor": self.postprocessor,
                "postprocessor_code": _code_hashes(
                    POSTPROCESSORS, (self.postprocessor,) if self.postprocessor else ()
                ),
                "prompt_builder": self.prompt_builder,
                "prompt_builder_code": _code_hashes(
                    PROMPTS, (self.prompt_builder,) if self.prompt_builder else ()
                ),
                "vocabulary_ref": self.vocabulary_ref,
                "confidence": self.confidence.to_canonical(),
                "reads_fields": sorted(self.reads_fields),
            }
        )

    def bump(self, version: str, **changes: Any) -> "ExtractionRecipe":
        """A new version of the same recipe — the ordinary way to edit a prompt."""

        return replace(self, version=version, **changes)

    def render_prompt(self, variables: Mapping[str, Any] | None = None) -> str:
        if self.prompt_builder:
            return str(PROMPTS.get(self.prompt_builder)(self, dict(variables or {})))
        try:
            return self.prompt_template.format(**dict(variables or {}))
        except KeyError as exc:
            raise KeyError(
                f"prompt for {self.ref} references {exc.args[0]!r}, which the step did not supply"
            ) from exc

    def to_canonical(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "writes": list(self.writes),
            "prompt_template": self.prompt_template,
            "system": self.system,
            "output_schema": self.output_schema,
            "model_policy": self.model_policy.to_canonical() if self.model_policy else None,
            "context_policy": self.context_policy.to_canonical(),
            "validators": list(self.validators),
            "postprocessor": self.postprocessor,
            "prompt_builder": self.prompt_builder,
            "vocabulary_ref": self.vocabulary_ref,
            "confidence": self.confidence.to_canonical(),
            "reads_fields": list(self.reads_fields),
            "description": self.description,
        }

    @classmethod
    def from_canonical(cls, data: Mapping[str, Any]) -> "ExtractionRecipe":
        return cls(
            name=data["name"],
            version=str(data["version"]),
            writes=tuple(data.get("writes") or ()),
            prompt_template=data.get("prompt_template", ""),
            system=data.get("system", ""),
            output_schema=dict(data.get("output_schema") or {}),
            model_policy=ModelPolicy.from_canonical(data["model_policy"])
            if data.get("model_policy")
            else None,
            context_policy=ContextPolicy.from_canonical(data.get("context_policy") or {})
            if isinstance(data.get("context_policy"), dict)
            and "sequence" in (data.get("context_policy") or {})
            else policy_from_spec(data.get("context_policy")),
            validators=tuple(data.get("validators") or ()),
            postprocessor=data.get("postprocessor"),
            prompt_builder=data.get("prompt_builder"),
            vocabulary_ref=data.get("vocabulary_ref"),
            confidence=ConfidenceSpec.from_canonical(data.get("confidence") or {}),
            reads_fields=tuple(data.get("reads_fields") or ()),
            description=data.get("description", ""),
        )


def _code_hashes(registry: Any, refs: Sequence[str | None]) -> dict[str, str]:
    """Hash the *code* behind named validators/postprocessors when available.

    A validator whose body changed without a version bump still changes the
    recipe hash, which is what "test invalidation" in §19 asks for.  A name
    that is not registered in this process contributes nothing rather than
    raising: a deterministic-only run must not need the semantic extensions
    loaded (FR-STO-005).
    """

    out: dict[str, str] = {}
    for ref in refs:
        if not ref:
            continue
        entry = registry.maybe(ref)
        if entry is not None:
            out[ref] = entry.code_hash
    return out


@dataclass(frozen=True)
class Vocabulary:
    """A controlled vocabulary with its own version (terminology §6)."""

    name: str
    version: str
    values: tuple[str, ...] = ()
    #: ``old_value -> new_value`` for values this version renamed.
    replaces: dict[str, str] = field(default_factory=dict)
    #: Values that this version made ambiguous and that must be re-decided.
    reopens: tuple[str, ...] = ()
    descriptions: dict[str, str] = field(default_factory=dict)

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.version}"

    def vocabulary_hash(self) -> str:
        return content_hash(
            {
                "name": self.name,
                "version": self.version,
                "values": sorted(self.values),
                "replaces": dict(sorted(self.replaces.items())),
                "reopens": sorted(self.reopens),
                "descriptions": {k: v for k, v in sorted(self.descriptions.items())},
            }
        )

    def contains(self, value: str) -> bool:
        return value in self.values

    def to_canonical(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "values": list(self.values),
            "replaces": dict(sorted(self.replaces.items())),
            "reopens": list(self.reopens),
            "descriptions": dict(sorted(self.descriptions.items())),
        }

    @classmethod
    def from_canonical(cls, data: Mapping[str, Any]) -> "Vocabulary":
        return cls(
            name=data["name"],
            version=str(data["version"]),
            values=tuple(data.get("values") or ()),
            replaces=dict(data.get("replaces") or {}),
            reopens=tuple(data.get("reopens") or ()),
            descriptions=dict(data.get("descriptions") or {}),
        )


def load_recipes(path: Any) -> list[ExtractionRecipe]:
    """Read recipes from a YAML file, or from a ``module:attr`` factory.

    A recipe is ordinary version-controlled configuration, so the common case
    is a file; the dotted form exists because a prompt built from code is
    sometimes the honest way to write one.
    """

    return _load(path, "recipes", ExtractionRecipe)


def load_vocabularies(path: Any) -> list[Vocabulary]:
    return _load(path, "vocabularies", Vocabulary)


def _load(path: Any, key: str, cls: Any) -> list[Any]:
    import importlib
    from pathlib import Path

    import yaml

    text = str(path)
    if ":" in text and not Path(text).exists():
        module_name, _, attr = text.partition(":")
        factory = getattr(importlib.import_module(module_name), attr)
        produced = factory() if callable(factory) else factory
        return list(produced) if isinstance(produced, (list, tuple)) else [produced]

    docs = [d for d in yaml.safe_load_all(Path(text).read_text(encoding="utf-8")) if d]
    out: list[Any] = []
    for doc in docs:
        if isinstance(doc, list):
            out.extend(cls.from_canonical(d) for d in doc)
        elif key in doc:
            out.extend(cls.from_canonical(d) for d in doc[key])
        else:
            out.append(cls.from_canonical(doc))
    return out


__all__ = [
    "ConfidenceSpec",
    "ExtractionRecipe",
    "Vocabulary",
    "load_recipes",
    "load_vocabularies",
]
