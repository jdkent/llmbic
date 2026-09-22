"""Context policies: what a semantic migration is allowed to look at.

FR-CTX-001..010.  A policy is an *ordered fallback chain*, never an implicit
widening: each source is tried in turn and the first one that satisfies the
policy wins.  Sending the whole article is a separate, explicitly-declared
decision (``full_document``) that most policies forbid.

Every source is a versioned selector with its own identity, so changing how
evidence windows are cut invalidates exactly the artifacts that used them
(FR-CTX-005) and an LLM-backed selector is itself a semantic computation with
provenance and cost (FR-CTX-008).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

from ..ids import content_hash
from ..source import UnitKind


class FallbackMode(str, Enum):
    FORBIDDEN = "forbidden"
    ALLOWED = "allowed"
    REQUIRED = "required"


class OnMissingContext(str, Enum):
    """What to do when no source in the chain satisfies the policy."""

    REVIEW = "review"
    FAIL = "fail"
    #: Run the step with no source context at all.  Only legitimate when the
    #: recipe can abstain, which the validator should check.
    PROCEED = "proceed"


class OnBudget(str, Enum):
    #: Keep units in selection order until the budget is spent.
    TRUNCATE = "truncate"
    #: Treat the source as unsatisfied and fall through to the next one.
    SKIP = "skip"
    FAIL = "fail"


@dataclass(frozen=True)
class ContextSource:
    """One step of a fallback chain.

    A dataclass rather than a class hierarchy so a policy round-trips through
    YAML and a plan without needing a registry of subclasses.  ``kind`` selects
    the resolver; ``params`` is the resolver's configuration; ``version`` is
    part of the selector identity and therefore of the cache key.
    """

    kind: str
    params: dict[str, Any] = field(default_factory=dict)
    version: str = "1"
    #: Minimum units this source must yield to count as satisfying the policy.
    min_units: int = 1

    @property
    def selector_ref(self) -> str:
        return f"{self.kind}@{self.version}"

    def selector_hash(self) -> str:
        return content_hash(
            {"kind": self.kind, "version": self.version, "params": self.params}
        )

    def to_canonical(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "version": self.version,
            "params": dict(sorted(self.params.items())),
            "min_units": self.min_units,
        }

    @classmethod
    def from_canonical(cls, data: dict[str, Any]) -> "ContextSource":
        return cls(
            kind=data["kind"],
            params=dict(data.get("params") or {}),
            version=str(data.get("version", "1")),
            min_units=int(data.get("min_units", 1)),
        )

    def describe(self) -> str:
        if not self.params:
            return self.kind
        bits = ", ".join(f"{k}={v!r}" for k, v in sorted(self.params.items()))
        return f"{self.kind}({bits})"


# ---- constructors for the FR-CTX-002 source list ------------------------

def no_context() -> ContextSource:
    return ContextSource("none", min_units=0)


def structured_fields(*field_ids: str, include_status: bool = True) -> ContextSource:
    """Existing structured values, rendered as auditable units."""

    return ContextSource(
        "structured_fields",
        {"fields": list(field_ids), "include_status": include_status},
    )


def prior_evidence(expand_chars: int = 0) -> ContextSource:
    """Stored evidence spans for the fields this step reads.

    Product principle 4 — evidence before regeneration — is spelled here: this
    is the source a well-written policy tries first.
    """

    return ContextSource("prior_evidence", {"expand_chars": expand_chars})


def sections(*names: str) -> ContextSource:
    return ContextSource("sections", {"names": [n.lower() for n in names]})


def units_of_kind(*kinds: str | UnitKind) -> ContextSource:
    return ContextSource(
        "unit_kinds",
        {"kinds": [k.value if isinstance(k, UnitKind) else str(k) for k in kinds]},
    )


def evidence_window(chars: int = 500) -> ContextSource:
    """A deterministic text window around stored evidence."""

    return ContextSource("evidence_window", {"chars": chars})


def retrieve(
    query: str,
    top_k: int = 8,
    *,
    retriever: str = "lexical",
    version: str = "1",
) -> ContextSource:
    return ContextSource(
        "retrieve",
        {"query": query, "top_k": top_k, "retriever": retriever},
        version=version,
    )


def full_document() -> ContextSource:
    return ContextSource("full_document")


def raw_document() -> ContextSource:
    """The original binary, for adapters that accept it."""

    return ContextSource("raw_document")


@dataclass(frozen=True)
class ContextBudget:
    """Token, character, section and monetary ceilings (FR-CTX-006)."""

    max_input_tokens: int | None = None
    max_chars: int | None = None
    max_units: int | None = None
    max_sections: int | None = None
    max_cost_usd: float | None = None
    #: Characters per token for estimation when no tokeniser is configured.
    chars_per_token: float = 4.0

    def estimate_tokens(self, n_chars: int) -> int:
        return int(n_chars / self.chars_per_token + 0.999)

    def to_canonical(self) -> dict[str, Any]:
        return {
            "max_input_tokens": self.max_input_tokens,
            "max_chars": self.max_chars,
            "max_units": self.max_units,
            "max_sections": self.max_sections,
            "max_cost_usd": self.max_cost_usd,
            "chars_per_token": self.chars_per_token,
        }

    @classmethod
    def from_canonical(cls, data: dict[str, Any]) -> "ContextBudget":
        return cls(
            max_input_tokens=data.get("max_input_tokens"),
            max_chars=data.get("max_chars"),
            max_units=data.get("max_units"),
            max_sections=data.get("max_sections"),
            max_cost_usd=data.get("max_cost_usd"),
            chars_per_token=float(data.get("chars_per_token", 4.0)),
        )


@dataclass(frozen=True)
class PrivacyPolicy:
    """Jurisdictional / data-handling restrictions (FR-CTX-009, NFR-SEC-003).

    Enforced in the resolver, before anything leaves the local trust boundary:
    a unit carrying a forbidden label is never placed in a request, and a
    provider that does not satisfy ``require_provider_attributes`` is refused
    outright rather than being sent redacted text.
    """

    #: Unit labels (from ``DocumentUnit.metadata["labels"]``) that may not be
    #: transmitted to any external provider.
    forbidden_unit_labels: tuple[str, ...] = ()
    #: Provider ids explicitly allowed; empty means "any".
    allowed_providers: tuple[str, ...] = ()
    forbidden_providers: tuple[str, ...] = ()
    #: Attributes an adapter must declare, e.g. ``{"retains_data": False}``.
    require_provider_attributes: dict[str, Any] = field(default_factory=dict)

    def permits_provider(self, provider_id: str, attributes: dict[str, Any]) -> tuple[bool, str]:
        if provider_id in self.forbidden_providers:
            return False, f"provider {provider_id!r} is forbidden by policy"
        if self.allowed_providers and provider_id not in self.allowed_providers:
            return False, f"provider {provider_id!r} is not in the allow-list"
        for key, want in sorted(self.require_provider_attributes.items()):
            if attributes.get(key) != want:
                return (
                    False,
                    f"provider {provider_id!r} declares {key}={attributes.get(key)!r}, "
                    f"policy requires {want!r}",
                )
        return True, ""

    def to_canonical(self) -> dict[str, Any]:
        return {
            "forbidden_unit_labels": list(self.forbidden_unit_labels),
            "allowed_providers": list(self.allowed_providers),
            "forbidden_providers": list(self.forbidden_providers),
            "require_provider_attributes": dict(sorted(self.require_provider_attributes.items())),
        }

    @classmethod
    def from_canonical(cls, data: dict[str, Any]) -> "PrivacyPolicy":
        return cls(
            forbidden_unit_labels=tuple(data.get("forbidden_unit_labels") or ()),
            allowed_providers=tuple(data.get("allowed_providers") or ()),
            forbidden_providers=tuple(data.get("forbidden_providers") or ()),
            require_provider_attributes=dict(data.get("require_provider_attributes") or {}),
        )


@dataclass(frozen=True)
class ContextPolicy:
    """An ordered chain plus the rules for leaving it."""

    sequence: tuple[ContextSource, ...] = ()
    full_document_fallback: FallbackMode = FallbackMode.FORBIDDEN
    on_missing_context: OnMissingContext = OnMissingContext.REVIEW
    on_budget: OnBudget = OnBudget.TRUNCATE
    budget: ContextBudget = field(default_factory=ContextBudget)
    privacy: PrivacyPolicy = field(default_factory=PrivacyPolicy)
    #: Stop at the first satisfying source (the default) or concatenate all of
    #: them.  Concatenation is still bounded by the budget.
    accumulate: bool = False
    version: str = "1"

    @property
    def policy_hash(self) -> str:
        return content_hash(self.to_canonical())

    @property
    def effective_sequence(self) -> tuple[ContextSource, ...]:
        """The chain with the full-document fallback appended when allowed."""

        seq = list(self.sequence)
        if self.full_document_fallback is FallbackMode.REQUIRED:
            return (full_document(),)
        if self.full_document_fallback is FallbackMode.ALLOWED:
            if not any(s.kind in ("full_document", "raw_document") for s in seq):
                seq.append(full_document())
        return tuple(seq)

    def permits_full_document(self) -> bool:
        return self.full_document_fallback is not FallbackMode.FORBIDDEN or any(
            s.kind in ("full_document", "raw_document") for s in self.sequence
        )

    def to_canonical(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "sequence": [s.to_canonical() for s in self.sequence],
            "full_document_fallback": self.full_document_fallback.value,
            "on_missing_context": self.on_missing_context.value,
            "on_budget": self.on_budget.value,
            "accumulate": self.accumulate,
            "budget": self.budget.to_canonical(),
            "privacy": self.privacy.to_canonical(),
        }

    @classmethod
    def from_canonical(cls, data: dict[str, Any]) -> "ContextPolicy":
        return cls(
            sequence=tuple(ContextSource.from_canonical(s) for s in data.get("sequence") or ()),
            full_document_fallback=FallbackMode(data.get("full_document_fallback", "forbidden")),
            on_missing_context=OnMissingContext(data.get("on_missing_context", "review")),
            on_budget=OnBudget(data.get("on_budget", "truncate")),
            accumulate=bool(data.get("accumulate", False)),
            budget=ContextBudget.from_canonical(data.get("budget") or {}),
            privacy=PrivacyPolicy.from_canonical(data.get("privacy") or {}),
            version=str(data.get("version", "1")),
        )

    def describe(self) -> str:
        chain = " -> ".join(s.describe() for s in self.sequence) or "(empty)"
        return (
            f"{chain} | full_document={self.full_document_fallback.value} "
            f"| on_missing={self.on_missing_context.value}"
        )


#: Context that needs no document at all — for derived fields and vocabulary
#: remaps that only read structured values.
STRUCTURED_ONLY = ContextPolicy(
    sequence=(no_context(),),
    full_document_fallback=FallbackMode.FORBIDDEN,
    on_missing_context=OnMissingContext.PROCEED,
)


def policy_from_spec(spec: dict[str, Any] | None) -> ContextPolicy:
    """Build a policy from the YAML shape shown in the requirements §10.

    ::

        context:
          sequence:
            - prior_evidence
            - sections: [methods, supplement]
            - retrieve: {query: "...", top_k: 8}
          full_document_fallback: forbidden
          max_input_tokens: 12000
    """

    if not spec:
        return ContextPolicy()
    sources: list[ContextSource] = []
    for item in spec.get("sequence") or ():
        sources.append(_source_from_spec(item))

    budget_keys = {
        "max_input_tokens",
        "max_chars",
        "max_units",
        "max_sections",
        "max_cost_usd",
        "chars_per_token",
    }
    budget = ContextBudget.from_canonical(
        {**(spec.get("budget") or {}), **{k: v for k, v in spec.items() if k in budget_keys}}
    )
    return ContextPolicy(
        sequence=tuple(sources),
        full_document_fallback=FallbackMode(spec.get("full_document_fallback", "forbidden")),
        on_missing_context=OnMissingContext(spec.get("on_missing_context", "review")),
        on_budget=OnBudget(spec.get("on_budget", "truncate")),
        accumulate=bool(spec.get("accumulate", False)),
        budget=budget,
        privacy=PrivacyPolicy.from_canonical(spec.get("privacy") or {}),
        version=str(spec.get("version", "1")),
    )


def _source_from_spec(item: Any) -> ContextSource:
    if isinstance(item, str):
        return ContextSource(item, min_units=0 if item == "none" else 1)
    if isinstance(item, dict):
        if "kind" in item:
            return ContextSource.from_canonical(item)
        if len(item) != 1:
            raise ValueError(f"ambiguous context source spec: {item!r}")
        (kind, params), = item.items()
        if isinstance(params, Sequence) and not isinstance(params, (str, bytes)):
            key = {"sections": "names", "unit_kinds": "kinds", "structured_fields": "fields"}.get(
                kind, "values"
            )
            return ContextSource(kind, {key: list(params)})
        if isinstance(params, dict):
            return ContextSource(kind, dict(params))
        return ContextSource(kind, {"value": params})
    raise ValueError(f"cannot read context source from {item!r}")
