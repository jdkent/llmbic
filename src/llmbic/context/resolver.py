"""Resolving a :class:`ContextPolicy` into the exact units a model will see.

The resolver is deliberately *pure and offline*: it reads the parsed source,
stored evidence and prior field values, and it never calls a billable model.
That is what lets FR-PLN-005 hold — a dry run can report exactly which units
would be transmitted, and their hashes, without spending anything.

An LLM-backed selector (FR-CTX-008) plugs in through :func:`register_retriever`
and is only invoked when ``allow_model_selectors=True``, which the planner
never passes during an ordinary dry run.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from ..errors import ContextError, ErrorCode, PolicyError
from ..ids import content_hash, text_hash
from ..provenance import FieldArtifact
from ..source import DocumentUnit, EvidenceReference, ParsedSource, SourceArtifact
from .policy import ContextPolicy, ContextSource, OnBudget, OnMissingContext


@dataclass(frozen=True)
class ContextUnit:
    """One addressable piece of context, document-derived or synthesised."""

    unit_id: str
    text: str
    kind: str
    origin: str
    section: str | None = None
    unit_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_chars(self) -> int:
        return len(self.text)

    def to_canonical(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "kind": self.kind,
            "origin": self.origin,
            "section": self.section,
            "unit_hash": self.unit_hash or text_hash(self.text),
            "n_chars": self.n_chars,
        }

    @classmethod
    def from_document_unit(cls, unit: DocumentUnit, origin: str) -> "ContextUnit":
        return cls(
            unit_id=unit.unit_id,
            text=unit.text,
            kind=unit.kind.value,
            origin=origin,
            section=unit.section,
            unit_hash=unit.content_hash(),
            metadata=dict(unit.metadata),
        )


@dataclass(frozen=True)
class ContextResolution:
    """What a policy actually selected, and what it cost to select it."""

    units: tuple[ContextUnit, ...] = ()
    satisfied: bool = False
    #: Selector refs that contributed units, in order.
    used_sources: tuple[str, ...] = ()
    #: Every selector tried, including ones that yielded nothing.
    attempted: tuple[str, ...] = ()
    truncated: bool = False
    blocked_reason: str | None = None
    blocked_code: str | None = None
    includes_full_document: bool = False
    selector_hashes: tuple[str, ...] = ()
    #: Index, within the policy's effective sequence, of the last source that
    #: contributed.  ``None`` when nothing did.  A step asking for more context
    #: resumes from the next position.
    used_index: int | None = None

    @property
    def n_chars(self) -> int:
        return sum(u.n_chars for u in self.units)

    def estimate_tokens(self, chars_per_token: float = 4.0) -> int:
        return int(math.ceil(self.n_chars / chars_per_token)) if self.units else 0

    @property
    def unit_ids(self) -> tuple[str, ...]:
        return tuple(u.unit_id for u in self.units)

    def context_hash(self) -> str:
        """Identity of the exact context supplied — part of every cache key."""

        return content_hash([u.to_canonical() for u in self.units])

    def render(self, separator: str = "\n\n---\n\n") -> str:
        return separator.join(
            f"[{u.unit_id}{' ' + u.section if u.section else ''}]\n{u.text}" for u in self.units
        )

    def to_canonical(self) -> dict[str, Any]:
        return {
            "units": [u.to_canonical() for u in self.units],
            "satisfied": self.satisfied,
            "used_sources": list(self.used_sources),
            "attempted": list(self.attempted),
            "truncated": self.truncated,
            "blocked_reason": self.blocked_reason,
            "blocked_code": self.blocked_code,
            "includes_full_document": self.includes_full_document,
            "context_hash": self.context_hash(),
            "n_chars": self.n_chars,
        }


@dataclass
class SelectionInput:
    """Everything a selector is allowed to read."""

    record_id: str
    parsed: ParsedSource | None = None
    source: SourceArtifact | None = None
    #: Stored evidence for the fields this step declares it reads.
    prior_evidence: tuple[EvidenceReference, ...] = ()
    #: ``field_id -> artifact`` for the structured values this step reads.
    prior_fields: Mapping[str, FieldArtifact] = field(default_factory=dict)
    field_id: str | None = None
    entity: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class Retriever(Protocol):
    """Pluggable passage selection (FR-CTX-008, phase 3)."""

    #: True when the retriever itself calls a billable model.
    billable: bool

    def retrieve(
        self, query: str, units: Sequence[DocumentUnit], top_k: int
    ) -> list[DocumentUnit]: ...


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class LexicalRetriever:
    """Deterministic BM25-lite ranking.  No network, no model, no cost."""

    billable = False
    k1 = 1.2
    b = 0.75

    def retrieve(
        self, query: str, units: Sequence[DocumentUnit], top_k: int
    ) -> list[DocumentUnit]:
        if not units:
            return []
        q = _tokens(query)
        if not q:
            return []
        docs = [_tokens(u.text) for u in units]
        avg_len = sum(len(d) for d in docs) / len(docs) or 1.0
        df = Counter()
        for d in docs:
            for term in set(d):
                df[term] += 1
        n = len(docs)

        scored: list[tuple[float, int, DocumentUnit]] = []
        for i, (unit, doc) in enumerate(zip(units, docs)):
            tf = Counter(doc)
            score = 0.0
            for term in q:
                if term not in tf:
                    continue
                idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
                numer = tf[term] * (self.k1 + 1)
                denom = tf[term] + self.k1 * (1 - self.b + self.b * len(doc) / avg_len)
                score += idf * numer / denom
            if score > 0:
                scored.append((score, i, unit))
        # Ties break on document order, so the selection is reproducible.
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [u for _, _, u in scored[:top_k]]


_RETRIEVERS: dict[str, Retriever] = {"lexical": LexicalRetriever()}


def register_retriever(name: str, retriever: Retriever) -> None:
    _RETRIEVERS[name] = retriever


def get_retriever(name: str) -> Retriever:
    if name not in _RETRIEVERS:
        raise ContextError(
            f"unknown retriever {name!r}; registered: {sorted(_RETRIEVERS)}",
            code=ErrorCode.CONFIG_INVALID,
        )
    return _RETRIEVERS[name]


SelectorFn = Callable[[ContextSource, SelectionInput], list[ContextUnit]]
_SELECTORS: dict[str, SelectorFn] = {}


def selector(kind: str) -> Callable[[SelectorFn], SelectorFn]:
    def decorate(fn: SelectorFn) -> SelectorFn:
        _SELECTORS[kind] = fn
        return fn

    return decorate


# ---- built-in selectors ------------------------------------------------

@selector("none")
def _select_none(src: ContextSource, inp: SelectionInput) -> list[ContextUnit]:
    return []


@selector("structured_fields")
def _select_structured(src: ContextSource, inp: SelectionInput) -> list[ContextUnit]:
    wanted = src.params.get("fields") or sorted(inp.prior_fields)
    include_status = src.params.get("include_status", True)
    out: list[ContextUnit] = []
    for fid in wanted:
        artifact = inp.prior_fields.get(fid)
        if artifact is None:
            continue
        value = artifact.value
        body = f"{fid}: {value.value!r}" if value.status.has_value else f"{fid}: <no value>"
        if include_status:
            body += f"  [status={value.status.value}]"
        out.append(
            ContextUnit(
                unit_id=f"field:{fid}@{artifact.entity or 'root'}",
                text=body,
                kind="structured_field",
                origin=src.selector_ref,
                unit_hash=artifact.value_hash(),
            )
        )
    return out


@selector("prior_evidence")
def _select_prior_evidence(src: ContextSource, inp: SelectionInput) -> list[ContextUnit]:
    if inp.parsed is None:
        return []
    expand = int(src.params.get("expand_chars", 0))
    out: list[ContextUnit] = []
    seen: set[str] = set()
    for ref in inp.prior_evidence:
        for span in ref.spans:
            unit = inp.parsed.unit(span.unit_id)
            if unit is None:
                continue
            start = max(0, span.start_char - expand)
            end = min(len(unit.text), span.end_char + expand)
            text = unit.text[start:end]
            if not text:
                continue
            uid = f"{unit.unit_id}:{start}-{end}"
            if uid in seen:
                continue
            seen.add(uid)
            out.append(
                ContextUnit(
                    unit_id=uid,
                    text=text,
                    kind="evidence_span",
                    origin=src.selector_ref,
                    section=unit.section,
                    unit_hash=text_hash(text),
                    metadata={"locator": ref.locator} if ref.locator else {},
                )
            )
    return out


@selector("evidence_window")
def _select_evidence_window(src: ContextSource, inp: SelectionInput) -> list[ContextUnit]:
    widened = ContextSource(
        "prior_evidence",
        {"expand_chars": int(src.params.get("chars", 500))},
        version=src.version,
    )
    units = _select_prior_evidence(widened, inp)
    return [
        ContextUnit(
            u.unit_id, u.text, "evidence_window", src.selector_ref, u.section, u.unit_hash, u.metadata
        )
        for u in units
    ]


@selector("sections")
def _select_sections(src: ContextSource, inp: SelectionInput) -> list[ContextUnit]:
    if inp.parsed is None:
        return []
    names = src.params.get("names") or []
    units = inp.parsed.units_in_sections(names)
    return [ContextUnit.from_document_unit(u, src.selector_ref) for u in units]


@selector("unit_kinds")
def _select_unit_kinds(src: ContextSource, inp: SelectionInput) -> list[ContextUnit]:
    if inp.parsed is None:
        return []
    kinds = src.params.get("kinds") or []
    try:
        units = inp.parsed.units_of_kind(kinds)
    except ValueError as exc:
        raise ContextError(str(exc), code=ErrorCode.CONFIG_INVALID) from exc
    return [ContextUnit.from_document_unit(u, src.selector_ref) for u in units]


@selector("retrieve")
def _select_retrieve(src: ContextSource, inp: SelectionInput) -> list[ContextUnit]:
    if inp.parsed is None:
        return []
    retriever = get_retriever(str(src.params.get("retriever", "lexical")))
    restrict = src.params.get("sections")
    pool: Sequence[DocumentUnit]
    if restrict:
        pool = inp.parsed.units_in_sections(restrict)
    else:
        pool = inp.parsed.units
    hits = retriever.retrieve(
        str(src.params.get("query", "")), pool, int(src.params.get("top_k", 8))
    )
    return [ContextUnit.from_document_unit(u, src.selector_ref) for u in hits]


@selector("full_document")
def _select_full_document(src: ContextSource, inp: SelectionInput) -> list[ContextUnit]:
    if inp.parsed is None:
        return []
    return [ContextUnit.from_document_unit(u, src.selector_ref) for u in inp.parsed.units]


@selector("raw_document")
def _select_raw_document(src: ContextSource, inp: SelectionInput) -> list[ContextUnit]:
    if inp.source is None:
        return []
    return [
        ContextUnit(
            unit_id=f"raw:{inp.source.ref}",
            text="",
            kind="raw_document",
            origin=src.selector_ref,
            unit_hash=inp.source.content_hash,
            metadata={"media_type": inp.source.media_type, "uri": inp.source.uri},
        )
    ]


_BILLABLE_KINDS = {"retrieve"}


def _is_billable(src: ContextSource) -> bool:
    if src.kind != "retrieve":
        return False
    return getattr(get_retriever(str(src.params.get("retriever", "lexical"))), "billable", False)


def resolve_context(
    policy: ContextPolicy,
    inp: SelectionInput,
    *,
    allow_model_selectors: bool = False,
    start_at: int = 0,
) -> ContextResolution:
    """Walk the fallback chain and return the first satisfying selection.

    Never widens silently: the full document is only reachable when the policy
    names it or sets ``full_document_fallback`` to something other than
    ``forbidden`` (FR-CTX-003).

    ``start_at`` skips the first *n* sources.  It is how a step that asked for
    more context gets it: the engine advances one position down the chain the
    migration already declared, rather than reaching for anything the policy
    did not authorise.  The chain's end is still the end.
    """

    attempted: list[str] = []
    used: list[str] = []
    selector_hashes: list[str] = []
    accumulated: list[ContextUnit] = []
    truncated = False
    includes_full = False

    sequence = policy.effective_sequence[start_at:]
    if not sequence:
        return ContextResolution(
            satisfied=policy.on_missing_context is OnMissingContext.PROCEED,
            attempted=(),
            blocked_reason=None
            if policy.on_missing_context is OnMissingContext.PROCEED
            else "context policy declares no sources",
            blocked_code=None
            if policy.on_missing_context is OnMissingContext.PROCEED
            else ErrorCode.CONTEXT_UNAVAILABLE.value,
        )

    used_index: int | None = None
    for offset, src in enumerate(sequence):
        index = start_at + offset
        attempted.append(src.selector_ref)
        if src.kind not in _SELECTORS:
            raise ContextError(
                f"unknown context source kind {src.kind!r}", code=ErrorCode.CONFIG_INVALID
            )
        if _is_billable(src) and not allow_model_selectors:
            # A dry run must not spend money resolving context (FR-PLN-005).
            continue

        units = _SELECTORS[src.kind](src, inp)
        units = _drop_forbidden(units, policy)

        if src.kind in ("full_document", "raw_document"):
            if not policy.permits_full_document():
                continue
            includes_full = includes_full or bool(units)

        if len(units) < src.min_units:
            continue

        candidate = (accumulated + units) if policy.accumulate else units
        kept, was_truncated, overflow = _apply_budget(candidate, policy)
        if overflow and policy.on_budget is OnBudget.FAIL:
            return ContextResolution(
                units=tuple(kept),
                satisfied=False,
                used_sources=tuple(used),
                attempted=tuple(attempted),
                truncated=True,
                blocked_reason=(
                    f"selector {src.selector_ref} exceeds the context budget "
                    f"({len(candidate)} units, {sum(u.n_chars for u in candidate)} chars)"
                ),
                blocked_code=ErrorCode.CONTEXT_BUDGET_EXCEEDED.value,
                includes_full_document=includes_full,
                selector_hashes=tuple(selector_hashes),
            )
        if overflow and policy.on_budget is OnBudget.SKIP:
            continue

        truncated = truncated or was_truncated
        selector_hashes.append(src.selector_hash())
        used.append(src.selector_ref)
        used_index = index
        accumulated = kept

        if not policy.accumulate and kept:
            return ContextResolution(
                units=tuple(kept),
                satisfied=True,
                used_sources=tuple(used),
                attempted=tuple(attempted),
                truncated=truncated,
                includes_full_document=includes_full,
                selector_hashes=tuple(selector_hashes),
                used_index=used_index,
            )
        if not policy.accumulate and src.min_units == 0:
            # ``none`` satisfies the policy by design.
            return ContextResolution(
                units=(),
                satisfied=True,
                used_sources=tuple(used),
                attempted=tuple(attempted),
                includes_full_document=False,
                selector_hashes=tuple(selector_hashes),
                used_index=used_index,
            )

    if accumulated:
        return ContextResolution(
            units=tuple(accumulated),
            satisfied=True,
            used_sources=tuple(used),
            attempted=tuple(attempted),
            truncated=truncated,
            includes_full_document=includes_full,
            selector_hashes=tuple(selector_hashes),
            used_index=used_index,
        )

    if policy.on_missing_context is OnMissingContext.PROCEED:
        return ContextResolution(
            units=(),
            satisfied=True,
            used_sources=(),
            attempted=tuple(attempted),
            selector_hashes=tuple(selector_hashes),
        )

    return ContextResolution(
        units=(),
        satisfied=False,
        used_sources=(),
        attempted=tuple(attempted),
        blocked_reason=(
            "no context source satisfied the policy; tried " + ", ".join(attempted)
        ),
        blocked_code=ErrorCode.CONTEXT_UNAVAILABLE.value,
        selector_hashes=tuple(selector_hashes),
    )


def _drop_forbidden(units: list[ContextUnit], policy: ContextPolicy) -> list[ContextUnit]:
    """Enforce unit-level privacy before anything is assembled (NFR-SEC-003)."""

    forbidden = set(policy.privacy.forbidden_unit_labels)
    if not forbidden:
        return units
    kept = []
    for u in units:
        labels = set(u.metadata.get("labels") or ())
        if labels & forbidden:
            continue
        kept.append(u)
    return kept


def _apply_budget(
    units: list[ContextUnit], policy: ContextPolicy
) -> tuple[list[ContextUnit], bool, bool]:
    """Return ``(kept, truncated, overflowed)``."""

    budget = policy.budget
    kept: list[ContextUnit] = []
    chars = 0
    sections: set[str] = set()
    overflow = False

    for u in units:
        if budget.max_units is not None and len(kept) >= budget.max_units:
            overflow = True
            break
        next_chars = chars + u.n_chars
        if budget.max_chars is not None and next_chars > budget.max_chars:
            overflow = True
            break
        if (
            budget.max_input_tokens is not None
            and budget.estimate_tokens(next_chars) > budget.max_input_tokens
        ):
            overflow = True
            break
        next_sections = sections | ({u.section} if u.section else set())
        if budget.max_sections is not None and len(next_sections) > budget.max_sections:
            overflow = True
            break
        kept.append(u)
        chars = next_chars
        sections = next_sections

    return kept, overflow, overflow


def check_provider(policy: ContextPolicy, provider_id: str, attributes: Mapping[str, Any]) -> None:
    """Raise when a provider is not permitted to see this context."""

    ok, reason = policy.privacy.permits_provider(provider_id, dict(attributes))
    if not ok:
        raise PolicyError(reason, code=ErrorCode.PRIVACY_POLICY_FORBIDS)


def evidence_for(
    artifacts: Iterable[FieldArtifact], field_ids: Iterable[str]
) -> tuple[EvidenceReference, ...]:
    wanted = set(field_ids)
    out: list[EvidenceReference] = []
    for a in artifacts:
        if a.field_id in wanted:
            out.extend(a.evidence)
    return tuple(out)


__all__ = [
    "ContextResolution",
    "ContextUnit",
    "LexicalRetriever",
    "Retriever",
    "SelectionInput",
    "check_provider",
    "evidence_for",
    "get_retriever",
    "register_retriever",
    "resolve_context",
    "selector",
]
