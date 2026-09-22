"""Immutable sources and their parsed representations.

Decision 18.5 — "which parsed-document representation provides stable section,
table, figure and span identifiers" — is answered here with a deliberately
small structure: a :class:`ParsedSource` is an ordered list of
:class:`DocumentUnit`, each with an id that is stable for a given
``parse_version``.  llmbic does not parse anything itself (non-goal 4.4); a
parser adapter produces this structure.

Offsets are always relative to the *unit*, and each unit records its offset in
the whole document, so a re-parse that shifts global offsets can be
re-anchored unit by unit (FR-PROV-008) instead of invalidating every span.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Sequence

from .ids import content_hash, text_hash


class UnitKind(str, Enum):
    SECTION = "section"
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"
    TABLE = "table"
    TABLE_ROW = "table_row"
    FIGURE = "figure"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    SUPPLEMENT = "supplement"
    TITLE = "title"
    ABSTRACT = "abstract"
    OTHER = "other"


@dataclass(frozen=True)
class DocumentUnit:
    """One addressable piece of a parsed document."""

    unit_id: str
    kind: UnitKind
    text: str
    #: Normalised section name ("methods", "results", "supplement"), lowercase.
    section: str | None = None
    #: Offset of this unit's first character within the whole parsed document.
    doc_start: int = 0
    #: Ordering within the document, used for deterministic selection.
    ordinal: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def doc_end(self) -> int:
        return self.doc_start + len(self.text)

    @property
    def n_chars(self) -> int:
        return len(self.text)

    def content_hash(self) -> str:
        return text_hash(self.text)

    def to_canonical(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "kind": self.kind.value,
            "text": self.text,
            "section": self.section,
            "doc_start": self.doc_start,
            "ordinal": self.ordinal,
            "metadata": dict(sorted(self.metadata.items())),
        }

    @classmethod
    def from_canonical(cls, data: dict[str, Any]) -> "DocumentUnit":
        return cls(
            unit_id=data["unit_id"],
            kind=UnitKind(data["kind"]),
            text=data["text"],
            section=data.get("section"),
            doc_start=int(data.get("doc_start", 0)),
            ordinal=int(data.get("ordinal", 0)),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class SourceArtifact:
    """The immutable original document (FR-PROV-002, entity ``SourceArtifact``)."""

    source_id: str
    source_version: str
    content_hash: str
    uri: str | None = None
    media_type: str = "application/pdf"
    #: Jurisdiction / sensitivity labels a context policy can refuse to send.
    labels: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ref(self) -> str:
        return f"{self.source_id}@{self.source_version}"

    def to_canonical(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_version": self.source_version,
            "content_hash": self.content_hash,
            "uri": self.uri,
            "media_type": self.media_type,
            "labels": list(self.labels),
            "metadata": dict(sorted(self.metadata.items())),
        }

    @classmethod
    def from_canonical(cls, data: dict[str, Any]) -> "SourceArtifact":
        return cls(
            source_id=data["source_id"],
            source_version=data["source_version"],
            content_hash=data["content_hash"],
            uri=data.get("uri"),
            media_type=data.get("media_type", "application/pdf"),
            labels=tuple(data.get("labels") or ()),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class ParsedSource:
    """Immutable parser output with stable unit identifiers."""

    source_id: str
    source_version: str
    parse_version: str
    units: tuple[DocumentUnit, ...] = ()
    #: Sections available, in document order, for context policies to name.
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ref(self) -> str:
        return f"{self.source_id}@{self.source_version}/{self.parse_version}"

    def unit(self, unit_id: str) -> DocumentUnit | None:
        for u in self.units:
            if u.unit_id == unit_id:
                return u
        return None

    def sections(self) -> list[str]:
        seen: list[str] = []
        for u in self.units:
            if u.section and u.section not in seen:
                seen.append(u.section)
        return seen

    def units_in_sections(self, names: Iterable[str]) -> list[DocumentUnit]:
        wanted = {n.lower() for n in names}
        return [u for u in self.units if (u.section or "").lower() in wanted]

    def units_of_kind(self, kinds: Iterable[UnitKind | str]) -> list[DocumentUnit]:
        wanted = {UnitKind(k) if not isinstance(k, UnitKind) else k for k in kinds}
        return [u for u in self.units if u.kind in wanted]

    def total_chars(self) -> int:
        return sum(u.n_chars for u in self.units)

    def parse_hash(self) -> str:
        return content_hash([u.unit_id + ":" + u.content_hash() for u in self.units])

    def to_canonical(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_version": self.source_version,
            "parse_version": self.parse_version,
            "units": [u.to_canonical() for u in self.units],
            "metadata": dict(sorted(self.metadata.items())),
        }

    @classmethod
    def from_canonical(cls, data: dict[str, Any]) -> "ParsedSource":
        return cls(
            source_id=data["source_id"],
            source_version=data["source_version"],
            parse_version=data["parse_version"],
            units=tuple(DocumentUnit.from_canonical(u) for u in data.get("units", [])),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class EvidenceSpan:
    """An exact span inside one document unit."""

    unit_id: str
    start_char: int
    end_char: int
    text: str = ""

    def to_canonical(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "text": self.text,
        }

    @classmethod
    def from_canonical(cls, data: dict[str, Any]) -> "EvidenceSpan":
        return cls(
            unit_id=data["unit_id"],
            start_char=int(data["start_char"]),
            end_char=int(data["end_char"]),
            text=data.get("text", ""),
        )


@dataclass(frozen=True)
class EvidenceReference:
    """One independently sufficient set of spans supporting a value.

    Stored separately from the value (FR-PROV-003) so a structural migration
    that moves the value keeps the evidence untouched.
    """

    source_id: str
    source_version: str
    parse_version: str
    spans: tuple[EvidenceSpan, ...] = ()
    #: Which locator produced the set ("model_quote", "retriever",
    #: "literal_match", "repair_pass"); free text, mirroring study_schema's
    #: ``EvidenceSource`` without hard-coding its vocabulary.
    locator: str | None = None
    #: Quotes proposed for this value that no locator could place.  Presence is
    #: the claim: it distinguishes a recall failure from a fidelity failure.
    unlocated_quotes: int = 0

    @property
    def source_ref(self) -> str:
        return f"{self.source_id}@{self.source_version}/{self.parse_version}"

    def unit_ids(self) -> tuple[str, ...]:
        return tuple(s.unit_id for s in self.spans)

    def resolve(self, parsed: ParsedSource) -> list[str]:
        """The literal text each span points at in ``parsed``.

        Returns ``""`` for a span whose unit is gone — a re-anchoring case, not
        an exception, because an absent unit is exactly the fact a re-anchoring
        migration needs to see.
        """

        out: list[str] = []
        for span in self.spans:
            unit = parsed.unit(span.unit_id)
            out.append("" if unit is None else unit.text[span.start_char : span.end_char])
        return out

    def is_anchored(self, parsed: ParsedSource) -> bool:
        """Whether every span still resolves to the text it recorded."""

        if self.parse_version != parsed.parse_version:
            return False
        for span, actual in zip(self.spans, self.resolve(parsed)):
            if span.text and span.text != actual:
                return False
            if not span.text and not actual:
                return False
        return bool(self.spans)

    def to_canonical(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_version": self.source_version,
            "parse_version": self.parse_version,
            "spans": [s.to_canonical() for s in self.spans],
            "locator": self.locator,
            "unlocated_quotes": self.unlocated_quotes,
        }

    @classmethod
    def from_canonical(cls, data: dict[str, Any]) -> "EvidenceReference":
        return cls(
            source_id=data["source_id"],
            source_version=data["source_version"],
            parse_version=data["parse_version"],
            spans=tuple(EvidenceSpan.from_canonical(s) for s in data.get("spans", [])),
            locator=data.get("locator"),
            unlocated_quotes=int(data.get("unlocated_quotes", 0) or 0),
        )


def build_parsed_source(
    source: SourceArtifact,
    sections: Sequence[tuple[str, str]],
    *,
    parse_version: str = "parse@1",
    split_sentences: bool = False,
) -> ParsedSource:
    """Convenience parser for tests and fixtures.

    ``sections`` is a sequence of ``(section_name, text)``.  Unit ids are
    ``<section>:<ordinal>`` so they are stable for a given input and readable
    in a plan.
    """

    units: list[DocumentUnit] = []
    offset = 0
    ordinal = 0
    for name, text in sections:
        if split_sentences:
            pieces = [p.strip() for p in _split_sentences(text) if p.strip()]
        else:
            pieces = [text]
        for piece in pieces:
            units.append(
                DocumentUnit(
                    unit_id=f"{name}:{ordinal}",
                    kind=UnitKind.SECTION if not split_sentences else UnitKind.SENTENCE,
                    text=piece,
                    section=name.lower(),
                    doc_start=offset,
                    ordinal=ordinal,
                )
            )
            offset += len(piece) + 1
            ordinal += 1
    return ParsedSource(
        source_id=source.source_id,
        source_version=source.source_version,
        parse_version=parse_version,
        units=tuple(units),
    )


def _split_sentences(text: str) -> list[str]:
    out: list[str] = []
    buf: list[str] = []
    for ch in text:
        buf.append(ch)
        if ch in ".!?":
            out.append("".join(buf))
            buf = []
    if buf:
        out.append("".join(buf))
    return out


def find_span(parsed: ParsedSource, needle: str) -> EvidenceReference | None:
    """Locate ``needle`` verbatim, the ``literal_match`` locator.

    Returns ``None`` when the text does not occur exactly once, because a
    value occurring twice has not been located — it has been guessed at.
    """

    hits: list[tuple[DocumentUnit, int]] = []
    for unit in parsed.units:
        start = unit.text.find(needle)
        while start >= 0:
            hits.append((unit, start))
            start = unit.text.find(needle, start + 1)
    if len(hits) != 1:
        return None
    unit, start = hits[0]
    return EvidenceReference(
        source_id=parsed.source_id,
        source_version=parsed.source_version,
        parse_version=parsed.parse_version,
        spans=(EvidenceSpan(unit.unit_id, start, start + len(needle), needle),),
        locator="literal_match",
    )
