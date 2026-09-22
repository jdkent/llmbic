"""Decomposing records into field artifacts, and assembling them back.

FR-DEP-005: record assembly is separate from field extraction, so adding one
field does not invalidate unrelated fields.  That separation is this module.
Import turns a nested JSON record into one artifact per leaf value plus one
:class:`RecordEntity` per collection member; export does the reverse.

The :class:`ValueCodec` indirection is what lets llmbic read a corpus whose
values are wrapped — study_schema's ``ExtractedValue`` carries the value beside
its status and evidence — without the migration model knowing anything about
that wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol, Sequence

from .ids import now
from .provenance import (
    Actor,
    FieldArtifact,
    FieldProvenance,
    RecordEntity,
    RecordVersion,
)
from .schema.normalized import (
    CollectionDefinition,
    FieldDefinition,
    NormalizedSchema,
    parent_collection,
    path_segments,
)
from .source import EvidenceReference, EvidenceSpan
from .values import FieldValue, ValueStatus

_MISSING = object()

#: Local id of the sentinel entity that records "this collection is present and
#: empty".  Without it, an empty list and an absent list would be the same on
#: export, which is the collapse FR-PROV-009 forbids one level up.
EMPTY = "∅"


class ValueCodec(Protocol):
    """Reads and writes the on-disk shape of one slot's value."""

    def decode(
        self, raw: Any, field: FieldDefinition
    ) -> tuple[FieldValue, tuple[EvidenceReference, ...]]: ...

    def encode(
        self, value: FieldValue, evidence: Sequence[EvidenceReference], field: FieldDefinition
    ) -> Any: ...


class PlainCodec:
    """Values stored bare.

    An explicit ``null`` is read as NOT_REPORTED and an absent key as
    NOT_EXTRACTED — the only two states a bare JSON document can express.  A
    corpus that needs the other five should wrap its values; that is what
    :class:`ExtractedValueCodec` is for.
    """

    def decode(
        self, raw: Any, field: FieldDefinition
    ) -> tuple[FieldValue, tuple[EvidenceReference, ...]]:
        if raw is _MISSING:
            return FieldValue(ValueStatus.NOT_EXTRACTED), ()
        if raw is None:
            return FieldValue(ValueStatus.NOT_REPORTED), ()
        return FieldValue.present(raw), ()

    def encode(
        self, value: FieldValue, evidence: Sequence[EvidenceReference], field: FieldDefinition
    ) -> Any:
        return value.value if value.status.has_value else None


_STATUS_FROM_EXTRACTION = {
    "extracted": ValueStatus.PRESENT,
    "not_reported": ValueStatus.NOT_REPORTED,
    "not_applicable": ValueStatus.NOT_APPLICABLE,
    "failed": ValueStatus.EXTRACTION_FAILED,
}

_EXTRACTION_FROM_STATUS = {
    ValueStatus.PRESENT: "extracted",
    ValueStatus.REVIEW_REQUIRED: "extracted",
    ValueStatus.NOT_REPORTED: "not_reported",
    ValueStatus.NOT_APPLICABLE: "not_applicable",
    ValueStatus.UNKNOWN: "not_reported",
    ValueStatus.NOT_EXTRACTED: "not_reported",
    ValueStatus.EXTRACTION_FAILED: "not_reported",
}


class ExtractedValueCodec:
    """study_schema's ``ExtractedValue`` wrapper.

    ::

        {"extraction_status": "extracted",
         "value": 42,
         "value_source": "reported",
         "unreported_reason": null,
         "evidence": {"status": "present",
                      "sets": [{"source": "model_quote",
                                "spans": [{"unit_id": ..., "start_char": 10,
                                           "end_char": 30, "text": "..."}]}],
                      "unlocated_quotes": 0}}

    ``unreported_reason`` maps onto :attr:`FieldValue.reason`, and
    ``undetermined`` — the one value that reports on the extraction rather than
    on the source — becomes :data:`ValueStatus.UNKNOWN`, which is exactly the
    distinction FR-PROV-009 forbids collapsing.
    """

    def __init__(self, source_id: str = "", source_version: str = "", parse_version: str = "") -> None:
        self.source_id = source_id
        self.source_version = source_version
        self.parse_version = parse_version

    def decode(
        self, raw: Any, field: FieldDefinition
    ) -> tuple[FieldValue, tuple[EvidenceReference, ...]]:
        if raw is _MISSING or raw is None:
            return FieldValue(ValueStatus.NOT_EXTRACTED), ()
        if not isinstance(raw, Mapping):
            # A native slot (identifier, type designator) — not wrapped.
            return FieldValue.present(raw), ()

        status_text = str(raw.get("extraction_status", "not_reported"))
        reason = raw.get("unreported_reason")
        status = _STATUS_FROM_EXTRACTION.get(status_text, ValueStatus.NOT_REPORTED)
        if status is ValueStatus.NOT_REPORTED and reason == "undetermined":
            status = ValueStatus.UNKNOWN

        value = (
            FieldValue(
                status,
                raw.get("value"),
                reason=reason,
                annotations=(
                    {"value_source": raw["value_source"]} if raw.get("value_source") else {}
                ),
            )
            if status.has_value
            else FieldValue(status, None, reason=reason)
        )

        evidence: list[EvidenceReference] = []
        ev = raw.get("evidence") or {}
        unlocated = int(ev.get("unlocated_quotes") or 0)
        for s in ev.get("sets") or ():
            spans = tuple(
                EvidenceSpan(
                    unit_id=sp.get("unit_id", ""),
                    start_char=int(sp.get("start_char", 0)),
                    end_char=int(sp.get("end_char", 0)),
                    text=sp.get("text", ""),
                )
                for sp in s.get("spans") or ()
            )
            if not spans:
                continue
            evidence.append(
                EvidenceReference(
                    source_id=s.get("source_id", self.source_id),
                    source_version=s.get("source_version", self.source_version),
                    parse_version=s.get("parse_version", self.parse_version),
                    spans=spans,
                    locator=s.get("source"),
                    unlocated_quotes=unlocated,
                )
            )
        if not evidence and unlocated:
            evidence.append(
                EvidenceReference(
                    source_id=self.source_id,
                    source_version=self.source_version,
                    parse_version=self.parse_version,
                    spans=(),
                    locator=None,
                    unlocated_quotes=unlocated,
                )
            )
        return value, tuple(evidence)

    def encode(
        self, value: FieldValue, evidence: Sequence[EvidenceReference], field: FieldDefinition
    ) -> Any:
        if not field.evidence_bearing:
            return value.value if value.status.has_value else None

        sets = [
            {
                "source": e.locator,
                "spans": [s.to_canonical() for s in e.spans],
            }
            for e in evidence
            if e.spans
        ]
        unlocated = sum(e.unlocated_quotes for e in evidence)
        if value.status.has_value:
            ev_status = "present" if sets else "not_found"
        else:
            ev_status = "not_applicable"

        out: dict[str, Any] = {
            "extraction_status": _EXTRACTION_FROM_STATUS[value.status],
            "evidence": {"status": ev_status},
        }
        if value.status.has_value:
            out["value"] = value.value
            source = value.annotations.get("value_source")
            if source:
                out["value_source"] = source
        if value.reason:
            out["unreported_reason"] = value.reason
        if sets:
            out["evidence"]["sets"] = sets
        if unlocated:
            out["evidence"]["unlocated_quotes"] = unlocated
        # The status llmbic knows and the wrapper cannot express is kept beside
        # it rather than dropped, so nothing is collapsed on export.
        if value.status in (
            ValueStatus.NOT_EXTRACTED,
            ValueStatus.EXTRACTION_FAILED,
            ValueStatus.UNKNOWN,
            ValueStatus.REVIEW_REQUIRED,
        ):
            out["llmbic_value_status"] = value.status.value
        return out


@dataclass
class DecomposedRecord:
    artifacts: list[FieldArtifact]
    entities: list[RecordEntity]

    def entity_ids(self) -> set[str]:
        return {e.entity for e in self.entities}


def collection_parent(collection: CollectionDefinition | str) -> str:
    """The collection that encloses ``collection``, ``""`` at the root."""

    path = collection if isinstance(collection, str) else collection.path
    segs = path_segments(path)
    return parent_collection(".".join(segs[:-1])) if len(segs) > 1 else ""


def decompose(
    record: Mapping[str, Any],
    schema: NormalizedSchema,
    *,
    record_id: str,
    codec: ValueCodec | None = None,
    provenance: FieldProvenance | None = None,
    actor: Actor = Actor.IMPORT,
) -> DecomposedRecord:
    """Split a nested record into per-field artifacts and entity rows."""

    codec = codec or PlainCodec()
    base = provenance or FieldProvenance(schema_version=schema.ref, actor=actor)
    artifacts: list[FieldArtifact] = []
    entities: list[RecordEntity] = []

    by_parent: dict[str, list[CollectionDefinition]] = {}
    for coll in schema.collections:
        by_parent.setdefault(collection_parent(coll), []).append(coll)

    def walk(node: Any, collection_path: str, entity: str) -> None:
        if not isinstance(node, Mapping):
            return
        for fdef in schema.fields_in(collection_path):
            relative = _relative(fdef.path, collection_path)
            raw = _navigate(node, relative)
            value, evidence = codec.decode(raw, fdef)
            artifacts.append(
                FieldArtifact(
                    record_id=record_id,
                    field_id=fdef.field_id,
                    entity=entity,
                    value=value,
                    evidence=evidence,
                    provenance=FieldProvenance(
                        schema_version=base.schema_version,
                        recipe_ref=base.recipe_ref or fdef.recipe_ref,
                        source_ref=base.source_ref,
                        parse_version=base.parse_version,
                        migration_id=base.migration_id,
                        execution_id=base.execution_id,
                        actor=base.actor,
                        actor_id=base.actor_id,
                        created_at=base.created_at or now(),
                        software_version=base.software_version,
                        input_hashes=dict(base.input_hashes),
                    ),
                )
            )

        for coll in by_parent.get(collection_path, ()):
            relative = _relative(coll.path, collection_path).removesuffix("[]")
            items = _navigate(node, relative)
            if items is _MISSING or items is None:
                continue
            if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
                continue
            if not items:
                # "This collection is present and empty" is a fact about the
                # record, distinct from "this collection was never extracted".
                entities.append(
                    RecordEntity(
                        entity=f"{entity}/{coll.path}={EMPTY}" if entity else f"{coll.path}={EMPTY}",
                        collection_path=coll.path,
                        local_id=EMPTY,
                        position=-1,
                        parent=entity,
                    )
                )
                continue
            for position, item in enumerate(items):
                local_id = _local_id(item, coll, position)
                child = f"{entity}/{coll.path}={local_id}" if entity else f"{coll.path}={local_id}"
                entities.append(
                    RecordEntity(
                        entity=child,
                        collection_path=coll.path,
                        local_id=local_id,
                        position=position,
                        parent=entity,
                    )
                )
                walk(item, coll.path, child)

    walk(record, "", "")
    return DecomposedRecord(artifacts=artifacts, entities=entities)


def assemble(
    artifacts: Iterable[FieldArtifact],
    entities: Iterable[RecordEntity],
    schema: NormalizedSchema,
    *,
    codec: ValueCodec | None = None,
    include_absent: bool = False,
) -> dict[str, Any]:
    """Build the nested record view from artifacts.

    ``include_absent=False`` omits slots that were never extracted, which is
    what a consumer wants; ``True`` writes every declared slot with its status,
    which is what an audit export wants.
    """

    codec = codec or PlainCodec()
    art_list = list(artifacts)
    ent_list = sorted(entities, key=lambda e: (e.parent.count("/"), e.parent, e.position))

    root: dict[str, Any] = {}
    containers: dict[str, dict[str, Any]] = {"": root}

    for ent in ent_list:
        parent = containers.get(ent.parent)
        if parent is None:
            continue
        relative = _relative(ent.collection_path, _collection_of(ent.parent)).removesuffix("[]")
        holder = _ensure_path(parent, relative, list)
        if ent.local_id == EMPTY:
            continue
        item: dict[str, Any] = {}
        holder.append(item)
        containers[ent.entity] = item

    for artifact in art_list:
        fdef = schema.get(artifact.field_id)
        if fdef is None:
            continue
        container = containers.get(artifact.entity)
        if container is None:
            continue
        if not include_absent and artifact.value.status is ValueStatus.NOT_EXTRACTED:
            continue
        relative = _relative(fdef.path, fdef.collection_path)
        encoded = codec.encode(artifact.value, artifact.evidence, fdef)
        if encoded is None and not include_absent and not artifact.value.status.has_value:
            if isinstance(codec, PlainCodec):
                continue
        _set_path(container, relative, encoded)

    return root


def assemble_version(
    version: RecordVersion,
    artifacts: Iterable[FieldArtifact],
    schema: NormalizedSchema,
    *,
    codec: ValueCodec | None = None,
    include_absent: bool = False,
) -> dict[str, Any]:
    wanted = set(version.artifact_ids)
    selected = [a for a in artifacts if a.artifact_id in wanted] if wanted else list(artifacts)
    return assemble(selected, version.entities, schema, codec=codec, include_absent=include_absent)


# ---- path plumbing -------------------------------------------------------

def _relative(path: str, collection_path: str) -> str:
    if not collection_path:
        return path
    if path == collection_path:
        return path
    prefix = collection_path + "."
    return path[len(prefix) :] if path.startswith(prefix) else path


def _collection_of(entity: str) -> str:
    if not entity:
        return ""
    return entity.rsplit("/", 1)[-1].split("=", 1)[0]


def _navigate(node: Any, relative: str) -> Any:
    current: Any = node
    for seg in path_segments(relative):
        if not isinstance(current, Mapping) or seg not in current:
            return _MISSING
        current = current[seg]
    return current


def _ensure_path(node: dict[str, Any], relative: str, leaf_factory: type) -> Any:
    segs = list(path_segments(relative))
    current = node
    for seg in segs[:-1]:
        current = current.setdefault(seg, {})
    last = segs[-1]
    if last not in current:
        current[last] = leaf_factory()
    return current[last]


def _set_path(node: dict[str, Any], relative: str, value: Any) -> None:
    segs = list(path_segments(relative))
    current = node
    for seg in segs[:-1]:
        nxt = current.get(seg)
        if not isinstance(nxt, dict):
            nxt = {}
            current[seg] = nxt
        current = nxt
    current[segs[-1]] = value


def _local_id(item: Any, coll: CollectionDefinition, position: int) -> str:
    """The member's persistent logical identity (FR-DEP-008).

    Falls back to the ordinal only when the collection declares no identity
    field *and* the member carries nothing usable — the case where reordering
    a list does rewrite artifacts, which is why the schema should name one.
    """

    if isinstance(item, Mapping) and coll.identity_field:
        raw = item.get(coll.identity_field)
        if isinstance(raw, Mapping):
            raw = raw.get("value")
        if isinstance(raw, (str, int)) and str(raw):
            return str(raw)
    return f"#{position}"


__all__ = [
    "DecomposedRecord",
    "ExtractedValueCodec",
    "PlainCodec",
    "ValueCodec",
    "assemble",
    "assemble_version",
    "collection_parent",
    "decompose",
]
