"""LinkML -> :class:`NormalizedSchema`.

Written against the shape neurostuff's ``study_schema`` uses, which is the
first corpus llmbic targets: a set of module files importing one another, one
class marked ``tree_root``, closed and open enums, and every source-derived
slot wrapped in an ``ExtractedValue`` subclass that carries the value beside
its status and evidence.

Two LinkML-specific facts are folded into the normalized form:

* A class deriving from ``ExtractedValue`` is *unwrapped*: its
  ``slot_usage.value`` supplies the real type and cardinality, and the field is
  marked ``evidence_bearing``.  The wrapper is a transport detail; the planner
  cares about the value's type.
* ``in_subset: [deterministic]`` marks a slot filled by code, so no extraction
  recipe is required for it and a recipe change never invalidates it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import yaml

from ...errors import ErrorCode, SchemaError
from ..normalized import (
    BaseType,
    CollectionDefinition,
    Constraints,
    FieldDefinition,
    NormalizedSchema,
)

_SCALAR_RANGES = {
    "string": BaseType.STRING,
    "uriorcurie": BaseType.STRING,
    "uri": BaseType.STRING,
    "date": BaseType.STRING,
    "datetime": BaseType.STRING,
    "integer": BaseType.INTEGER,
    "float": BaseType.NUMBER,
    "double": BaseType.NUMBER,
    "decimal": BaseType.NUMBER,
    "boolean": BaseType.BOOLEAN,
    "Any": BaseType.ANY,
}

_EXTRACTED_ROOT = "ExtractedValue"
_MAX_DEPTH = 10


class LinkMLBundle:
    """A LinkML schema and every module it imports, merged.

    LinkML's import graph is resolved relative to the importing file, the same
    way ``linkml-runtime`` does it, but without taking a dependency on it: the
    subset used here is ``classes``, ``enums``, ``slots`` and ``subsets``.
    """

    def __init__(self) -> None:
        self.classes: dict[str, Any] = {}
        self.enums: dict[str, Any] = {}
        self.slots: dict[str, Any] = {}
        self.meta: dict[str, Any] = {}
        self._loaded: set[str] = set()

    @classmethod
    def load(cls, entry: str | os.PathLike[str]) -> "LinkMLBundle":
        bundle = cls()
        bundle._load_file(Path(entry).resolve())
        return bundle

    def _load_file(self, path: Path) -> None:
        key = str(path)
        if key in self._loaded:
            return
        if not path.exists():
            for candidate in (path.with_suffix(".yaml"), path.with_suffix(".yml")):
                if candidate.exists():
                    path = candidate
                    key = str(path)
                    break
            else:
                raise SchemaError(f"LinkML module not found: {path}", code=ErrorCode.SCHEMA_ADAPTER)
        self._loaded.add(key)

        with path.open("r", encoding="utf-8") as fh:
            doc = yaml.safe_load(fh) or {}

        if not self.meta:
            self.meta = {
                k: doc.get(k) for k in ("id", "name", "title", "description", "version", "license")
            }

        for imp in doc.get("imports") or []:
            if imp.startswith("linkml:"):
                continue
            self._load_file((path.parent / imp).resolve())

        _merge(self.classes, doc.get("classes") or {})
        _merge(self.enums, doc.get("enums") or {})
        _merge(self.slots, doc.get("slots") or {})

    # ---- class model helpers --------------------------------------------
    def ancestry(self, class_name: str) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        current: str | None = class_name
        while current and current in self.classes and current not in seen:
            seen.add(current)
            out.append(current)
            current = self.classes[current].get("is_a")
        if current and current not in seen:
            out.append(current)
        return out

    def is_extracted_wrapper(self, class_name: str) -> bool:
        return _EXTRACTED_ROOT in self.ancestry(class_name)[1:] or class_name == _EXTRACTED_ROOT

    def attributes(self, class_name: str) -> dict[str, Any]:
        """Attributes of a class, with ``is_a`` inheritance and ``slot_usage`` applied."""

        chain = list(reversed(self.ancestry(class_name)))
        merged: dict[str, Any] = {}
        for name in chain:
            spec = self.classes.get(name) or {}
            for slot_name in spec.get("slots") or []:
                if slot_name in self.slots:
                    merged.setdefault(slot_name, {}).update(self.slots[slot_name])
            for attr_name, attr in (spec.get("attributes") or {}).items():
                merged.setdefault(attr_name, {})
                merged[attr_name] = {**merged[attr_name], **(attr or {})}
            for attr_name, usage in (spec.get("slot_usage") or {}).items():
                merged.setdefault(attr_name, {})
                merged[attr_name] = {**merged[attr_name], **(usage or {})}
        return merged

    def tree_root(self) -> str:
        roots = [name for name, spec in self.classes.items() if (spec or {}).get("tree_root")]
        if len(roots) == 1:
            return roots[0]
        if not roots:
            raise SchemaError("no class marked tree_root", code=ErrorCode.SCHEMA_ADAPTER)
        raise SchemaError(
            f"multiple tree_root classes: {sorted(roots)}", code=ErrorCode.SCHEMA_ADAPTER
        )


def _is_inlined(slot: Mapping[str, Any]) -> bool:
    """Whether a class-ranged slot nests the entity or names it by identifier.

    ``inlined: false`` is study_schema's way of saying "a document-local
    reference": the slot holds a ``local_id`` string, and the entity's own
    fields live where the entity is declared.  Expanding such a slot would
    duplicate every field of the target class under a second path and give one
    logical value two field identities.
    """

    if "inlined_as_list" in slot:
        return bool(slot["inlined_as_list"])
    if "inlined" in slot:
        return bool(slot["inlined"])
    return True


def _merge(target: MutableMapping[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            target[key] = {**target[key], **value}
        else:
            target[key] = value


def from_linkml(
    entry: str | os.PathLike[str],
    *,
    name: str | None = None,
    version: str | None = None,
    root_class: str | None = None,
    identity_map: Mapping[str, str] | None = None,
    max_depth: int = _MAX_DEPTH,
) -> NormalizedSchema:
    bundle = LinkMLBundle.load(entry)
    return from_linkml_bundle(
        bundle,
        name=name,
        version=version,
        root_class=root_class,
        identity_map=identity_map,
        max_depth=max_depth,
    )


def from_linkml_bundle(
    bundle: LinkMLBundle,
    *,
    name: str | None = None,
    version: str | None = None,
    root_class: str | None = None,
    identity_map: Mapping[str, str] | None = None,
    max_depth: int = _MAX_DEPTH,
) -> NormalizedSchema:
    root = root_class or bundle.tree_root()
    fields: list[FieldDefinition] = []
    collections: list[CollectionDefinition] = []
    _walk_class(bundle, root, "", fields, collections, depth=0, seen=frozenset({root}), max_depth=max_depth)

    return NormalizedSchema(
        name=name or bundle.meta.get("name") or root.lower(),
        version=version or bundle.meta.get("version") or "0",
        title=bundle.meta.get("title") or "",
        description=bundle.meta.get("description") or "",
        fields=tuple(fields),
        collections=tuple(collections),
        annotations={"source_format": "linkml", "root_class": root},
    ).rekey(dict(identity_map or {}))


def _walk_class(
    bundle: LinkMLBundle,
    class_name: str,
    prefix: str,
    fields: list[FieldDefinition],
    collections: list[CollectionDefinition],
    *,
    depth: int,
    seen: frozenset[str],
    max_depth: int,
) -> None:
    if depth > max_depth:
        return
    for slot_name, slot in bundle.attributes(class_name).items():
        slot = slot or {}
        path = f"{prefix}.{slot_name}" if prefix else slot_name
        rng = slot.get("range")
        multivalued = bool(slot.get("multivalued"))

        if (
            isinstance(rng, str)
            and rng in bundle.classes
            and not bundle.is_extracted_wrapper(rng)
            and _is_inlined(slot)
        ):
            if multivalued:
                coll_path = f"{path}[]"
                collections.append(
                    CollectionDefinition(
                        path=coll_path,
                        identity_field="local_id"
                        if "local_id" in bundle.attributes(rng)
                        else None,
                        description=str(slot.get("description") or ""),
                    )
                )
                if rng in seen:
                    continue
                _walk_class(
                    bundle,
                    rng,
                    coll_path,
                    fields,
                    collections,
                    depth=depth + 1,
                    seen=seen | {rng},
                    max_depth=max_depth,
                )
            else:
                if rng in seen:
                    continue
                _walk_class(
                    bundle,
                    rng,
                    path,
                    fields,
                    collections,
                    depth=depth + 1,
                    seen=seen | {rng},
                    max_depth=max_depth,
                )
            continue

        fields.append(_leaf(bundle, path, slot))


def _leaf(bundle: LinkMLBundle, path: str, slot: Mapping[str, Any]) -> FieldDefinition:
    description = str(slot.get("description") or "")
    subsets = set(slot.get("in_subset") or ())
    rng = slot.get("range")
    multivalued = bool(slot.get("multivalued"))
    evidence_bearing = False
    any_of = slot.get("any_of")

    if isinstance(rng, str) and rng in bundle.classes and bundle.is_extracted_wrapper(rng):
        evidence_bearing = True
        inner = bundle.attributes(rng).get("value") or {}
        wrapper_desc = str((bundle.classes.get(rng) or {}).get("description") or "")
        if not description:
            description = wrapper_desc
        multivalued = bool(inner.get("multivalued")) or multivalued
        any_of = inner.get("any_of")
        rng = inner.get("range")

    enum_values: tuple[str, ...] | None = None
    open_vocabulary = False
    vocabulary_ref: str | None = None

    if any_of:
        values: list[str] = []
        for option in any_of:
            opt_range = (option or {}).get("range")
            if opt_range in bundle.enums:
                values.extend((bundle.enums[opt_range].get("permissible_values") or {}).keys())
                vocabulary_ref = opt_range
            elif opt_range == "string":
                open_vocabulary = True
        if values:
            enum_values = tuple(values)
            # The wrapper's inherited ``range: Any`` says nothing; the any_of
            # branch does.
            if rng in (None, "Any"):
                rng = vocabulary_ref

    references: str | None = None
    if isinstance(rng, str) and rng in bundle.enums:
        enum_values = tuple((bundle.enums[rng].get("permissible_values") or {}).keys())
        vocabulary_ref = rng
        base = BaseType.STRING
    elif isinstance(rng, str) and rng in bundle.classes:
        # A non-inlined class range: the slot holds the target's local_id.
        references = rng
        base = BaseType.STRING
    else:
        base = _SCALAR_RANGES.get(str(rng), BaseType.STRING if enum_values else BaseType.ANY)

    constraints = Constraints(
        enum=enum_values,
        open_vocabulary=open_vocabulary,
        vocabulary_ref=vocabulary_ref,
        minimum=slot.get("minimum_value"),
        maximum=slot.get("maximum_value"),
        pattern=slot.get("pattern"),
        min_items=slot.get("minimum_cardinality") if multivalued else None,
        max_items=slot.get("maximum_cardinality") if multivalued else None,
    )

    annotations: dict[str, Any] = {}
    if references:
        annotations["references"] = references
    if subsets:
        annotations["in_subset"] = sorted(subsets)
    if slot.get("annotations"):
        annotations.update({str(k): v for k, v in slot["annotations"].items()})

    recipe_ref = annotations.pop("recipe", None)

    return FieldDefinition(
        field_id=path,
        path=path,
        base_type=base,
        multivalued=multivalued,
        required=bool(slot.get("required")),
        description=description,
        constraints=constraints,
        recipe_ref=recipe_ref,
        deterministic="deterministic" in subsets or not evidence_bearing,
        evidence_bearing=evidence_bearing,
        annotations=annotations,
    )
