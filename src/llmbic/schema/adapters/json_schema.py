"""JSON Schema -> :class:`NormalizedSchema`.

Supports the subset that extraction schemas actually use: objects, arrays of
objects (which become collections), arrays of scalars (which become
``multivalued`` fields), ``enum``, ``$ref`` into ``$defs``/``definitions``, and
``anyOf`` used as "closed vocabulary or the source's own wording".

Vendor extensions llmbic reads, all optional:

``x-llmbic-field-id``
    Pin a stable field identity in the document itself.
``x-llmbic-recipe``
    ``name@version`` of the recipe required to produce the field.
``x-llmbic-deterministic``
    The field is filled by code, so no recipe currency applies.
``x-llmbic-vocabulary``
    ``name@version`` of the controlled vocabulary behind an enum.
``x-llmbic-identity-field``
    On an array-of-objects, the member slot carrying the logical id.
"""

from __future__ import annotations

from typing import Any, Mapping

from ...errors import ErrorCode, SchemaError
from ..normalized import (
    BaseType,
    CollectionDefinition,
    Constraints,
    FieldDefinition,
    NormalizedSchema,
)

_TYPE_MAP = {
    "string": BaseType.STRING,
    "integer": BaseType.INTEGER,
    "number": BaseType.NUMBER,
    "boolean": BaseType.BOOLEAN,
    "object": BaseType.OBJECT,
}

_MAX_DEPTH = 12


def from_json_schema(
    document: Mapping[str, Any],
    *,
    name: str,
    version: str,
    identity_map: Mapping[str, str] | None = None,
) -> NormalizedSchema:
    """Normalize ``document``.

    ``identity_map`` maps a path to the stable ``field_id`` it should carry.
    It is how a rename keeps its identity across versions; without it the
    field_id defaults to the path.
    """

    defs: dict[str, Any] = {}
    for key in ("$defs", "definitions"):
        defs.update(document.get(key) or {})

    fields: list[FieldDefinition] = []
    collections: list[CollectionDefinition] = []
    _walk(document, "", defs, fields, collections, depth=0, seen=frozenset())

    schema = NormalizedSchema(
        name=name,
        version=version,
        title=document.get("title", ""),
        description=document.get("description", ""),
        fields=tuple(fields),
        collections=tuple(collections),
        annotations={"source_format": "json_schema"},
    )
    if identity_map:
        schema = schema.rekey(dict(identity_map))
    return schema


def _resolve(node: Mapping[str, Any], defs: Mapping[str, Any]) -> tuple[Mapping[str, Any], str | None]:
    ref = node.get("$ref")
    if not ref:
        return node, None
    key = ref.rsplit("/", 1)[-1]
    if key not in defs:
        raise SchemaError(f"unresolvable $ref {ref!r}", code=ErrorCode.SCHEMA_ADAPTER)
    target = dict(defs[key])
    # Sibling keywords beside a $ref (description, title) override the target's.
    for k, v in node.items():
        if k != "$ref":
            target[k] = v
    return target, key


def _walk(
    node: Mapping[str, Any],
    prefix: str,
    defs: Mapping[str, Any],
    fields: list[FieldDefinition],
    collections: list[CollectionDefinition],
    *,
    depth: int,
    seen: frozenset[str],
) -> None:
    if depth > _MAX_DEPTH:
        raise SchemaError(
            f"schema nesting exceeds {_MAX_DEPTH} levels at {prefix!r}",
            code=ErrorCode.SCHEMA_ADAPTER,
        )
    props: Mapping[str, Any] = node.get("properties") or {}
    required = set(node.get("required") or ())

    for prop_name in props:
        raw = props[prop_name]
        resolved, def_key = _resolve(raw, defs)
        path = f"{prefix}.{prop_name}" if prefix else prop_name
        is_required = prop_name in required
        kind = _kind_of(resolved)

        if kind == "collection":
            item, item_key = _resolve(resolved.get("items") or {}, defs)
            coll_path = f"{path}[]"
            collections.append(
                CollectionDefinition(
                    path=coll_path,
                    identity_field=resolved.get("x-llmbic-identity-field")
                    or item.get("x-llmbic-identity-field")
                    or ("local_id" if "local_id" in (item.get("properties") or {}) else None),
                    description=resolved.get("description", ""),
                )
            )
            next_seen = seen | {item_key} if item_key else seen
            if item_key and item_key in seen:
                # Recursive definition: stop expanding, keep the collection.
                continue
            _walk(item, coll_path, defs, fields, collections, depth=depth + 1, seen=next_seen)
        elif kind == "object":
            next_seen = seen | {def_key} if def_key else seen
            if def_key and def_key in seen:
                continue
            _walk(resolved, path, defs, fields, collections, depth=depth + 1, seen=next_seen)
        else:
            fields.append(_leaf(path, resolved, is_required, defs))


def _kind_of(node: Mapping[str, Any]) -> str:
    t = node.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), None)
    if t == "array":
        item = node.get("items") or {}
        item_t = item.get("type")
        if item_t == "object" or ("properties" in item) or ("$ref" in item and "enum" not in item):
            return "collection"
        return "scalar"
    if t == "object" and ("properties" in node):
        return "object"
    return "scalar"


def _leaf(
    path: str,
    node: Mapping[str, Any],
    required: bool,
    defs: Mapping[str, Any],
) -> FieldDefinition:
    t = node.get("type")
    nullable = isinstance(t, list) and "null" in t
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), None)

    multivalued = t == "array"
    value_node: Mapping[str, Any] = node
    if multivalued:
        value_node, _ = _resolve(node.get("items") or {}, defs)
        t = value_node.get("type")

    enum_values, open_vocab = _enum_of(value_node, defs)
    base = _TYPE_MAP.get(t or "", BaseType.ANY)
    if enum_values and base is BaseType.ANY:
        base = BaseType.STRING

    constraints = Constraints(
        enum=enum_values,
        open_vocabulary=open_vocab,
        vocabulary_ref=value_node.get("x-llmbic-vocabulary") or node.get("x-llmbic-vocabulary"),
        minimum=value_node.get("minimum"),
        maximum=value_node.get("maximum"),
        min_length=value_node.get("minLength"),
        max_length=value_node.get("maxLength"),
        pattern=value_node.get("pattern"),
        min_items=node.get("minItems") if multivalued else None,
        max_items=node.get("maxItems") if multivalued else None,
    )

    annotations: dict[str, Any] = {}
    for key, ann in (("x-llmbic-annotations", None),):
        extra = node.get(key)
        if isinstance(extra, Mapping):
            annotations.update(extra)
    if nullable:
        annotations["nullable"] = True

    return FieldDefinition(
        field_id=node.get("x-llmbic-field-id") or path,
        path=path,
        base_type=base,
        multivalued=multivalued,
        required=required,
        description=node.get("description", ""),
        constraints=constraints,
        recipe_ref=node.get("x-llmbic-recipe"),
        deterministic=bool(node.get("x-llmbic-deterministic", False)),
        evidence_bearing=bool(node.get("x-llmbic-evidence", True)),
        annotations=annotations,
    )


def _enum_of(
    node: Mapping[str, Any], defs: Mapping[str, Any]
) -> tuple[tuple[str, ...] | None, bool]:
    """Read an enum, including the ``anyOf: [enum, string]`` open-vocabulary idiom."""

    if "enum" in node:
        return tuple(str(v) for v in node["enum"]), bool(node.get("x-llmbic-open-vocabulary"))

    any_of = node.get("anyOf") or node.get("oneOf")
    if not any_of:
        return None, False

    values: list[str] = []
    bare_string = False
    for option in any_of:
        resolved, _ = _resolve(option, defs)
        if "enum" in resolved:
            values.extend(str(v) for v in resolved["enum"])
        elif resolved.get("type") == "string":
            bare_string = True
    if not values:
        return None, False
    # Preserve declaration order, drop duplicates.
    seen: set[str] = set()
    ordered = tuple(v for v in values if not (v in seen or seen.add(v)))
    return ordered, bare_string
