"""Pydantic v2 models -> :class:`NormalizedSchema`.

Pydantic is not a core dependency of the migration model; this adapter simply
asks the model for its JSON Schema and hands that to
:func:`llmbic.schema.adapters.json_schema.from_json_schema`, so exactly one
normalizer exists.  Field-level extras (``json_schema_extra``) carry the
``x-llmbic-*`` keys.
"""

from __future__ import annotations

from typing import Any, Mapping

from ...errors import ErrorCode, SchemaError
from ..normalized import NormalizedSchema
from .json_schema import from_json_schema


def from_pydantic(
    model: Any,
    *,
    name: str | None = None,
    version: str,
    identity_map: Mapping[str, str] | None = None,
) -> NormalizedSchema:
    if not hasattr(model, "model_json_schema"):
        raise SchemaError(
            f"{model!r} is not a pydantic v2 model", code=ErrorCode.SCHEMA_ADAPTER
        )
    document = model.model_json_schema(ref_template="#/$defs/{model}")
    document.setdefault("title", getattr(model, "__name__", "Record"))
    if not document.get("description") and getattr(model, "__doc__", None):
        document["description"] = model.__doc__.strip()
    return from_json_schema(
        document,
        name=name or getattr(model, "__name__", "record"),
        version=version,
        identity_map=identity_map,
    )


def schema_for(model: Any) -> dict[str, Any]:
    """The raw JSON Schema a structured-output adapter should constrain against."""

    return model.model_json_schema(ref_template="#/$defs/{model}")


__all__ = ["from_pydantic", "schema_for", "NormalizedSchema"]
