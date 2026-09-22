"""Authoring-format adapters.

Each adapter produces a :class:`llmbic.schema.normalized.NormalizedSchema` and
nothing else.  Adding one never touches the migration graph (NFR-MNT-003).
"""

from .json_schema import from_json_schema
from .linkml import LinkMLBundle, from_linkml, from_linkml_bundle
from .pydantic_models import from_pydantic, schema_for

__all__ = [
    "from_json_schema",
    "from_linkml",
    "from_linkml_bundle",
    "LinkMLBundle",
    "from_pydantic",
    "schema_for",
]
