"""Content addressing.

Every identity in llmbic that must survive a process restart is either a
user-supplied immutable string or a content hash computed here.  The hash is
over a canonical JSON encoding so that two structurally equivalent inputs
produce the same digest regardless of dict ordering (a property test in
``tests/test_properties.py`` pins this).
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any

HASH_PREFIX = "sha256"
_SHORT_LEN = 16


def canonical(value: Any) -> Any:
    """Convert ``value`` to a JSON-safe structure with deterministic ordering.

    Sets are sorted, mappings are key-sorted by ``json.dumps(sort_keys=True)``
    at encode time, dataclasses and enums are unwrapped.  ``None`` is preserved
    (it is a meaningful value, distinct from an absent key).
    """

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return canonical(value.value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    if is_dataclass(value) and not isinstance(value, type):
        if hasattr(value, "to_canonical"):
            return canonical(value.to_canonical())
        return canonical(asdict(value))
    if hasattr(value, "to_canonical"):
        return canonical(value.to_canonical())
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted((canonical(v) for v in value), key=_sort_key)
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    if hasattr(value, "model_dump"):  # pydantic BaseModel
        return canonical(value.model_dump(mode="json"))
    raise TypeError(f"cannot canonicalise {type(value)!r}")


def _sort_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def canonical_json(value: Any) -> str:
    return json.dumps(canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_hash(value: Any, *, short: bool = True) -> str:
    """Stable ``sha256:<hex>`` digest of ``value``."""

    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    if short:
        digest = digest[:_SHORT_LEN]
    return f"{HASH_PREFIX}:{digest}"


def text_hash(text: str, *, short: bool = True) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if short:
        digest = digest[:_SHORT_LEN]
    return f"{HASH_PREFIX}:{digest}"


def now() -> str:
    """UTC timestamp, ISO-8601 with a ``Z`` suffix."""

    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


_SLUG_RE = re.compile(r"[^a-zA-Z0-9_.@-]+")


def slug(text: str) -> str:
    return _SLUG_RE.sub("-", text).strip("-")


def version_ref(name: str, version: str) -> str:
    """``name@version`` — the canonical spelling for every versioned identity."""

    return f"{name}@{version}"


def parse_version_ref(ref: str) -> tuple[str, str]:
    if "@" not in ref:
        raise ValueError(f"expected 'name@version', got {ref!r}")
    name, _, version = ref.rpartition("@")
    return name, version
