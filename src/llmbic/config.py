"""Version-controlled configuration with environment-based secret references.

FR-API-004 and NFR-SEC-001: the config file lives in the repository and names
its secrets; it never contains them.  ``${env:NAME}`` is resolved at load time
and the resolved value is kept out of every payload llmbic serialises — plans,
provenance and logs see the *reference*, not the value.

A minimal ``llmbic.yaml``::

    store: ./study.db
    schemas:
      - path: ./schemas/study_v1.json
        name: study
        version: "1.0"
        format: json_schema
    migrations:
      - ./migrations/*.yaml
    adapters:
      main:
        kind: llmbic.models.mock:ScriptedAdapter
        options: {provider: mock}
    policy:
      allow_lossy: false
      budget_usd: 25.0
"""

from __future__ import annotations

import importlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from .errors import ErrorCode, LlmbicError
from .migration.spec import ExecutionPolicy

_ENV_RE = re.compile(r"\$\{env:([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}")

#: Config keys whose values are never echoed back, even resolved.
SECRET_KEYS = frozenset({"api_key", "token", "secret", "password", "authorization"})


@dataclass
class SchemaSpec:
    path: str
    name: str
    version: str
    format: str = "json_schema"
    root_class: str | None = None
    identity_map: dict[str, str] = field(default_factory=dict)


@dataclass
class AdapterSpec:
    name: str
    kind: str
    options: dict[str, Any] = field(default_factory=dict)

    def redacted(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "options": {
                k: ("<redacted>" if k.lower() in SECRET_KEYS else v)
                for k, v in sorted(self.options.items())
            },
        }


@dataclass
class CodecSpec:
    """How stored values are shaped on disk.

    ``plain`` is a bare value; ``extracted_value`` is study_schema's wrapper,
    which carries the value beside its status and evidence.  Anything else is a
    dotted ``module:factory`` producing a :class:`llmbic.records.ValueCodec`.
    """

    kind: str = "plain"
    options: dict[str, Any] = field(default_factory=dict)

    def build(self) -> Any:
        from .records import ExtractedValueCodec, PlainCodec

        builtin = {"plain": PlainCodec, "extracted_value": ExtractedValueCodec}
        if self.kind in builtin:
            return builtin[self.kind](**self.options)
        return _instantiate(AdapterSpec(name="codec", kind=self.kind, options=self.options))


@dataclass
class Config:
    store: str = ":memory:"
    codec: CodecSpec = field(default_factory=CodecSpec)
    schemas: list[SchemaSpec] = field(default_factory=list)
    migrations: list[str] = field(default_factory=list)
    recipes: list[str] = field(default_factory=list)
    adapters: list[AdapterSpec] = field(default_factory=list)
    policy: ExecutionPolicy = field(default_factory=ExecutionPolicy)
    max_workers: int = 4
    #: Dotted module paths imported before anything runs, so that
    #: ``@transform`` / ``@validator`` registrations happen.
    extensions: list[str] = field(default_factory=list)
    root: Path = field(default_factory=Path.cwd)

    def resolve(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else (self.root / p)

    def redacted(self) -> dict[str, Any]:
        """Safe to print and to embed in a report (NFR-SEC-002)."""

        return {
            "store": self.store,
            "codec": {"kind": self.codec.kind, "options": dict(self.codec.options)},
            "schemas": [vars(s) for s in self.schemas],
            "migrations": list(self.migrations),
            "recipes": list(self.recipes),
            "adapters": [a.redacted() for a in self.adapters],
            "policy": self.policy.to_canonical(),
            "max_workers": self.max_workers,
            "extensions": list(self.extensions),
        }

    def load_extensions(self) -> list[str]:
        loaded = []
        for dotted in self.extensions:
            importlib.import_module(dotted)
            loaded.append(dotted)
        return loaded

    def build_adapters(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for spec in self.adapters:
            out[spec.name] = _instantiate(spec)
        return out


def expand_env(value: Any, *, strict: bool = True) -> Any:
    """Resolve ``${env:NAME}`` / ``${env:NAME:default}`` recursively."""

    if isinstance(value, str):
        def sub(m: re.Match[str]) -> str:
            name, default = m.group(1), m.group(2)
            if name in os.environ:
                return os.environ[name]
            if default is not None:
                return default
            if strict:
                raise LlmbicError(
                    f"configuration references ${{env:{name}}}, which is not set",
                    code=ErrorCode.CONFIG_INVALID,
                    details={"variable": name},
                )
            return ""

        return _ENV_RE.sub(sub, value)
    if isinstance(value, Mapping):
        return {k: expand_env(v, strict=strict) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [expand_env(v, strict=strict) for v in value]
    return value


def load_config(path: str | Path, *, strict_env: bool = True) -> Config:
    p = Path(path)
    if not p.exists():
        raise LlmbicError(f"no configuration at {p}", code=ErrorCode.CONFIG_INVALID)
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    data = expand_env(raw, strict=strict_env)

    schemas = [
        SchemaSpec(
            path=s["path"],
            name=s["name"],
            version=str(s["version"]),
            format=s.get("format", "json_schema"),
            root_class=s.get("root_class"),
            identity_map=dict(s.get("identity_map") or {}),
        )
        for s in data.get("schemas") or ()
    ]
    adapters = [
        AdapterSpec(name=name, kind=spec["kind"], options=dict(spec.get("options") or {}))
        for name, spec in sorted((data.get("adapters") or {}).items())
    ]
    raw_codec = data.get("codec") or {}
    codec = (
        CodecSpec(kind=raw_codec)
        if isinstance(raw_codec, str)
        else CodecSpec(
            kind=raw_codec.get("kind", "plain"), options=dict(raw_codec.get("options") or {})
        )
    )
    return Config(
        store=data.get("store", ":memory:"),
        codec=codec,
        schemas=schemas,
        migrations=list(data.get("migrations") or ()),
        recipes=list(data.get("recipes") or ()),
        adapters=adapters,
        policy=ExecutionPolicy.from_canonical(data.get("policy") or {}),
        max_workers=int(data.get("max_workers", 4)),
        extensions=list(data.get("extensions") or ()),
        root=p.parent.resolve(),
    )


def find_config(start: str | Path | None = None) -> Path | None:
    current = Path(start or Path.cwd()).resolve()
    for directory in [current, *current.parents]:
        for name in ("llmbic.yaml", "llmbic.yml", ".llmbic.yaml"):
            candidate = directory / name
            if candidate.exists():
                return candidate
    return None


def _instantiate(spec: AdapterSpec) -> Any:
    module_name, _, attr = spec.kind.partition(":")
    if not attr:
        module_name, _, attr = spec.kind.rpartition(".")
    try:
        module = importlib.import_module(module_name)
        factory = getattr(module, attr)
    except (ImportError, AttributeError) as exc:
        raise LlmbicError(
            f"cannot load model adapter {spec.kind!r}: {exc}",
            code=ErrorCode.CONFIG_INVALID,
        ) from exc
    return factory(**spec.options)


def load_schema_spec(spec: SchemaSpec, config: Config) -> Any:
    """Read one schema using the adapter its ``format`` names."""

    from .schema.adapters import from_json_schema, from_linkml

    path = config.resolve(spec.path)
    if spec.format in ("linkml", "yaml-linkml"):
        return from_linkml(
            path,
            name=spec.name,
            version=spec.version,
            root_class=spec.root_class,
            identity_map=spec.identity_map,
        )
    if spec.format in ("json_schema", "json", "jsonschema"):
        import json

        document = json.loads(path.read_text(encoding="utf-8"))
        return from_json_schema(
            document, name=spec.name, version=spec.version, identity_map=spec.identity_map
        )
    if spec.format in ("pydantic",):
        from .schema.adapters import from_pydantic

        module_name, _, attr = spec.path.partition(":")
        model = getattr(importlib.import_module(module_name), attr)
        return from_pydantic(
            model, name=spec.name, version=spec.version, identity_map=spec.identity_map
        )
    raise LlmbicError(
        f"unknown schema format {spec.format!r}", code=ErrorCode.CONFIG_INVALID
    )


__all__ = [
    "AdapterSpec",
    "Config",
    "SECRET_KEYS",
    "SchemaSpec",
    "expand_env",
    "find_config",
    "load_config",
    "load_schema_spec",
]
