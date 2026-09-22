"""The code escape hatch, with an identity.

FR-MIG-009 wants migrations to be ordinary version-controlled code; NFR-REP-003
wants every function's identity content-addressed or tied to an immutable
revision.  Both hold here: a transform is registered under ``name@version`` and
the registry refuses to rebind a name/version pair to different code, so a
cache key that names ``split_characteristics@2`` cannot silently start meaning
something else.
"""

from __future__ import annotations

import inspect
import textwrap
from dataclasses import dataclass
from typing import Any, Callable, Generic, Iterable, TypeVar

from .errors import ErrorCode, LlmbicError
from .ids import content_hash, text_hash

F = TypeVar("F", bound=Callable[..., Any])


@dataclass(frozen=True)
class RegisteredFunction:
    name: str
    version: str
    fn: Callable[..., Any]
    #: Hash of the function's source, so an edited body is a different identity
    #: even when the author forgot to bump the version.
    code_hash: str
    doc: str = ""

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.version}"

    def identity(self) -> dict[str, str]:
        return {"ref": self.ref, "code_hash": self.code_hash}

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)


class FunctionRegistry(Generic[F]):
    def __init__(self, label: str) -> None:
        self.label = label
        self._items: dict[str, RegisteredFunction] = {}

    def register(self, name: str, version: str = "1") -> Callable[[F], F]:
        def decorate(fn: F) -> F:
            self.add(name, fn, version=version)
            return fn

        return decorate

    def add(self, name: str, fn: Callable[..., Any], *, version: str = "1") -> RegisteredFunction:
        ref = f"{name}@{version}"
        entry = RegisteredFunction(
            name=name,
            version=version,
            fn=fn,
            code_hash=_code_hash(fn),
            doc=textwrap.dedent(fn.__doc__ or "").strip(),
        )
        existing = self._items.get(ref)
        if existing is not None and existing.code_hash != entry.code_hash:
            raise LlmbicError(
                f"{self.label} {ref!r} is already registered with different code; "
                "bump the version rather than rebinding it",
                code=ErrorCode.CONFIG_INVALID,
                details={"ref": ref, "existing": existing.code_hash, "new": entry.code_hash},
            )
        self._items[ref] = entry
        return entry

    def get(self, ref: str) -> RegisteredFunction:
        if ref in self._items:
            return self._items[ref]
        # Allow an unversioned reference when exactly one version exists.
        matches = [v for k, v in self._items.items() if v.name == ref]
        if len(matches) == 1:
            return matches[0]
        raise LlmbicError(
            f"unknown {self.label} {ref!r}"
            + (f"; candidates: {[m.ref for m in matches]}" if matches else ""),
            code=ErrorCode.TRANSFORM_NOT_FOUND,
            details={"ref": ref, "registered": sorted(self._items)},
        )

    def maybe(self, ref: str) -> RegisteredFunction | None:
        try:
            return self.get(ref)
        except LlmbicError:
            return None

    def identity_of(self, ref: str) -> dict[str, str]:
        return self.get(ref).identity()

    def refs(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, ref: object) -> bool:
        return isinstance(ref, str) and self.maybe(ref) is not None

    def hashes_for(self, refs: Iterable[str]) -> dict[str, str]:
        return {r: self.get(r).code_hash for r in sorted(set(refs))}


def _code_hash(fn: Callable[..., Any]) -> str:
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):  # builtins, C functions, lambdas from exec
        return content_hash({"repr": repr(fn), "qualname": getattr(fn, "__qualname__", "")})
    return text_hash(textwrap.dedent(source).strip())


#: Deterministic record/field transforms used by structural, derived and
#: vocabulary steps.
TRANSFORMS: FunctionRegistry[Callable[..., Any]] = FunctionRegistry("transform")

#: Field- and record-level semantic validators (FR-VAL-002).
VALIDATORS: FunctionRegistry[Callable[..., Any]] = FunctionRegistry("validator")

#: Postprocessors applied to a model's structured output before validation.
POSTPROCESSORS: FunctionRegistry[Callable[..., Any]] = FunctionRegistry("postprocessor")

#: Prompt builders, when a recipe needs more than template substitution.
PROMPTS: FunctionRegistry[Callable[..., Any]] = FunctionRegistry("prompt builder")


def transform(name: str, version: str = "1") -> Callable[[F], F]:
    return TRANSFORMS.register(name, version)


def validator(name: str, version: str = "1") -> Callable[[F], F]:
    return VALIDATORS.register(name, version)


def postprocessor(name: str, version: str = "1") -> Callable[[F], F]:
    return POSTPROCESSORS.register(name, version)


def prompt_builder(name: str, version: str = "1") -> Callable[[F], F]:
    return PROMPTS.register(name, version)


__all__ = [
    "POSTPROCESSORS",
    "PROMPTS",
    "TRANSFORMS",
    "VALIDATORS",
    "FunctionRegistry",
    "RegisteredFunction",
    "postprocessor",
    "prompt_builder",
    "transform",
    "validator",
]
