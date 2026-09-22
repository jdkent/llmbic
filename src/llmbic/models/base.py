"""Model adapters: one request/response contract for every provider.

FR-LLM-002.  The core never imports a provider SDK; an adapter implements
:class:`ModelAdapter` and declares its data-handling attributes so a privacy
policy can refuse it *before* the request is built (NFR-SEC-003/004).

Structured output is the default path (FR-LLM-001): a request carries the JSON
Schema the answer must satisfy, and an adapter that cannot constrain natively
is expected to validate and raise :class:`SchemaValidationError` so the retry
policy can see a schema failure as distinct from a transport failure
(FR-LLM-004).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from ..context.resolver import ContextUnit
from ..ids import content_hash


@dataclass(frozen=True)
class ModelIdentity:
    provider: str
    model: str
    #: Provider-reported identity of the served weights, when available.
    #: Decision 18.8: a stable model name is not a stable model.
    fingerprint: str | None = None

    @property
    def ref(self) -> str:
        return f"{self.provider}/{self.model}"

    def to_canonical(self) -> dict[str, Any]:
        return {"provider": self.provider, "model": self.model, "fingerprint": self.fingerprint}


@dataclass(frozen=True)
class DataPolicyAttributes:
    """What the adapter promises about the data it is sent (NFR-SEC-004)."""

    retains_data: bool = True
    trains_on_data: bool = False
    locality: str = "unknown"  # "local", "us", "eu", ...
    accepts_binary: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "retains_data": self.retains_data,
            "trains_on_data": self.trains_on_data,
            "locality": self.locality,
            "accepts_binary": self.accepts_binary,
        }


@dataclass(frozen=True)
class ModelRequest:
    """Everything that affects the answer, and nothing that does not."""

    prompt: str
    #: JSON Schema the response must conform to.
    output_schema: dict[str, Any]
    parameters: dict[str, Any] = field(default_factory=dict)
    context_units: tuple[ContextUnit, ...] = ()
    system: str = ""
    tools: tuple[dict[str, Any], ...] = ()
    #: Opaque, for adapters that need it; excluded from the request hash only
    #: when the caller marks it non-output-affecting.
    metadata: dict[str, Any] = field(default_factory=dict)

    def request_hash(self) -> str:
        return content_hash(
            {
                "prompt": self.prompt,
                "system": self.system,
                "output_schema": self.output_schema,
                "parameters": dict(sorted(self.parameters.items())),
                "context": [u.to_canonical() for u in self.context_units],
                "tools": list(self.tools),
            }
        )

    def context_text(self) -> str:
        return "\n\n".join(f"[{u.unit_id}]\n{u.text}" for u in self.context_units)

    def estimated_input_tokens(self, chars_per_token: float = 4.0) -> int:
        n = len(self.prompt) + len(self.system) + sum(u.n_chars for u in self.context_units)
        return int(n / chars_per_token + 0.999)


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cost_usd + other.cost_usd,
        )

    def to_canonical(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 8),
        }


@dataclass(frozen=True)
class ModelResponse:
    output: Any
    identity: ModelIdentity
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    provider_request_id: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ModelAdapter(Protocol):
    identity: ModelIdentity
    data_policy: DataPolicyAttributes

    def generate(self, request: ModelRequest) -> ModelResponse: ...


@dataclass(frozen=True)
class Pricing:
    """USD per million tokens, for estimation and accounting (FR-PLN-003)."""

    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok
        ) / 1_000_000


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    #: Seconds; the engine multiplies by ``backoff_factor ** (attempt - 1)``.
    initial_backoff: float = 0.5
    backoff_factor: float = 2.0
    max_backoff: float = 30.0
    #: Retry a schema-invalid answer by asking again, which is a different
    #: decision from retrying a dropped connection.
    retry_schema_failures: bool = True
    retry_semantic_failures: bool = False

    def backoff_for(self, attempt: int) -> float:
        return min(self.initial_backoff * (self.backoff_factor ** max(0, attempt - 1)), self.max_backoff)

    def to_canonical(self) -> dict[str, Any]:
        return {
            "max_attempts": self.max_attempts,
            "initial_backoff": self.initial_backoff,
            "backoff_factor": self.backoff_factor,
            "max_backoff": self.max_backoff,
            "retry_schema_failures": self.retry_schema_failures,
            "retry_semantic_failures": self.retry_semantic_failures,
        }

    @classmethod
    def from_canonical(cls, data: Mapping[str, Any]) -> "RetryPolicy":
        return cls(
            max_attempts=int(data.get("max_attempts", 3)),
            initial_backoff=float(data.get("initial_backoff", 0.5)),
            backoff_factor=float(data.get("backoff_factor", 2.0)),
            max_backoff=float(data.get("max_backoff", 30.0)),
            retry_schema_failures=bool(data.get("retry_schema_failures", True)),
            retry_semantic_failures=bool(data.get("retry_semantic_failures", False)),
        )


@dataclass(frozen=True)
class ModelPolicy:
    """Provider selection, parameters, retries, limits and budget (FR-LLM-003).

    ``adapter`` and ``fallbacks`` are *names*, resolved through the registry at
    execution time, so a policy stays serialisable and a plan can be inspected
    without a provider credential (FR-STO-005).
    """

    adapter: str
    parameters: dict[str, Any] = field(default_factory=lambda: {"temperature": 0.0})
    fallbacks: tuple[str, ...] = ()
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    #: Maximum in-flight requests for this policy; the engine also has a global
    #: ceiling and the smaller of the two applies.
    max_concurrency: int = 4
    #: Requests per second, 0 for unlimited.
    rate_limit_rps: float = 0.0
    budget_usd: float | None = None
    pricing: Pricing = field(default_factory=Pricing)
    #: When set, an artifact produced under a different fingerprint stops being
    #: semantically current.
    pin_fingerprint: str | None = None

    def policy_hash(self) -> str:
        return content_hash(self.to_canonical())

    def to_canonical(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter,
            "parameters": dict(sorted(self.parameters.items())),
            "fallbacks": list(self.fallbacks),
            "retry": self.retry.to_canonical(),
            "max_concurrency": self.max_concurrency,
            "rate_limit_rps": self.rate_limit_rps,
            "budget_usd": self.budget_usd,
            "pricing": {
                "input_per_mtok": self.pricing.input_per_mtok,
                "output_per_mtok": self.pricing.output_per_mtok,
            },
            "pin_fingerprint": self.pin_fingerprint,
        }

    @classmethod
    def from_canonical(cls, data: Mapping[str, Any]) -> "ModelPolicy":
        pricing = data.get("pricing") or {}
        return cls(
            adapter=data["adapter"],
            parameters=dict(data.get("parameters") or {}),
            fallbacks=tuple(data.get("fallbacks") or ()),
            retry=RetryPolicy.from_canonical(data.get("retry") or {}),
            max_concurrency=int(data.get("max_concurrency", 4)),
            rate_limit_rps=float(data.get("rate_limit_rps", 0.0)),
            budget_usd=data.get("budget_usd"),
            pricing=Pricing(
                float(pricing.get("input_per_mtok", 0.0)),
                float(pricing.get("output_per_mtok", 0.0)),
            ),
            pin_fingerprint=data.get("pin_fingerprint"),
        )

    def candidates(self) -> tuple[str, ...]:
        return (self.adapter,) + self.fallbacks


class AdapterRegistry:
    """Name -> adapter.  Populated by configuration, never by the core."""

    def __init__(self) -> None:
        self._adapters: dict[str, ModelAdapter] = {}

    def register(self, name: str, adapter: ModelAdapter) -> None:
        self._adapters[name] = adapter

    def get(self, name: str) -> ModelAdapter | None:
        return self._adapters.get(name)

    def names(self) -> list[str]:
        return sorted(self._adapters)

    def __contains__(self, name: object) -> bool:
        return name in self._adapters


DEFAULT_REGISTRY = AdapterRegistry()


class RateLimiter:
    """Simple deterministic token-bucket, shared across worker threads."""

    def __init__(self, rps: float) -> None:
        self.rps = rps
        self._next_at = 0.0

    def acquire(self, sleep=time.sleep, clock=time.monotonic) -> None:
        if self.rps <= 0:
            return
        interval = 1.0 / self.rps
        nowt = clock()
        wait = self._next_at - nowt
        if wait > 0:
            sleep(wait)
            nowt = clock()
        self._next_at = max(nowt, self._next_at) + interval


def sequence_of(units: Sequence[ContextUnit]) -> tuple[str, ...]:
    return tuple(u.unit_id for u in units)
