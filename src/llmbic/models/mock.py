"""Offline adapters: deterministic stand-ins for hosted providers.

NFR-MNT-002 requires the core tests to run without network access, and FR
15.3 asks integration tests to exercise "one local model or mocked
structured-output adapter" *and* "one hosted-provider-compatible mock".  Both
live here.

None of these is a toy: they enforce the same structured-output contract,
report usage and request ids, and fail in the four distinguishable ways
FR-LLM-004 names, so retry, caching and budget logic is exercised for real.
"""

from __future__ import annotations

import json
import random
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import jsonschema

from ..errors import (
    RateLimitedError,
    SchemaValidationError,
    TransportError,
    UnsupportedContextError,
)
from ..ids import content_hash
from .base import (
    DataPolicyAttributes,
    ModelIdentity,
    ModelRequest,
    ModelResponse,
    Pricing,
    Usage,
)


def validate_output(output: Any, schema: Mapping[str, Any]) -> None:
    try:
        jsonschema.validate(output, dict(schema))
    except jsonschema.ValidationError as exc:
        raise SchemaValidationError(
            f"model output does not satisfy the target schema: {exc.message}",
            details={"path": list(exc.absolute_path), "validator": exc.validator},
        ) from exc


@dataclass
class CallLog:
    """Every request an adapter saw, for asserting on call counts."""

    entries: list[dict[str, Any]] = field(default_factory=list)

    def record(self, request: ModelRequest, outcome: str, **extra: Any) -> None:
        self.entries.append(
            {
                "request_hash": request.request_hash(),
                "prompt_hash": content_hash(request.prompt),
                "context_units": [u.unit_id for u in request.context_units],
                "outcome": outcome,
                **extra,
            }
        )

    @property
    def successes(self) -> list[dict[str, Any]]:
        return [e for e in self.entries if e["outcome"] == "ok"]

    def __len__(self) -> int:
        return len(self.entries)

    def clear(self) -> None:
        self.entries.clear()

    def distinct_requests(self) -> set[str]:
        return {e["request_hash"] for e in self.entries}


ResponderFn = Callable[[ModelRequest], Any]


class ScriptedAdapter:
    """Answers from a table keyed by a caller-supplied selector.

    The default selector is the request hash, so the same question always gets
    the same answer and a duplicate billable call is detectable by counting
    entries in :attr:`calls`.
    """

    def __init__(
        self,
        responses: Mapping[str, Any] | None = None,
        *,
        provider: str = "mock",
        model: str = "scripted-1",
        fingerprint: str | None = "fp-1",
        key: Callable[[ModelRequest], str] | None = None,
        default: Any | ResponderFn | None = None,
        pricing: Pricing = Pricing(0.5, 1.5),
        latency_ms: float = 1.0,
        data_policy: DataPolicyAttributes | None = None,
    ) -> None:
        self.identity = ModelIdentity(provider, model, fingerprint)
        self.data_policy = data_policy or DataPolicyAttributes(
            retains_data=False, trains_on_data=False, locality="local"
        )
        self.responses: dict[str, Any] = dict(responses or {})
        self._key = key or (lambda r: r.request_hash())
        self._default = default
        self.pricing = pricing
        self.latency_ms = latency_ms
        self.calls = CallLog()
        self._lock = threading.Lock()

    def add(self, key: str, response: Any) -> None:
        self.responses[key] = response

    def generate(self, request: ModelRequest) -> ModelResponse:
        key = self._key(request)
        if key in self.responses:
            output = self.responses[key]
        elif callable(self._default):
            output = self._default(request)
        elif self._default is not None:
            output = self._default
        else:
            with self._lock:
                self.calls.record(request, "no_response", key=key)
            raise UnsupportedContextError(
                f"scripted adapter has no response for key {key!r}",
                details={"key": key, "known": sorted(self.responses)[:10]},
            )

        validate_output(output, request.output_schema)
        usage = self._usage(request, output)
        with self._lock:
            self.calls.record(request, "ok", key=key)
            request_id = f"mock-req-{len(self.calls.entries):06d}"
        return ModelResponse(
            output=output,
            identity=self.identity,
            usage=usage,
            latency_ms=self.latency_ms,
            provider_request_id=request_id,
        )

    def _usage(self, request: ModelRequest, output: Any) -> Usage:
        in_tok = request.estimated_input_tokens()
        out_tok = max(1, len(json.dumps(output, default=str)) // 4)
        return Usage(in_tok, out_tok, self.pricing.cost(in_tok, out_tok))


class RuleBasedExtractor:
    """A local "model" that genuinely reads its context.

    Rules are ``(compiled pattern, builder)``; the first pattern matching any
    supplied context unit wins and its builder produces the structured answer
    from the match.  When nothing matches, the adapter *abstains* — which is
    the behaviour acceptance criterion 10 depends on, since a record with no
    usable context must not receive an invented value.
    """

    def __init__(
        self,
        rules: Sequence[tuple[str, Callable[[re.Match[str], Any], Any]]] = (),
        *,
        abstention: Any | None = None,
        provider: str = "local",
        model: str = "rules-1",
        pricing: Pricing = Pricing(0.0, 0.0),
        flags: int = re.IGNORECASE,
    ) -> None:
        self.identity = ModelIdentity(provider, model, "rules-v1")
        self.data_policy = DataPolicyAttributes(
            retains_data=False, trains_on_data=False, locality="local"
        )
        self.rules = [(re.compile(p, flags), fn) for p, fn in rules]
        self.abstention = abstention
        self.pricing = pricing
        self.calls = CallLog()
        self._lock = threading.Lock()

    def generate(self, request: ModelRequest) -> ModelResponse:
        output = self.abstention
        matched_unit = None
        for pattern, build in self.rules:
            for unit in request.context_units:
                m = pattern.search(unit.text)
                if m:
                    output = build(m, unit)
                    matched_unit = unit.unit_id
                    break
            if matched_unit:
                break

        if output is None:
            raise UnsupportedContextError(
                "no rule matched the supplied context and no abstention value is configured",
                details={"units": [u.unit_id for u in request.context_units]},
            )

        validate_output(output, request.output_schema)
        in_tok = request.estimated_input_tokens()
        out_tok = max(1, len(json.dumps(output, default=str)) // 4)
        with self._lock:
            self.calls.record(request, "ok", matched_unit=matched_unit)
            request_id = f"local-{len(self.calls.entries):06d}"
        return ModelResponse(
            output=output,
            identity=self.identity,
            usage=Usage(in_tok, out_tok, self.pricing.cost(in_tok, out_tok)),
            latency_ms=0.5,
            provider_request_id=request_id,
            raw={"matched_unit": matched_unit},
        )


class HostedMockAdapter:
    """A hosted provider simulated down to its failure modes.

    Fails transiently, rate-limits, occasionally returns something that does
    not satisfy the schema, reports token usage and a provider request id, and
    refuses binary context unless configured to accept it.  Deterministic: the
    failure schedule is seeded from the request hash, so a test that resumes a
    job sees the same failures in the same places.
    """

    def __init__(
        self,
        responder: ResponderFn,
        *,
        provider: str = "hosted-mock",
        model: str = "big-1",
        fingerprint: str = "2026-05-01",
        transient_failure_rate: float = 0.0,
        rate_limit_rate: float = 0.0,
        schema_failure_rate: float = 0.0,
        pricing: Pricing = Pricing(3.0, 15.0),
        accepts_binary: bool = False,
        max_input_tokens: int | None = None,
        latency_ms: float = 2.0,
        locality: str = "us",
    ) -> None:
        self.identity = ModelIdentity(provider, model, fingerprint)
        self.data_policy = DataPolicyAttributes(
            retains_data=True, trains_on_data=False, locality=locality, accepts_binary=accepts_binary
        )
        self.responder = responder
        self.transient_failure_rate = transient_failure_rate
        self.rate_limit_rate = rate_limit_rate
        self.schema_failure_rate = schema_failure_rate
        self.pricing = pricing
        self.max_input_tokens = max_input_tokens
        self.latency_ms = latency_ms
        self.calls = CallLog()
        self._attempts: dict[str, int] = {}
        self._lock = threading.Lock()

    def generate(self, request: ModelRequest) -> ModelResponse:
        rh = request.request_hash()
        with self._lock:
            attempt = self._attempts.get(rh, 0) + 1
            self._attempts[rh] = attempt
        rng = random.Random(f"{rh}:{attempt}")

        if any(u.kind == "raw_document" for u in request.context_units) and not self.data_policy.accepts_binary:
            with self._lock:
                self.calls.record(request, "unsupported_context")
            raise UnsupportedContextError(
                f"{self.identity.ref} does not accept binary documents"
            )

        in_tok = request.estimated_input_tokens()
        if self.max_input_tokens is not None and in_tok > self.max_input_tokens:
            with self._lock:
                self.calls.record(request, "context_too_large", input_tokens=in_tok)
            raise UnsupportedContextError(
                f"{in_tok} input tokens exceeds the model's {self.max_input_tokens} limit",
                details={"input_tokens": in_tok, "limit": self.max_input_tokens},
            )

        if rng.random() < self.rate_limit_rate:
            with self._lock:
                self.calls.record(request, "rate_limited", attempt=attempt)
            raise RateLimitedError(
                f"{self.identity.ref} rate limited", details={"retry_after": 0.01}
            )
        if rng.random() < self.transient_failure_rate:
            with self._lock:
                self.calls.record(request, "transport_error", attempt=attempt)
            raise TransportError(f"{self.identity.ref} connection reset", details={"attempt": attempt})

        output = self.responder(request)
        if rng.random() < self.schema_failure_rate:
            output = {"__garbage__": True}

        validate_output(output, request.output_schema)

        out_tok = max(1, len(json.dumps(output, default=str)) // 4)
        time.sleep(0)  # keep the call a real yield point for the thread pool
        with self._lock:
            self.calls.record(request, "ok", attempt=attempt, input_tokens=in_tok)
            request_id = f"{self.identity.provider}-{len(self.calls.entries):06d}"
        return ModelResponse(
            output=output,
            identity=self.identity,
            usage=Usage(in_tok, out_tok, self.pricing.cost(in_tok, out_tok)),
            latency_ms=self.latency_ms,
            provider_request_id=request_id,
        )


class FailingAdapter:
    """Always raises, to exercise fallback selection and blocked dispositions."""

    def __init__(
        self,
        error: Exception | None = None,
        *,
        provider: str = "broken",
        model: str = "down-1",
    ) -> None:
        self.identity = ModelIdentity(provider, model, None)
        self.data_policy = DataPolicyAttributes()
        self.error = error or TransportError(f"{provider}/{model} is unavailable")
        self.calls = CallLog()

    def generate(self, request: ModelRequest) -> ModelResponse:
        self.calls.record(request, "error")
        raise self.error


class RecordingAdapter:
    """Wraps another adapter and counts requests, for duplicate-call assertions."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.identity = inner.identity
        self.data_policy = inner.data_policy
        self.calls = CallLog()
        self._lock = threading.Lock()

    def generate(self, request: ModelRequest) -> ModelResponse:
        try:
            response = self.inner.generate(request)
        except Exception:
            with self._lock:
                self.calls.record(request, "error")
            raise
        with self._lock:
            self.calls.record(request, "ok")
        return response


__all__ = [
    "CallLog",
    "FailingAdapter",
    "HostedMockAdapter",
    "RecordingAdapter",
    "RuleBasedExtractor",
    "ScriptedAdapter",
    "validate_output",
]
