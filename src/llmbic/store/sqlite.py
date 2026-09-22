"""SQLite reference backend (FR-STO-002).

Everything llmbic persists goes through this one class.  It is the development
and single-machine production backend; a PostgreSQL adapter implements the
same :class:`llmbic.store.base.Store` protocol without the core noticing.

Notes on the schema below:

* ``field_artifact`` and ``record_version`` are insert-only.  ``INSERT OR
  IGNORE`` on a content-addressed primary key is what makes re-running a
  migration idempotent (product principle 7).
* ``cache.cache_key`` is ``PRIMARY KEY``, which is FR-STO-004's uniqueness
  requirement, and ``cache_put`` reports whether it actually inserted — that
  is how the engine proves it never paid twice for the same call.
* The internal table layout has its own version in ``meta`` (NFR-MNT-004);
  it is *not* the user's schema version and the two never mix.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from ..errors import ErrorCode, StorageError
from ..ids import now
from ..provenance import FieldArtifact, RecordState, RecordVersion, ReviewEvent
from ..source import ParsedSource, SourceArtifact
from .base import CacheEntry, RecordFilter, RecordIndexEntry, StepState

STORE_FORMAT_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS registry (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    payload TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS source_artifact (
    source_id TEXT NOT NULL,
    source_version TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (source_id, source_version)
);

CREATE TABLE IF NOT EXISTS parsed_source (
    source_id TEXT NOT NULL,
    source_version TEXT NOT NULL,
    parse_version TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (source_id, source_version, parse_version)
);

CREATE TABLE IF NOT EXISTS field_artifact (
    artifact_id TEXT PRIMARY KEY,
    record_id TEXT NOT NULL,
    field_id TEXT NOT NULL,
    entity TEXT NOT NULL DEFAULT '',
    recipe_ref TEXT,
    value_status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_artifact_record ON field_artifact (record_id);
CREATE INDEX IF NOT EXISTS ix_artifact_field ON field_artifact (record_id, field_id, entity);

CREATE TABLE IF NOT EXISTS record_version (
    version_id TEXT PRIMARY KEY,
    record_id TEXT NOT NULL,
    schema_ref TEXT NOT NULL,
    state TEXT NOT NULL,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_version_record ON record_version (record_id);

CREATE TABLE IF NOT EXISTS record_pointer (
    record_id TEXT PRIMARY KEY,
    version_id TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS record_index (
    record_id TEXT PRIMARY KEY,
    schema_ref TEXT NOT NULL,
    source_id TEXT,
    source_version TEXT,
    parse_version TEXT,
    state TEXT NOT NULL,
    version_id TEXT,
    labels TEXT NOT NULL DEFAULT '[]',
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_index_schema ON record_index (schema_ref);

CREATE TABLE IF NOT EXISTS execution (
    execution_id TEXT PRIMARY KEY,
    plan_id TEXT,
    state TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS execution_step (
    execution_id TEXT NOT NULL,
    step_key TEXT NOT NULL,
    record_id TEXT NOT NULL DEFAULT '',
    state TEXT NOT NULL,
    artifact_id TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    detail TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (execution_id, step_key)
);
CREATE INDEX IF NOT EXISTS ix_step_state ON execution_step (execution_id, state);

CREATE TABLE IF NOT EXISTS attempt (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    execution_id TEXT NOT NULL,
    step_key TEXT NOT NULL,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_attempt_step ON attempt (execution_id, step_key);

CREATE TABLE IF NOT EXISTS cache (
    cache_key TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    execution_id TEXT,
    usage TEXT NOT NULL DEFAULT '{}',
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS review_item (
    item_id TEXT PRIMARY KEY,
    execution_id TEXT,
    record_id TEXT NOT NULL,
    field_id TEXT NOT NULL,
    entity TEXT NOT NULL DEFAULT '',
    state TEXT NOT NULL DEFAULT 'open',
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_review_state ON review_item (state);

CREATE TABLE IF NOT EXISTS review_event (
    event_id TEXT PRIMARY KEY,
    record_id TEXT NOT NULL,
    field_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_event_record ON review_event (record_id);

CREATE TABLE IF NOT EXISTS plan (
    plan_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL
);
"""


class SqliteStore:
    """Thread-safe SQLite store.

    One connection guarded by a re-entrant lock: the engine's concurrency is
    in model calls, not in the database, and a single writer keeps every state
    transition transactional (NFR-REL-001).
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(_SCHEMA)
            self._conn.execute(
                "INSERT OR IGNORE INTO meta (key, value) VALUES ('store_format_version', ?)",
                (str(STORE_FORMAT_VERSION),),
            )
            self._conn.commit()
        self._check_format()

    # ---- plumbing --------------------------------------------------------
    def _check_format(self) -> None:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = 'store_format_version'"
        ).fetchone()
        found = int(row["value"]) if row else 0
        if found != STORE_FORMAT_VERSION:
            raise StorageError(
                f"store was written by format version {found}, this build speaks "
                f"{STORE_FORMAT_VERSION}; run the store's own migration first "
                "(this is unrelated to your record schema versions)",
                code=ErrorCode.STORAGE_CONFLICT,
            )

    def close(self) -> None:
        with self._lock:
            self._conn.commit()
            self._conn.close()

    def __enter__(self) -> "SqliteStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _exec(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, tuple(params))
            self._conn.commit()
            return cur

    def _query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute(sql, tuple(params)).fetchall())

    # ---- registry --------------------------------------------------------
    def save_registry(self, payload: Mapping[str, Any]) -> None:
        self._exec(
            "INSERT INTO registry (id, payload, updated_at) VALUES (1, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET payload = excluded.payload, "
            "updated_at = excluded.updated_at",
            (json.dumps(payload), now()),
        )

    def load_registry(self) -> dict[str, Any]:
        rows = self._query("SELECT payload FROM registry WHERE id = 1")
        return json.loads(rows[0]["payload"]) if rows else {}

    # ---- sources ---------------------------------------------------------
    def put_source(self, source: SourceArtifact) -> None:
        self._exec(
            "INSERT OR REPLACE INTO source_artifact (source_id, source_version, payload) "
            "VALUES (?, ?, ?)",
            (source.source_id, source.source_version, json.dumps(source.to_canonical())),
        )

    def get_source(self, source_id: str, source_version: str) -> SourceArtifact | None:
        rows = self._query(
            "SELECT payload FROM source_artifact WHERE source_id = ? AND source_version = ?",
            (source_id, source_version),
        )
        return SourceArtifact.from_canonical(json.loads(rows[0]["payload"])) if rows else None

    def put_parsed(self, parsed: ParsedSource) -> None:
        self._exec(
            "INSERT OR REPLACE INTO parsed_source "
            "(source_id, source_version, parse_version, payload) VALUES (?, ?, ?, ?)",
            (
                parsed.source_id,
                parsed.source_version,
                parsed.parse_version,
                json.dumps(parsed.to_canonical()),
            ),
        )

    def get_parsed(
        self, source_id: str, source_version: str, parse_version: str | None = None
    ) -> ParsedSource | None:
        if parse_version:
            rows = self._query(
                "SELECT payload FROM parsed_source WHERE source_id = ? AND source_version = ? "
                "AND parse_version = ?",
                (source_id, source_version, parse_version),
            )
        else:
            rows = self._query(
                "SELECT payload FROM parsed_source WHERE source_id = ? AND source_version = ? "
                "ORDER BY parse_version DESC LIMIT 1",
                (source_id, source_version),
            )
        return ParsedSource.from_canonical(json.loads(rows[0]["payload"])) if rows else None

    def list_parse_versions(self, source_id: str, source_version: str) -> list[str]:
        rows = self._query(
            "SELECT parse_version FROM parsed_source WHERE source_id = ? AND source_version = ? "
            "ORDER BY parse_version",
            (source_id, source_version),
        )
        return [r["parse_version"] for r in rows]

    # ---- records ---------------------------------------------------------
    def put_artifacts(self, artifacts: Sequence[FieldArtifact]) -> list[str]:
        written: list[str] = []
        with self._lock:
            for a in artifacts:
                payload = a.to_canonical()
                self._conn.execute(
                    "INSERT OR IGNORE INTO field_artifact "
                    "(artifact_id, record_id, field_id, entity, recipe_ref, value_status, "
                    " created_at, payload) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        payload["artifact_id"],
                        a.record_id,
                        a.field_id,
                        a.entity,
                        a.provenance.recipe_ref,
                        a.value.status.value,
                        a.provenance.created_at,
                        json.dumps(payload),
                    ),
                )
                written.append(payload["artifact_id"])
            self._conn.commit()
        return written

    def get_artifacts(self, record_id: str) -> list[FieldArtifact]:
        rows = self._query(
            "SELECT payload FROM field_artifact WHERE record_id = ? ORDER BY created_at, artifact_id",
            (record_id,),
        )
        return [FieldArtifact.from_canonical(json.loads(r["payload"])) for r in rows]

    def get_artifact(self, artifact_id: str) -> FieldArtifact | None:
        rows = self._query(
            "SELECT payload FROM field_artifact WHERE artifact_id = ?", (artifact_id,)
        )
        return FieldArtifact.from_canonical(json.loads(rows[0]["payload"])) if rows else None

    def get_artifacts_by_ids(self, artifact_ids: Sequence[str]) -> list[FieldArtifact]:
        if not artifact_ids:
            return []
        out: list[FieldArtifact] = []
        for i in range(0, len(artifact_ids), 400):
            chunk = artifact_ids[i : i + 400]
            marks = ",".join("?" * len(chunk))
            rows = self._query(
                f"SELECT payload FROM field_artifact WHERE artifact_id IN ({marks})", chunk
            )
            out.extend(FieldArtifact.from_canonical(json.loads(r["payload"])) for r in rows)
        return out

    def put_record_version(self, version: RecordVersion) -> str:
        payload = version.to_canonical()
        self._exec(
            "INSERT OR IGNORE INTO record_version "
            "(version_id, record_id, schema_ref, state, created_at, payload) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                payload["version_id"],
                version.record_id,
                version.schema_ref,
                version.state.value,
                version.created_at,
                json.dumps(payload),
            ),
        )
        return payload["version_id"]

    def get_record_version(self, version_id: str) -> RecordVersion | None:
        rows = self._query(
            "SELECT payload FROM record_version WHERE version_id = ?", (version_id,)
        )
        return RecordVersion.from_canonical(json.loads(rows[0]["payload"])) if rows else None

    def record_versions(self, record_id: str) -> list[RecordVersion]:
        rows = self._query(
            "SELECT payload FROM record_version WHERE record_id = ? ORDER BY created_at, version_id",
            (record_id,),
        )
        return [RecordVersion.from_canonical(json.loads(r["payload"])) for r in rows]

    def publish(self, version: RecordVersion) -> None:
        """Atomically make ``version`` the record consumers see (FR-EXE-007).

        The version row and the pointer move in one transaction, so a crash
        between them is impossible: a consumer sees the prior complete record
        or the new complete one.
        """

        if version.state is not RecordState.PUBLISHED:
            version = RecordVersion(
                **{**_as_kwargs(version), "state": RecordState.PUBLISHED}
            )
        payload = version.to_canonical()
        stamp = now()
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                self._conn.execute(
                    "INSERT OR IGNORE INTO record_version "
                    "(version_id, record_id, schema_ref, state, created_at, payload) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        payload["version_id"],
                        version.record_id,
                        version.schema_ref,
                        version.state.value,
                        version.created_at,
                        json.dumps(payload),
                    ),
                )
                prior = self._conn.execute(
                    "SELECT version_id FROM record_pointer WHERE record_id = ?",
                    (version.record_id,),
                ).fetchone()
                if prior and prior["version_id"] != payload["version_id"]:
                    row = self._conn.execute(
                        "SELECT payload FROM record_version WHERE version_id = ?",
                        (prior["version_id"],),
                    ).fetchone()
                    superseded = json.loads(row["payload"]) if row else {}
                    superseded["state"] = RecordState.SUPERSEDED.value
                    self._conn.execute(
                        "UPDATE record_version SET state = ?, payload = ? WHERE version_id = ?",
                        (
                            RecordState.SUPERSEDED.value,
                            json.dumps(superseded),
                            prior["version_id"],
                        ),
                    )
                self._conn.execute(
                    "INSERT INTO record_pointer (record_id, version_id, updated_at) "
                    "VALUES (?, ?, ?) ON CONFLICT(record_id) DO UPDATE SET "
                    "version_id = excluded.version_id, updated_at = excluded.updated_at",
                    (version.record_id, payload["version_id"], stamp),
                )
                self._conn.execute(
                    "INSERT INTO record_index "
                    "(record_id, schema_ref, source_id, source_version, parse_version, "
                    " state, version_id, labels, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, '[]', ?) "
                    "ON CONFLICT(record_id) DO UPDATE SET schema_ref = excluded.schema_ref, "
                    "state = excluded.state, version_id = excluded.version_id, "
                    "updated_at = excluded.updated_at",
                    (
                        version.record_id,
                        version.schema_ref,
                        (version.source_ref or "@").split("@")[0] or None,
                        _source_version_of(version.source_ref),
                        None,
                        RecordState.PUBLISHED.value,
                        payload["version_id"],
                        stamp,
                    ),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def current_version(self, record_id: str) -> RecordVersion | None:
        rows = self._query(
            "SELECT rv.payload FROM record_pointer rp "
            "JOIN record_version rv ON rv.version_id = rp.version_id "
            "WHERE rp.record_id = ?",
            (record_id,),
        )
        return RecordVersion.from_canonical(json.loads(rows[0]["payload"])) if rows else None

    def upsert_index(self, entry: RecordIndexEntry) -> None:
        self._exec(
            "INSERT INTO record_index "
            "(record_id, schema_ref, source_id, source_version, parse_version, state, "
            " version_id, labels, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(record_id) DO UPDATE SET schema_ref = excluded.schema_ref, "
            "source_id = excluded.source_id, source_version = excluded.source_version, "
            "parse_version = excluded.parse_version, state = excluded.state, "
            "version_id = excluded.version_id, labels = excluded.labels, "
            "updated_at = excluded.updated_at",
            (
                entry.record_id,
                entry.schema_ref,
                entry.source_id,
                entry.source_version,
                entry.parse_version,
                entry.state,
                entry.version_id,
                json.dumps(list(entry.labels)),
                entry.updated_at or now(),
            ),
        )

    def index(self, filt: RecordFilter | None = None) -> Iterator[RecordIndexEntry]:
        filt = filt or RecordFilter()
        sql = "SELECT * FROM record_index"
        clauses: list[str] = []
        params: list[Any] = []
        if filt.record_ids:
            marks = ",".join("?" * len(filt.record_ids))
            clauses.append(f"record_id IN ({marks})")
            params.extend(filt.record_ids)
        if filt.schema_ref:
            clauses.append("schema_ref = ?")
            params.append(filt.schema_ref)
        if filt.source_version:
            clauses.append("source_version = ?")
            params.append(filt.source_version)
        if filt.state:
            clauses.append("state = ?")
            params.append(filt.state)
        if filt.failed_in_execution:
            clauses.append(
                "record_id IN (SELECT record_id FROM execution_step WHERE execution_id = ? "
                "AND state IN ('failed', 'blocked', 'review_needed'))"
            )
            params.append(filt.failed_in_execution)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY record_id"
        if filt.limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params.extend([filt.limit, filt.offset])
        elif filt.offset:
            sql += " LIMIT -1 OFFSET ?"
            params.append(filt.offset)

        for row in self._query(sql, params):
            yield RecordIndexEntry(
                record_id=row["record_id"],
                schema_ref=row["schema_ref"],
                source_id=row["source_id"],
                source_version=row["source_version"],
                parse_version=row["parse_version"],
                state=row["state"],
                version_id=row["version_id"],
                updated_at=row["updated_at"],
                labels=tuple(json.loads(row["labels"] or "[]")),
            )

    def count_records(self, filt: RecordFilter | None = None) -> int:
        return sum(1 for _ in self.index(filt))

    # ---- execution -------------------------------------------------------
    def create_execution(self, execution_id: str, payload: Mapping[str, Any]) -> None:
        stamp = now()
        self._exec(
            "INSERT OR IGNORE INTO execution "
            "(execution_id, plan_id, state, created_at, updated_at, payload) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                execution_id,
                payload.get("plan_id"),
                payload.get("state", "planned"),
                stamp,
                stamp,
                json.dumps(dict(payload)),
            ),
        )

    def update_execution(self, execution_id: str, **changes: Any) -> None:
        current = self.get_execution(execution_id)
        if current is None:
            raise StorageError(
                f"unknown execution {execution_id!r}", code=ErrorCode.EXECUTION_NOT_FOUND
            )
        merged = {**current, **changes}
        self._exec(
            "UPDATE execution SET state = ?, updated_at = ?, payload = ? WHERE execution_id = ?",
            (merged.get("state", "running"), now(), json.dumps(merged), execution_id),
        )

    def get_execution(self, execution_id: str) -> dict[str, Any] | None:
        rows = self._query(
            "SELECT payload FROM execution WHERE execution_id = ?", (execution_id,)
        )
        return json.loads(rows[0]["payload"]) if rows else None

    def list_executions(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._query(
            "SELECT payload FROM execution ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        return [json.loads(r["payload"]) for r in rows]

    def put_step_state(self, state: StepState) -> None:
        self._exec(
            "INSERT INTO execution_step "
            "(execution_id, step_key, record_id, state, artifact_id, attempts, updated_at, detail) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(execution_id, step_key) DO UPDATE SET state = excluded.state, "
            "record_id = excluded.record_id, artifact_id = excluded.artifact_id, "
            "attempts = excluded.attempts, updated_at = excluded.updated_at, "
            "detail = excluded.detail",
            (
                state.execution_id,
                state.step_key,
                state.record_id,
                state.state,
                state.artifact_id,
                state.attempts,
                state.updated_at or now(),
                json.dumps(state.detail),
            ),
        )

    def get_step_states(self, execution_id: str) -> list[StepState]:
        rows = self._query(
            "SELECT * FROM execution_step WHERE execution_id = ? ORDER BY step_key",
            (execution_id,),
        )
        return [_row_to_step(r) for r in rows]

    def get_step_state(self, execution_id: str, step_key: str) -> StepState | None:
        rows = self._query(
            "SELECT * FROM execution_step WHERE execution_id = ? AND step_key = ?",
            (execution_id, step_key),
        )
        return _row_to_step(rows[0]) if rows else None

    def put_attempt(self, execution_id: str, step_key: str, payload: Mapping[str, Any]) -> None:
        self._exec(
            "INSERT INTO attempt (execution_id, step_key, created_at, payload) VALUES (?, ?, ?, ?)",
            (execution_id, step_key, now(), json.dumps(dict(payload))),
        )

    def get_attempts(
        self, execution_id: str, step_key: str | None = None
    ) -> list[dict[str, Any]]:
        if step_key:
            rows = self._query(
                "SELECT payload FROM attempt WHERE execution_id = ? AND step_key = ? ORDER BY id",
                (execution_id, step_key),
            )
        else:
            rows = self._query(
                "SELECT payload FROM attempt WHERE execution_id = ? ORDER BY id", (execution_id,)
            )
        return [json.loads(r["payload"]) for r in rows]

    # ---- cache -----------------------------------------------------------
    def cache_get(self, cache_key: str) -> CacheEntry | None:
        rows = self._query("SELECT * FROM cache WHERE cache_key = ?", (cache_key,))
        if not rows:
            return None
        row = rows[0]
        return CacheEntry(
            cache_key=row["cache_key"],
            payload=json.loads(row["payload"]),
            created_at=row["created_at"],
            execution_id=row["execution_id"],
            usage=json.loads(row["usage"] or "{}"),
        )

    def cache_put(self, entry: CacheEntry) -> bool:
        """Insert if absent.  Returns True when this call actually wrote."""

        with self._lock:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO cache (cache_key, created_at, execution_id, usage, payload) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    entry.cache_key,
                    entry.created_at or now(),
                    entry.execution_id,
                    json.dumps(entry.usage),
                    json.dumps(entry.payload),
                ),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def cache_size(self) -> int:
        return int(self._query("SELECT COUNT(*) AS n FROM cache")[0]["n"])

    # ---- review ----------------------------------------------------------
    def put_review_item(self, item: Mapping[str, Any]) -> None:
        self._exec(
            "INSERT INTO review_item "
            "(item_id, execution_id, record_id, field_id, entity, state, created_at, payload) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(item_id) DO UPDATE SET state = excluded.state, "
            "execution_id = excluded.execution_id, payload = excluded.payload",
            (
                item["item_id"],
                item.get("execution_id"),
                item["record_id"],
                item["field_id"],
                item.get("entity", ""),
                item.get("state", "open"),
                item.get("created_at") or now(),
                json.dumps(dict(item)),
            ),
        )

    def get_review_items(
        self, execution_id: str | None = None, state: str | None = None
    ) -> list[dict[str, Any]]:
        clauses, params = [], []
        if execution_id:
            clauses.append("execution_id = ?")
            params.append(execution_id)
        if state:
            clauses.append("state = ?")
            params.append(state)
        sql = "SELECT payload FROM review_item"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at, item_id"
        return [json.loads(r["payload"]) for r in self._query(sql, params)]

    def update_review_item(self, item_id: str, **changes: Any) -> None:
        rows = self._query("SELECT payload FROM review_item WHERE item_id = ?", (item_id,))
        if not rows:
            raise StorageError(f"unknown review item {item_id!r}", code=ErrorCode.RECORD_NOT_FOUND)
        merged = {**json.loads(rows[0]["payload"]), **changes}
        self._exec(
            "UPDATE review_item SET state = ?, payload = ? WHERE item_id = ?",
            (merged.get("state", "open"), json.dumps(merged), item_id),
        )

    def put_review_event(self, event: ReviewEvent) -> None:
        payload = event.to_canonical()
        self._exec(
            "INSERT OR IGNORE INTO review_event "
            "(event_id, record_id, field_id, created_at, payload) VALUES (?, ?, ?, ?, ?)",
            (event.event_id, event.record_id, event.field_id, event.created_at, json.dumps(payload)),
        )

    def get_review_events(self, record_id: str | None = None) -> list[ReviewEvent]:
        if record_id:
            rows = self._query(
                "SELECT payload FROM review_event WHERE record_id = ? ORDER BY created_at",
                (record_id,),
            )
        else:
            rows = self._query("SELECT payload FROM review_event ORDER BY created_at")
        return [ReviewEvent.from_canonical(json.loads(r["payload"])) for r in rows]

    # ---- plans -----------------------------------------------------------
    def put_plan(self, plan_id: str, payload: Mapping[str, Any]) -> None:
        self._exec(
            "INSERT OR REPLACE INTO plan (plan_id, created_at, payload) VALUES (?, ?, ?)",
            (plan_id, now(), json.dumps(dict(payload))),
        )

    def get_plan(self, plan_id: str) -> dict[str, Any] | None:
        rows = self._query("SELECT payload FROM plan WHERE plan_id = ?", (plan_id,))
        return json.loads(rows[0]["payload"]) if rows else None

    def list_plans(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._query(
            "SELECT payload FROM plan ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        return [json.loads(r["payload"]) for r in rows]


def _row_to_step(row: sqlite3.Row) -> StepState:
    return StepState(
        execution_id=row["execution_id"],
        step_key=row["step_key"],
        state=row["state"],
        record_id=row["record_id"],
        artifact_id=row["artifact_id"],
        attempts=int(row["attempts"]),
        detail=json.loads(row["detail"] or "{}"),
        updated_at=row["updated_at"],
    )


def _as_kwargs(version: RecordVersion) -> dict[str, Any]:
    return {
        "record_id": version.record_id,
        "schema_ref": version.schema_ref,
        "artifact_ids": version.artifact_ids,
        "entities": version.entities,
        "state": version.state,
        "parent_version_id": version.parent_version_id,
        "execution_id": version.execution_id,
        "source_ref": version.source_ref,
        "created_at": version.created_at,
        "current_field_ids": version.current_field_ids,
        "notes": version.notes,
    }


def _source_version_of(source_ref: str | None) -> str | None:
    if not source_ref:
        return None
    return source_ref.split("@", 1)[1].split("/", 1)[0] if "@" in source_ref else None
