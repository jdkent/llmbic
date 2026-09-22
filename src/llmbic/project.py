"""The Python API (FR-API-001).

One object ties the six layers together: registry, store, planner, engine,
validation and review.  Everything the CLI does, it does through this class,
which is the point — the CLI is an interface, not a second implementation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


from .diffing import MigrationReport, RecordDiff, build_report, diff_records
from .errors import ErrorCode, LlmbicError
from .execution.engine import SOFTWARE_VERSION, Engine
from .execution.state import EventHook, ExecutionResult
from .ids import now
from .migration.spec import ExecutionPolicy, Migration
from .models.base import AdapterRegistry, ModelAdapter
from .planner.dependencies import DependencyContext, semantic_currency_of_record
from .planner.plan import ExecutionPlan
from .planner.planner import Planner, PlannerOptions
from .provenance import FieldArtifact, RecordState, RecordVersion, latest_per_key
from .recipe import ExtractionRecipe, Vocabulary
from .records import PlainCodec, ValueCodec, assemble, decompose
from .registry import Registry
from .review import ReviewQueue
from .schema.diff import SchemaDiff
from .schema.normalized import NormalizedSchema
from .source import ParsedSource, SourceArtifact
from .store.base import RecordFilter, RecordIndexEntry
from .store.sqlite import SqliteStore
from .validation import validate_artifacts


@dataclass
class IngestResult:
    record_id: str
    version_id: str
    n_artifacts: int
    n_entities: int
    validation: list[dict[str, Any]]


class Project:
    """A store plus the registry it holds, and the operations over both."""

    def __init__(
        self,
        store: Any | str | Path = ":memory:",
        *,
        registry: Registry | None = None,
        adapters: AdapterRegistry | None = None,
        codec: ValueCodec | None = None,
        max_workers: int = 4,
        on_event: EventHook | None = None,
        shadow: Mapping[str, str] | None = None,
    ) -> None:
        self.store = (
            SqliteStore(store) if isinstance(store, (str, Path)) else store
        )
        self.adapters = adapters or AdapterRegistry()
        self.codec = codec or PlainCodec()
        self.max_workers = max_workers
        self.on_event = on_event
        #: ``recipe_ref -> alternative`` run in shadow mode (FR-LLM-010).
        self.shadow: dict[str, str] = dict(shadow or {})

        if registry is not None:
            self.registry = registry
        else:
            payload = self.store.load_registry()
            self.registry = Registry.from_canonical(payload) if payload else Registry()

    # ---- lifecycle -------------------------------------------------------
    def save(self) -> None:
        """Persist the registry.  Records and artifacts are already durable."""

        self.store.save_registry(self.registry.to_canonical())

    def close(self) -> None:
        self.save()
        self.store.close()

    def __enter__(self) -> "Project":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---- registration ----------------------------------------------------
    def register_schema(self, schema: NormalizedSchema) -> NormalizedSchema:
        out = self.registry.register_schema(schema)
        self.save()
        return out

    def register_migration(self, migration: Migration, *, validate: bool = True) -> Migration:
        out = self.registry.register_migration(migration, validate=validate)
        self.save()
        return out

    def register_recipe(self, recipe: ExtractionRecipe) -> ExtractionRecipe:
        out = self.registry.register_recipe(recipe)
        self.save()
        return out

    def register_vocabulary(self, vocab: Vocabulary) -> Vocabulary:
        out = self.registry.register_vocabulary(vocab)
        self.save()
        return out

    def register_adapter(self, name: str, adapter: ModelAdapter) -> None:
        self.adapters.register(name, adapter)

    def diff(self, from_ref: str, to_ref: str, **kwargs: Any) -> SchemaDiff:
        return self.registry.diff(from_ref, to_ref, **kwargs)

    # ---- sources ---------------------------------------------------------
    def add_source(
        self, source: SourceArtifact, parsed: ParsedSource | None = None
    ) -> SourceArtifact:
        self.store.put_source(source)
        if parsed is not None:
            self.store.put_parsed(parsed)
        return source

    # ---- ingest ----------------------------------------------------------
    def ingest(
        self,
        record: Mapping[str, Any],
        *,
        schema_ref: str,
        record_id: str,
        source: SourceArtifact | None = None,
        parsed: ParsedSource | None = None,
        codec: ValueCodec | None = None,
        publish: bool = True,
        assert_currency: bool = True,
    ) -> IngestResult:
        """Import an existing extracted record into the artifact store.

        ``assert_currency`` records, for every imported value, the dependency
        hashes that hold *now* under ``schema_ref`` and its declared recipes —
        i.e. the import asserts "these values were produced under this schema
        and these recipes".  That is what makes a later schema change able to
        invalidate exactly the fields it touched.

        Pass ``False`` for a corpus whose provenance you cannot vouch for: the
        values import without dependency hashes and the planner will treat them
        as having no provenance, which conservatively routes them to
        regeneration or review (FR-DEP-006).
        """

        schema = self.registry.schema(schema_ref)
        codec = codec or self.codec
        decomposed = decompose(record, schema, record_id=record_id, codec=codec)
        if assert_currency:
            decomposed.artifacts = self._stamp_currency(
                decomposed.artifacts, schema, source=source, parsed=parsed
            )
        self.store.put_artifacts(decomposed.artifacts)

        if source is not None:
            self.store.put_source(source)
        if parsed is not None:
            self.store.put_parsed(parsed)

        results = validate_artifacts(decomposed.artifacts, schema)
        version = RecordVersion(
            record_id=record_id,
            schema_ref=schema.ref,
            artifact_ids=tuple(sorted(a.artifact_id for a in decomposed.artifacts)),
            entities=tuple(decomposed.entities),
            state=RecordState.PUBLISHED if publish else RecordState.DRAFT,
            source_ref=source.ref if source else None,
            current_field_ids=tuple(
                sorted({a.field_id for a in decomposed.artifacts if a.value.status.has_value})
            ),
            notes={"validation": [r.to_canonical() for r in results]},
        )
        if publish:
            self.store.publish(version)
        else:
            self.store.put_record_version(version)

        self.store.upsert_index(
            RecordIndexEntry(
                record_id=record_id,
                schema_ref=schema.ref,
                source_id=source.source_id if source else None,
                source_version=source.source_version if source else None,
                parse_version=parsed.parse_version if parsed else None,
                state=RecordState.PUBLISHED.value if publish else RecordState.DRAFT.value,
                version_id=version.version_id,
                updated_at=now(),
            )
        )
        return IngestResult(
            record_id=record_id,
            version_id=version.version_id,
            n_artifacts=len(decomposed.artifacts),
            n_entities=len(decomposed.entities),
            validation=[r.to_canonical() for r in results],
        )

    def _stamp_currency(
        self,
        artifacts: Sequence[FieldArtifact],
        schema: NormalizedSchema,
        *,
        source: SourceArtifact | None,
        parsed: ParsedSource | None,
    ) -> list[FieldArtifact]:
        from dataclasses import replace

        from .planner.dependencies import dependency_hashes

        ctx = DependencyContext(
            schema=schema,
            source=source,
            parsed=parsed,
            recipes={r.ref: r for r in self.registry.recipes()},
            vocabularies={v.ref: v for v in self.registry.vocabularies()},
            artifacts={(a.field_id, a.entity): a for a in artifacts},
        )
        out: list[FieldArtifact] = []
        for artifact in artifacts:
            fdef = schema.get(artifact.field_id)
            if fdef is None:
                out.append(artifact)
                continue
            recipe = self.registry.maybe_recipe(fdef.recipe_ref)
            hashes = dependency_hashes(
                None, fdef, ctx, entity=artifact.entity, recipe=recipe
            )
            out.append(
                replace(
                    artifact,
                    provenance=replace(
                        artifact.provenance,
                        recipe_ref=fdef.recipe_ref,
                        source_ref=source.ref if source else None,
                        parse_version=parsed.parse_version if parsed else None,
                        input_hashes=hashes,
                    ),
                )
            )
        return out

    def ingest_jsonl(
        self,
        path: str | Path,
        *,
        schema_ref: str,
        id_field: str = "record_id",
        codec: ValueCodec | None = None,
    ) -> list[IngestResult]:
        out: list[IngestResult] = []
        with Path(path).open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                record = payload.get("record", payload)
                source = (
                    SourceArtifact.from_canonical(payload["source"])
                    if payload.get("source")
                    else None
                )
                parsed = (
                    ParsedSource.from_canonical(payload["parsed"])
                    if payload.get("parsed")
                    else None
                )
                out.append(
                    self.ingest(
                        record,
                        schema_ref=payload.get("schema_ref", schema_ref),
                        record_id=str(payload[id_field])
                        if id_field in payload
                        else str(record[id_field]),
                        source=source,
                        parsed=parsed,
                        codec=codec,
                    )
                )
        return out

    # ---- planning and execution -----------------------------------------
    def plan(
        self,
        target_schema: str,
        *,
        filt: RecordFilter | None = None,
        policy: ExecutionPolicy | None = None,
        options: PlannerOptions | None = None,
    ) -> ExecutionPlan:
        planner = Planner(
            self.registry,
            self.store,
            options=options or PlannerOptions(software_version=SOFTWARE_VERSION),
            adapters=self.adapters,
        )
        plan = planner.plan(target_schema, filt=filt, policy=policy)
        self.store.put_plan(plan.plan_id, plan.to_canonical())
        return plan

    def run(
        self,
        plan: ExecutionPlan,
        *,
        execution_id: str | None = None,
        resume: bool = False,
        record_ids: Sequence[str] | None = None,
        max_workers: int | None = None,
    ) -> ExecutionResult:
        engine = Engine(
            self.registry,
            self.store,
            adapters=self.adapters,
            max_workers=max_workers or self.max_workers,
            on_event=self.on_event,
            codec=self.codec,
            shadow=self.shadow,
        )
        return engine.run(plan, execution_id=execution_id, resume=resume, record_ids=record_ids)

    def resume(self, execution_id: str, *, max_workers: int | None = None) -> ExecutionResult:
        engine = Engine(
            self.registry,
            self.store,
            adapters=self.adapters,
            max_workers=max_workers or self.max_workers,
            on_event=self.on_event,
            codec=self.codec,
            shadow=self.shadow,
        )
        return engine.resume(execution_id)

    def migrate(
        self,
        target_schema: str,
        *,
        filt: RecordFilter | None = None,
        policy: ExecutionPolicy | None = None,
        options: PlannerOptions | None = None,
        dry_run: bool = False,
    ) -> tuple[ExecutionPlan, ExecutionResult | None]:
        """Plan and, unless ``dry_run``, run.  The common path."""

        plan = self.plan(target_schema, filt=filt, policy=policy, options=options)
        if dry_run:
            return plan, None
        return plan, self.run(plan)

    # ---- reading ---------------------------------------------------------
    def records(self, filt: RecordFilter | None = None) -> Iterator[RecordIndexEntry]:
        return self.store.index(filt)

    def artifacts(self, record_id: str, *, latest: bool = True) -> list[FieldArtifact]:
        rows = self.store.get_artifacts(record_id)
        return latest_per_key(rows) if latest else rows

    def get_record(
        self,
        record_id: str,
        *,
        codec: ValueCodec | None = None,
        include_absent: bool = False,
        version_id: str | None = None,
    ) -> dict[str, Any] | None:
        version = (
            self.store.get_record_version(version_id)
            if version_id
            else self.store.current_version(record_id)
        )
        if version is None:
            return None
        schema = self.registry.schema(version.schema_ref)
        wanted = set(version.artifact_ids)
        selected = [a for a in self.store.get_artifacts(record_id) if a.artifact_id in wanted]
        return assemble(
            selected,
            version.entities,
            schema,
            codec=codec or self.codec,
            include_absent=include_absent,
        )

    def provenance(
        self, record_id: str, field_id: str, entity: str = ""
    ) -> list[dict[str, Any]]:
        """Every artifact ever written for one slot, oldest first (FR-PROV-007)."""

        out = []
        for artifact in self.store.get_artifacts(record_id):
            if artifact.field_id == field_id and artifact.entity == entity:
                out.append(artifact.to_canonical())
        return out

    def currency(self, record_id: str) -> dict[str, str]:
        """Per-field semantic currency against the record's own schema version."""

        version = self.store.current_version(record_id)
        if version is None:
            raise LlmbicError(
                f"unknown record {record_id!r}", code=ErrorCode.RECORD_NOT_FOUND
            )
        schema = self.registry.schema(version.schema_ref)
        artifacts = latest_per_key(self.store.get_artifacts(record_id))
        ctx = DependencyContext(
            schema=schema,
            recipes={r.ref: r for r in self.registry.recipes()},
            vocabularies={v.ref: v for v in self.registry.vocabularies()},
            artifacts={(a.field_id, a.entity): a for a in artifacts},
        )
        entities = [""] + [e.entity for e in version.entities]
        return {
            k: v.currency.value for k, v in semantic_currency_of_record(ctx, entities).items()
        }

    def diff_record(
        self, record_id: str, *, before: str | None = None, after: str | None = None
    ) -> RecordDiff:
        """Compare two record versions field by field."""

        versions = self.store.record_versions(record_id)
        if len(versions) < 2 and (before is None or after is None):
            raise LlmbicError(
                f"{record_id} has fewer than two versions to compare",
                code=ErrorCode.RECORD_NOT_FOUND,
            )
        v_before = (
            self.store.get_record_version(before) if before else versions[-2]
        )
        v_after = self.store.get_record_version(after) if after else versions[-1]
        if v_before is None or v_after is None:
            raise LlmbicError("unknown record version", code=ErrorCode.RECORD_NOT_FOUND)

        all_artifacts = self.store.get_artifacts(record_id)
        by_id = {a.artifact_id: a for a in all_artifacts}
        return diff_records(
            [by_id[i] for i in v_before.artifact_ids if i in by_id],
            [by_id[i] for i in v_after.artifact_ids if i in by_id],
            record_id=record_id,
            from_schema=self.registry.schema(v_before.schema_ref),
            to_schema=self.registry.schema(v_after.schema_ref),
        )

    def report(
        self, execution_id: str, *, target_schema: str | None = None
    ) -> MigrationReport:
        execution = self.store.get_execution(execution_id)
        if execution is None:
            raise LlmbicError(
                f"unknown execution {execution_id!r}", code=ErrorCode.EXECUTION_NOT_FOUND
            )
        diffs: list[RecordDiff] = []
        artifacts: list[FieldArtifact] = []
        for record_id, state in sorted((execution.get("records") or {}).items()):
            artifacts.extend(latest_per_key(self.store.get_artifacts(record_id)))
            try:
                diffs.append(self.diff_record(record_id))
            except LlmbicError:
                continue
        schema_ref = target_schema or execution.get("target_schema")
        schema = self.registry.schema(schema_ref) if schema_ref else None
        return build_report(diffs, artifacts, schema, metrics=execution.get("metrics") or {})

    def status(self, execution_id: str | None = None) -> dict[str, Any]:
        if execution_id is None:
            return {"executions": self.store.list_executions(20)}
        execution = self.store.get_execution(execution_id)
        if execution is None:
            raise LlmbicError(
                f"unknown execution {execution_id!r}", code=ErrorCode.EXECUTION_NOT_FOUND
            )
        steps = self.store.get_step_states(execution_id)
        by_state: dict[str, int] = {}
        for s in steps:
            by_state[s.state] = by_state.get(s.state, 0) + 1
        return {
            **execution,
            "steps_by_state": dict(sorted(by_state.items())),
            "open_reviews": len(self.store.get_review_items(execution_id, "open")),
        }

    # ---- export ----------------------------------------------------------
    def export_jsonl(
        self,
        path: str | Path,
        *,
        filt: RecordFilter | None = None,
        codec: ValueCodec | None = None,
        include_provenance: bool = True,
        include_absent: bool = True,
    ) -> int:
        """FR-STO-007 / FR-PROV-007.

        Provenance is a *sidecar* on each line, not embedded in the record
        (decision 18.2): the record stays the shape consumers expect and the
        audit trail joins on ``field_id`` + ``entity``.
        """

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with p.open("w", encoding="utf-8") as fh:
            for entry in self.store.index(filt):
                version = self.store.current_version(entry.record_id)
                if version is None:
                    continue
                schema = self.registry.schema(version.schema_ref)
                wanted = set(version.artifact_ids)
                selected = [
                    a
                    for a in self.store.get_artifacts(entry.record_id)
                    if a.artifact_id in wanted
                ]
                line: dict[str, Any] = {
                    "record_id": entry.record_id,
                    "schema_ref": version.schema_ref,
                    "version_id": version.version_id,
                    "source_ref": version.source_ref,
                    "record": assemble(
                        selected,
                        version.entities,
                        schema,
                        codec=codec or self.codec,
                        include_absent=include_absent,
                    ),
                }
                if include_provenance:
                    line["provenance"] = [
                        {
                            "field_id": a.field_id,
                            "entity": a.entity,
                            "path": schema.get(a.field_id).path if schema.get(a.field_id) else None,
                            "value_status": a.value.status.value,
                            "artifact_id": a.artifact_id,
                            "evidence": [e.to_canonical() for e in a.evidence],
                            **a.provenance.to_canonical(),
                        }
                        for a in sorted(selected, key=lambda x: (x.field_id, x.entity))
                    ]
                fh.write(json.dumps(line, ensure_ascii=False) + "\n")
                count += 1
        return count

    # ---- review ----------------------------------------------------------
    @property
    def review(self) -> ReviewQueue:
        return ReviewQueue(self.store)


__all__ = ["IngestResult", "Project"]
