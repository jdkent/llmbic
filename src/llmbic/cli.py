"""Command-line interface (FR-API-002/003).

Every command takes ``--json`` so it is usable from a pipeline, and every
command goes through :class:`llmbic.project.Project` so the CLI never grows
behaviour the Python API does not have.

``llmbic plan`` is a dry run by definition: it invokes no model and writes no
record state.  ``llmbic run`` is the only command that spends money, and it
refuses to start when the registry has moved since the plan was made.
"""

from __future__ import annotations

import glob
import json as jsonlib
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import click

from . import __version__
from .config import Config, find_config, load_config
from .errors import LlmbicError
from .migration.loader import dump_migration, load_migration_file, save_migration
from .migration.scaffold import scaffold_migration
from .migration.spec import ExecutionPolicy
from .planner.planner import PlannerOptions
from .project import Project

from .registry import PathPreference
from .schema.adapters import from_json_schema, from_linkml
from .store.base import RecordFilter


class Ctx:
    def __init__(self, config: Config | None, store: str | None, as_json: bool) -> None:
        self.config = config
        self.as_json = as_json
        # A store path in the config file is relative to that file, not to
        # wherever the command happened to be run from.
        if store:
            self.store_path = store
        elif config and config.store != ":memory:":
            self.store_path = str(config.resolve(config.store))
        else:
            self.store_path = ":memory:"
        self._project: Project | None = None

    @property
    def project(self) -> Project:
        if self._project is None:
            adapters = {}
            if self.config is not None:
                self.config.load_extensions()
                adapters = self.config.build_adapters()
            self._project = Project(
                self.store_path,
                max_workers=self.config.max_workers if self.config else 4,
                codec=self.config.codec.build() if self.config else None,
            )
            for name, adapter in adapters.items():
                self._project.register_adapter(name, adapter)
        return self._project

    def emit(self, payload: Any, text: str = "") -> None:
        if self.as_json:
            click.echo(jsonlib.dumps(payload, indent=2, sort_keys=True, default=str))
        else:
            click.echo(text or _pretty(payload))


def _pretty(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    return jsonlib.dumps(payload, indent=2, sort_keys=True, default=str)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="llmbic")
@click.option("--config", "-c", type=click.Path(), default=None, help="Path to llmbic.yaml.")
@click.option("--store", "-s", type=click.Path(), default=None, help="Override the store path.")
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output.")
@click.pass_context
def main(ctx: click.Context, config: str | None, store: str | None, as_json: bool) -> None:
    """Semantic schema migration for LLM-extracted records."""

    path = Path(config) if config else find_config()
    cfg = load_config(path) if path else None
    ctx.obj = Ctx(cfg, store, as_json)


# ---- schema --------------------------------------------------------------

@main.group()
def schema() -> None:
    """Register and compare schema versions."""


@schema.command("register")
@click.argument("path", type=click.Path(exists=True))
@click.option("--name", required=True)
@click.option("--version", "version_", required=True)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["json_schema", "linkml"]),
    default="json_schema",
)
@click.option("--root-class", default=None, help="LinkML tree root, when not marked.")
@click.option(
    "--rename",
    "renames",
    multiple=True,
    metavar="NEW_PATH=FIELD_ID",
    help="Pin a path to an existing field identity so a rename keeps its value.",
)
@click.pass_obj
def schema_register(
    obj: Ctx,
    path: str,
    name: str,
    version_: str,
    fmt: str,
    root_class: str | None,
    renames: Sequence[str],
) -> None:
    """Register an immutable schema version."""

    identity_map = dict(r.split("=", 1) for r in renames)
    if fmt == "linkml":
        normalized = from_linkml(
            path, name=name, version=version_, root_class=root_class, identity_map=identity_map
        )
    else:
        document = jsonlib.loads(Path(path).read_text(encoding="utf-8"))
        normalized = from_json_schema(
            document, name=name, version=version_, identity_map=identity_map
        )
    obj.project.register_schema(normalized)
    obj.emit(
        {
            "ref": normalized.ref,
            "fields": len(normalized.fields),
            "collections": len(normalized.collections),
            "structural_hash": normalized.structural_hash(),
            "semantic_hash": normalized.semantic_hash(),
        },
        f"registered {normalized.ref}: {len(normalized.fields)} fields, "
        f"{len(normalized.collections)} collections",
    )


@schema.command("list")
@click.pass_obj
def schema_list(obj: Ctx) -> None:
    """List registered schema versions."""

    rows = [
        {"ref": s.ref, "fields": len(s.fields), "hash": s.schema_hash()}
        for s in obj.project.registry.schemas()
    ]
    obj.emit(rows, "\n".join(f"{r['ref']:<28} {r['fields']:>5} fields  {r['hash']}" for r in rows))


@schema.command("diff")
@click.argument("from_ref")
@click.argument("to_ref")
@click.option("--rename", "renames", multiple=True, metavar="OLD_PATH=NEW_PATH")
@click.pass_obj
def schema_diff(obj: Ctx, from_ref: str, to_ref: str, renames: Sequence[str]) -> None:
    """Compare two registered schema versions."""

    mapping = dict(r.split("=", 1) for r in renames)
    diff = obj.project.diff(from_ref, to_ref, renames=mapping or None)
    obj.emit(diff.to_canonical(), diff.render())


# ---- migration -----------------------------------------------------------

@main.group()
def migration() -> None:
    """Author, validate and inspect migrations."""


@migration.command("new")
@click.argument("from_ref")
@click.argument("to_ref")
@click.option("--id", "migration_id", default=None)
@click.option("--out", "-o", type=click.Path(), default=None, help="Write the scaffold here.")
@click.pass_obj
def migration_new(
    obj: Ctx, from_ref: str, to_ref: str, migration_id: str | None, out: str | None
) -> None:
    """Scaffold an incomplete migration from a schema diff.

    The scaffold never guesses semantics: every step it writes is marked TODO
    and the migration will not validate until a human fills it in.
    """

    reg = obj.project.registry
    diff = reg.diff(from_ref, to_ref)
    scaffolded, notes = scaffold_migration(
        diff, migration_id=migration_id, old=reg.schema(from_ref), new=reg.schema(to_ref)
    )
    text = dump_migration(scaffolded)
    if out:
        save_migration(scaffolded, out)
    payload = {"migration": scaffolded.to_canonical(), "notes": notes, "written_to": out}
    obj.emit(
        payload,
        (f"# written to {out}\n" if out else "")
        + text
        + "\n# notes\n"
        + "\n".join(f"# - {n}" for n in notes),
    )


@migration.command("register")
@click.argument("paths", nargs=-1, type=click.Path())
@click.option("--no-validate", is_flag=True, help="Register without static checks.")
@click.pass_obj
def migration_register(obj: Ctx, paths: Sequence[str], no_validate: bool) -> None:
    """Register migrations from YAML files (globs accepted)."""

    registered = []
    for pattern in paths:
        for path in sorted(glob.glob(pattern)) or [pattern]:
            for m in load_migration_file(path):
                obj.project.register_migration(m, validate=not no_validate)
                registered.append(m.id)
    obj.emit({"registered": registered}, "\n".join(f"registered {m}" for m in registered))


@migration.command("validate")
@click.argument("migration_id", required=False)
@click.pass_obj
def migration_validate(obj: Ctx, migration_id: str | None) -> None:
    """Check that every changed field has an explicit disposition."""

    reg = obj.project.registry
    targets = [reg.migration(migration_id)] if migration_id else reg.migrations()
    results = []
    ok = True
    for m in targets:
        problems = reg.validate_migration(m)
        ok = ok and not problems
        results.append(
            {
                "id": m.id,
                "from": m.from_schema,
                "to": m.to_schema,
                "fidelity": m.fidelity.value,
                "steps": len(m.steps),
                "problems": problems,
            }
        )
    text = "\n".join(
        f"{r['id']:<36} {'OK' if not r['problems'] else 'INVALID'}  "
        f"({r['steps']} steps, {r['fidelity']})"
        + ("".join(f"\n    - {p}" for p in r["problems"]))
        for r in results
    )
    obj.emit(results, text)
    if not ok:
        sys.exit(1)


@migration.command("list")
@click.pass_obj
def migration_list(obj: Ctx) -> None:
    """List registered migrations."""

    rows = [
        {
            "id": m.id,
            "from": m.from_schema,
            "to": m.to_schema,
            "fidelity": m.fidelity.value,
            "steps": len(m.steps),
            "approved": m.approved,
            "branch": m.branch,
        }
        for m in obj.project.registry.migrations()
    ]
    obj.emit(
        rows,
        "\n".join(
            f"{r['id']:<36} {r['from']} -> {r['to']:<16} {r['fidelity']:<18} "
            f"{r['steps']} steps" + ("" if r["approved"] else "  [unapproved]")
            for r in rows
        ),
    )


@migration.command("path")
@click.argument("from_ref")
@click.argument("to_ref")
@click.option("--optimise", type=click.Choice(["fidelity", "cost", "hops", "latency"]), default="fidelity")
@click.pass_obj
def migration_path(obj: Ctx, from_ref: str, to_ref: str, optimise: str) -> None:
    """Show the approved path between two schema versions."""

    pref = PathPreference(optimise=optimise)
    paths = obj.project.registry.find_paths(from_ref, to_ref, preference=pref)
    chosen = obj.project.registry.find_path(from_ref, to_ref, preference=pref)
    payload = {
        "chosen": [m.id for m in chosen],
        "alternatives": [[m.id for m in p] for p in paths],
    }
    obj.emit(
        payload,
        "chosen: " + " > ".join(payload["chosen"] or ["(already there)"])
        + ("\nalternatives:\n  " + "\n  ".join(" > ".join(p) for p in payload["alternatives"])
           if len(paths) > 1 else ""),
    )


@main.group()
def recipe() -> None:
    """Register extraction recipes and controlled vocabularies."""


@recipe.command("register")
@click.argument("paths", nargs=-1)
@click.pass_obj
def recipe_register(obj: Ctx, paths: Sequence[str]) -> None:
    """Register recipes from YAML files or ``module:factory`` references."""

    from .recipe import load_recipes

    registered = []
    for pattern in paths:
        for path in sorted(glob.glob(pattern)) or [pattern]:
            for r in load_recipes(path):
                obj.project.register_recipe(r)
                registered.append(r.ref)
    obj.emit({"registered": registered}, "\n".join(f"registered {r}" for r in registered))


@recipe.command("list")
@click.pass_obj
def recipe_list(obj: Ctx) -> None:
    """List registered recipes with their identity hashes."""

    rows = [
        {
            "ref": r.ref,
            "writes": list(r.writes),
            "context": r.context_policy.describe(),
            "hash": r.recipe_hash(),
        }
        for r in obj.project.registry.recipes()
    ]
    obj.emit(
        rows,
        "\n".join(
            f"{r['ref']:<28} {r['hash']}  -> {', '.join(r['writes']) or '-'}\n"
            f"{'':28}   {r['context']}"
            for r in rows
        ),
    )


@main.group()
def vocabulary() -> None:
    """Register controlled vocabularies."""


@vocabulary.command("register")
@click.argument("paths", nargs=-1)
@click.pass_obj
def vocabulary_register(obj: Ctx, paths: Sequence[str]) -> None:
    from .recipe import load_vocabularies

    registered = []
    for pattern in paths:
        for path in sorted(glob.glob(pattern)) or [pattern]:
            for v in load_vocabularies(path):
                obj.project.register_vocabulary(v)
                registered.append(v.ref)
    obj.emit({"registered": registered}, "\n".join(f"registered {v}" for v in registered))


# ---- records -------------------------------------------------------------

@main.command("ingest")
@click.argument("path", type=click.Path(exists=True))
@click.option("--schema", "schema_ref", required=True)
@click.option("--id-field", default="record_id")
@click.pass_obj
def ingest(obj: Ctx, path: str, schema_ref: str, id_field: str) -> None:
    """Import extracted records from JSON Lines."""

    results = obj.project.ingest_jsonl(path, schema_ref=schema_ref, id_field=id_field)
    payload = [
        {"record_id": r.record_id, "artifacts": r.n_artifacts, "entities": r.n_entities}
        for r in results
    ]
    obj.emit(payload, f"ingested {len(results)} records into {schema_ref}")


@main.command("plan")
@click.argument("target_schema")
@click.option("--record", "record_ids", multiple=True, help="Restrict to these record ids.")
@click.option("--from-schema", default=None, help="Only records at this schema version.")
@click.option("--source-version", default=None)
@click.option("--state", default=None)
@click.option("--failed-in", default=None, help="Only records that failed in this execution.")
@click.option("--limit", type=int, default=None)
@click.option("--allow-lossy", is_flag=True)
@click.option("--allow-destructive", is_flag=True)
@click.option("--approve", "approved", multiple=True, help="Destructive migration ids to permit.")
@click.option("--allow-full-document", is_flag=True)
@click.option("--budget", type=float, default=None, help="USD ceiling for the run.")
@click.option("--max-calls", type=int, default=None)
@click.option("--on-stale", type=click.Choice(["report", "review"]), default="report")
@click.option("--verbose", "-v", is_flag=True)
@click.option("--out", "-o", type=click.Path(), default=None, help="Write the plan JSON here.")
@click.pass_obj
def plan_cmd(
    obj: Ctx,
    target_schema: str,
    record_ids: Sequence[str],
    from_schema: str | None,
    source_version: str | None,
    state: str | None,
    failed_in: str | None,
    limit: int | None,
    allow_lossy: bool,
    allow_destructive: bool,
    approved: Sequence[str],
    allow_full_document: bool,
    budget: float | None,
    max_calls: int | None,
    on_stale: str,
    verbose: bool,
    out: str | None,
) -> None:
    """Produce a migration plan.  Calls no model and mutates no record."""

    plan = obj.project.plan(
        target_schema,
        filt=RecordFilter(
            record_ids=list(record_ids) or None,
            schema_ref=from_schema,
            source_version=source_version,
            state=state,
            failed_in_execution=failed_in,
            limit=limit,
        ),
        policy=ExecutionPolicy(
            allow_lossy=allow_lossy,
            allow_destructive=allow_destructive,
            approved_migrations=tuple(approved),
            allow_full_document=allow_full_document,
            budget_usd=budget,
            max_model_calls=max_calls,
        ),
        options=PlannerOptions(on_stale=on_stale),
    )
    if out:
        Path(out).write_text(jsonlib.dumps(plan.to_canonical(), indent=2), encoding="utf-8")
    obj.emit(plan.to_canonical(), plan.render(verbose=verbose))


@main.command("run")
@click.argument("target_schema", required=False)
@click.option("--plan", "plan_id", default=None, help="Run a stored plan by id.")
@click.option("--plan-file", type=click.Path(exists=True), default=None)
@click.option("--execution-id", default=None)
@click.option("--record", "record_ids", multiple=True)
@click.option("--allow-lossy", is_flag=True)
@click.option("--allow-destructive", is_flag=True)
@click.option("--approve", "approved", multiple=True)
@click.option("--allow-full-document", is_flag=True)
@click.option("--budget", type=float, default=None)
@click.option("--workers", type=int, default=None)
@click.pass_obj
def run_cmd(
    obj: Ctx,
    target_schema: str | None,
    plan_id: str | None,
    plan_file: str | None,
    execution_id: str | None,
    record_ids: Sequence[str],
    allow_lossy: bool,
    allow_destructive: bool,
    approved: Sequence[str],
    allow_full_document: bool,
    budget: float | None,
    workers: int | None,
) -> None:
    """Execute a plan."""

    from .planner.plan import ExecutionPlan

    project = obj.project
    if plan_file:
        plan = ExecutionPlan.from_canonical(
            jsonlib.loads(Path(plan_file).read_text(encoding="utf-8"))
        )
    elif plan_id:
        payload = project.store.get_plan(plan_id)
        if payload is None:
            raise click.ClickException(f"no stored plan {plan_id!r}")
        plan = ExecutionPlan.from_canonical(payload)
    elif target_schema:
        plan = project.plan(
            target_schema,
            policy=ExecutionPolicy(
                allow_lossy=allow_lossy,
                allow_destructive=allow_destructive,
                approved_migrations=tuple(approved),
                allow_full_document=allow_full_document,
                budget_usd=budget,
            ),
        )
    else:
        raise click.ClickException("give a target schema, --plan or --plan-file")

    result = project.run(
        plan,
        execution_id=execution_id,
        record_ids=list(record_ids) or None,
        max_workers=workers,
    )
    obj.emit(
        result.to_canonical(),
        f"{result.execution_id}: {result.status.value}\n"
        + f"  published {len(result.published)}, held {len(result.held)}\n"
        + f"  model calls {result.metrics.model_calls}, cache hits {result.metrics.cache_hits}, "
        + f"cost ${result.metrics.usage.cost_usd:.4f}",
    )


@main.command("resume")
@click.argument("execution_id")
@click.pass_obj
def resume_cmd(obj: Ctx, execution_id: str) -> None:
    """Continue an interrupted execution from its checkpoints."""

    result = obj.project.resume(execution_id)
    obj.emit(
        result.to_canonical(),
        f"{result.execution_id}: {result.status.value}; "
        f"{result.metrics.steps_skipped} steps already done, "
        f"{result.metrics.model_calls} new model calls",
    )


@main.command("status")
@click.argument("execution_id", required=False)
@click.pass_obj
def status_cmd(obj: Ctx, execution_id: str | None) -> None:
    """Show execution state."""

    payload = obj.project.status(execution_id)
    if execution_id is None:
        text = "\n".join(
            f"{e['execution_id']:<24} {e.get('state', '?'):<12} plan={e.get('plan_id', '')[:20]}"
            for e in payload["executions"]
        )
    else:
        text = jsonlib.dumps(payload, indent=2, sort_keys=True, default=str)
    obj.emit(payload, text)


@main.command("diff")
@click.argument("record_id")
@click.option("--before", default=None)
@click.option("--after", default=None)
@click.option("--all", "include_unchanged", is_flag=True)
@click.pass_obj
def diff_cmd(
    obj: Ctx, record_id: str, before: str | None, after: str | None, include_unchanged: bool
) -> None:
    """Field-level diff between two versions of a record."""

    diff = obj.project.diff_record(record_id, before=before, after=after)
    obj.emit(diff.to_canonical(), diff.render(include_unchanged=include_unchanged))


@main.command("record")
@click.argument("record_id")
@click.option("--absent/--no-absent", default=False, help="Include unextracted slots.")
@click.pass_obj
def record_cmd(obj: Ctx, record_id: str, absent: bool) -> None:
    """Show the assembled record as consumers see it."""

    record = obj.project.get_record(record_id, include_absent=absent)
    if record is None:
        raise click.ClickException(f"no current version for {record_id!r}")
    obj.emit(record)


@main.command("export")
@click.argument("path", type=click.Path())
@click.option("--schema", "schema_ref", default=None)
@click.option("--provenance/--no-provenance", default=True)
@click.pass_obj
def export_cmd(obj: Ctx, path: str, schema_ref: str | None, provenance: bool) -> None:
    """Export records as JSON Lines with a provenance sidecar."""

    n = obj.project.export_jsonl(
        path, filt=RecordFilter(schema_ref=schema_ref), include_provenance=provenance
    )
    obj.emit({"written": n, "path": path}, f"wrote {n} records to {path}")


@main.command("evaluate")
@click.argument("target_schema")
@click.argument("gold_path", type=click.Path(exists=True))
@click.option(
    "--gate",
    "gates",
    multiple=True,
    metavar="METRIC[:FIELD][>=MIN][<=MAX]",
    help="Rollout gate, e.g. 'precision>=0.9' or 'change_rate<=0.2'.",
)
@click.option("--allow-lossy", is_flag=True)
@click.pass_obj
def evaluate_cmd(
    obj: Ctx, target_schema: str, gold_path: str, gates: Sequence[str], allow_lossy: bool
) -> None:
    """Score a semantic migration against a frozen adjudicated sample."""

    from .evaluation import GoldCorpus, evaluate

    gold = GoldCorpus.load(gold_path)
    report = evaluate(
        obj.project,
        gold,
        target_schema,
        gates=[_parse_gate(g) for g in gates],
        policy=ExecutionPolicy(allow_lossy=allow_lossy),
    )
    obj.emit(report.to_canonical(), report.render())
    if not report.passed:
        sys.exit(1)


_GATE_RE = re.compile(
    r"^(?P<metric>[a-z_0-9]+)(?::(?P<field>[^<>]+))?"
    r"(?:>=(?P<min>-?[0-9.]+))?(?:<=(?P<max>-?[0-9.]+))?$"
)


def _parse_gate(text: str):
    from .evaluation import RolloutGate

    m = _GATE_RE.match(text.strip())
    if not m or (m.group("min") is None and m.group("max") is None):
        raise click.ClickException(
            f"cannot read gate {text!r}; expected e.g. 'precision>=0.9' or "
            "'precision:tasks[].stimulus_modality>=0.9'"
        )
    return RolloutGate(
        metric=m.group("metric"),
        field_id=m.group("field"),
        min_value=float(m.group("min")) if m.group("min") else None,
        max_value=float(m.group("max")) if m.group("max") else None,
    )


@main.command("reanchor")
@click.argument("source_id")
@click.argument("source_version")
@click.argument("parse_version")
@click.option("--apply/--dry-run", default=False, help="Write the re-anchored artifacts.")
@click.pass_obj
def reanchor_cmd(
    obj: Ctx, source_id: str, source_version: str, parse_version: str, apply: bool
) -> None:
    """Move stored evidence spans onto a new parse of the same source."""

    from .reanchor import reanchor_artifacts

    project = obj.project
    parsed = project.store.get_parsed(source_id, source_version, parse_version)
    if parsed is None:
        raise click.ClickException(
            f"no parse {parse_version!r} stored for {source_id}@{source_version}"
        )
    records = [
        e.record_id for e in project.records() if e.source_id == source_id
    ]
    summary: dict[str, int] = {}
    written = 0
    for record_id in records:
        changed, totals = reanchor_artifacts(project.artifacts(record_id), parsed)
        for k, v in totals.items():
            summary[k] = summary.get(k, 0) + v
        if apply and changed:
            project.store.put_artifacts(changed)
            written += len(changed)
    payload = {
        "records": len(records),
        "outcomes": dict(sorted(summary.items())),
        "written": written,
        "applied": apply,
    }
    obj.emit(
        payload,
        f"{len(records)} records; "
        + ", ".join(f"{k}={v}" for k, v in sorted(summary.items()))
        + (f"; wrote {written} artifacts" if apply else "; dry run, nothing written"),
    )


@main.command("report")
@click.argument("execution_id")
@click.pass_obj
def report_cmd(obj: Ctx, execution_id: str) -> None:
    """Coverage, change rate, failures, review rate and cost for a run."""

    report = obj.project.report(execution_id)
    obj.emit(report.to_canonical(), report.render())


# ---- provenance ----------------------------------------------------------

@main.group()
def provenance() -> None:
    """Inspect where a value came from."""


@provenance.command("show")
@click.argument("record_id")
@click.argument("field_id")
@click.option("--entity", default="")
@click.pass_obj
def provenance_show(obj: Ctx, record_id: str, field_id: str, entity: str) -> None:
    """Every artifact ever written for one slot, oldest first."""

    rows = obj.project.provenance(record_id, field_id, entity)
    if not rows:
        raise click.ClickException(f"no artifacts for {record_id}/{field_id}")
    text_lines = []
    for row in rows:
        p = row["provenance"]
        call = p.get("model_call") or {}
        text_lines.append(
            f"{p.get('created_at', '')}  {row['value']['status']:<16} "
            f"actor={p.get('actor')} recipe={p.get('recipe_ref')} "
            f"migration={p.get('migration_id')} model={call.get('model')} "
            f"units={len(p.get('context_units') or [])} "
            f"evidence={sum(len(e['spans']) for e in row['evidence'])}"
        )
    obj.emit(rows, "\n".join(text_lines))


@main.command("currency")
@click.argument("record_id")
@click.pass_obj
def currency_cmd(obj: Ctx, record_id: str) -> None:
    """Which fields are semantically current, and which merely validate."""

    result = obj.project.currency(record_id)
    counts: dict[str, int] = {}
    for state in result.values():
        counts[state] = counts.get(state, 0) + 1
    obj.emit(
        {"fields": result, "counts": counts},
        "\n".join(f"{v:<16} {k}" for k, v in sorted(result.items()) if v != "current")
        + f"\n\n{counts}",
    )


# ---- review --------------------------------------------------------------

@main.group()
def review() -> None:
    """Export and import curator decisions."""


@review.command("export")
@click.argument("path", type=click.Path())
@click.option("--execution", "execution_id", default=None)
@click.option("--state", default="open")
@click.pass_obj
def review_export(obj: Ctx, path: str, execution_id: str | None, state: str) -> None:
    """Write the review queue as JSON Lines."""

    n = obj.project.review.export_jsonl(path, execution_id=execution_id, state=state)
    obj.emit({"written": n, "path": path}, f"wrote {n} review items to {path}")


@review.command("import")
@click.argument("path", type=click.Path(exists=True))
@click.option("--actor", required=True, help="Who made these decisions.")
@click.option("--schema", "schema_ref", default="")
@click.pass_obj
def review_import(obj: Ctx, path: str, actor: str, schema_ref: str) -> None:
    """Apply curator decisions from JSON Lines."""

    events = obj.project.review.import_jsonl(path, actor_id=actor, schema_ref=schema_ref)
    payload = [e.to_canonical() for e in events]
    obj.emit(payload, f"applied {len(events)} decisions by {actor}")


@review.command("list")
@click.option("--execution", "execution_id", default=None)
@click.option("--state", default="open")
@click.pass_obj
def review_list(obj: Ctx, execution_id: str | None, state: str) -> None:
    """Show the review queue."""

    items = obj.project.review.items(execution_id=execution_id, state=state)
    payload = [i.to_row() for i in items]
    obj.emit(
        payload,
        "\n".join(
            f"{i.item_id[:16]}  {i.record_id:<20} {i.field_id:<28} {i.reason[:60]}"
            for i in items
        ),
    )


def cli() -> None:  # pragma: no cover - console entry point
    try:
        main(standalone_mode=False)
    except LlmbicError as exc:
        click.echo(jsonlib.dumps(exc.to_dict()), err=True)
        sys.exit(2)
    except click.ClickException as exc:
        exc.show()
        sys.exit(exc.exit_code)
    except click.Abort:
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    cli()
