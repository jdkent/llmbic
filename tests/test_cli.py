"""The command line, end to end (FR-API-002/003)."""

from __future__ import annotations

import json


import pytest
import yaml
from click.testing import CliRunner

import studybed as sb
from llmbic.cli import main
from llmbic.migration.loader import migration_to_dict


@pytest.fixture
def workspace(tmp_path):
    """A directory with schemas, migrations, records and a config file."""

    root = tmp_path / "ws"
    (root / "schemas").mkdir(parents=True)
    (root / "migrations").mkdir()

    for version in ("1.0", "1.1", "1.2", "1.3", "1.4"):
        (root / "schemas" / f"study_{version}.json").write_text(
            json.dumps(sb.SCHEMA_DOCS[version]())
        )
    for builder in (
        sb.migration_1_0_to_1_1,
        sb.migration_1_1_to_1_2,
        sb.migration_1_2_to_1_3,
        sb.migration_1_3_to_1_4,
    ):
        m = builder()
        (root / "migrations" / f"{m.id}.yaml").write_text(
            yaml.safe_dump(migration_to_dict(m), sort_keys=False)
        )

    (root / "recipes.yaml").write_text(
        yaml.safe_dump(
            {
                "recipes": [
                    sb.stimulus_recipe("1").to_canonical(),
                    sb.stimulus_recipe("2").to_canonical(),
                    sb.assignment_escalation_recipe().to_canonical(),
                ]
            },
            sort_keys=False,
        )
    )
    (root / "vocabularies.yaml").write_text(
        yaml.safe_dump(
            {"vocabularies": [v.to_canonical() for v in sb.VOCABULARIES.values()]},
            sort_keys=False,
        )
    )

    papers = sb.corpus(8)
    with (root / "records.jsonl").open("w") as fh:
        for paper in papers:
            fh.write(
                json.dumps(
                    {
                        "record_id": paper.record_id,
                        "record": paper.record,
                        "source": paper.source.to_canonical(),
                        "parsed": paper.parsed.to_canonical(),
                    }
                )
                + "\n"
            )

    (root / "llmbic.yaml").write_text(
        yaml.safe_dump(
            {
                "store": "study.db",
                "codec": "extracted_value",
                "adapters": {
                    "main": {"kind": "studybed:stimulus_adapter"},
                    "backup": {
                        "kind": "studybed:stimulus_adapter",
                        "options": {"provider": "b", "model": "r2"},
                    },
                },
                "extensions": ["studybed"],
                "max_workers": 2,
            }
        )
    )
    return root, papers


@pytest.fixture
def run(workspace):
    root, papers = workspace
    runner = CliRunner()

    def _run(*args, expect_ok: bool = True, json_out: bool = False):
        argv = ["--config", str(root / "llmbic.yaml")]
        if json_out:
            argv.append("--json")
        argv.extend(args)
        result = runner.invoke(main, argv, catch_exceptions=False)
        if expect_ok and result.exit_code != 0:
            raise AssertionError(f"{argv} failed ({result.exit_code}):\n{result.output}")
        return result

    return _run


def _bootstrap(run, root):
    for version in ("1.0", "1.1", "1.2", "1.3", "1.4"):
        args = [
            "schema",
            "register",
            str(root / "schemas" / f"study_{version}.json"),
            "--name",
            "study",
            "--version",
            version,
        ]
        if version != "1.0":
            args += ["--rename", "tasks[].response_modality=tasks[].response_mode"]
        run(*args)
    run("vocabulary", "register", str(root / "vocabularies.yaml"))
    run("recipe", "register", str(root / "recipes.yaml"))
    run("migration", "register", str(root / "migrations" / "*.yaml"))
    run("ingest", str(root / "records.jsonl"), "--schema", "study@1.0")


def test_the_whole_workflow_from_the_command_line(run, workspace):
    root, papers = workspace
    _bootstrap(run, root)

    listed = run("schema", "list").output
    assert "study@1.0" in listed and "study@1.4" in listed

    diff = run("schema", "diff", "study@1.0", "study@1.1").output
    assert "renamed" in diff and "tasks[].stimulus_modality" in diff

    validated = run("migration", "validate").output
    assert "INVALID" not in validated

    plan = run("plan", "study@1.1", json_out=True)
    payload = json.loads(plan.output)
    assert payload["summary"]["n_model_calls"] > 0
    assert payload["summary"]["n_full_document_transmissions"] == 0

    ran = run("run", "study@1.1", json_out=True)
    result = json.loads(ran.output)
    assert result["status"] in ("succeeded", "partial")
    assert result["metrics"]["model_calls"] > 0

    again = json.loads(run("plan", "study@1.1", json_out=True).output)
    assert again["summary"]["n_model_calls"] == 0


def test_plan_is_a_dry_run(run, workspace):
    root, papers = workspace
    _bootstrap(run, root)
    before = json.loads(run("record", papers[0].record_id, json_out=True).output)
    run("plan", "study@1.1")
    after = json.loads(run("record", papers[0].record_id, json_out=True).output)
    assert before == after


def test_migration_new_scaffolds_a_file(run, workspace, tmp_path):
    root, _ = workspace
    _bootstrap(run, root)
    out = tmp_path / "scaffold.yaml"
    result = run("migration", "new", "study@1.0", "study@1.1", "--out", str(out))
    assert out.exists()
    assert "TODO" in out.read_text()
    assert "notes" in result.output or "#" in result.output


def test_migration_validate_exits_nonzero_on_an_invalid_migration(run, workspace, tmp_path):
    root, _ = workspace
    _bootstrap(run, root)
    out = tmp_path / "scaffold.yaml"
    run("migration", "new", "study@1.0", "study@1.1", "--out", str(out))
    run("migration", "register", str(out), "--no-validate")
    result = run("migration", "validate", expect_ok=False)
    assert result.exit_code == 1
    assert "INVALID" in result.output


def test_migration_path_shows_the_chosen_route(run, workspace):
    root, _ = workspace
    _bootstrap(run, root)
    out = run("migration", "path", "study@1.0", "study@1.4").output
    assert "study-1.0-to-1.1 > study-1.1-to-1.2" in out


def test_diff_and_provenance_and_currency(run, workspace):
    root, papers = workspace
    _bootstrap(run, root)
    run("run", "study@1.1")

    record_id = papers[0].record_id
    diff = run("diff", record_id).output
    assert "study@1.0 -> study@1.1" in diff

    prov = json.loads(
        run(
            "provenance",
            "show",
            record_id,
            "tasks[].stimulus_modality",
            "--entity",
            "tasks[]=t1",
            json_out=True,
        ).output
    )
    assert prov[-1]["provenance"]["recipe_ref"] == "stimulus-modality@1"

    currency = json.loads(run("currency", record_id, json_out=True).output)
    assert "counts" in currency and currency["counts"]


def test_review_export_and_import_round_trip(run, workspace, tmp_path):
    root, _ = workspace
    _bootstrap(run, root)
    run("run", "study@1.1")
    run("run", "study@1.2", "--allow-lossy")
    run("run", "study@1.3", "--allow-lossy")

    queue = tmp_path / "queue.jsonl"
    run("review", "export", str(queue))
    rows = [json.loads(l) for l in queue.read_text().splitlines()]
    assert rows, "the corpus should have produced review items by study@1.3"
    for row in rows:
        row["decision"] = "accept"
        row["rationale"] = "checked"
    queue.write_text("\n".join(json.dumps(r) for r in rows))

    out = run(
        "review", "import", str(queue), "--actor", "curator", "--schema", "study@1.3"
    ).output
    assert "curator" in out
    assert not json.loads(run("review", "list", json_out=True).output)


def test_export_writes_records_with_a_provenance_sidecar(run, workspace, tmp_path):
    root, _ = workspace
    _bootstrap(run, root)
    run("run", "study@1.1")
    out = tmp_path / "export.jsonl"
    run("export", str(out))
    lines = [json.loads(l) for l in out.read_text().splitlines()]
    assert lines
    assert "record" in lines[0] and "provenance" in lines[0]
    assert lines[0]["provenance"][0]["field_id"]


def test_status_and_report(run, workspace):
    root, _ = workspace
    _bootstrap(run, root)
    result = json.loads(run("run", "study@1.1", json_out=True).output)
    execution_id = result["execution_id"]

    status = json.loads(run("status", execution_id, json_out=True).output)
    assert status["steps_by_state"]

    report = run("report", execution_id).output
    assert "change rate" in report


def test_evaluate_gates_a_rollout(run, workspace, tmp_path):
    root, papers = workspace
    _bootstrap(run, root)

    gold = tmp_path / "gold.jsonl"
    with gold.open("w") as fh:
        fh.write(json.dumps({"name": "gold-v1"}) + "\n")
        for paper in papers:
            fh.write(
                json.dumps(
                    {
                        "record_id": paper.record_id,
                        "field_id": "tasks[].stimulus_modality",
                        "entity": "tasks[]=t1",
                        "status": "present" if paper.truth else "not_extracted",
                        "value": [paper.truth] if paper.truth else None,
                    }
                )
                + "\n"
            )

    ok = run("evaluate", "study@1.1", str(gold), "--gate", "precision>=0.9")
    assert "all gates passed" in ok.output

    failing = run(
        "evaluate", "study@1.1", str(gold), "--gate", "cost_usd<=-1", expect_ok=False
    )
    assert failing.exit_code == 1
    assert "GATES FAILED" in failing.output


def test_an_unreadable_gate_is_a_usage_error(run, workspace, tmp_path):
    root, _ = workspace
    _bootstrap(run, root)
    gold = tmp_path / "gold.jsonl"
    gold.write_text(json.dumps({"name": "g"}) + "\n")
    result = run("evaluate", "study@1.1", str(gold), "--gate", "nonsense", expect_ok=False)
    assert result.exit_code != 0
    assert "cannot read gate" in result.output


def test_reanchor_is_a_dry_run_by_default(run, workspace):
    root, papers = workspace
    _bootstrap(run, root)
    payload = json.loads(
        run("reanchor", papers[0].record_id, "v1", "parse@1", json_out=True).output
    )
    assert payload["applied"] is False
    assert payload["written"] == 0
    assert payload["outcomes"]["unchanged"] > 0

    missing = run(
        "reanchor", papers[0].record_id, "v1", "parse@99", expect_ok=False
    )
    assert "no parse" in missing.output


def test_json_output_is_machine_readable_everywhere(run, workspace):
    root, papers = workspace
    _bootstrap(run, root)
    for args in (
        ("schema", "list"),
        ("migration", "list"),
        ("schema", "diff", "study@1.0", "study@1.1"),
        ("plan", "study@1.1"),
        ("record", papers[0].record_id),
        ("status",),
    ):
        text = run(*args, json_out=True).output
        json.loads(text)  # raises if it is not JSON
