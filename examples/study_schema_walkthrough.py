#!/usr/bin/env python3
"""A complete walkthrough on the study_schema test bed.

Run it:

    python examples/study_schema_walkthrough.py

It ingests a small corpus at ``study@1.0`` and migrates it through four schema
versions, each taken from a real commit in neurostuff/study_schema, printing
what the planner decided and why at every step.  Nothing here touches the
network: the "model" is a local rule-based extractor that genuinely reads the
context it is handed and abstains when nothing matches.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

import studybed as sb  # noqa: E402

from llmbic import (  # noqa: E402
    ExecutionPolicy,
    ExtractedValueCodec,
    GoldCorpus,
    GoldValue,
    Project,
    RolloutGate,
    ValueStatus,
    evaluate,
)
from llmbic.models.base import AdapterRegistry  # noqa: E402

LOSSY = ExecutionPolicy(allow_lossy=True)
RULE = "\n" + "=" * 72 + "\n"


def banner(text: str) -> None:
    print(f"{RULE}{text}{RULE}")


def build() -> tuple[Project, list[sb.Paper]]:
    adapters = AdapterRegistry()
    adapters.register("main", sb.stimulus_adapter())
    adapters.register("backup", sb.stimulus_adapter(provider="backup", model="rules-2"))

    project = Project(
        ":memory:",
        registry=sb.build_registry(),
        adapters=adapters,
        # study_schema wraps every source-derived value in an ExtractedValue
        # carrying its status and evidence; this codec reads that shape.
        codec=ExtractedValueCodec(),
    )

    papers = sb.corpus(8)
    for paper in papers:
        project.ingest(
            paper.record,
            schema_ref="study@1.0",
            record_id=paper.record_id,
            source=paper.source,
            parsed=paper.parsed,
        )
    return project, papers


def main() -> int:
    project, papers = build()

    banner("1. What changed between study@1.0 and study@1.1")
    print(project.diff("study@1.0", "study@1.1").render())
    print(
        "\nNote the rename is marked identity_preserved: the value and its "
        "evidence\nmove with the field, so no step is needed and nothing is "
        "recomputed."
    )

    banner("2. The plan, before anything runs")
    plan = project.plan("study@1.1")
    print(plan.render(verbose=True, max_records=2))
    print(
        "\nThe only field any step writes is tasks[].stimulus_modality. "
        "Everything\nelse in the record is left alone."
    )

    banner("3. Running through four schema versions")
    for target in ("study@1.1", "study@1.2", "study@1.3", "study@1.4"):
        plan, result = project.migrate(target, policy=LOSSY)
        m = result.metrics
        print(
            f"{target:<12} {result.status.value:<9} "
            f"model_calls={m.model_calls:<3} reused={m.steps_reused:<3} "
            f"published={m.records_published:<3} held={m.records_held:<3} "
            f"review={m.steps_review}"
        )
    print(
        "\n1.2 and 1.3 needed no model at all: one derives a field from another, "
        "and\nthe other remaps a vocabulary in code, escalating only what the "
        "record\ncannot settle."
    )

    banner("4. Where one value came from")
    paper = next(p for p in papers if p.truth)
    rows = project.provenance(paper.record_id, "tasks[].stimulus_modality", "tasks[]=t1")
    p = rows[-1]["provenance"]
    print(f"record          {paper.record_id}")
    print(f"value           {rows[-1]['value']}")
    print(f"schema version  {p['schema_version']}")
    print(f"recipe          {p['recipe_ref']}")
    print(f"migration/step  {p['migration_id']} / {p['step_id']}")
    print(f"context         {p['context_selector_ref']} -> {p['context_units']}")
    print(f"model           {p['model_call']['provider']}/{p['model_call']['model']}")
    print(f"request id      {p['model_call']['provider_request_id']}")
    print(f"validators      {[v['validator'] for v in p['validations']]}")
    print(f"evidence spans  {sum(len(e['spans']) for e in rows[-1]['evidence'])}")

    banner("5. What is current, and what merely validates")
    currency = project.currency(papers[0].record_id)
    stale = {k: v for k, v in currency.items() if v != "current"}
    print(f"{len(currency)} field slots, {len(stale)} not semantically current:")
    for key, state in sorted(stale.items()):
        print(f"  {state:<16} {key}")
    print(
        "\ntasks[].stimuli had its description rewritten in 1.1 and no migration\n"
        "addressed it. The record validates against study@1.4; that field does "
        "not\nanswer the question study@1.4 asks. llmbic reports the two "
        "separately."
    )

    banner("6. What a curator has to look at")
    for item in project.review.items():
        print(f"{item.record_id}  {item.field_id}")
        print(f"   old      {item.old_value}")
        print(f"   proposed {item.proposed_value}")
        print(f"   because  {item.reason[:100]}")

    banner("7. Evaluating the semantic migration against a gold sample")
    fresh, fresh_papers = build()
    gold = GoldCorpus(
        name="stimulus-modality-v1",
        values=tuple(
            GoldValue(
                record_id=p.record_id,
                field_id="tasks[].stimulus_modality",
                entity="tasks[]=t1",
                status=ValueStatus.PRESENT if p.truth else ValueStatus.NOT_EXTRACTED,
                value=[p.truth] if p.truth else None,
            )
            for p in fresh_papers
        ),
    )
    report = evaluate(
        fresh,
        gold,
        "study@1.1",
        gates=[
            RolloutGate(metric="precision", min_value=0.9),
            RolloutGate(metric="evidence_support", min_value=0.8),
        ],
    )
    print(report.render())

    banner("8. Re-running changes nothing and costs nothing")
    plan, result = project.migrate("study@1.4", policy=LOSSY)
    print(
        f"model calls: {result.metrics.model_calls}, "
        f"reused: {result.metrics.steps_reused}, "
        f"cost: ${result.metrics.usage.cost_usd:.4f}"
    )

    project.close()
    fresh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
