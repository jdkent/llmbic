# llmbic

**Semantic schema migration for LLM-extracted records.**

An Alembic-like migration layer for schema-driven information extraction. When
your extraction schema changes, llmbic works out — field by field, record by
record — what is still valid, what can be transformed with code, what can be
derived from values or stored evidence you already have, and what genuinely
needs another look at the source. Then it does only that, and records enough to
explain the decision afterwards.

It exists because the usual assumption behind schema migration fails for
extracted data. A conventional migration produces the new record from the old
one deterministically. Adding a field to a scientific-extraction schema may
instead require another model inference over the article — and sending 20,000
articles back through a model because one field's description was reworded is
the outcome this tool is built to avoid.

```
$ llmbic plan study@1.1
plan sha256:9a5757ecc3681653  ->  study@1.1
  records:          1817
  steps:            3634
    reuse           1810
    semantic        1817
    validate        1817
  model calls:      1817 (0 already cached)
  est. tokens:      2841k in / 454k out
  est. cost:        $15.34
  source access:    1817 records; full documents transmitted: 0
  worst fidelity:   lossless
  stale fields with no step: 1817 (structurally valid, not semantically current)
```

Nothing has run yet. That is the whole point of the output: you can see what a
migration will cost, which parts of which documents it will transmit, and which
values it intends to leave alone, before you spend anything.

---

## Install

```bash
pip install -e .          # plus: pip install -e '.[test]' for the test suite
```

Python 3.10+. The core has four dependencies (pydantic, click, PyYAML,
jsonschema) and no provider SDK: model adapters are plugins.

## The shape of it

```python
from llmbic import Project, from_json_schema

project = Project("study.db")
project.register_schema(from_json_schema(v1_document, name="study", version="1.0"))
project.ingest(record, schema_ref="study@1.0", record_id="pmid:12345",
               source=source, parsed=parsed)

# Register the new version. The identity map is what makes a rename free:
# it says "this new path is the same logical field", so the value and its
# evidence move with it and nothing is recomputed.
project.register_schema(from_json_schema(
    v2_document, name="study", version="1.1",
    identity_map={"tasks[].response_modality": "tasks[].response_mode"},
))
project.register_recipe(stimulus_modality_recipe)
project.register_migration(migration_1_0_to_1_1)

plan = project.plan("study@1.1")     # a dry run: no model calls, no writes
print(plan.render(verbose=True))

plan, result = project.migrate("study@1.1")
```

Everything the CLI does goes through this class, so there is one implementation
of the behaviour and two ways to reach it.

## What a migration looks like

```yaml
id: study-1.0-to-1.1
from_schema: study@1.0
to_schema: study@1.1
description: |
  response_mode -> response_modality keeps its identity, so it needs no step
  and costs nothing. stimulus_modality is new and has to be read out of the
  paper, evidence first and never the whole document.
renames:
  tasks[].response_mode: tasks[].response_modality
acknowledged:
  tasks[].stimuli: reworded, not redefined; existing values stand

steps:
  - id: extract_stimulus_modality
    kind: source_semantic
    reads:
      - field:tasks[].stimuli
      - evidence:tasks[].stimuli
      - source:parsed
    writes: [field:tasks[].stimulus_modality]
    recipe: stimulus-modality@1
    entity_scope: tasks[]
    context:
      sequence:
        - prior_evidence
        - sections: [methods, supplement]
      full_document_fallback: forbidden
      max_input_tokens: 3000
    on_missing_context: review
    validators: [require_evidence]
```

Three declarations carry most of the weight:

- **`renames`** turns a remove/add pair into a rename. Nothing else does:
  similar names are reported as *candidates* and never acted on, because
  "`response_mode` looks like `response_modality`" is a hint for a person, not
  a licence to move data.
- **`acknowledged`** is how a change gets an explicit disposition without
  getting a step. Tightening `minimum` from 0 to 1 is a validation matter; the
  rationale here is the record of that decision. A migration whose diff
  contains a change with neither a step nor an acknowledgement will not
  validate.
- **`context`** is an ordered fallback chain with a hard ceiling. Try stored
  evidence; if there is none, try the Methods section; never send the article.
  A record whose context chain comes up empty goes to review rather than to a
  model with nothing to read.

## What makes a value stale

This is the part that decides whether your corpus is re-read or not, so it is
worth stating precisely. Every artifact records a hash per dependency —
the field definition, the recipe, the prompt, the model policy, the context
policy, each validator, the vocabulary, the source, the parse, and the exact
prior values it was computed from. A value is reusable when the hashes that
hold now equal the ones it recorded.

The consequences, in order of how often they matter:

| Change | Effect |
|---|---|
| Rename a field (identity preserved) | Nothing. Not even a step. |
| Reflow a description's line breaks | Nothing: whitespace is normalised first. |
| Reword a description | That field is stale. Nothing else is. |
| Change a prompt (new recipe version) | Only artifacts produced by that recipe. |
| Add a permissible value to a vocabulary | That field is stale; stored values stay valid. |
| Remove a permissible value | That field is stale *and* stored values may now be invalid. |
| Raise a `minimum` | Nothing is stale. Validation may now reject a record. |
| Add a field | Only the new field is computed. |
| Re-parse the source | Evidence needs re-anchoring; values are untouched. |

The last column is the useful one: a schema change that touches one field's
wording costs one field's worth of model calls, not a corpus.

A record that validates against the newest schema is **not** thereby
semantically current, and llmbic never conflates the two. `llmbic currency
<record>` reports them separately.

## Value status is not null

Five different facts get flattened into `null` by most tooling, and they have
opposite fixes:

- `not_reported` — examined, the source says nothing.
- `not_applicable` — the question does not apply here.
- `unknown` — the pass could not settle it. A fact about the extraction.
- `not_extracted` — no pass has looked yet.
- `extraction_failed` — a pass ran and broke.
- `review_required` — there is a value, and a person has to decide.

llmbic keeps all six distinct through migration and through export. For a
corpus like [study_schema](https://github.com/neurostuff/study_schema), whose
values arrive wrapped in an `ExtractedValue` carrying `extraction_status`,
`unreported_reason` and evidence spans, `ExtractedValueCodec` reads and writes
that wrapper directly.

## Selective recomputation, concretely

Given a corpus at `study@1.0` and a target of `study@1.4` — four sequential
schema versions taken from real commits in study_schema — a run does this:

```
study@1.1: calls=7 reuse=0 published=7 held=1 review=1     # new field, one per record
study@1.2: calls=0 reuse=0 published=7 held=1 review=1     # derived + structural, no model
study@1.3: calls=0 reuse=2 published=6 held=2 review=2     # vocabulary remap, ambiguous ones escalated
study@1.4: calls=7 reuse=3 published=6 held=2 review=3     # prompt bumped; only that recipe re-runs
```

Seven records, fourteen model calls across four schema versions — not
twenty-eight, and not four corpus-wide re-extractions. The held records are
held because the tool declined to invent a value: one paper has no Methods
section, and one study's `parallel` design could not be re-classified under the
narrowed vocabulary without a person.

## The six layers

The reference architecture from the requirements, one module group each:

| Layer | Module | What it owns |
|---|---|---|
| Registry | `llmbic.registry`, `llmbic.schema`, `llmbic.migration` | Schema versions, field identities, recipes, the migration graph |
| Artifact store | `llmbic.store`, `llmbic.provenance`, `llmbic.records` | Immutable sources, parsed units, field artifacts, evidence, record versions |
| Planner | `llmbic.planner` | Dependency resolution, invalidation, reuse, context planning, cost |
| Engine | `llmbic.execution`, `llmbic.models`, `llmbic.context` | Transforms, model adapters, retries, caching, checkpoints, concurrency |
| Validation & review | `llmbic.validation`, `llmbic.review`, `llmbic.diffing`, `llmbic.evaluation` | Structural and semantic checks, diffs, curator decisions, rollout gates |
| Interfaces | `llmbic.project`, `llmbic.cli`, `llmbic.config` | Python API, CLI, configuration |

The core migration model imports no workflow engine and no provider SDK. Model
adapters, retrievers, parsers, codecs, storage backends and transforms are all
registered from outside.

## Command line

```
llmbic schema register schema.json --name study --version 1.1 \
    --rename 'tasks[].response_modality=tasks[].response_mode'
llmbic schema diff study@1.0 study@1.1
llmbic migration new study@1.0 study@1.1 -o migrations/1_1.yaml   # scaffold, marked TODO
llmbic migration validate
llmbic plan study@1.1 --verbose                # dry run
llmbic run study@1.1 --budget 25.00
llmbic resume exec-3f9a2c
llmbic status exec-3f9a2c
llmbic diff pmid:12345
llmbic currency pmid:12345
llmbic provenance show pmid:12345 'tasks[].stimulus_modality' --entity 'tasks[]=t1'
llmbic review export queue.jsonl                    # edit, then:
llmbic review import queue.jsonl --actor curator@example.org
llmbic evaluate study@1.1 gold.jsonl --gate 'precision>=0.9' --gate 'change_rate<=0.2'
llmbic reanchor pmid:12345 v1 parse@2 --apply
llmbic export records.jsonl
```

Every command takes `--json`. `plan` never calls a model and never writes
record state; `run` refuses to start if the registry has moved since the plan
was made.

### Configuration

`llmbic.yaml`, version-controlled, with secrets referenced rather than
contained:

```yaml
store: ./study.db
codec: extracted_value
adapters:
  main:
    kind: myproject.adapters:AnthropicAdapter
    options:
      model: claude-sonnet-5
      api_key: ${env:ANTHROPIC_API_KEY}
extensions: [myproject.transforms]      # imported so @transform registers
policy:
  allow_lossy: false
  budget_usd: 25.0
```

`${env:NAME}` resolves at load; the resolved value never reaches a plan, a
provenance record or a log.

## Safety properties

The ones worth knowing before you point this at a corpus that cost real money:

- **A dry run is a dry run.** `plan` resolves context policies offline. A
  retriever that would itself call a model is skipped, not invoked.
- **Publication is atomic.** A record version becomes current in one
  transaction, or the previous one stays. A partially migrated record is never
  visible; a failed attempt is retained as an inspectable draft.
- **Resumption pays nothing twice.** Work is addressed by a cache key over
  every output-affecting input. Interrupting a 1,000-record job and resuming it
  makes no duplicate successful model calls.
- **Lossy and destructive migrations need permission.** A potentially lossy
  migration needs `allow_lossy`; a destructive one needs `allow_destructive`
  *and* to be named in `approved_migrations`.
- **Budgets stop scheduling, not execution.** When the cap is reached, no new
  billable call starts; in-flight results finish and are checkpointed.
- **Privacy is enforced before transmission.** Units carrying a forbidden label
  are never assembled into a request, and a provider that does not satisfy the
  policy's declared attributes is refused rather than sent redacted text.
- **Human corrections are never mistakable for model output.** They are
  separate artifacts with `actor: human` and a durable review event.

## Testing

```bash
python -m pytest            # 326 tests, no network
```

The suite is unit, property-based (hypothesis), integration, golden and
evaluation tests, in the categories §15 of the requirements lays out. Two
things about it are worth calling out:

**The test bed is study_schema.** `tests/studybed.py` is a faithful miniature
of [neurostuff/study_schema](https://github.com/neurostuff/study_schema) with
four schema versions, each taken from a real commit in that repository:

- `1.0 → 1.1` — commit `5f282f0`, *"Track stimulus modality, and name it
  symmetrically with response"*: a rename that keeps its identity, a new field
  that must be read from the source, and a description rewritten in place.
- `1.1 → 1.2` — commits `6b20e30` / `653cd5b`: `is_healthy` stops being asked
  and starts being derived; the catch-all `population_characteristics` is
  partitioned so a filterable trait and an unremarkable one stop sharing a
  field.
- `1.2 → 1.3` — commit `dc5752d`: `AssignmentStructure` gains
  `observational_cohorts` and `parallel` narrows, so stored values must be
  remapped where the record settles it and escalated where it does not.
- `1.3 → 1.4` — a prompt revision with no shape change, plus a tightened
  constraint that must trigger validation without triggering re-extraction.

**And the real schema is tested directly.** `tests/test_study_schema.py` runs
the LinkML adapter, the diff, and a complete migration over the *actual*
study_schema at those commits, materialised with `git archive`. It skips
cleanly when no checkout is present (`LLMBIC_STUDY_SCHEMA` to point it
elsewhere).

`tests/test_acceptance.py` has one test per item in §14's acceptance list, named
after it.

## Status

Phases 1–3 of the requirements are implemented and tested: the deterministic
core, semantic field migrations, retrieval and review. Selected phase-4 items
are in (shadow migrations, alternative-path optimisation, evaluation and
rollout gates); batch provider APIs, distributed executors, a service API and a
PostgreSQL adapter are not. `docs/decisions.md` records the answers to §18's
ten questions, and `docs/requirements-coverage.md` maps every requirement
identifier to where it lives and what tests it.

## Licence

MIT.
