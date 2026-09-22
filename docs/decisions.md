# Decisions

§18 of the requirements lists ten questions the maintainers must answer before
implementation. Here are the answers this implementation makes, with the
reasoning, so that changing one is a deliberate act rather than a discovery.

## 1. Canonical schema authoring format

**Neither Pydantic nor JSON Schema is the source of truth. The normalized form
is, and adapters produce it.**

`llmbic.schema.normalized.NormalizedSchema` is a flat list of
`FieldDefinition`, each with a stable `field_id`, a current `path`, a base
type, cardinality, requiredness, constraints, description, and the recipe it
requires. Everything in the system — diffing, planning, validation, invalidation
— reads that and nothing else.

Three adapters exist: JSON Schema (the reference; Pydantic delegates to it via
`model_json_schema`) and LinkML (for study_schema). Adding a fourth touches no
other module, which is NFR-MNT-003.

The alternative — designating one authoring format canonical — would have
forced study_schema to express its `ExtractedValue` wrapper, its `in_subset`
provenance marks and its open vocabularies through a format that has no words
for them. The normalized form has words for exactly the things the planner
needs, and the adapter's job is to find them wherever the authoring format
hides them.

## 2. Where field-level provenance lives

**Normalized sidecar, joined on export.**

`FieldArtifact` is the unit of storage; a record is an assembled view over
artifacts (`llmbic.records.assemble`). Export writes one JSON line per record
with `record` and a `provenance` array keyed by `field_id` + `entity`.

Embedding provenance in the record would have made the record unusable by
consumers who only want the values, and would have made the *record* the unit
of immutability — which is the monolithic-extraction failure §19 lists first.

## 3. Nested entity identity

**Entity keys built from `(collection_path, local_id)` pairs.**

`groups[]=g1`, `tasks[]=t1/tasks[].conditions[]=c2`. The collection path is part
of the key so two collections reusing the same local ids never collide. The
`local_id` comes from the collection's declared `identity_field`, so reordering
a list rewrites nothing — a property test pins this.

When a collection declares no identity field and its members carry nothing
usable, the key falls back to the ordinal (`#0`), and reordering *does* rewrite
artifacts. That is a real cost and the schema should name an identity field; the
fallback exists so an unidentified collection degrades rather than fails.

A present-and-empty collection gets a sentinel entity, because "no assessments"
and "assessments not extracted" are different facts and an empty list is the
only place JSON can say the first one.

## 4. Branching vs. linear history

**Branching is allowed in development; production requires a unique approved
path.**

The registry refuses a second approved, `main`-branch migration between the
same source/target pair. A rival lives on a branch and is reachable only when
the planner is told to look there (`PathPreference(branches=...)`), or is
registered unapproved.

Cycles are rejected outright in the upgrade graph. A reverse edge must be
marked `is_downgrade=True`, which stores it apart from upgrade planning
entirely.

## 5. Parsed-document representation

**`ParsedSource`: an ordered list of `DocumentUnit` with ids stable for a given
`parse_version`.**

A unit has a kind (section, paragraph, sentence, table, table row, figure,
caption, footnote, supplement), an optional normalised section name, its offset
within the whole document, and an ordinal. Offsets in an `EvidenceSpan` are
*unit-relative*, and each unit records its own document offset, so a re-parse
that shifts global offsets can be re-anchored unit by unit rather than
invalidating everything.

llmbic parses nothing itself. A parser adapter produces this structure; an old
`ParsedSource` is retained beside a new one, and `llmbic.reanchor` moves spans
across, re-anchoring only what it can find exactly once and counting the rest.

## 6. Review system

**JSONL round trip.**

`llmbic review export` writes one row per open item carrying the old value, the
proposed value, the evidence, the context that produced it, the migration
identity, the validator results and the escalation reason, with blank
`decision` / `rationale` / `edited_value` fields. A curator fills those in;
`llmbic review import` applies them.

It needs no service, it diffs in review, and it is a few lines of adapter away
from an annotation platform. A decision becomes a durable `ReviewEvent` *and*,
where it changes a value, a new artifact attributed to a person — the model's
answer stays in the store beside the correction.

## 7. Direct execution vs. a workflow substrate

**Direct, with the domain model kept portable.**

`llmbic.execution.Engine` is a bounded thread pool over records, with durable
per-step checkpoints, a content-addressed cache, retry and fallback policies,
rate limiting, budget enforcement and cancellation. Roughly 600 lines.

The reason not to delegate is §19's last row: a workflow framework would have
become the migration model's owner. The plan is a serialisable data structure
and the engine consumes it, so a DataChain or CocoIndex executor is an
alternative consumer of the same plan rather than a rewrite.

## 8. Hosted models changing behind a stable name

**Record the fingerprint; let a recipe pin it.**

Every `ModelCall` records the provider-reported `model_fingerprint`.
`ModelPolicy.pin_fingerprint` makes it part of the recipe hash, so a change
invalidates that recipe's artifacts; and at execution, an answer whose
fingerprint does not match the pin is routed to review rather than committed.

Unpinned is the default, because most recipes do not want to re-extract a
corpus every time a provider ships a point release. The fingerprint is recorded
either way, so the question "which weights produced this value" always has an
answer.

## 9. What may be sent to a provider

**A per-corpus `PrivacyPolicy` inside the context policy, enforced before
assembly.**

Three mechanisms, all pre-transmission:

- `forbidden_unit_labels` — a unit carrying a matching label is dropped during
  selection and never enters a request.
- `allowed_providers` / `forbidden_providers` — by provider id.
- `require_provider_attributes` — an adapter declares `retains_data`,
  `trains_on_data`, `locality` and `accepts_binary`; a policy that requires
  `retains_data: False` refuses an adapter that admits to retaining.

Separately, `full_document_fallback` defaults to `forbidden`, so transmitting a
whole article is an explicit decision in the migration *and* needs
`allow_full_document` on the run.

## 10. What counts as semantic invalidation

**Description, prompt, validators, vocabulary and recipe — not numeric
bounds.**

`FieldDefinition.semantic_key()` covers the description (whitespace-normalised,
so reflowing is free), the base type, cardinality, the vocabulary (permissible
values, open/closed, vocabulary ref), the required recipe, and annotations.
`structural_key()` covers everything the shape depends on, bounds included.

The split follows §18.10's own list, which names descriptions, prompts,
validators and vocabularies and does not name constraints. Raising a `minimum`
does not change what the field is asking for; it changes which stored answers
are acceptable, which is validation's job. `llmbic migration validate` still
demands an explicit disposition for it — a step, or an `acknowledged` entry
with a rationale — so the decision is recorded either way.

A field's path is deliberately *excluded* from the semantic key. That is
FR-DEP-003, and it is what makes a rename free.

---

## Two decisions the requirements did not ask for

### In-place transforms are excluded from their own dependency hash

A step that both reads and writes `population_characteristics` would, if its
own prior value were hashed as a dependency, be permanently stale against its
own output and re-run forever. The prior value is recorded in
`provenance.notes["in_place_inputs"]` instead — auditable, and outside the
content address. Idempotence (principle 7) wins over completeness of the hash
in the one case where they conflict.

### A step is answerable to the schema it targets

When a record catches up across several versions at once, each step stamps its
artifact with its *own* migration's target schema, not the run's final
destination. Otherwise the `1.0 → 1.1` step would claim currency for `1.4`'s
definition of a field it never saw, and a replay of a pending intermediate step
would produce a near-duplicate artifact.
