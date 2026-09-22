# Requirements coverage

Every identifier from the requirements document, where it lives, and what tests
it. Status is one of **done**, **partial** (with what is missing) or **not
implemented** (with why it was deferred).

Priorities are the requirements' own: **P0** is "required for a usable first
release", P1 "important next capability", P2 "desirable extension". **Every P0
is done.**

## 8.1 Schema registry and comparison

| ID | P | Status | Where | Tests |
|---|---|---|---|---|
| FR-SCH-001 | P0 | done | `registry.Registry.register_schema` — re-registering a ref with different content is refused | `test_registry.py::test_a_version_identifier_cannot_change_meaning` |
| FR-SCH-002 | P0 | done | `schema.adapters.{json_schema,pydantic_models,linkml}` | `test_schema.py`, `test_study_schema.py` |
| FR-SCH-003 | P0 | done | `schema.diff.diff_schemas` — 14 change kinds | `test_schema.py`, `test_study_schema.py::test_the_rename_commit_diffs_the_way_its_message_describes` |
| FR-SCH-004 | P0 | done | Renames come only from `identity_map` or `Migration.renames`; similarity yields `rename_candidates` only | `test_schema.py::test_similar_names_are_reported_as_candidates_but_never_applied` |
| FR-SCH-005 | P0 | done | `FieldDefinition.field_id` vs `.path`; `NormalizedSchema.rekey` | `test_schema.py::test_rename_keeps_its_field_identity_when_declared` |
| FR-SCH-006 | P1 | done | `FieldDefinition.annotations`, `constraints.vocabulary_ref`, `recipe_ref`; `recipe.Vocabulary` | `test_schema.py`, `test_planner.py` |
| FR-SCH-007 | P1 | done | `FieldDefinition.semantic_key` — description changes detected with no structural change | `test_schema.py::test_rewording_a_description_is_a_semantic_change` |
| FR-SCH-008 | P1 | done | `SchemaDiff.compatibility()` — shape / values / semantics, independently | `test_schema.py::test_compatibility_is_reported_on_three_independent_axes` |

## 8.2 Migration declaration and graph

| ID | P | Status | Where | Tests |
|---|---|---|---|---|
| FR-MIG-001 | P0 | done | `migration.spec.Migration` / `MigrationStep`; `migration_hash()` | `test_registry.py`, `test_migration_files.py` |
| FR-MIG-002 | P0 | done | `Registry.find_path` / `find_paths` | `test_registry.py::test_path_spans_every_intermediate_version` |
| FR-MIG-003 | P0 | done | Deterministic ordering, lexicographic tiebreak, `PathPreference.key` | `test_registry.py::test_path_selection_is_deterministic_across_repeated_calls` |
| FR-MIG-004 | P0 | done | `_assert_no_cycle`; downgrades stored in a separate index | `test_registry.py::test_cycles_are_rejected`, `::test_a_downgrade_is_stored_apart_from_upgrade_planning` |
| FR-MIG-005 | P0 | done | `Fidelity`; `Migration.fidelity` is the worst of its steps | `test_registry.py::test_fidelity_of_a_migration_is_the_worst_of_its_steps` |
| FR-MIG-006 | P0 | done | `ExecutionPolicy.permits` — lossy needs a flag, destructive needs naming | `test_planner.py::test_a_lossy_migration_is_blocked_without_an_explicit_policy`, `::test_a_destructive_migration_must_also_be_named` |
| FR-MIG-007 | P0 | done | `StepKind` — all seven kinds plus validation and assembly | `test_registry.py`, `test_engine.py` |
| FR-MIG-008 | P0 | done | Each step is a separate `PlannedStep` with its own disposition | `test_planner.py`, `test_acceptance.py` |
| FR-MIG-009 | P0 | done | YAML in `migration.loader`; code via `functions.TRANSFORMS` etc., with code hashes | `test_migration_files.py::test_the_specification_example_loads_verbatim` |
| FR-MIG-010 | P1 | done | `migration.scaffold.scaffold_migration` — steps marked TODO, does not validate | `test_migration_files.py::test_a_scaffold_does_not_pretend_to_know_the_semantics` |
| FR-MIG-011 | P1 | partial | `Migration.downgrade` and `is_downgrade` are declared and stored apart; executing a downgrade path is not wired into the planner | `test_registry.py::test_a_downgrade_is_stored_apart_from_upgrade_planning` |
| FR-MIG-012 | P1 | done | `_assert_unique_approved`; `PathPreference.branches` | `test_registry.py::test_two_approved_production_edges_between_the_same_pair_are_refused` |

## 8.3 Dependency and invalidation

| ID | P | Status | Where | Tests |
|---|---|---|---|---|
| FR-DEP-001 | P0 | done | `planner.dependencies.dependency_hashes` — one hash per dependency kind | `test_planner.py::test_dependency_hashes_name_what_changed_not_merely_that_something_did` |
| FR-DEP-002 | P0 | done | `assess_currency` + `Planner._plan_step` | `test_acceptance.py::test_03_adding_a_field_runs_only_that_fields_recipe` |
| FR-DEP-003 | P0 | done | `path` excluded from `semantic_key` | `test_acceptance.py::test_02_a_rename_completes_with_zero_model_calls_and_keeps_its_lineage` |
| FR-DEP-004 | P0 | done | Separate hash keys for recipe, prompt, context policy, model policy, each validator, vocabulary | `test_acceptance.py::test_05_changing_a_prompt_invalidates_only_that_recipes_artifacts` |
| FR-DEP-005 | P0 | done | `records.decompose` / `assemble` — assembly is separate from extraction | `test_records.py`, `test_acceptance.py::test_03` |
| FR-DEP-006 | P0 | done | `Currency.NO_PROVENANCE`; `Project.ingest(assert_currency=False)` | `test_planner.py::test_missing_provenance_is_never_evidence_of_currency` |
| FR-DEP-007 | P1 | done | `ReviewQueue.accept_legacy_value`; `FieldProvenance.accepted_under` | `test_review.py::test_accepting_a_legacy_value_records_the_decision_without_rewriting_history` |
| FR-DEP-008 | P1 | done | Entity keys from `(collection_path, local_id)`; `RecordEntity` | `test_records.py::test_reordering_a_list_does_not_change_entity_keys` |

## 8.4 Context selection

| ID | P | Status | Where | Tests |
|---|---|---|---|---|
| FR-CTX-001 | P0 | done | `MigrationStep.context` or the recipe's; a semantic step always resolves one | `test_context.py` |
| FR-CTX-002 | P0 | done | Selectors: `none`, `structured_fields`, `prior_evidence`, `sections`, `unit_kinds`, `evidence_window`, `retrieve`, `full_document`, `raw_document` | `test_context.py` (one test per selector) |
| FR-CTX-003 | P0 | done | `ContextPolicy.sequence` is ordered; `full_document_fallback` defaults to forbidden | `test_context.py::test_a_forbidden_full_document_is_never_reached` |
| FR-CTX-004 | P0 | done | `FieldProvenance.context_units` + `context_hash`; `ContextUnit.unit_hash` | `test_acceptance.py::test_09_every_new_semantic_value_is_fully_traceable` |
| FR-CTX-005 | P0 | done | `ContextSource.version` / `selector_hash`, part of the policy hash | `test_context.py::test_changing_a_selector_version_changes_its_identity` |
| FR-CTX-006 | P0 | done | `ContextBudget` — tokens, chars, units, sections, cost; `OnBudget` | `test_context.py` (four budget tests) |
| FR-CTX-007 | P0 | done | `OnMissingContext.{REVIEW,FAIL,PROCEED}` | `test_acceptance.py::test_10_records_without_required_context_are_escalated_not_invented` |
| FR-CTX-008 | P1 | done | `register_retriever`; `billable` retrievers skipped unless `allow_model_selectors` | `test_context.py::test_a_billable_selector_is_skipped_during_a_dry_run` |
| FR-CTX-009 | P1 | done | `PrivacyPolicy` — unit labels, provider allow/deny, required attributes | `test_context.py` (three privacy tests) |
| FR-CTX-010 | P1 | done | `ContextPreview.est_input_tokens`; `PlannedStep.est_cost_usd` | `test_acceptance.py::test_07_a_dry_run_reports_everything_and_mutates_nothing` |

## 8.5 Model-assisted execution

| ID | P | Status | Where | Tests |
|---|---|---|---|---|
| FR-LLM-001 | P0 | done | `ModelRequest.output_schema`; adapters validate with jsonschema | `test_engine.py` |
| FR-LLM-002 | P0 | done | `models.base.ModelAdapter` protocol; `AdapterRegistry` | `test_engine.py::test_a_fallback_model_takes_over_when_the_primary_is_down` |
| FR-LLM-003 | P0 | done | `ModelPolicy` — provider, fallbacks, params, `RetryPolicy`, rate limit, budget | `test_engine.py` |
| FR-LLM-004 | P0 | done | `TransportError`, `RateLimitedError`, `SchemaValidationError`, `UnsupportedContextError`, `SemanticValidationError` | `test_engine.py::test_a_schema_invalid_answer_is_a_different_failure_from_a_dropped_connection` |
| FR-LLM-005 | P0 | done | One `step_key`, one `attempt` row per try | `test_engine.py::test_transient_failures_are_retried_and_every_attempt_is_recorded` |
| FR-LLM-006 | P0 | done | `planner.dependencies.cache_key` | `test_planner.py::test_cache_keys_differ_for_every_output_affecting_input`, `test_properties.py` |
| FR-LLM-007 | P0 | done | Cache lookup by that key only | `test_engine.py::test_the_planner_reports_a_cache_hit_as_its_own_disposition` |
| FR-LLM-008 | P0 | done | `_artifacts_from_output` validates before committing | `test_validation.py::test_a_failing_validator_stops_a_value_from_being_committed` |
| FR-LLM-009 | P1 | not implemented | Provider batch APIs. The adapter contract is per-request; a batching adapter can buffer behind it without the engine noticing, but nothing ships one | — |
| FR-LLM-010 | P1 | done | `Engine(shadow={recipe: alternative})` — recorded as attempts, never committed | `test_evaluation.py::test_shadow_mode_records_disagreement_without_committing_it` |
| FR-LLM-011 | P1 | done | `ModelCall` — tokens, latency, cost, `provider_request_id`, attempts | `test_acceptance.py::test_09` |
| FR-LLM-012 | P2 | not implemented | Conditional model routing | — |
| FR-LLM-013 | P1 | done | `ConfidenceSpec` — an uncalibrated self-rating is recorded as exactly that | `test_schema.py` (recipe hashing), `recipe.ConfidenceSpec.is_calibrated` |

## 8.6 Evidence and provenance

| ID | P | Status | Where | Tests |
|---|---|---|---|---|
| FR-PROV-001 | P0 | done | `FieldArtifact.evidence: tuple[EvidenceReference, ...]` | `test_records.py` |
| FR-PROV-002 | P0 | done | `EvidenceReference` carries source id/version/parse version; `EvidenceSpan` unit-relative offsets | `test_reanchor.py` |
| FR-PROV-003 | P0 | done | Evidence stored on the artifact, not inside the value; `_single_evidence` carries it through a structural move | `test_engine.py::test_a_structural_move_carries_the_evidence_with_it` |
| FR-PROV-004 | P0 | done | `FieldProvenance` — all fourteen listed items | `test_acceptance.py::test_09` |
| FR-PROV-005 | P0 | done | `FieldProvenance.lineage` records the exact prior value hashes | `test_engine.py::test_a_derived_field_records_the_exact_values_it_came_from` |
| FR-PROV-006 | P0 | done | `Actor.HUMAN` on review artifacts; durable `ReviewEvent` | `test_review.py::test_a_human_correction_is_never_indistinguishable_from_model_output` |
| FR-PROV-007 | P1 | done | `Project.export_jsonl` with a provenance sidecar | `test_acceptance.py::test_13`, `test_cli.py::test_export_writes_records_with_a_provenance_sidecar` |
| FR-PROV-008 | P1 | done | `llmbic.reanchor`; old parses retained | `test_reanchor.py` (10 tests) |
| FR-PROV-009 | P0 | done | `ValueStatus` — six distinct states, preserved through migration and export | `test_acceptance.py::test_13_distinct_missingness_and_failure_states_survive_migration_and_export` |

## 8.7 Planning and dry runs

| ID | P | Status | Where | Tests |
|---|---|---|---|---|
| FR-PLN-001 | P0 | done | `ExecutionPlan` / `RecordPlan` / `PlannedStep` with `depends_on` | `test_planner.py` |
| FR-PLN-002 | P0 | done | `PlanSummary.by_disposition` / `fields_by_disposition` | `test_planner.py::test_the_plan_counts_records_fields_and_calls` |
| FR-PLN-003 | P0 | done | `est_input_tokens`, `est_output_tokens`, `est_cost_usd` from `Pricing` | `test_acceptance.py::test_07` |
| FR-PLN-004 | P0 | done | `ContextPreview.unit_ids` / `includes_full_document`; `n_full_document_transmissions` | `test_planner.py::test_the_plan_reports_context_access_before_anything_runs` |
| FR-PLN-005 | P0 | done | `Planner` never calls an adapter; billable selectors skipped | `test_acceptance.py::test_07`, `test_cli.py::test_plan_is_a_dry_run` |
| FR-PLN-006 | P0 | done | `RecordFilter` — ids, schema, source version, state, failed-in-execution, limit/offset | `test_store.py::test_the_index_supports_every_documented_filter` |
| FR-PLN-007 | P1 | done | `PathPreference.optimise` — fidelity / cost / hops / latency | `test_registry.py::test_alternative_paths_are_compared_by_declared_policy` |
| FR-PLN-008 | P1 | done | `ExecutionPlan.plan_id` / `signature()`; the engine refuses a moved registry | `test_engine.py::test_running_a_stale_plan_is_refused` |

## 8.8 Corpus execution and state

| ID | P | Status | Where | Tests |
|---|---|---|---|---|
| FR-EXE-001 | P0 | done | `_checkpoint` per step; `Engine.resume` | `test_acceptance.py::test_08_interrupt_and_resume_creates_no_duplicate_successful_calls` |
| FR-EXE-002 | P0 | done | `_run_record_guarded` isolates per record | `test_engine.py::test_one_records_failure_does_not_roll_back_the_others` |
| FR-EXE-003 | P0 | done | `ThreadPoolExecutor`, `RateLimiter`, `RetryPolicy.backoff_for`, per-policy concurrency | `test_engine.py::test_concurrency_produces_the_same_result_as_serial` |
| FR-EXE-004 | P0 | done | `cache_key`; content-addressed `artifact_id` | `test_engine.py::test_re_executing_an_approved_plan_hits_the_cache_instead_of_the_model` |
| FR-EXE-005 | P0 | done | `Engine.cancel` + `resume` | `test_engine.py::test_cancellation_stops_the_run_and_leaves_it_resumable` |
| FR-EXE-006 | P0 | done | Insert-only `field_artifact` and `record_version` | `test_engine.py::test_prior_artifacts_are_never_mutated`, `test_properties.py` |
| FR-EXE-007 | P0 | done | `SqliteStore.publish` — version row and pointer in one transaction | `test_store.py::test_publication_moves_the_pointer_atomically` |
| FR-EXE-008 | P0 | done | Held records stored as `FAILED` / `REVIEW_NEEDED` drafts, pointer unmoved | `test_acceptance.py::test_11_a_partially_failed_job_exposes_no_partial_record` |
| FR-EXE-009 | P1 | partial | Local synchronous and bounded-thread execution exist; the plan is serialisable so a distributed executor is a consumer of the same structure, but no such executor ships | — |
| FR-EXE-010 | P1 | done | `_reserve_budget` blocks new billable calls; in-flight results finish and checkpoint | `test_engine.py::test_a_budget_stops_scheduling_new_billable_calls` |
| FR-EXE-011 | P1 | not implemented | Prioritisation and fair scheduling across jobs | — |

## 8.9 Validation, comparison, review

| ID | P | Status | Where | Tests |
|---|---|---|---|---|
| FR-VAL-001 | P0 | done | `validation.validate_artifacts` at ingest and before publication | `test_validation.py` |
| FR-VAL-002 | P0 | done | `functions.VALIDATORS`; `ValidationContext` | `test_validation.py` |
| FR-VAL-003 | P0 | done | `ValidationResult.Outcome.{PASS,FAIL,REVIEW}` with structured details | `test_validation.py` |
| FR-VAL-004 | P0 | done | `diffing.diff_records`; `Project.diff_record` | `test_review.py::test_a_field_diff_shows_what_changed_and_what_did_not` |
| FR-VAL-005 | P0 | done | `ReviewItem` carries old, proposed, evidence, context, migration, validations, reason | `test_review.py::test_a_queue_row_carries_everything_needed_to_decide` |
| FR-VAL-006 | P0 | done | `ReviewDecision.{ACCEPT,EDIT,REJECT,DEFER,REQUEST_CONTEXT}` | `test_review.py` (one test per decision) |
| FR-VAL-007 | P0 | done | `ReviewEvent` persisted with actor, timestamp, rationale | `test_review.py` |
| FR-VAL-008 | P1 | done | `evaluation.GoldCorpus` — frozen JSONL | `test_evaluation.py` |
| FR-VAL-009 | P1 | done | `diffing.MigrationReport` — coverage, missingness, change rate, failure rate, review rate, cost | `test_review.py::test_a_migration_report_covers_change_rate_and_cost` |
| FR-VAL-010 | P1 | done | `evaluation.RolloutGate`; `llmbic evaluate` exits non-zero | `test_evaluation.py`, `test_cli.py::test_evaluate_gates_a_rollout` |

## 8.10 Interfaces

| ID | P | Status | Where | Tests |
|---|---|---|---|---|
| FR-API-001 | P0 | done | `project.Project` | every integration test |
| FR-API-002 | P0 | done | All twelve listed commands, plus `currency`, `evaluate`, `reanchor`, `recipe`, `vocabulary`, `record`, `export`, `report`, `ingest` | `test_cli.py` |
| FR-API-003 | P0 | done | `--json` on every command | `test_cli.py::test_json_output_is_machine_readable_everywhere` |
| FR-API-004 | P0 | done | `config.load_config`; `${env:NAME}` references | `test_config.py` |
| FR-API-005 | P1 | not implemented | Service API. The Python API is the seam it would sit on | — |
| FR-API-006 | P1 | done | `execution.state.emit`; `Project(on_event=...)` | `test_engine.py::test_events_are_emitted_with_identifiers_and_no_prompt_text` |

## 8.11 Storage and adapters

| ID | P | Status | Where | Tests |
|---|---|---|---|---|
| FR-STO-001 | P0 | done | `store.base.Store` protocol over all nine concerns | `test_store.py` |
| FR-STO-002 | P0 | done | `store.sqlite.SqliteStore` | `test_store.py` (18 tests) |
| FR-STO-003 | P1 | not implemented | PostgreSQL + object storage. The protocol is the seam | — |
| FR-STO-004 | P0 | done | `cache.cache_key PRIMARY KEY`; `cache_put` reports whether it inserted | `test_store.py::test_cache_keys_are_unique_and_the_writer_is_told` |
| FR-STO-005 | P0 | done | No adapter needed to read, diff, plan deterministically or assemble | `test_store.py::test_reading_deterministic_history_needs_no_model_credential` |
| FR-STO-006 | P1 | not implemented | DataChain / CocoIndex adapters. The plan is serialisable so they are consumers, not rewrites | — |
| FR-STO-007 | P1 | done | `Project.export_jsonl` / `ingest_jsonl`, with source and parse | `test_cli.py` |

## 9 Conceptual data model

All sixteen entities exist with distinct responsibilities:
`SchemaFamily` (`registry`), `SchemaVersion` (`NormalizedSchema`),
`FieldDefinition`, `Migration`, `MigrationStep`, `ExtractionRecipe`,
`ContextPolicy`, `SourceArtifact`, `ParsedSource`, `RecordVersion`,
`FieldArtifact`, `EvidenceReference`, `ExecutionPlan`, `Execution`
(`ExecutionResult` + the `execution` table), `Attempt` (the `attempt` table),
`ValidationResult`, `ReviewEvent`.

## 13 Non-functional

| ID | Status | Notes |
|---|---|---|
| NFR-REP-001 | done | Deterministic steps reproduce from retained inputs; transform code hashed |
| NFR-REP-002 | done | Request, context unit hashes, params and fingerprint retained |
| NFR-REP-003 | done | `ids.content_hash` everywhere; `functions` hashes function source |
| NFR-PERF-001 | done | Planning reads `record_index` rows and artifacts; parsed sources are loaded only when a step needs context |
| NFR-PERF-002 | done | `Store.index` is a generator; `store.base.batched` pages |
| NFR-PERF-003 | partial | The design is O(fields) per record with indexed lookups; not benchmarked at 100k/1M in this suite |
| NFR-PERF-004 | done | `test_acceptance.py::test_08` asserts exactly this |
| NFR-PERF-005 | done | `test_engine.py::test_deterministic_migrations_need_no_model_runtime` |
| NFR-REL-001 | done | Single-writer SQLite with explicit transactions |
| NFR-REL-002 | done | `StepStatus` covers all eight states |
| NFR-REL-003 | done | `ErrorCode` + `LlmbicError.to_dict()` |
| NFR-REL-004 | done | Retry with backoff, fallback adapters |
| NFR-SEC-001 | done | `${env:}` references; `Config.redacted()` |
| NFR-SEC-002 | done | `SECRET_KEYS` redaction |
| NFR-SEC-003 | done | `PrivacyPolicy` enforced during selection, before assembly |
| NFR-SEC-004 | done | `DataPolicyAttributes` |
| NFR-SEC-005 | done | `ReviewEvent.actor_id`; `ExecutionPolicy.actor_id` |
| NFR-SEC-006 | partial | Source labels propagate to context selection; llmbic has no user model of its own to enforce record-level ACLs |
| NFR-MNT-001 | done | `Store`, `ModelAdapter`, `Retriever`, `ValueCodec` are protocols |
| NFR-MNT-002 | done | No test touches the network |
| NFR-MNT-003 | done | Adapters and selectors register from outside the graph |
| NFR-MNT-004 | done | `STORE_FORMAT_VERSION` in `meta`, refused on mismatch, distinct from user schema versions |
| NFR-OBS-001 | done | `execution.state.Metrics` |
| NFR-OBS-002 | done | Events and attempts carry execution / record / step / attempt ids |
| NFR-OBS-003 | done | `test_engine.py::test_events_are_emitted_with_identifiers_and_no_prompt_text` |

## 14 Acceptance criteria

`tests/test_acceptance.py` has one test per criterion, named for it. All
thirteen pass.

## 15 Testing requirements

| Section | Where |
|---|---|
| 15.1 Unit | `test_schema.py`, `test_registry.py`, `test_records.py`, `test_context.py`, `test_store.py`, `test_validation.py`, `test_migration_files.py`, `test_config.py` |
| 15.2 Property-based | `test_properties.py` — all six named properties, with hypothesis |
| 15.3 Integration | `test_engine.py` — `RuleBasedExtractor` (local model), `HostedMockAdapter` (hosted-compatible), SQLite, parsing, evidence selection, cancellation, resumption, review round trip |
| 15.4 Golden migration | `test_study_schema.py` — the real schema at real commits, with expected records, context behaviour and operation counts |
| 15.5 Evaluation | `test_evaluation.py` — precision, recall, evidence support, abstention, schema validity, change rate, cost, and gates |

## 16 Implementation phases

Phase 1 (deterministic core), phase 2 (semantic field migrations) and phase 3
(retrieval and review, rollout gates, migration reports) are complete. From
phase 4: shadow migrations and alternative-path optimisation are in; batch
inference, workflow-engine adapters, a service API and distributed execution
are not.

## 4 Non-goals

All seven hold. Notably: llmbic never decides two differently-worded fields
mean the same thing (`rename_candidates` are advisory), never infers a
destructive migration, treats JSON Schema compatibility as a *separate* verdict
from semantic compatibility, and never accepts model output without validation
and retained evidence.
