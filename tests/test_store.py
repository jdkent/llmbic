"""SQLite reference backend (§8.11)."""

from __future__ import annotations

import pytest

import studybed as sb
from llmbic import FieldValue, RecordFilter, SqliteStore
from llmbic.errors import ErrorCode, LlmbicError
from llmbic.provenance import (
    FieldArtifact,
    FieldProvenance,
    RecordState,
    RecordVersion,
    ReviewDecision,
    ReviewEvent,
)
from llmbic.store.base import CacheEntry, RecordIndexEntry, StepState


@pytest.fixture
def store():
    s = SqliteStore(":memory:")
    yield s
    s.close()


def _artifact(record_id="r", field_id="f", value=1, entity=""):
    return FieldArtifact(
        record_id=record_id,
        field_id=field_id,
        entity=entity,
        value=FieldValue.present(value),
        provenance=FieldProvenance(schema_version="study@1.0"),
    )


def test_artifacts_are_insert_only(store):
    a = _artifact()
    assert store.put_artifacts([a]) == [a.artifact_id]
    store.put_artifacts([a])
    assert len(store.get_artifacts("r")) == 1


def test_two_identical_computations_are_one_artifact(store):
    assert _artifact().artifact_id == _artifact().artifact_id


def test_a_different_value_is_a_different_artifact(store):
    assert _artifact(value=1).artifact_id != _artifact(value=2).artifact_id


def test_cache_keys_are_unique_and_the_writer_is_told(store):
    """FR-STO-004: the store, not the caller, enforces the uniqueness."""

    entry = CacheEntry(cache_key="k", payload={"output": 1}, created_at="")
    assert store.cache_put(entry) is True
    assert store.cache_put(entry) is False
    assert store.cache_get("k").payload == {"output": 1}
    assert store.cache_size() == 1


def test_publication_moves_the_pointer_atomically(store):
    a = _artifact()
    store.put_artifacts([a])
    v1 = RecordVersion(record_id="r", schema_ref="study@1.0", artifact_ids=(a.artifact_id,))
    store.publish(v1)
    assert store.current_version("r").version_id == v1.version_id

    b = _artifact(value=2)
    store.put_artifacts([b])
    v2 = RecordVersion(
        record_id="r",
        schema_ref="study@1.1",
        artifact_ids=(b.artifact_id,),
        parent_version_id=v1.version_id,
    )
    store.publish(v2)
    assert store.current_version("r").version_id == v2.version_id
    assert store.get_record_version(v1.version_id).state is RecordState.SUPERSEDED


def test_a_draft_version_never_becomes_current(store):
    a = _artifact()
    store.put_artifacts([a])
    draft = RecordVersion(
        record_id="r",
        schema_ref="study@1.1",
        artifact_ids=(a.artifact_id,),
        state=RecordState.FAILED,
    )
    store.put_record_version(draft)
    assert store.current_version("r") is None
    assert store.get_record_version(draft.version_id).state is RecordState.FAILED


def test_the_index_supports_every_documented_filter(store):
    for i in range(4):
        store.upsert_index(
            RecordIndexEntry(
                record_id=f"r{i}",
                schema_ref="study@1.0" if i < 2 else "study@1.1",
                source_id=f"s{i}",
                source_version="v1" if i % 2 == 0 else "v2",
                state="published",
            )
        )
    assert len(list(store.index())) == 4
    assert [e.record_id for e in store.index(RecordFilter(record_ids=["r1"]))] == ["r1"]
    assert len(list(store.index(RecordFilter(schema_ref="study@1.1")))) == 2
    assert len(list(store.index(RecordFilter(source_version="v2")))) == 2
    assert len(list(store.index(RecordFilter(limit=2)))) == 2
    assert [e.record_id for e in store.index(RecordFilter(limit=2, offset=2))] == ["r2", "r3"]


def test_filtering_by_failed_step_selects_the_right_records(store):
    store.upsert_index(RecordIndexEntry(record_id="r0", schema_ref="s@1"))
    store.upsert_index(RecordIndexEntry(record_id="r1", schema_ref="s@1"))
    store.create_execution("e1", {"plan_id": "p", "state": "running"})
    store.put_step_state(
        StepState(execution_id="e1", step_key="k0", state="succeeded", record_id="r0")
    )
    store.put_step_state(
        StepState(execution_id="e1", step_key="k1", state="failed", record_id="r1")
    )
    got = [e.record_id for e in store.index(RecordFilter(failed_in_execution="e1"))]
    assert got == ["r1"]


def test_step_state_is_upserted_not_duplicated(store):
    store.create_execution("e", {"plan_id": "p"})
    store.put_step_state(StepState(execution_id="e", step_key="k", state="running"))
    store.put_step_state(StepState(execution_id="e", step_key="k", state="succeeded"))
    states = store.get_step_states("e")
    assert len(states) == 1
    assert states[0].state == "succeeded"


def test_attempts_accumulate_under_one_step(store):
    store.create_execution("e", {"plan_id": "p"})
    for n in range(3):
        store.put_attempt("e", "k", {"attempt": n})
    assert [a["attempt"] for a in store.get_attempts("e", "k")] == [0, 1, 2]


def test_updating_an_unknown_execution_is_an_error(store):
    with pytest.raises(LlmbicError) as exc:
        store.update_execution("nope", state="running")
    assert exc.value.code is ErrorCode.EXECUTION_NOT_FOUND


def test_sources_and_parsed_versions_are_addressable(store):
    paper = sb.make_paper(0)
    store.put_source(paper.source)
    store.put_parsed(paper.parsed)
    assert store.get_source("pmid:100000", "v1").content_hash == paper.source.content_hash
    assert store.get_parsed("pmid:100000", "v1", "parse@1").parse_version == "parse@1"
    # The newest parse wins when none is named.
    assert store.get_parsed("pmid:100000", "v1") is not None
    assert store.list_parse_versions("pmid:100000", "v1") == ["parse@1"]


def test_an_old_parse_is_retained_beside_a_new_one(store):
    """FR-PROV-008: offsets change, the old parse stays."""

    import dataclasses

    paper = sb.make_paper(0)
    store.put_parsed(paper.parsed)
    store.put_parsed(dataclasses.replace(paper.parsed, parse_version="parse@2"))
    assert store.list_parse_versions("pmid:100000", "v1") == ["parse@1", "parse@2"]
    assert store.get_parsed("pmid:100000", "v1", "parse@1") is not None


def test_review_events_are_durable_and_ordered(store):
    for i in range(2):
        store.put_review_event(
            ReviewEvent(
                record_id="r",
                field_id="f",
                decision=ReviewDecision.ACCEPT,
                actor_id=f"c{i}",
            )
        )
    events = store.get_review_events("r")
    assert [e.actor_id for e in events] == ["c0", "c1"]


def test_the_registry_persists_across_connections(tmp_path):
    path = tmp_path / "s.db"
    with SqliteStore(path) as store:
        store.save_registry(sb.build_registry().to_canonical())
    with SqliteStore(path) as store:
        from llmbic import Registry

        reg = Registry.from_canonical(store.load_registry())
        assert len(reg.schemas()) == 5
        assert len(reg.migrations()) == 4


def test_a_store_written_by_a_future_format_is_refused(tmp_path):
    path = tmp_path / "s.db"
    with SqliteStore(path) as store:
        store._exec("UPDATE meta SET value = '99' WHERE key = 'store_format_version'")
    with pytest.raises(LlmbicError) as exc:
        SqliteStore(path)
    assert exc.value.code is ErrorCode.STORAGE_CONFLICT
    assert "unrelated to your record schema versions" in exc.value.message


def test_reading_deterministic_history_needs_no_model_credential(tmp_path, registry, papers):
    """FR-STO-005."""

    from llmbic import ExtractedValueCodec, Project

    path = tmp_path / "s.db"
    project = Project(path, registry=registry, codec=ExtractedValueCodec())
    project.ingest(
        papers[0].record,
        schema_ref="study@1.0",
        record_id=papers[0].record_id,
        source=papers[0].source,
        parsed=papers[0].parsed,
    )
    project.close()

    reopened = Project(path, codec=ExtractedValueCodec())  # no adapters at all
    assert reopened.get_record(papers[0].record_id) is not None
    assert reopened.registry.migrations()
    assert reopened.diff("study@1.0", "study@1.1").changes
    reopened.close()


def test_paging_does_not_materialise_the_corpus(store):
    from llmbic.store.base import batched

    for i in range(10):
        store.upsert_index(RecordIndexEntry(record_id=f"r{i:02d}", schema_ref="s@1"))
    pages = list(batched(store.index(), 3))
    assert [len(p) for p in pages] == [3, 3, 3, 1]
