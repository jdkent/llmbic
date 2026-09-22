"""Shared fixtures.

NFR-MNT-002: none of these touches the network.  The "hosted" adapter is a
local simulation of one, which is the point — its failure modes are what the
retry, resume and budget tests need.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import studybed  # noqa: E402
from llmbic import ExtractedValueCodec, Project  # noqa: E402
from llmbic.models.base import AdapterRegistry  # noqa: E402
from llmbic.models.mock import FailingAdapter  # noqa: E402


@pytest.fixture
def registry():
    return studybed.build_registry()


@pytest.fixture
def adapters():
    reg = AdapterRegistry()
    reg.register("main", studybed.stimulus_adapter())
    reg.register("backup", studybed.stimulus_adapter(provider="backup", model="rules-2"))
    return reg


@pytest.fixture
def broken_adapters():
    reg = AdapterRegistry()
    reg.register("main", FailingAdapter())
    reg.register("backup", FailingAdapter(provider="backup"))
    return reg


@pytest.fixture
def project(tmp_path, registry, adapters):
    """A project at study@1.0 with a small corpus already ingested."""

    p = Project(
        tmp_path / "study.db",
        registry=registry,
        adapters=adapters,
        codec=ExtractedValueCodec(),
    )
    p.save()
    yield p
    p.close()


@pytest.fixture
def papers():
    return studybed.corpus(8)


@pytest.fixture
def loaded(project, papers):
    """``project`` with ``papers`` ingested at study@1.0."""

    for paper in papers:
        project.ingest(
            paper.record,
            schema_ref="study@1.0",
            record_id=paper.record_id,
            source=paper.source,
            parsed=paper.parsed,
        )
    return project


@pytest.fixture
def bed():
    return studybed
