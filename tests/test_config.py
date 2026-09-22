"""Configuration and secret handling (FR-API-004, NFR-SEC-001/002)."""

from __future__ import annotations


import pytest
import yaml

from llmbic.config import (
    SECRET_KEYS,
    AdapterSpec,
    Config,
    SchemaSpec,
    expand_env,
    find_config,
    load_config,
    load_schema_spec,
)
from llmbic.errors import ErrorCode, LlmbicError
from llmbic.records import ExtractedValueCodec, PlainCodec


def _write(tmp_path, payload, name="llmbic.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload))
    return path


def test_environment_references_are_resolved(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMBIC_TEST_KEY", "s3cret")
    path = _write(
        tmp_path,
        {
            "store": "./s.db",
            "adapters": {
                "main": {
                    "kind": "llmbic.models.mock:ScriptedAdapter",
                    "options": {"api_key": "${env:LLMBIC_TEST_KEY}"},
                }
            },
        },
    )
    config = load_config(path)
    assert config.adapters[0].options["api_key"] == "s3cret"


def test_a_missing_environment_variable_is_a_configuration_error(tmp_path, monkeypatch):
    monkeypatch.delenv("LLMBIC_ABSENT", raising=False)
    path = _write(tmp_path, {"store": "${env:LLMBIC_ABSENT}"})
    with pytest.raises(LlmbicError) as exc:
        load_config(path)
    assert exc.value.code is ErrorCode.CONFIG_INVALID
    assert "LLMBIC_ABSENT" in exc.value.message


def test_a_default_may_be_supplied_inline(monkeypatch):
    monkeypatch.delenv("LLMBIC_ABSENT", raising=False)
    assert expand_env("${env:LLMBIC_ABSENT:fallback}") == "fallback"


def test_non_strict_expansion_leaves_an_empty_string(monkeypatch):
    monkeypatch.delenv("LLMBIC_ABSENT", raising=False)
    assert expand_env("${env:LLMBIC_ABSENT}", strict=False) == ""


def test_secrets_are_redacted_from_anything_printable(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMBIC_TEST_KEY", "s3cret")
    path = _write(
        tmp_path,
        {
            "adapters": {
                "main": {
                    "kind": "llmbic.models.mock:ScriptedAdapter",
                    "options": {
                        "api_key": "${env:LLMBIC_TEST_KEY}",
                        "provider": "mock",
                    },
                }
            }
        },
    )
    redacted = load_config(path).redacted()
    blob = yaml.safe_dump(redacted)
    assert "s3cret" not in blob
    assert redacted["adapters"][0]["options"]["api_key"] == "<redacted>"
    assert redacted["adapters"][0]["options"]["provider"] == "mock"


def test_every_documented_secret_key_is_redacted():
    spec = AdapterSpec(name="a", kind="k", options={k: "x" for k in SECRET_KEYS})
    assert set(spec.redacted()["options"].values()) == {"<redacted>"}


def test_paths_resolve_relative_to_the_config_file(tmp_path):
    (tmp_path / "sub").mkdir()
    path = _write(tmp_path / "sub", {"store": "s.db"})
    config = load_config(path)
    assert config.resolve("s.db") == tmp_path / "sub" / "s.db"
    assert config.resolve("/abs/s.db").is_absolute()


def test_a_missing_config_file_is_an_error(tmp_path):
    with pytest.raises(LlmbicError):
        load_config(tmp_path / "nope.yaml")


def test_find_config_walks_upward(tmp_path, monkeypatch):
    _write(tmp_path, {"store": "s.db"})
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    assert find_config(nested) == tmp_path / "llmbic.yaml"


def test_find_config_returns_none_when_there_is_none(tmp_path):
    assert find_config(tmp_path) is None


@pytest.mark.parametrize(
    "spec,expected",
    [("plain", PlainCodec), ("extracted_value", ExtractedValueCodec)],
)
def test_the_codec_is_configurable(tmp_path, spec, expected):
    path = _write(tmp_path, {"codec": spec})
    assert isinstance(load_config(path).codec.build(), expected)


def test_a_codec_may_take_options(tmp_path):
    path = _write(
        tmp_path,
        {"codec": {"kind": "extracted_value", "options": {"source_id": "pmid:1"}}},
    )
    codec = load_config(path).codec.build()
    assert codec.source_id == "pmid:1"


def test_adapters_are_instantiated_from_dotted_paths(tmp_path):
    path = _write(
        tmp_path,
        {
            "adapters": {
                "main": {
                    "kind": "llmbic.models.mock:ScriptedAdapter",
                    "options": {"provider": "cfg", "model": "m1"},
                }
            }
        },
    )
    adapters = load_config(path).build_adapters()
    assert adapters["main"].identity.provider == "cfg"


def test_an_unloadable_adapter_is_a_configuration_error(tmp_path):
    path = _write(tmp_path, {"adapters": {"main": {"kind": "nope.nothing:Missing"}}})
    with pytest.raises(LlmbicError) as exc:
        load_config(path).build_adapters()
    assert exc.value.code is ErrorCode.CONFIG_INVALID


def test_extensions_are_imported_so_transforms_register(tmp_path):
    path = _write(tmp_path, {"extensions": ["studybed"]})
    assert load_config(path).load_extensions() == ["studybed"]
    from llmbic.functions import TRANSFORMS

    assert "derive_is_healthy@1" in TRANSFORMS.refs()


def test_a_schema_spec_loads_json_schema(tmp_path):
    import json

    import studybed as sb

    (tmp_path / "s.json").write_text(json.dumps(sb.SCHEMA_DOCS["1.0"]()))
    config = Config(root=tmp_path)
    schema = load_schema_spec(
        SchemaSpec(path="s.json", name="study", version="1.0"), config
    )
    assert schema.ref == "study@1.0"
    assert schema.by_path("tasks[].response_mode") is not None


def test_a_schema_spec_loads_linkml(tmp_path):
    config = Config(root=tmp_path)
    schema = load_schema_spec(
        SchemaSpec(
            path="/home/user/study_schema/neuroimaging-study-extraction.yaml",
            name="study",
            version="0.5.0",
            format="linkml",
        ),
        config,
    )
    assert schema.ref == "study@0.5.0"
    assert len(schema.fields) > 100


def test_an_unknown_schema_format_is_refused(tmp_path):
    with pytest.raises(LlmbicError):
        load_schema_spec(
            SchemaSpec(path="x", name="n", version="1", format="runes"), Config(root=tmp_path)
        )
