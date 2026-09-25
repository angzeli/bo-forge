"""Adversarial parser, schema, and scalar cases for the inspection contract."""

import json
from decimal import Decimal
from fractions import Fraction

import jsonschema
import numpy as np
import pandas as pd
import pytest

from bo_forge._cli.output import SerializationError, json_value, render, table_payload
from bo_forge.cli import run
from bo_forge.session import CampaignSession
from tests.test_cli_json import SCHEMA, example_args, response, validate_contract


@pytest.mark.parametrize("flags", [
    ["--for=json"], ["--for", "json"],
    ["--format=json", "--format", "--bad"],
    ["--format=json", "--format=invalid"],
    ["--format=json", "--for"],
    ["--unknown", "--for=json"], ["--for=json", "--unknown"],
    ["--format=text", "--format=json"],
    ["--format=json", "--", "--format=text"],
])
def test_argument_failures_preserve_json_intent(flags, monkeypatch, capsys):
    monkeypatch.setattr(CampaignSession, "from_files", lambda *a, **kw: pytest.fail("loaded"))
    assert run(["validate", *flags]) == 2
    assert response(capsys)[0]["error"]["code"] == "argument_error"


@pytest.mark.parametrize("flags", [
    ["--format=json", "--format=text"],
    ["--for=json", "--for", "text"],
    ["--", "--format=json"], ["--format=invalid"],
])
def test_text_or_terminated_format_intent(flags, capsys):
    assert run(["validate", *flags]) == 2
    assert capsys.readouterr().out == ""


def test_abbreviated_format_success_and_ambiguity(capsys):
    assert run(["validate", *example_args("validate"), "--for=json"]) == 0
    assert response(capsys)[0]["data"] == {"valid": True}
    # Both --profile and --format exist here; an ambiguous abbreviation is not JSON intent.
    assert run(["model-compare", "--=json"]) == 2
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("value", [Fraction(1, 3), Fraction(10**1000), Decimal("0.1"),
                                   np.array([1.0])])
def test_unsupported_numeric_objects_are_controlled_failures(value, monkeypatch, capsys):
    with pytest.raises(SerializationError):
        json_value(value)
    frame = pd.DataFrame({"value": pd.Series([value], dtype=object)})
    monkeypatch.setattr(CampaignSession, "summary", lambda self: frame)
    assert run(["summary", *example_args("summary"), "--format=json"]) == 1
    payload, _ = response(capsys)
    assert payload["data"] is None
    assert payload["error"]["code"] == "serialization_error"


@pytest.mark.parametrize("scalar", [np.int8(-3), np.uint64(2**64 - 1),
                                   np.float16(0.5), np.float32(0.1), np.float64(0.1),
                                   10**1000, np.bool_(False)])
def test_supported_scalar_values_are_preserved(scalar):
    value = json_value(scalar)
    assert value == scalar
    assert json.loads(json.dumps(value, allow_nan=False)) == value


def test_extended_precision_is_not_silently_rounded():
    value = np.longdouble("0.1234567890123456789")
    if value.dtype.itemsize > 8:
        with pytest.raises(SerializationError, match="Extended-precision"):
            json_value(value)
    else:
        assert json_value(value) == float(value)


@pytest.mark.parametrize("command", ["summary", "status", "validate", "provenance"])
def test_additive_fields_do_not_conflict_with_other_payload_shapes(command):
    data = {"columns": ["value"], "records": [{"value": None}],
            "status": "complete", "valid": True, "future": {"field": 1}}
    payload = json.loads(render(command, data))
    payload["future"] = True
    jsonschema.validate(payload, SCHEMA)


@pytest.mark.parametrize("command,data", [
    ("validate", {"valid": True}), ("summary", {"columns": [], "records": []}),
    ("status", {"status": "ready"}), ("provenance", {"valid": True}),
])
def test_failure_payloads_cannot_masquerade_as_success(command, data):
    error = {"code": "config_error", "message": "bad", "hint": None, "details": {}}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(json.loads(render(command, data, error)), SCHEMA)


def test_provenance_failure_can_retain_table():
    error = {"code": "log_conflict_error", "message": "stale", "hint": None, "details": {}}
    for data in (None, {"columns": ["field"], "records": [{"field": "blocked"}]}):
        jsonschema.validate(json.loads(render("provenance", data, error)), SCHEMA)


def test_table_records_match_columns_and_reject_ambiguous_labels():
    data = table_payload(pd.DataFrame({"value": [1, 2], "name": ["001", "002"]}))
    assert all(list(record) == data["columns"] for record in data["records"])
    for columns in (["value", "value"], ["value", 1]):
        with pytest.raises(SerializationError, match="unique strings"):
            table_payload(pd.DataFrame([[1, 2]], columns=columns))


@pytest.mark.parametrize("record", [{"wrong": None}, {}, {"value": 1, "extra": 2}])
def test_table_key_correspondence_is_checked_beyond_schema(record):
    payload = json.loads(render("summary", {"columns": ["value"], "records": [record]}))
    # Standard JSON Schema cannot refer from a record's keys to columns' values.
    jsonschema.validate(payload, SCHEMA)
    with pytest.raises(AssertionError):
        validate_contract(payload)


@pytest.mark.parametrize("command,data", [
    ("validate", {"status": "ready"}), ("status", {"valid": True}),
    ("summary", {"status": "ready"}), ("provenance", None),
])
def test_success_payload_shape_is_selected_by_command(command, data):
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(json.loads(render(command, data)), SCHEMA)
