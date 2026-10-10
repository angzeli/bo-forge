"""Runtime boundaries and static-fixture coverage for the documented core API."""

import ast
import inspect
import json
import tomllib
from pathlib import Path

import pandas as pd
import pytest
from packaging.requirements import Requirement

import bo_forge
from tests._public_typing_contract import public_calls
from tests._typing_install_probe import expected_diagnostics

ROOT = Path(__file__).parents[1]
FIXTURES = ROOT / "tests" / "typing_contract"


@pytest.mark.parametrize("mixed", [False, True])
def test_evaluate_cost_accepts_pandas_rows_without_mutation(mixed):
    variables = (bo_forge.VariableConfig("x", "continuous", 0, 1),)
    if mixed:
        variables += (bo_forge.VariableConfig("solvent", "categorical", values=("A", "B")),)
        row = pd.DataFrame([{"solvent": "B", "x": 0.5}]).iloc[0]
        expression, expected = "1 + x + (solvent == 'B')", 2.5
    else:
        row = pd.Series({"x": 0.5})
        expression, expected = "1 + x", 1.5
    config = bo_forge.CampaignConfig(
        "cost", bo_forge.ObjectiveConfig("score", "maximize"), variables,
        bo_forge.BOConfig(), cost=bo_forge.CostConfig(expression),
    )
    original = row.copy(deep=True)

    assert bo_forge.evaluate_cost(config, row) == expected
    pd.testing.assert_series_equal(row, original)


def test_public_parameters_defaults_and_binding_match_v342():
    baseline = json.loads((ROOT / "tests/fixtures/public_calls_v342.json").read_text())
    assert public_calls() == baseline


def test_public_annotations_are_complete():
    targets = {}
    for name in bo_forge.__all__:
        value = getattr(bo_forge, name)
        if inspect.isfunction(value):
            targets[name] = value
        elif isinstance(value, type):
            for member_name, descriptor in vars(value).items():
                if member_name.startswith("_") and member_name != "__init__":
                    continue
                member = descriptor.fget if isinstance(descriptor, property) else getattr(
                    value, member_name,
                )
                if callable(member):
                    targets[f"{name}.{member_name}"] = member
    for name, value in targets.items():
        signature = inspect.signature(value)
        assert signature.return_annotation is not inspect.Signature.empty, name
        for parameter in signature.parameters.values():
            if parameter.name not in {"self", "cls"}:
                assert parameter.annotation is not inspect.Parameter.empty, (name, parameter.name)


def test_consumers_cover_every_export_and_session_method():
    tree = ast.parse((FIXTURES / "positive.py").read_text())
    exports = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
               and isinstance(node.value, ast.Name) and node.value.id == "bo"}
    assert set(bo_forge.__all__) <= exports
    methods = {name for name, member in inspect.getmembers(bo_forge.CampaignSession)
               if not name.startswith("_") and callable(member)}
    exercised = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
                 and isinstance(node.value, ast.Name) and node.value.id == "session"}
    assert methods <= exercised | {"initialize", "from_files"}
    assert len(expected_diagnostics(FIXTURES / "negative.py")) >= 25


def test_consumer_config_does_not_suppress_core_or_missing_import_errors():
    import configparser

    config = configparser.ConfigParser()
    config.read(FIXTURES / "mypy.ini")
    assert config.sections() == ["mypy"]
    assert config.getboolean("mypy", "strict")
    assert config.getboolean("mypy", "warn_unused_ignores")
    assert config.getboolean("mypy", "disallow_any_unimported")
    assert not config.getboolean("mypy", "ignore_missing_imports")


def test_typing_dependencies_and_marker_stay_within_declared_scope():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    metadata = project["project"]
    def names(requirements):
        return {Requirement(item).name for item in requirements}

    tools = {"mypy", "pandas-stubs"}
    assert tools <= names(metadata["optional-dependencies"]["dev"])
    assert not tools & names(metadata["dependencies"])
    for extra in ("app", "api"):
        assert not tools & names(metadata["optional-dependencies"][extra])
    assert project["tool"]["setuptools"]["package-data"] == {"bo_forge": ["py.typed"]}
    assert (ROOT / "bo_forge/py.typed").read_bytes() == b""


def test_lazy_declarations_cover_runtime_exports_without_static_fallback():
    tree = ast.parse((ROOT / "bo_forge/__init__.py").read_text())
    declarations = next(node for node in tree.body if isinstance(node, ast.If)
                        and isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING")
    declared = {alias.asname or alias.name for node in declarations.body
                if isinstance(node, ast.ImportFrom) for alias in node.names}
    assert declared == set(bo_forge._LAZY_EXPORTS)
    runtime = next(node for node in tree.body if isinstance(node, ast.If)
                  and isinstance(node.test, ast.UnaryOp))
    assert isinstance(runtime.test.op, ast.Not)
    assert runtime.test.operand.id == "TYPE_CHECKING"
    assert any(isinstance(node, ast.FunctionDef) and node.name == "__getattr__"
               for node in runtime.body)
