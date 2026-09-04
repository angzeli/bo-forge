"""Architecture and compatibility gates for the v3 maintainability baseline."""

from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import json
from pathlib import Path

import bo_forge
from tests._v3_signature_contract import (
    PUBLIC_EXCEPTION_EXPORTS,
    PUBLIC_SIGNATURE_DIGEST,
    PUBLIC_SIGNATURE_NAMES,
    SESSION_METHOD_NAMES,
    SESSION_METHOD_SIGNATURE_DIGEST,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_ROOTS = (
    PROJECT_ROOT / "bo_forge",
    PROJECT_ROOT / "bo_forge_app",
    PROJECT_ROOT / "bo_forge_api",
)
EXPECTED_PUBLIC_EXPORTS = {
    "BOConfig",
    "BOForgeError",
    "CampaignConfig",
    "CampaignSession",
    "ConfigError",
    "ConstraintConfig",
    "ContextConfig",
    "CostConfig",
    "FidelityConfig",
    "LogBusyError",
    "LogConflictError",
    "LogValidationError",
    "LogWriteError",
    "ModelConfig",
    "ObjectiveConfig",
    "ProvenanceError",
    "ReplicateConfig",
    "ReviewConfig",
    "StageConfig",
    "SuggestionError",
    "VariableConfig",
    "__version__",
    "active_variables_for_stage",
    "append_suggestions",
    "aggregate_observed_replicates",
    "best_replicate_group",
    "configured_stage_names",
    "context_summary",
    "evaluate_cost",
    "fidelity_coverage",
    "fidelity_summary",
    "get_observed_data",
    "hypervolume",
    "hypervolume_progress",
    "is_structured_campaign",
    "load_campaign_log",
    "mark_observed",
    "model_profile_comparison",
    "model_summary",
    "pareto_front",
    "pareto_summary",
    "provenance_summary",
    "qlog_nei_summary",
    "replicate_summary",
    "review_suggestion",
    "stage_summary",
    "suggest_next",
    "suggestion_quality_summary",
    "validate_campaign_data",
}
COMPATIBILITY_MODULES = (
    "bo_forge.config",
    "bo_forge.logs",
    "bo_forge.validation",
    "bo_forge.suggestions",
    "bo_forge.diagnostics",
    "bo_forge.session",
    "bo_forge_app.api",
    "bo_forge_app.api_cli",
    "bo_forge_app.service",
    "bo_forge_app.stages",
)


def _python_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)


def _parsed(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _signature_digest(signatures: dict[str, str]) -> str:
    payload = json.dumps(signatures, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def test_production_modules_stay_below_physical_line_limit() -> None:
    oversized = {
        path.relative_to(PROJECT_ROOT).as_posix(): len(
            path.read_text(encoding="utf-8").splitlines()
        )
        for root in PRODUCTION_ROOTS
        for path in _python_files(root)
        if len(path.read_text(encoding="utf-8").splitlines()) > 800
    }

    assert oversized == {}


def test_production_functions_stay_below_line_limit() -> None:
    oversized: dict[str, int] = {}
    for root in PRODUCTION_ROOTS:
        for path in _python_files(root):
            for node in ast.walk(_parsed(path)):
                if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    continue
                length = (node.end_lineno or node.lineno) - node.lineno + 1
                if length > 120:
                    key = f"{path.relative_to(PROJECT_ROOT).as_posix()}:{node.lineno}:{node.name}"
                    oversized[key] = length

    assert oversized == {}


def test_test_modules_stay_below_physical_line_limit() -> None:
    oversized = {
        path.relative_to(PROJECT_ROOT).as_posix(): len(
            path.read_text(encoding="utf-8").splitlines()
        )
        for path in _python_files(PROJECT_ROOT / "tests")
        if len(path.read_text(encoding="utf-8").splitlines()) > 1_500
    }

    assert oversized == {}


def test_collected_test_modules_use_feature_names_and_explicit_support_modules() -> None:
    tests_root = PROJECT_ROOT / "tests"
    assert list(tests_root.glob("test_*_part*.py")) == []

    support_modules = sorted(tests_root.glob("_*_support.py"))
    assert support_modules
    for path in support_modules:
        tree = _parsed(path)
        assert not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_")
            for node in tree.body
        ), path.name


def test_core_and_optional_packages_respect_layer_boundaries() -> None:
    violations: list[str] = []
    compatibility_shims = {
        Path("bo_forge_app/api.py"),
        Path("bo_forge_app/api_cli.py"),
        Path("bo_forge_app/stages.py"),
    }
    forbidden_by_root = {
        "bo_forge": {"bo_forge_app", "bo_forge_api", "fastapi", "streamlit", "uvicorn"},
        "bo_forge_app": {"bo_forge_api", "fastapi", "uvicorn"},
        "bo_forge_api": {"bo_forge_app", "streamlit"},
    }
    for root in PRODUCTION_ROOTS:
        for path in _python_files(root):
            for node in ast.walk(_parsed(path)):
                imported: str | None = None
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imported = alias.name.split(".", maxsplit=1)[0]
                        if imported in forbidden_by_root[root.name]:
                            if path.relative_to(PROJECT_ROOT) in compatibility_shims:
                                continue
                            violations.append(
                                f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}:{alias.name}"
                            )
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported = node.module.split(".", maxsplit=1)[0]
                    if imported in forbidden_by_root[root.name]:
                        if path.relative_to(PROJECT_ROOT) in compatibility_shims:
                            continue
                        violations.append(
                            f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}:{node.module}"
                        )

    assert violations == []


def test_broad_exception_catches_explain_their_boundary() -> None:
    unexplained: list[str] = []
    for root in PRODUCTION_ROOTS:
        for path in _python_files(root):
            lines = path.read_text(encoding="utf-8").splitlines()
            for node in ast.walk(_parsed(path)):
                if not isinstance(node, ast.ExceptHandler):
                    continue
                if not isinstance(node.type, ast.Name) or node.type.id not in {
                    "Exception",
                    "BaseException",
                }:
                    continue
                start = max(0, node.lineno - 2)
                end = min(len(lines), (node.end_lineno or node.lineno) + 1)
                nearby = "\n".join(lines[start:end]).lower()
                if not any(
                    marker in nearby
                    for marker in (
                        "boundary",
                        "diagnostic",
                        "rollback",
                        "transaction",
                        "reservation",
                    )
                ):
                    unexplained.append(
                        f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}"
                    )

    assert unexplained == []


def test_public_exports_and_compatibility_modules_remain_available() -> None:
    assert set(bo_forge.__all__) == EXPECTED_PUBLIC_EXPORTS
    namespace: dict[str, object] = {}
    exec("from bo_forge import *", namespace)
    assert EXPECTED_PUBLIC_EXPORTS.issubset(namespace)
    assert EXPECTED_PUBLIC_EXPORTS.issubset(dir(bo_forge))
    for module_name in COMPATIBILITY_MODULES:
        assert importlib.import_module(module_name) is not None


def test_core_public_call_signatures_remain_keyword_compatible() -> None:
    callable_exports = {
        name for name in bo_forge.__all__ if callable(getattr(bo_forge, name))
    }
    signatures = {
        name: str(inspect.signature(getattr(bo_forge, name)))
        for name in PUBLIC_SIGNATURE_NAMES
    }

    assert callable_exports == PUBLIC_SIGNATURE_NAMES | PUBLIC_EXCEPTION_EXPORTS
    assert _signature_digest(signatures) == PUBLIC_SIGNATURE_DIGEST, signatures


def test_campaign_session_public_method_signatures_remain_compatible() -> None:
    session_type = bo_forge.CampaignSession
    public_methods = {
        name: member
        for name, member in inspect.getmembers(session_type)
        if not name.startswith("_") and callable(member)
    }

    signatures = {
        name: str(inspect.signature(member)) for name, member in public_methods.items()
    }

    assert set(public_methods) == SESSION_METHOD_NAMES
    assert _signature_digest(signatures) == SESSION_METHOD_SIGNATURE_DIGEST, signatures


def test_compatibility_facades_limit_assignment_propagation(monkeypatch) -> None:
    import bo_forge.suggestions as suggestions
    from bo_forge._optimization import common, single_objective

    original_retry_limit = common.MAX_DECODE_RETRIES
    original_optimizer = single_objective.optimize_log_ei
    replacement_optimizer = object()

    with monkeypatch.context() as patch:
        patch.setattr(suggestions, "MAX_DECODE_RETRIES", original_retry_limit + 1)
        patch.setattr(suggestions, "optimize_log_ei", replacement_optimizer)
        assert common.MAX_DECODE_RETRIES == original_retry_limit
        assert single_objective.optimize_log_ei is replacement_optimizer

    assert common.MAX_DECODE_RETRIES == original_retry_limit
    assert single_objective.optimize_log_ei is original_optimizer
