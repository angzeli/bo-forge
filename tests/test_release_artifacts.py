"""Distribution, installation, and release-document readiness tests."""

import ast
import re

from tests._release_readiness_support import (
    PROJECT_ROOT,
    PROJECT_VERSION,
    PYPROJECT,
    RELEASE_DEPENDENCY_PROBE_ENV,
    CampaignSession,
    Path,
    _assert_sdist_contains_release_assets,
    _assert_sdist_test_fixture_works,
    _assert_wheel_package_boundaries,
    _install_api_extra_and_probe,
    _install_app_extra_and_probe,
    _install_core_only_app_missing_streamlit_probe,
    _install_distribution_and_probe,
    bo_forge,
    os,
    packaged_streamlit_app_path,
    pd,
    shutil,
    subprocess,
    sys,
)


def test_version_and_production_complexity_gate_match_project_metadata() -> None:
    lint = PYPROJECT["tool"]["ruff"]["lint"]

    assert bo_forge.__version__ == PROJECT_VERSION
    assert "C901" in lint["select"]
    assert lint["mccabe"]["max-complexity"] == 12
    assert lint["per-file-ignores"]["tests/**/*.py"] == ["C901"]
    assert "LogBusyError" in bo_forge.__all__
    assert "LogConflictError" in bo_forge.__all__
    assert issubclass(bo_forge.LogBusyError, bo_forge.LogWriteError)
    assert issubclass(bo_forge.LogConflictError, bo_forge.LogWriteError)
    assert "Programming Language :: Python :: 3.12" in PYPROJECT["project"]["classifiers"]
    assert "Development Status :: 4 - Beta" in PYPROJECT["project"]["classifiers"]
    assert "Development Status :: 5 - Production/Stable" not in PYPROJECT["project"][
        "classifiers"
    ]

def test_license_file_exists() -> None:
    assert (PROJECT_ROOT / "LICENSE").is_file()

def test_no_duplicate_release_artifacts_in_worktree() -> None:
    duplicate_artifacts = [
        path
        for directory in ["configs", "examples", "notebooks"]
        for path in (PROJECT_ROOT / directory).glob("* 2.*")
    ]

    assert duplicate_artifacts == []

def test_manifest_does_not_reference_removed_screenshot_assets() -> None:
    manifest = (PROJECT_ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    assert "recursive-include docs *.md" in manifest
    assert "*.png" not in manifest

def test_manifest_uses_expected_release_directives() -> None:
    manifest = (PROJECT_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    lines = [
        line.strip()
        for line in manifest.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    expected = {
        "include CHANGELOG.md",
        "include CONTRIBUTING.md",
        "include LICENSE",
        "include README.md",
        "include ROADMAP_V0_TO_V1.md",
        "include ROADMAP_V1_X.md",
        "include ROADMAP_V2_X.md",
        "include ROADMAP_V3_X.md",
        "include SECURITY.md",
        "recursive-include configs *.yaml",
        "recursive-include docs *.md",
        "include examples/quickstart.py",
        "recursive-include examples *_campaign_log.csv",
        "recursive-include notebooks *.ipynb",
        "recursive-include requirements *.md *.txt",
        "include tests/conftest.py",
    }

    assert set(lines) == expected
    assert all(line.split()[0] in {"include", "recursive-include"} for line in lines)
    assert not any(line.startswith("inclgitude") for line in lines)

def test_structured_tutorial_assets_are_tracked_release_files() -> None:
    release_assets = [
        "configs/14_structured_campaign_tutorial.yaml",
        "examples/14_structured_campaign_tutorial_campaign_log.csv",
        "notebooks/14_structured_campaign_tutorial.ipynb",
    ]

    for relative_path in release_assets:
        assert (PROJECT_ROOT / relative_path).is_file()

    completed = subprocess.run(
        ["git", "ls-files", "--error-unmatch", *release_assets],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr

def test_multi_fidelity_assets_are_tracked_release_files() -> None:
    release_assets = [
        "configs/15_multi_fidelity_qmfkg.yaml",
        "examples/15_multi_fidelity_qmfkg_campaign_log.csv",
        "notebooks/15_multi_fidelity_qmfkg_campaign.ipynb",
    ]

    for relative_path in release_assets:
        assert (PROJECT_ROOT / relative_path).is_file()

    completed = subprocess.run(
        ["git", "ls-files", "--error-unmatch", *release_assets],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr

def test_contextual_assets_are_tracked_release_files() -> None:
    release_assets = [
        "configs/16_contextual_logei.yaml",
        "examples/16_contextual_logei_campaign_log.csv",
        "notebooks/16_contextual_logei_campaign.ipynb",
    ]

    for relative_path in release_assets:
        assert (PROJECT_ROOT / relative_path).is_file()

    completed = subprocess.run(
        ["git", "ls-files", "--error-unmatch", *release_assets],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr

def test_model_profile_assets_are_tracked_release_files() -> None:
    release_assets = [
        "configs/17_model_profile_logei.yaml",
        "examples/17_model_profile_campaign_log.csv",
        "notebooks/17_model_profile_logei_campaign.ipynb",
    ]

    for relative_path in release_assets:
        assert (PROJECT_ROOT / relative_path).is_file()

    completed = subprocess.run(
        ["git", "ls-files", "--error-unmatch", *release_assets],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr

def test_qlog_nei_assets_are_tracked_release_files() -> None:
    release_assets = [
        "configs/18_noisy_pending_qlognei.yaml",
        "examples/18_noisy_pending_qlognei_campaign_log.csv",
        "notebooks/18_noisy_pending_qlognei_campaign.ipynb",
    ]

    for relative_path in release_assets:
        assert (PROJECT_ROOT / relative_path).is_file()

    completed = subprocess.run(
        ["git", "ls-files", "--error-unmatch", *release_assets],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr

def test_qlog_nehvi_assets_and_scope_doc_are_release_files() -> None:
    release_assets = [
        "docs/QLOGNEHVI_FEASIBILITY.md",
        "configs/19_multi_objective_qlognehvi.yaml",
        "examples/19_multi_objective_qlognehvi_campaign_log.csv",
    ]
    for release_asset in release_assets:
        assert (PROJECT_ROOT / release_asset).is_file()

    path = PROJECT_ROOT / "docs/QLOGNEHVI_FEASIBILITY.md"
    assert path.is_file()

    content = path.read_text(encoding="utf-8")
    required_phrases = [
        "bo.acquisition: qlog_nehvi",
        "source=qlog_nehvi",
        "implements the conservative qLogNEHVI scope",
        "v2.2.3",
        "Implemented v2.2.3 Scope",
        "Multi-objective + review",
        "Multi-objective + cost",
        "replicates",
        "Structured + qLogNEHVI",
        "Contextual + qLogNEHVI",
        "Multi-fidelity + qLogNEHVI",
        "Decoupled objectives",
        "encoded observed design points as `X_baseline`",
        "objective values remain model outputs, not baseline inputs",
    ]
    for phrase in required_phrases:
        assert phrase in content
    assert "observed objective values as `X_baseline`" not in content
    assert "not a public acquisition yet" not in content

    completed = subprocess.run(
        ["git", "ls-files", "--error-unmatch", *release_assets],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr

def test_readme_contains_current_install_commands() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert "pip install bo-forge" in readme
    assert 'pip install "bo-forge[app]"' in readme
    assert 'pip install "bo-forge[api]"' in readme
    assert "bo-forge --version" in readme
    assert "bo-forge-app" in readme
    assert "bo-forge-api" in readme
    assert "python -m bo_forge_app" in readme
    assert "docs/STREAMLIT_DEPLOYMENT.md" in readme
    assert "docs/API_PROBE.md" in readme
    assert "docs/API_SECURITY.md" in readme
    assert "docs/CAPABILITY_MATRIX.md" in readme
    assert "docs/INSTALLATION.md" in readme

def test_v3_docs_describe_architecture_and_scientific_ux_reset() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    changelog = (PROJECT_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    streamlit_app_docs = (PROJECT_ROOT / "docs" / "STREAMLIT_APP.md").read_text(
        encoding="utf-8"
    )

    assert "# 🧪 BO Forge v3.0.1" in readme
    assert "v3.0.1 adds a CI-backed release foundation" in readme
    assert "## v3.0.1 - CI-Backed Release Foundation" in changelog
    assert "## v3.0.0 - Architecture And Scientific UX Reset" in changelog
    assert "## v2.5.3 - App And API Operational Closeout" in changelog
    assert "## v2.5.2 - Trusted Deployment Hardening" in changelog
    assert "## v2.5.1 - Stage Lifecycle Diagnostics And Usability Polish" in changelog
    assert "## v2.5.0 - Server-Managed Staging And Concurrent Write Safety" in changelog
    assert "server-managed staging is preferred" in readme
    assert "coordinated append, review, and observation writes" in readme
    assert "## v2.4.2 - Multi-Fidelity Diagnostic Polish" in changelog
    assert "## v2.4.1 - qMFKG Performance And Startup Hardening" in changelog
    assert "## v2.4.0 - Discrete And Batch Multi-Fidelity qMFKG" in changelog
    assert "## v2.3.3 - Code Quality And Error-Handling Refactor" in changelog
    assert "## v2.3.2 - Contextual Replicate-Aware BO" in changelog
    assert "context-matched active repeat selection" in changelog
    assert "qLogNEI for supported single-objective campaigns" in readme
    assert "qLogEHVI/qLogNEHVI for coupled multi-objective campaigns" in readme
    assert "notebooks/18_noisy_pending_qlognei_campaign.ipynb" in readme
    assert "docs/QLOGNEHVI_FEASIBILITY.md" in readme
    assert "bo.acquisition: qlog_nehvi" in readme
    assert "configs/19_multi_objective_qlognehvi.yaml" in readme
    assert "cost-aware qLogNEI" in readme
    assert "configs/17_model_profile_logei.yaml" in readme
    assert "bo-forge model-summary" in readme
    assert "bo-forge model-compare" in readme
    assert "bo-forge plot --kind model-diagnostics" in readme
    assert "bo-forge plot --kind model-comparison" in readme
    assert "does not automatically select a model" in readme
    assert "single-objective contextual LogEI/qLogEI" in readme
    assert "--allow-network-access" in readme
    assert "backward compatible with prior v1.x baselines" not in readme
    assert "ROADMAP_V2_X.md" in readme
    assert "ROADMAP_V3_X.md" in readme
    assert "docs/MIGRATION_V3.md" in readme
    assert "CAPABILITY_MATRIX.md" in readme
    assert "configs/16_contextual_logei.yaml" in readme
    assert "configs/20_contextual_cost_review_logei.yaml" in readme
    assert "notebooks/20_contextual_cost_review_logei_campaign.ipynb" in readme
    assert "configs/21_contextual_replicate_logei.yaml" in readme
    assert "examples/21_contextual_replicate_campaign_log.csv" in readme
    assert "configs/22_discrete_multi_fidelity_qmfkg.yaml" in readme
    assert "examples/22_discrete_multi_fidelity_qmfkg_campaign_log.csv" in readme
    assert "notebooks/22_discrete_multi_fidelity_qmfkg_campaign.ipynb" in readme
    assert "campaign-global budget accounting across contexts" in readme
    assert "CampaignSession.suggest_next(context_values={...})" in readme
    assert "unchanged from the v1.2.3 baseline" not in readme
    assert "BO Forge v3.0.1 provides a local Streamlit workbench" in streamlit_app_docs
    assert "`Campaign`, `Run`, and `Analyze`" in streamlit_app_docs
    assert "Fidelity Coverage" in streamlit_app_docs
    assert "Fidelity Progress" in streamlit_app_docs
    assert "ordered discrete fidelity levels" in streamlit_app_docs
    assert "batch sizes from one through four" in streamlit_app_docs
    assert "conditioned greedy mixed optimization" in readme
    assert "conditioned greedy mixed optimization" in streamlit_app_docs
    assert "Max optimizer iterations" in streamlit_app_docs
    assert "Limit acquisition" in streamlit_app_docs
    assert "v2.2.1 adds Streamlit-facing qLogNEI diagnostics" in streamlit_app_docs
    assert "qLogNEI Summary" in streamlit_app_docs
    assert "qLogNEI Diagnostics" in streamlit_app_docs
    assert "v2.2.3 adds backend qLogNEHVI support" in streamlit_app_docs
    assert "qLogNEHVI as `X_pending`" in streamlit_app_docs
    assert "The v2.2 line includes model-profile visibility" in streamlit_app_docs
    assert "bo.acquisition: log_ei" in streamlit_app_docs
    assert "qlog_nei" in streamlit_app_docs
    assert "Model Diagnostics" in streamlit_app_docs
    assert "Model Comparison" in streamlit_app_docs
    assert "adds Streamlit support for existing structured campaign semantics" in (
        streamlit_app_docs
    )
    assert "shows configured stages and active/inactive variables" in streamlit_app_docs
    assert "requires a stage selection before stage-aware dry-run suggestions" in (
        streamlit_app_docs
    )
    assert "automatic structured-stage transitions" in streamlit_app_docs
    assert "Campaign kind = Multi-fidelity qMFKG" in streamlit_app_docs
    assert "bo.acquisition: qmf_kg" in streamlit_app_docs
    assert "completed v1.5.x line closed the Streamlit-facing contextual BO workflow" in (
        streamlit_app_docs
    )
    assert "Campaign kind = Contextual LogEI" in streamlit_app_docs
    assert "context.default_values" in streamlit_app_docs
    assert "bo.acquisition: log_ei" in streamlit_app_docs
    assert "Contextual Campaigns" in streamlit_app_docs
    assert "Context Summary" in streamlit_app_docs
    assert "Context Diagnostics" in streamlit_app_docs
    assert "context values changed after staging" in streamlit_app_docs.lower()
    assert "exposes `new_only` and `uncertain_best` replicate policies" in streamlit_app_docs
    assert "campaign-scoped observation" in streamlit_app_docs
    assert "deterministic `cost:`, replicates" in streamlit_app_docs
    assert "actual-cost entry" in streamlit_app_docs
    assert "Contextual multi-objective BO" in streamlit_app_docs
    assert "contextual qLogNEI/qLogNEHVI" in streamlit_app_docs
    stale_streamlit_scope = (
        "keeps backend behavior and user-facing app workflow semantics unchanged from the v1.2"
    )
    assert stale_streamlit_scope not in streamlit_app_docs
    assert "no Streamlit multi-fidelity campaign creation" not in streamlit_app_docs

def test_botorch_minor_version_is_bounded_for_optimizer_compatibility() -> None:
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert '"botorch>=0.17,<0.18"' in pyproject

def test_v2_4_1_docs_cover_qmfkg_runtime_controls_and_lazy_startup() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    quickstart = (PROJECT_ROOT / "docs" / "QUICKSTART.md").read_text(encoding="utf-8")
    common_errors = (PROJECT_ROOT / "docs" / "COMMON_ERRORS.md").read_text(
        encoding="utf-8"
    )
    public_api = (PROJECT_ROOT / "docs" / "PUBLIC_API.md").read_text(
        encoding="utf-8"
    )
    performance = (PROJECT_ROOT / "docs" / "PERFORMANCE_BENCHMARKS.md").read_text(
        encoding="utf-8"
    )

    for content in (readme, quickstart, common_errors):
        assert "optimizer_timeout_seconds" in content
        assert "safety limit" in content
        assert "candidate-quality guarantee" in content
        assert "returned after" in content
        assert "in-flight" in content
    assert "optimizer_maxiter" in quickstart
    assert "returned after" in public_api
    assert "in-flight" in public_api
    assert "Top-level exports are resolved lazily" in public_api
    assert "does not load optimizer or plotting dependencies" in public_api
    assert "median of five warm-cache subprocess runs" in performance
    assert "Discrete qMFKG `q=1` suggestion" in performance
    assert "Discrete qMFKG `q=2` suggestion" in performance
    assert "Discrete qMFKG `q=4` suggestion" in performance

def test_capability_matrix_documents_supported_and_deferred_combinations() -> None:
    matrix = (PROJECT_ROOT / "docs" / "CAPABILITY_MATRIX.md").read_text(
        encoding="utf-8"
    )

    required_phrases = [
        "BO Forge v3.0.1",
        "supported",
        "read-only/reporting only",
        "rejected",
        "deferred",
        "Single-objective qLogNEI",
        "Single-objective model profiles",
        "qLogNEI + deterministic cost",
        "qLogNEI + replicate active repeats",
        "Coupled multi-objective qLogNEHVI",
        "qLogNEHVI + deterministic cost",
        "qLogNEHVI + replicates",
        "qLogNEHVI + structured stages",
        "qLogNEHVI + contextual",
        "qLogNEHVI + multi-fidelity",
        "Non-default model profile + multi-objective",
        "Multi-objective + deterministic cost",
        "Structured + contextual",
        "Contextual + multi-objective",
        "Contextual + multi-fidelity",
        "Contextual + deterministic cost",
        "Contextual + review + deterministic cost",
        "Contextual + replicates",
        "Structured + multi-fidelity",
        "Structured + cost",
        "Production API/auth/database",
        "CSV logs remain the source of truth",
    ]
    for phrase in required_phrases:
        assert phrase in matrix
    assert "| Contextual + replicates | supported |" in matrix

def test_contextual_replicate_assets_are_tracked_release_files() -> None:
    release_assets = [
        "configs/21_contextual_replicate_logei.yaml",
        "examples/21_contextual_replicate_campaign_log.csv",
        "tests/test_contextual_replicates.py",
    ]
    for relative_path in release_assets:
        assert (PROJECT_ROOT / relative_path).is_file()

    completed = subprocess.run(
        ["git", "ls-files", "--error-unmatch", *release_assets],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr

def test_core_docs_link_capability_matrix() -> None:
    docs = [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "docs" / "QUICKSTART.md",
        PROJECT_ROOT / "docs" / "PUBLIC_API.md",
        PROJECT_ROOT / "docs" / "STREAMLIT_APP.md",
        PROJECT_ROOT / "docs" / "API_PROBE.md",
        PROJECT_ROOT / "ROADMAP_V2_X.md",
    ]

    for path in docs:
        assert "CAPABILITY_MATRIX.md" in path.read_text(encoding="utf-8")

def test_streamlit_deployment_guide_exists_and_covers_safety_model() -> None:
    guide = (PROJECT_ROOT / "docs" / "STREAMLIT_DEPLOYMENT.md").read_text(
        encoding="utf-8"
    )

    required_phrases = [
        "no built-in auth",
        "trusted LAN",
        "VPN",
        "SSH tunnel",
        "no safe unauthenticated public internet exposure",
        "host filesystem access",
        "Back up CSV logs",
        "simultaneous writes from different hosts",
        "dedicated campaign working directory",
        "bypasses the launcher check",
    ]
    for phrase in required_phrases:
        assert phrase in guide

def test_core_docs_link_streamlit_deployment_guide() -> None:
    docs = [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "docs" / "STREAMLIT_APP.md",
        PROJECT_ROOT / "docs" / "INSTALLATION.md",
        PROJECT_ROOT / "docs" / "QUICKSTART.md",
        PROJECT_ROOT / "docs" / "RELEASE_CHECKLIST.md",
    ]

    for path in docs:
        assert "STREAMLIT_DEPLOYMENT.md" in path.read_text(encoding="utf-8")

def test_api_probe_guide_exists_and_covers_safety_model() -> None:
    guide = (PROJECT_ROOT / "docs" / "API_PROBE.md").read_text(encoding="utf-8")

    required_phrases = [
        "experimental",
        "not a stable public API",
        'pip install "bo-forge[api]"',
        "bo-forge-api --root . --host 127.0.0.1 --port 8765",
        "root-bound",
        "no built-in auth",
        "trusted LAN",
        "SSH tunnel",
        "Do not expose it directly to the public internet",
        "Streamlit remains the recommended local UI",
        "server-managed staging is preferred",
        "--stage-ttl-seconds",
        "--max-staged-batches",
        "GET    /campaign/stages",
        "GET    /campaign/stages/{stage_id}",
        "POST   /campaign/stages/{stage_id}/renew",
        "POST   /campaign/stages/{stage_id}/append",
        "DELETE /campaign/stages/{stage_id}",
        "remaining TTL",
        "Defaults and safely inferred single-stage selections are reported",
        "retained terminal",
        "Lifecycle totals reset",
        '"retryable": false',
        '"suggested_action"',
        "disappear on restart",
        "authentication credentials",
        "multi-worker stage sharing",
        "Client-Carried Compatibility Append",
        "log_busy",
        "--allow-network-access",
        "--server-stages-only",
        "--no-docs",
        "client_bundle_append_disabled",
        "structured `422 request_validation`",
        "deployment",
    ]
    for phrase in required_phrases:
        assert phrase in guide

    assert "v1.2.3" not in guide

def test_api_security_guide_covers_trust_boundary_and_deferred_controls() -> None:
    guide = (PROJECT_ROOT / "docs" / "API_SECURITY.md").read_text(encoding="utf-8")
    normalized = " ".join(guide.split())

    required_phrases = [
        "no built-in authentication",
        "campaign YAML files",
        "host compute time",
        "root-bound",
        "Unauthorized callers can mutate campaign logs and consume compute",
        "server-managed stages",
        "Stage IDs are opaque but are not credentials",
        "process restart",
        "not shared across workers",
        "--allow-network-access",
        "acknowledgement only",
        "--server-stages-only",
        "--no-docs",
        "SSH tunnel or VPN",
        "externally authenticated",
        "Signed client bundles are also deferred",
        "Do not expose this unauthenticated listener directly to the public internet",
    ]
    for phrase in required_phrases:
        assert phrase in normalized

def test_release_checklist_requires_clean_tracked_security_docs() -> None:
    checklist = (PROJECT_ROOT / "docs" / "RELEASE_CHECKLIST.md").read_text(
        encoding="utf-8"
    )

    assert "git status --short" in checklist
    assert "git ls-files --error-unmatch" in checklist
    assert "docs/API_SECURITY.md" in checklist
    assert "all intended files must be committed" in checklist.lower()

def test_v1_roadmap_line_is_completed_history_after_contextual_closeout() -> None:
    roadmap = (PROJECT_ROOT / "ROADMAP_V1_X.md").read_text(encoding="utf-8")

    assert "Current baseline: `v1.5.3`" in roadmap
    assert "Explicit stage-aware backend/session/CLI suggestions" in roadmap
    assert "Read-only stage summaries, structured report sections" in roadmap
    assert "Structured campaign tutorial config, seed log, and notebook" in roadmap
    assert "Streamlit stage display, stage-aware dry-run suggestions" in roadmap
    assert "multi-fidelity semantics remain deferred" not in roadmap
    assert "Streamlit structured campaign creation" not in roadmap
    assert (
        "`v1.3.4` | Patch | Streamlit structured campaign workflow wrapper "
        "with stage selector"
    ) in roadmap
    assert "`v1.4.0` | Minor | Single-objective continuous-fidelity qMFKG" in roadmap
    assert "`v1.4.1` | Patch | Read-only fidelity summaries" in roadmap
    assert "`v1.4.2` | Patch | Multi-fidelity qMFKG tutorial notebook" in roadmap
    assert "`v1.4.3` | Patch | Streamlit creation and qMFKG suggestion controls" in roadmap
    assert "`v1.5.0` | Minor | Contextual single-objective LogEI/qLogEI core" in roadmap
    assert "`v1.5.1` | Patch | Context summaries, context diagnostics" in roadmap
    assert "`v1.5.3` | Patch | Streamlit creation and suggestion controls" in roadmap
    assert "context-state safety and release polish" in roadmap
    assert "BoTorch `SingleTaskMultiFidelityGP` fitting" in roadmap
    assert "Context variables remain normal CSV variable columns" in roadmap
    assert "bo-forge suggest --context NAME=VALUE" in roadmap
    assert "bo-forge context-summary" in roadmap
    assert "notebooks/16_contextual_logei_campaign.ipynb" in roadmap
    assert "Streamlit can create single-objective Contextual LogEI campaigns" in roadmap
    assert re.search(
        r"## 🏗️ v1\.2 - App Launcher And Access Path\s+Status: completed",
        roadmap,
    )
    assert re.search(r"## 🧩 v1\.3 - Structured Campaigns\s+Status: completed", roadmap)
    assert re.search(
        r"## 🧪 v1\.4 - Single-Objective Multi-Fidelity qMFKG\s+Status: completed",
        roadmap,
    )
    assert re.search(
        r"## 🌐 v1\.5 - Contextual BO\s+Status: completed",
        roadmap,
    )

def test_v2_roadmap_is_completed_and_v3_baseline_is_active() -> None:
    roadmap = (PROJECT_ROOT / "ROADMAP_V2_X.md").read_text(encoding="utf-8")
    v3_roadmap = (PROJECT_ROOT / "ROADMAP_V3_X.md").read_text(encoding="utf-8")
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "Final v2 baseline: `v2.5.3`" in roadmap
    assert "ROADMAP_V3_X.md" in roadmap
    assert "coherence and controlled expansion" in roadmap
    assert "docs/CAPABILITY_MATRIX.md" in roadmap
    assert 'v210["v2.1.0<br/>Model profiles + diagnostics"]' in roadmap
    assert 'v211["v2.1.1<br/>Summary hardening + tutorial"]' in roadmap
    assert 'v212["v2.1.2<br/>Comparison diagnostics"]' in roadmap
    assert 'v213["v2.1.3<br/>Model-profile closeout"]' in roadmap
    assert 'v220["v2.2.0<br/>qLogNEI + X_pending"]' in roadmap
    assert 'v221["v2.2.1<br/>qLogNEI diagnostics + tutorial"]' in roadmap
    assert 'v222["v2.2.2<br/>qLogNEHVI feasibility review"]' in roadmap
    assert 'v223["v2.2.3<br/>Conservative qLogNEHVI"]' in roadmap
    assert 'v230["v2.3.0<br/>Contextual review + cost"]' in roadmap
    assert 'v231["v2.3.1<br/>Combination hardening"]' in roadmap
    assert 'v232["v2.3.2<br/>Contextual replicates"]' in roadmap
    assert 'v233["v2.3.3<br/>Code-quality closeout"]' in roadmap
    assert 'v240["v2.4.0<br/>Discrete + batch qMFKG"]' in roadmap
    assert 'v241["v2.4.1<br/>Performance hardening"]' in roadmap
    assert 'v242["v2.4.2<br/>Diagnostic polish"]' in roadmap
    assert 'v243["v2.4.3<br/>Release closeout"]' in roadmap
    assert 'v250["v2.5.0<br/>Server staging + write coordination"]' in roadmap
    assert 'v251["v2.5.1<br/>Stage lifecycle diagnostics + polish"]' in roadmap
    assert 'v252["v2.5.2<br/>Trusted deployment hardening"]' in roadmap
    assert 'v253["v2.5.3<br/>Operational closeout"]' in roadmap
    assert "class v20,v21,v22,v23,v24,v25 majorDone" in roadmap
    assert "class v210,v211,v212,v213 patchDone" in roadmap
    assert "class v220,v221,v222,v223 patchDone" in roadmap
    assert "class v230,v231,v232,v233 patchDone" in roadmap
    assert "class v240,v241,v242,v243 patchDone" in roadmap
    assert "class v250,v251,v252,v253 patchDone" in roadmap
    assert "classDef majorDone" in roadmap
    assert "classDef majorActive" in roadmap
    assert "classDef patchDone" in roadmap
    assert "classDef patchActive" in roadmap
    assert "classDef patchFuture" in roadmap
    assert "v2.0.x - Stable v2 Baseline" in roadmap
    assert "Status: completed" in roadmap
    assert "v2.1.x - Model Profiles And Advanced Surrogates" in roadmap
    assert "`v2.1.3` closes the model-profile line" in roadmap
    assert "bo-forge model-summary" in roadmap
    assert "bo-forge model-compare" in roadmap
    assert "plot --kind model-diagnostics" in roadmap
    assert "plot --kind model-comparison" in roadmap
    assert "Model comparison is diagnostic only" in roadmap
    assert "v2.2.x - Noisy And Pending-Aware BO" in roadmap
    assert "Status: completed" in roadmap
    assert "`v2.2.0` adds `bo.acquisition: qlog_nei`" in roadmap
    assert "`v2.2.1` adds `qlog_nei_summary`" in roadmap
    assert "`v2.2.2` adds [docs/QLOGNEHVI_FEASIBILITY.md]" in roadmap
    assert "`v2.2.3` implements conservative coupled multi-objective qLogNEHVI" in roadmap
    assert "cost-aware, replicate-aware, structured, contextual, multi-fidelity" in roadmap
    assert "v2.3.x - Controlled Feature Combinations" in roadmap
    assert re.search(
        r"## v2\.3\.x - Controlled Feature Combinations\s+Status: completed",
        roadmap,
    )
    assert "`v2.3.0` adds single-objective contextual LogEI support" in roadmap
    assert "`v2.3.1` hardens contextual combination staging" in roadmap
    assert "`v2.3.2` adds contextual replicate-aware group-mean fitting" in roadmap
    assert "`v2.3.3` closes the line with behavior-preserving" in roadmap
    assert "v2.4.x - Multi-Fidelity Expansion" in roadmap
    assert "`v2.4.3` closes the line" in roadmap
    assert re.search(
        r"## v2\.4\.x - Multi-Fidelity Expansion\s+Status: completed",
        roadmap,
    )
    assert "ordered numeric fidelity levels" in roadmap
    assert "batches from one through four" in roadmap
    assert "v2.5.x - App/API Operational Hardening" in roadmap
    assert re.search(
        r"## v2\.5\.x - App/API Operational Hardening\s+Status: completed",
        roadmap,
    )
    assert "exactly-once append claims" in roadmap
    assert "same-machine append, review, and observation mutations" in roadmap
    assert "deployment-mode and network-access behavior" in roadmap
    assert "No mandatory database" in roadmap
    assert "No unrestricted feature cross-product" in roadmap
    assert "No raw low-level kernel API as the first modeling extension" in roadmap
    expected_roadmap_url = (
        'Roadmap = "https://github.com/angzeli/bo-forge/blob/main/ROADMAP_V3_X.md"'
    )
    assert expected_roadmap_url in pyproject
    assert "Current prepared baseline: `v3.0.1`" in v3_roadmap
    assert 'v301["v3.0.1<br/>CI-backed release foundation"]' in v3_roadmap
    assert "docs/MIGRATION_V3.md" in v3_roadmap

def test_streamlit_service_layer_is_documented_as_internal_non_http() -> None:
    repository_structure = (PROJECT_ROOT / "docs" / "REPOSITORY_STRUCTURE.md").read_text(
        encoding="utf-8"
    )
    streamlit_app_docs = (PROJECT_ROOT / "docs" / "STREAMLIT_APP.md").read_text(
        encoding="utf-8"
    )
    public_api = (PROJECT_ROOT / "docs" / "PUBLIC_API.md").read_text(encoding="utf-8")

    assert "bo_forge.application" in repository_structure
    assert "internal, non-HTTP app service layer" in repository_structure
    assert "compatibility shim" in repository_structure
    assert "internal non-HTTP service layer" in streamlit_app_docs
    assert "bo_forge.application" in public_api
    assert "bo_forge_api" in public_api

def test_release_checklist_includes_fresh_install_pip_check() -> None:
    checklist = (PROJECT_ROOT / "docs" / "RELEASE_CHECKLIST.md").read_text(encoding="utf-8")

    assert "/tmp/bo-forge-release/bin/python -m pip check" in checklist
    assert "requirements/constraints-py312-linux-x86_64.txt" in checklist
    assert ".github/workflows/ci.yml" in checklist
    assert ".github/workflows/release-gate.yml" in checklist
    assert "exact release commit is the authoritative gate" in checklist
    assert "does not create a GitHub Release" in checklist


def test_release_checklist_pytest_nodes_reference_existing_tests() -> None:
    checklist = (PROJECT_ROOT / "docs" / "RELEASE_CHECKLIST.md").read_text(
        encoding="utf-8"
    )
    nodes = re.findall(r"(tests/[A-Za-z0-9_./-]+\.py)::([A-Za-z0-9_]+)", checklist)

    assert nodes
    for relative_path, test_name in nodes:
        test_path = PROJECT_ROOT / relative_path
        assert test_path.is_file(), relative_path
        tree = ast.parse(test_path.read_text(encoding="utf-8"), filename=str(test_path))
        assert any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == test_name
            for node in tree.body
        ), f"{relative_path}::{test_name}"


def test_release_checklist_names_runtime_packages_and_release_boundaries() -> None:
    checklist = (PROJECT_ROOT / "docs" / "RELEASE_CHECKLIST.md").read_text(
        encoding="utf-8"
    )

    assert "`bo_forge`, `bo_forge_app`,\nand `bo_forge_api` packages" in checklist
    assert "generated constraints remain outside the wheel" in checklist
    assert "inside the sdist" in checklist
    assert "requirements/constraints-py312-linux-x86_64.txt" in checklist
    assert "configs/18_noisy_pending_qlognei.yaml" in checklist
    assert "configs/22_discrete_multi_fidelity_qmfkg.yaml" in checklist

def test_release_checklist_isolated_quickstart_recipe_works(tmp_path: Path) -> None:
    probe_root = tmp_path / "bo_forge_quickstart_probe"
    shutil.copytree(PROJECT_ROOT / "configs", probe_root / "configs")
    shutil.copytree(
        PROJECT_ROOT / "examples",
        probe_root / "examples",
        ignore=shutil.ignore_patterns(
            "quickstart_working_log*.csv",
            "*latest_suggestions*.csv",
        ),
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    completed = subprocess.run(
        [sys.executable, "examples/quickstart.py"],
        cwd=probe_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (probe_root / "examples" / "quickstart_working_log.csv").is_file()

def test_generated_constraints_replace_direct_dependency_snapshot() -> None:
    assert not (PROJECT_ROOT / "requirements-lock.txt").exists()
    assert (PROJECT_ROOT / "requirements" / "README.md").is_file()

def test_installation_tutorial_covers_pip_install_paths() -> None:
    tutorial = (PROJECT_ROOT / "docs" / "INSTALLATION.md").read_text(encoding="utf-8")

    assert "pip install bo-forge" in tutorial
    assert 'pip install "bo-forge[app]"' in tutorial
    assert 'pip install "bo-forge[api]"' in tutorial
    assert "uv pip install --python" in tutorial
    assert "--no-deps --no-build-isolation -e ." in tutorial
    assert f"dist/bo_forge-{PROJECT_VERSION}-py3-none-any.whl" in tutorial
    assert f"dist/bo_forge-{PROJECT_VERSION}.tar.gz" in tutorial
    assert "pip check" in tutorial

def test_quickstart_has_no_stale_v0_4_current_feature_wording() -> None:
    quickstart = (PROJECT_ROOT / "docs" / "QUICKSTART.md").read_text(encoding="utf-8")

    stale_phrases = [
        "v0.4.3 adds optional deterministic cost",
        "v0.4.3 uses greedy",
        "v0.4.4 adds optional explicit replicate",
    ]
    for phrase in stale_phrases:
        assert phrase not in quickstart

def test_structured_stage_docs_use_working_log_suggestion_flow() -> None:
    cli_docs = (PROJECT_ROOT / "docs" / "CLI.md").read_text(encoding="utf-8")
    quickstart = (PROJECT_ROOT / "docs" / "QUICKSTART.md").read_text(encoding="utf-8")
    csv_schema = (PROJECT_ROOT / "docs" / "CSV_SCHEMA.md").read_text(encoding="utf-8")
    repository_structure = (PROJECT_ROOT / "docs" / "REPOSITORY_STRUCTURE.md").read_text(
        encoding="utf-8"
    )

    for content in (cli_docs, quickstart):
        assert "bo-forge init-log" in content
        assert "13_structured_campaign_core_working_log.csv" in content
        assert "--stage screen" in content
        assert "stage-summary" in content
        assert "stage-diagnostics" in content
    assert (
        "bo-forge suggest --config PATH --log PATH [--batch-size N] "
        "[--stage STAGE_NAME] [--context NAME=VALUE ...]"
    ) in cli_docs
    assert "Structured campaigns use `--stage`; contextual campaigns use repeatable" in cli_docs
    assert "14_structured_campaign_tutorial.yaml" in quickstart
    assert "14_structured_campaign_tutorial_campaign_log.csv" in quickstart
    assert "manually staged rows" not in quickstart
    assert "manually staged rows" not in repository_structure
    normalized_csv_schema = " ".join(csv_schema.split())
    assert "`stages:` cannot be combined with `cost:`." in normalized_csv_schema
    assert "contextual cost suggestions evaluate cost on the full candidate" in csv_schema
    assert "source,[stage],review_status" not in csv_schema

def test_app_created_campaign_tutorial_uses_current_streamlit_labels() -> None:
    tutorial = (PROJECT_ROOT / "docs" / "09_APP_CREATED_CAMPAIGN_TUTORIAL.md").read_text(
        encoding="utf-8"
    )

    assert "Campaign file action" in tutorial
    assert "Create Campaign" in tutorial
    assert "Contextual LogEI" in tutorial
    assert "Update YAML preview from form" in tutorial
    assert "`Campaign` area" in tutorial
    assert "Campaign Files" not in tutorial
    assert "`Campaign` panel" not in tutorial
    assert "Create Campaign tab" not in tutorial
    assert "Regenerate YAML from structured fields" not in tutorial

def test_multi_fidelity_docs_reference_example_and_qmfkg_contract() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    cli_docs = (PROJECT_ROOT / "docs" / "CLI.md").read_text(encoding="utf-8")
    quickstart = (PROJECT_ROOT / "docs" / "QUICKSTART.md").read_text(encoding="utf-8")
    csv_schema = (PROJECT_ROOT / "docs" / "CSV_SCHEMA.md").read_text(encoding="utf-8")
    common_errors = (PROJECT_ROOT / "docs" / "COMMON_ERRORS.md").read_text(
        encoding="utf-8"
    )
    public_api = (PROJECT_ROOT / "docs" / "PUBLIC_API.md").read_text(encoding="utf-8")
    api_probe = (PROJECT_ROOT / "docs" / "API_PROBE.md").read_text(encoding="utf-8")

    for content in (readme, cli_docs, quickstart):
        assert "22_discrete_multi_fidelity_qmfkg" in content
    for content in (readme, quickstart):
        assert "15_multi_fidelity_qmfkg" in content
    assert "fidelity-summary" in cli_docs
    assert "fidelity-coverage" in cli_docs
    assert "fidelity-diagnostics" in cli_docs
    assert "fidelity-progress" in cli_docs
    assert "campaign.fidelity_summary()" in quickstart
    assert "campaign.fidelity_coverage()" in quickstart
    assert "- `fidelity_coverage`" in public_api
    assert "CampaignSession.plot_fidelity_progress()" in public_api
    assert "`fidelity_summary` and `fidelity_coverage`" in api_probe
    assert "notebooks/15_multi_fidelity_qmfkg_campaign.ipynb" in quickstart
    assert "notebooks/15_multi_fidelity_qmfkg_campaign.ipynb" in readme
    assert "notebooks/22_discrete_multi_fidelity_qmfkg_campaign.ipynb" in quickstart
    assert "notebooks/22_discrete_multi_fidelity_qmfkg_campaign.ipynb" in readme
    assert "source=qmf_kg" in csv_schema
    assert "no new CSV columns" in csv_schema
    assert "fidelity cost is separate from BO Forge's `cost:`" in csv_schema
    assert "qMFKG supports batch_size from 1 through 4" in common_errors
    assert "fidelity.levels must be" in common_errors
    assert "has off-grid fidelity value" in common_errors
    assert "has ambiguous fidelity value" in common_errors
    assert "must match exactly one configured level" in csv_schema
    assert "maps each row to exactly one configured level" in csv_schema

def test_contextual_docs_reference_example_and_context_contract() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    cli_docs = (PROJECT_ROOT / "docs" / "CLI.md").read_text(encoding="utf-8")
    quickstart = (PROJECT_ROOT / "docs" / "QUICKSTART.md").read_text(encoding="utf-8")
    csv_schema = (PROJECT_ROOT / "docs" / "CSV_SCHEMA.md").read_text(encoding="utf-8")
    common_errors = (PROJECT_ROOT / "docs" / "COMMON_ERRORS.md").read_text(
        encoding="utf-8"
    )
    public_api = (PROJECT_ROOT / "docs" / "PUBLIC_API.md").read_text(encoding="utf-8")
    api_probe = (PROJECT_ROOT / "docs" / "API_PROBE.md").read_text(encoding="utf-8")

    for content in (readme, cli_docs, quickstart):
        assert "16_contextual_logei" in content
        assert "--context feedstock_acidity=0.25" in content
        assert "context-summary" in content
        assert "context-diagnostics" in content
    assert "notebooks/16_contextual_logei_campaign.ipynb" in quickstart
    assert "notebooks/16_contextual_logei_campaign.ipynb" in readme
    assert "context_values={...}" in public_api
    assert "ContextConfig" in public_api
    assert "context_summary" in public_api
    assert "context_values" in api_probe
    assert "no new CSV columns" in csv_schema
    assert "context variables are stored as normal CSV variable columns" in csv_schema
    assert "Contextual suggestions require values" in common_errors
    assert "context cannot be combined with" in common_errors

def test_model_profile_docs_reference_example_and_contract() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    cli_docs = (PROJECT_ROOT / "docs" / "CLI.md").read_text(encoding="utf-8")
    quickstart = (PROJECT_ROOT / "docs" / "QUICKSTART.md").read_text(encoding="utf-8")
    csv_schema = (PROJECT_ROOT / "docs" / "CSV_SCHEMA.md").read_text(encoding="utf-8")
    common_errors = (PROJECT_ROOT / "docs" / "COMMON_ERRORS.md").read_text(
        encoding="utf-8"
    )
    public_api = (PROJECT_ROOT / "docs" / "PUBLIC_API.md").read_text(encoding="utf-8")

    for content in (readme, cli_docs, quickstart):
        assert "17_model_profile_logei" in content
        assert "model-summary" in content
        assert "model-compare" in content
        assert "model-diagnostics" in content
        assert "model-comparison" in content
    assert "notebooks/17_model_profile_logei_campaign.ipynb" in readme
    assert "notebooks/17_model_profile_logei_campaign.ipynb" in quickstart
    assert "examples/17_model_profile_logei_working_log.csv" in quickstart
    assert "ModelConfig" in public_api
    assert "model_summary" in public_api
    assert "model_profile_comparison" in public_api
    assert "not_recorded" in public_api
    assert "model.profile" in csv_schema
    assert "does not add or remove CSV columns" in csv_schema
    assert "bo.acquisition: log_ei" in csv_schema
    assert "qlog_nei" in csv_schema
    assert "bo.acquisition: log_ei" in quickstart
    assert "qlog_nei" in quickstart
    assert "bo.acquisition: log_ei" in public_api
    assert "qlog_nei" in public_api
    assert "Non-default model profiles" in common_errors
    assert "bo.acquisition: log_ei" in common_errors
    assert "qlog_nei" in common_errors
    assert "model_profile_comparison() does not support" in common_errors

def test_qlog_nei_docs_reference_summary_diagnostics_and_notebook() -> None:
    cli_docs = (PROJECT_ROOT / "docs" / "CLI.md").read_text(encoding="utf-8")
    quickstart = (PROJECT_ROOT / "docs" / "QUICKSTART.md").read_text(
        encoding="utf-8"
    )
    public_api = (PROJECT_ROOT / "docs" / "PUBLIC_API.md").read_text(
        encoding="utf-8"
    )

    for content in (cli_docs, quickstart):
        assert "qlog-nei-summary" in content
        assert "qlog-nei-diagnostics" in content
        assert "18_noisy_pending_qlognei_campaign_log.csv" in content
    assert "qlog_nei_summary" in public_api
    assert "X_pending" in public_api
    assert "notebooks/18_noisy_pending_qlognei_campaign.ipynb" in quickstart

def test_cost_aware_multi_objective_notebook_uses_current_version_wording() -> None:
    notebook_text = (
        PROJECT_ROOT / "notebooks" / "12_cost_aware_multi_objective_qlogehvi_campaign.ipynb"
    ).read_text(encoding="utf-8")

    assert "v1.1 backend workflow" in notebook_text
    assert "v1.1.3 backend workflow" not in notebook_text

def test_replicate_ready_cli_demo_exercises_repeat_path() -> None:
    quickstart = (PROJECT_ROOT / "docs" / "QUICKSTART.md").read_text(encoding="utf-8")

    assert "/tmp/bo_forge_08_repeat_ready.csv" in quickstart
    assert "uncertain_best" in quickstart
    assert "rep_seed_3a" in quickstart
    assert "configs/08_replicate_aware_logei.yaml" in quickstart

def test_replicate_ready_demo_executes_repeat_path(tmp_path: Path) -> None:
    config_path = PROJECT_ROOT / "configs" / "08_replicate_aware_logei.yaml"
    seed_log_path = PROJECT_ROOT / "examples" / "08_replicate_aware_campaign_log.csv"
    working_log_path = tmp_path / "bo_forge_08_repeat_ready.csv"
    working_log_path.write_bytes(seed_log_path.read_bytes())
    df = pd.read_csv(working_log_path, keep_default_na=False)
    df.loc[len(df)] = [
        "rep_seed_3a",
        3,
        "observed",
        "manual",
        "rep_3",
        0,
        0.85,
        430,
        1.10,
        "",
        "",
        "",
    ]
    df.to_csv(working_log_path, index=False)

    campaign = CampaignSession.from_files(config_path, working_log_path)
    suggestions = campaign.suggest_next(batch_size=3)

    existing_groups = set(df["replicate_group"].astype(str))
    repeat_rows = suggestions[
        suggestions["replicate_group"].astype(str).isin(existing_groups)
    ]
    exploration_rows = suggestions[
        ~suggestions["replicate_group"].astype(str).isin(existing_groups)
    ]

    assert len(suggestions) == 3
    assert not repeat_rows.empty
    for group, group_suggestions in repeat_rows.groupby("replicate_group"):
        existing_indexes = df.loc[df["replicate_group"] == group, "replicate_index"].astype(
            int
        )
        expected_indexes = list(
            range(
                int(existing_indexes.max()) + 1,
                int(existing_indexes.max()) + 1 + len(group_suggestions),
            )
        )
        assert sorted(group_suggestions["replicate_index"].astype(int).tolist()) == (
            expected_indexes
        )
    assert len(exploration_rows) == 1
    exploration = exploration_rows.iloc[0]
    assert exploration["replicate_group"] == exploration["row_id"]
    assert int(exploration["replicate_index"]) == 0

def test_public_api_exports_are_importable() -> None:
    public_api = (PROJECT_ROOT / "docs" / "PUBLIC_API.md").read_text(encoding="utf-8")
    section = public_api.split("## ✅ Public Package Exports", maxsplit=1)[1].split(
        "## ",
        maxsplit=1,
    )[0]
    exports = re.findall(r"^- `([^`]+)`", section, flags=re.MULTILINE)

    assert exports
    for export in exports:
        assert hasattr(bo_forge, export), f"Missing public export from bo_forge: {export}"

def test_app_console_entrypoint_resolves_packaged_script() -> None:
    app_path = packaged_streamlit_app_path()

    assert app_path.name == "streamlit_app.py"
    assert app_path.is_file()
    assert app_path.parent.name == "bo_forge_app"

def test_built_distributions_install_from_outside_source_tree(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    resolve_dependencies = env.get(RELEASE_DEPENDENCY_PROBE_ENV) == "1"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--no-isolation",
            "--outdir",
            str(dist_dir),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
        text=True,
    )
    wheels = sorted(dist_dir.glob(f"bo_forge-{PROJECT_VERSION}-*.whl"))
    sdists = sorted(dist_dir.glob(f"bo_forge-{PROJECT_VERSION}.tar.gz"))
    assert wheels, f"No v{PROJECT_VERSION} wheel was built."
    assert sdists, f"No v{PROJECT_VERSION} sdist was built."

    _assert_wheel_package_boundaries(wheels[0])
    _assert_sdist_contains_release_assets(sdists[0])
    _assert_sdist_test_fixture_works(sdists[0], tmp_path / "sdist_tests", env)
    subprocess.run(
        [sys.executable, "-m", "twine", "check", str(wheels[0]), str(sdists[0])],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
        text=True,
    )
    _install_distribution_and_probe(
        artifact=wheels[0],
        probe_root=tmp_path / "wheel_probe",
        env=env,
        install_args=[],
        resolve_dependencies=resolve_dependencies,
    )
    _install_core_only_app_missing_streamlit_probe(
        wheel=wheels[0],
        probe_root=tmp_path / "core_app_probe",
        env=env,
    )
    _install_app_extra_and_probe(
        wheel=wheels[0],
        probe_root=tmp_path / "app_probe",
        env=env,
        resolve_dependencies=resolve_dependencies,
    )
    _install_api_extra_and_probe(
        wheel=wheels[0],
        probe_root=tmp_path / "api_probe",
        env=env,
        resolve_dependencies=resolve_dependencies,
    )
    _install_distribution_and_probe(
        artifact=sdists[0],
        probe_root=tmp_path / "sdist_probe",
        env=env,
        install_args=["--no-build-isolation"],
        resolve_dependencies=resolve_dependencies,
    )
