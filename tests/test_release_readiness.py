import os
import re
import subprocess
import sys
import sysconfig
import tarfile
import zipfile
from pathlib import Path

import pandas as pd

import bo_forge
from bo_forge.session import CampaignSession
from bo_forge_app.cli import packaged_streamlit_app_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
        "include LICENSE",
        "include README.md",
        "include ROADMAP_V0_TO_V1.md",
        "include ROADMAP_V1_X.md",
        "include ROADMAP_V2_X.md",
        "include requirements-lock.txt",
        "recursive-include configs *.yaml",
        "recursive-include docs *.md",
        "include examples/quickstart.py",
        "recursive-include examples *_campaign_log.csv",
        "recursive-include notebooks *.ipynb",
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
    assert "docs/CAPABILITY_MATRIX.md" in readme
    assert "docs/INSTALLATION.md" in readme


def test_v2_1_docs_describe_model_profile_release_scope() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    streamlit_app_docs = (PROJECT_ROOT / "docs" / "STREAMLIT_APP.md").read_text(
        encoding="utf-8"
    )

    assert "# 🧪 BO Forge v2.1.0" in readme
    assert "v2.1.0 adds curated single-objective model profiles" in readme
    assert "avoids raw BoTorch kernel passthrough" in readme
    assert "configs/17_model_profile_logei.yaml" in readme
    assert "bo-forge model-summary" in readme
    assert "bo-forge plot --kind model-diagnostics" in readme
    assert "single-objective contextual LogEI/qLogEI" in readme
    assert "backward compatible with prior v1.x baselines" in readme
    assert "ROADMAP_V2_X.md" in readme
    assert "CAPABILITY_MATRIX.md" in readme
    assert "configs/16_contextual_logei.yaml" in readme
    assert "CampaignSession.suggest_next(context_values={...})" in readme
    assert "unchanged from the v1.2.3 baseline" not in readme
    assert "BO Forge v2.1.0 provides a local Streamlit workbench" in streamlit_app_docs
    assert "v2.1.0 adds model-profile visibility" in streamlit_app_docs
    assert "Model Diagnostics" in streamlit_app_docs
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
    assert "no multi-objective, structured, cost-aware, replicate-aware" in (
        streamlit_app_docs
    )
    stale_streamlit_scope = (
        "keeps backend behavior and user-facing app workflow semantics unchanged from the v1.2"
    )
    assert stale_streamlit_scope not in streamlit_app_docs
    assert "no Streamlit multi-fidelity campaign creation" not in streamlit_app_docs


def test_capability_matrix_documents_supported_and_deferred_combinations() -> None:
    matrix = (PROJECT_ROOT / "docs" / "CAPABILITY_MATRIX.md").read_text(
        encoding="utf-8"
    )

    required_phrases = [
        "BO Forge v2.1.0",
        "supported",
        "read-only/reporting only",
        "rejected",
        "deferred",
        "Single-objective model profiles",
        "Non-default model profile + multi-objective",
        "Multi-objective + deterministic cost",
        "Structured + contextual",
        "Contextual + multi-objective",
        "Contextual + multi-fidelity",
        "Contextual + deterministic cost",
        "Contextual + replicates",
        "Structured + multi-fidelity",
        "Structured + cost",
        "Production API/auth/database",
        "CSV logs remain the source of truth",
    ]
    for phrase in required_phrases:
        assert phrase in matrix


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
        "Avoid simultaneous writes",
        "dedicated campaign working directory",
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
    ]
    for phrase in required_phrases:
        assert phrase in guide

    assert "v1.2.3" not in guide


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


def test_v2_roadmap_is_active_hardening_and_controlled_expansion_plan() -> None:
    roadmap = (PROJECT_ROOT / "ROADMAP_V2_X.md").read_text(encoding="utf-8")
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "Current baseline: `v2.1.0`" in roadmap
    assert "coherence and controlled expansion" in roadmap
    assert "docs/CAPABILITY_MATRIX.md" in roadmap
    assert "v2.0.x - Stable v2 Baseline" in roadmap
    assert "Status: completed" in roadmap
    assert "Status: active" in roadmap
    assert "v2.1.x - Model Profiles And Advanced Surrogates" in roadmap
    assert "bo-forge model-summary" in roadmap
    assert "plot --kind model-diagnostics" in roadmap
    assert "v2.2.x - Noisy And Pending-Aware BO" in roadmap
    assert "v2.3.x - Controlled Feature Combinations" in roadmap
    assert "v2.4.x - Multi-Fidelity Expansion" in roadmap
    assert "v2.5.x - App/API Operational Hardening" in roadmap
    assert "No mandatory database" in roadmap
    assert "No unrestricted feature cross-product" in roadmap
    assert "No raw low-level kernel API as the first modeling extension" in roadmap
    expected_roadmap_url = (
        'Roadmap = "https://github.com/angzeli/bo-forge/blob/main/ROADMAP_V2_X.md"'
    )
    assert expected_roadmap_url in pyproject


def test_streamlit_service_layer_is_documented_as_internal_non_http() -> None:
    repository_structure = (PROJECT_ROOT / "docs" / "REPOSITORY_STRUCTURE.md").read_text(
        encoding="utf-8"
    )
    streamlit_app_docs = (PROJECT_ROOT / "docs" / "STREAMLIT_APP.md").read_text(
        encoding="utf-8"
    )
    public_api = (PROJECT_ROOT / "docs" / "PUBLIC_API.md").read_text(encoding="utf-8")

    assert "bo_forge_app/service.py" in repository_structure
    assert "internal, non-HTTP app service layer" in repository_structure
    assert "not a stable public API" in repository_structure
    assert "internal non-HTTP service layer" in streamlit_app_docs
    assert "CampaignAppService" not in public_api
    assert "bo_forge_app.api" not in public_api


def test_release_checklist_includes_fresh_install_pip_check() -> None:
    checklist = (PROJECT_ROOT / "docs" / "RELEASE_CHECKLIST.md").read_text(encoding="utf-8")

    assert "/tmp/bo_forge_release_probe/bin/pip check" in checklist
    assert "/tmp/bo_forge_app_release_probe/bin/pip check" in checklist
    assert "/tmp/bo_forge_api_release_probe/bin/pip check" in checklist
    assert "/tmp/bo_forge_sdist_release_probe/bin/pip check" in checklist
    assert "fidelity-summary --config configs/15_multi_fidelity_qmfkg.yaml" in checklist
    assert "--kind fidelity-diagnostics" in checklist
    assert "context-summary --config configs/16_contextual_logei.yaml" in checklist
    assert "--kind context-diagnostics" in checklist
    assert "model-summary --config configs/17_model_profile_logei.yaml" in checklist
    assert "--kind model-diagnostics" in checklist


def test_requirements_lock_matches_current_release_snapshot() -> None:
    requirements_lock = (PROJECT_ROOT / "requirements-lock.txt").read_text(
        encoding="utf-8"
    )

    assert "BO Forge v2.1.0" in requirements_lock
    assert "v1.4.0 release" not in requirements_lock


def test_installation_tutorial_covers_pip_install_paths() -> None:
    tutorial = (PROJECT_ROOT / "docs" / "INSTALLATION.md").read_text(encoding="utf-8")

    assert "pip install bo-forge" in tutorial
    assert 'pip install "bo-forge[app]"' in tutorial
    assert 'pip install "bo-forge[api]"' in tutorial
    assert 'pip install -e ".[dev]"' in tutorial
    assert "dist/bo_forge-2.1.0-py3-none-any.whl" in tutorial
    assert "dist/bo_forge-2.1.0.tar.gz" in tutorial
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
    assert "In v1.5.0, `stages:` cannot be combined with `cost:`." in normalized_csv_schema
    assert "source,[stage],review_status" not in csv_schema


def test_app_created_campaign_tutorial_uses_current_streamlit_labels() -> None:
    tutorial = (PROJECT_ROOT / "docs" / "09_APP_CREATED_CAMPAIGN_TUTORIAL.md").read_text(
        encoding="utf-8"
    )

    assert "Campaign file action" in tutorial
    assert "Create Campaign" in tutorial
    assert "Contextual LogEI" in tutorial
    assert "Update YAML preview from form" in tutorial
    assert "Overview" in tutorial
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

    for content in (readme, cli_docs, quickstart):
        assert "15_multi_fidelity_qmfkg" in content
    assert "fidelity-summary" in cli_docs
    assert "fidelity-diagnostics" in cli_docs
    assert "campaign.fidelity_summary()" in quickstart
    assert "notebooks/15_multi_fidelity_qmfkg_campaign.ipynb" in quickstart
    assert "notebooks/15_multi_fidelity_qmfkg_campaign.ipynb" in readme
    assert "source=qmf_kg" in csv_schema
    assert "no new CSV columns" in csv_schema
    assert "fidelity cost is separate from BO Forge's `cost:`" in csv_schema
    assert "qMFKG model-based suggestions support batch_size=1" in common_errors


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
        assert "model-diagnostics" in content
    assert "ModelConfig" in public_api
    assert "model_summary" in public_api
    assert "model.profile" in csv_schema
    assert "does not add or remove CSV columns" in csv_schema
    assert "Non-default model profiles" in common_errors


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
    wheels = sorted(dist_dir.glob("bo_forge-2.1.0-*.whl"))
    sdists = sorted(dist_dir.glob("bo_forge-2.1.0.tar.gz"))
    assert wheels, "No v2.1.0 wheel was built."
    assert sdists, "No v2.1.0 sdist was built."

    _assert_wheel_package_boundaries(wheels[0])
    _assert_sdist_contains_release_assets(sdists[0])
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
    )
    _install_api_extra_and_probe(
        wheel=wheels[0],
        probe_root=tmp_path / "api_probe",
        env=env,
    )
    _install_distribution_and_probe(
        artifact=sdists[0],
        probe_root=tmp_path / "sdist_probe",
        env=env,
        install_args=["--no-build-isolation"],
    )


def _assert_wheel_package_boundaries(wheel_path: Path) -> None:
    with zipfile.ZipFile(wheel_path) as wheel:
        names = set(wheel.namelist())
        metadata = wheel.read("bo_forge-2.1.0.dist-info/METADATA").decode("utf-8")

    assert "bo_forge/__init__.py" in names
    assert "bo_forge/contextual.py" in names
    assert "bo_forge/multifidelity.py" in names
    assert "bo_forge/structured.py" in names
    assert "bo_forge_app/streamlit_app.py" in names
    assert "bo_forge_app/cli.py" in names
    assert "bo_forge_app/service.py" in names
    assert "bo_forge_app/api.py" in names
    assert "bo_forge_app/api_cli.py" in names
    assert "bo_forge_app/__main__.py" in names
    assert "bo_forge-2.1.0.dist-info/entry_points.txt" in names
    assert "bo_forge-2.1.0.dist-info/licenses/LICENSE" in names
    excluded_prefixes = ("docs/", "configs/", "examples/", "notebooks/", "tests/")
    assert not any(name.startswith(excluded_prefixes) for name in names)
    assert "Provides-Extra: app" in metadata
    assert "Provides-Extra: api" in metadata
    assert 'Requires-Dist: streamlit>=1.57; extra == "app"' in metadata
    assert 'Requires-Dist: fastapi>=0.115; extra == "api"' in metadata
    assert 'Requires-Dist: uvicorn>=0.30; extra == "api"' in metadata
    assert 'Requires-Dist: streamlit>=1.57\n' not in metadata
    assert 'Requires-Dist: fastapi>=0.115\n' not in metadata


def _assert_sdist_contains_release_assets(sdist_path: Path) -> None:
    with tarfile.open(sdist_path) as sdist:
        names = set(sdist.getnames())

    assert "bo_forge-2.1.0/README.md" in names
    assert "bo_forge-2.1.0/LICENSE" in names
    assert "bo_forge-2.1.0/requirements-lock.txt" in names
    assert "bo_forge-2.1.0/ROADMAP_V0_TO_V1.md" in names
    assert "bo_forge-2.1.0/ROADMAP_V1_X.md" in names
    assert "bo_forge-2.1.0/ROADMAP_V2_X.md" in names
    assert "bo_forge-2.1.0/docs/PUBLIC_API.md" in names
    assert "bo_forge-2.1.0/docs/STREAMLIT_DEPLOYMENT.md" in names
    assert "bo_forge-2.1.0/docs/API_PROBE.md" in names
    assert "bo_forge-2.1.0/docs/CAPABILITY_MATRIX.md" in names
    assert "bo_forge-2.1.0/examples/quickstart.py" in names
    assert "bo_forge-2.1.0/examples/01_simple_2d_maximise_logei_campaign_log.csv" in names
    assert (
        "bo_forge-2.1.0/examples/10_multi_objective_mixed_constrained_campaign_log.csv"
        in names
    )
    assert (
        "bo_forge-2.1.0/examples/11_four_objective_mixed_constrained_campaign_log.csv"
        in names
    )
    assert "bo_forge-2.1.0/examples/12_cost_aware_multi_objective_campaign_log.csv" in names
    assert "bo_forge-2.1.0/examples/13_structured_campaign_core_campaign_log.csv" in names
    assert "bo_forge-2.1.0/examples/14_structured_campaign_tutorial_campaign_log.csv" in names
    assert "bo_forge-2.1.0/examples/15_multi_fidelity_qmfkg_campaign_log.csv" in names
    assert "bo_forge-2.1.0/examples/16_contextual_logei_campaign_log.csv" in names
    assert "bo_forge-2.1.0/examples/17_model_profile_campaign_log.csv" in names
    assert "bo_forge-2.1.0/configs/10_multi_objective_mixed_constrained_qlogehvi.yaml" in names
    assert "bo_forge-2.1.0/configs/11_four_objective_mixed_constrained_qlogehvi.yaml" in names
    assert "bo_forge-2.1.0/configs/12_cost_aware_multi_objective_qlogehvi.yaml" in names
    assert "bo_forge-2.1.0/configs/13_structured_campaign_core.yaml" in names
    assert "bo_forge-2.1.0/configs/14_structured_campaign_tutorial.yaml" in names
    assert "bo_forge-2.1.0/configs/15_multi_fidelity_qmfkg.yaml" in names
    assert "bo_forge-2.1.0/configs/16_contextual_logei.yaml" in names
    assert "bo_forge-2.1.0/configs/17_model_profile_logei.yaml" in names
    assert "bo_forge-2.1.0/notebooks/01_maximisation_logei_campaign.ipynb" in names
    assert "bo_forge-2.1.0/notebooks/10_multi_objective_qlogehvi_campaign.ipynb" in names
    assert "bo_forge-2.1.0/notebooks/11_four_objective_qlogehvi_campaign.ipynb" in names
    assert "bo_forge-2.1.0/notebooks/12_cost_aware_multi_objective_qlogehvi_campaign.ipynb" in names
    assert "bo_forge-2.1.0/notebooks/14_structured_campaign_tutorial.ipynb" in names
    assert "bo_forge-2.1.0/notebooks/15_multi_fidelity_qmfkg_campaign.ipynb" in names
    assert "bo_forge-2.1.0/notebooks/16_contextual_logei_campaign.ipynb" in names
    assert not any("working_log" in name or "latest_suggestions" in name for name in names)


def _install_distribution_and_probe(
    *,
    artifact: Path,
    probe_root: Path,
    env: dict[str, str],
    install_args: list[str],
) -> None:
    venv_dir = probe_root / "venv"
    probe_dir = probe_root / "probe"
    probe_dir.mkdir(parents=True)

    subprocess.run(
        [sys.executable, "-m", "venv", "--system-site-packages", str(venv_dir)],
        env=env,
        check=True,
        text=True,
    )
    python = venv_dir / "bin" / "python"
    pip = venv_dir / "bin" / "pip"
    install_env = env
    if "--no-build-isolation" in install_args:
        install_env = dict(env)
        install_env["PYTHONPATH"] = sysconfig.get_paths()["purelib"]
    subprocess.run(
        [str(pip), "install", "--no-deps", *install_args, str(artifact)],
        cwd=probe_dir,
        env=install_env,
        check=True,
        text=True,
    )
    completed = subprocess.run(
        [str(venv_dir / "bin" / "bo-forge"), "--version"],
        cwd=probe_dir,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    assert completed.stdout == "bo-forge 2.1.0\n"
    subprocess.run(
        [str(venv_dir / "bin" / "bo-forge-api"), "--help"],
        cwd=probe_dir,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )

    source_root = str(PROJECT_ROOT.resolve())
    script = f"""
import builtins
from pathlib import Path
from importlib.metadata import entry_points
import bo_forge
import bo_forge_app

source_root = Path({source_root!r})
scripts = {{ep.name: ep.value for ep in entry_points(group="console_scripts")}}
assert scripts["bo-forge-app"] == "bo_forge_app.cli:main"
assert scripts["bo-forge-api"] == "bo_forge_app.api_cli:main"
for module in (bo_forge, bo_forge_app):
    module_path = Path(module.__file__).resolve()
    assert source_root not in module_path.parents, module_path
assert bo_forge.__version__ == "2.1.0"

real_import = builtins.__import__
def block_optional_app_deps(name, *args, **kwargs):
    if name == "streamlit" or name.startswith("streamlit."):
        raise AssertionError("doctor imported optional Streamlit dependencies")
    if name in {{"fastapi", "uvicorn"}} or name.startswith(("fastapi.", "uvicorn.")):
        raise AssertionError("doctor imported optional API dependencies")
    return real_import(name, *args, **kwargs)
builtins.__import__ = block_optional_app_deps
from bo_forge.cli import run
assert run(["doctor"]) == 0
"""
    subprocess.run(
        [str(python), "-c", script],
        cwd=probe_dir,
        env=env,
        check=True,
        text=True,
    )


def _install_app_extra_and_probe(
    *,
    wheel: Path,
    probe_root: Path,
    env: dict[str, str],
) -> None:
    venv_dir = probe_root / "venv"
    probe_dir = probe_root / "probe"
    probe_dir.mkdir(parents=True)
    subprocess.run(
        [sys.executable, "-m", "venv", "--system-site-packages", str(venv_dir)],
        env=env,
        check=True,
        text=True,
    )
    python = venv_dir / "bin" / "python"
    pip = venv_dir / "bin" / "pip"
    subprocess.run(
        [str(pip), "install", "--no-deps", f"{wheel}[app]"],
        cwd=probe_dir,
        env=env,
        check=True,
        text=True,
    )
    source_root = str(PROJECT_ROOT.resolve())
    script = f"""
from pathlib import Path
import streamlit
from bo_forge_app.cli import packaged_streamlit_app_path

source_root = Path({source_root!r})
app_path = packaged_streamlit_app_path()
assert app_path.name == "streamlit_app.py"
assert source_root not in app_path.resolve().parents, app_path
assert streamlit.__version__
"""
    subprocess.run(
        [str(python), "-c", script],
        cwd=probe_dir,
        env=env,
        check=True,
        text=True,
    )
    subprocess.run(
        [str(python), "-m", "bo_forge_app", "--help"],
        cwd=probe_dir,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    subprocess.run(
        [str(venv_dir / "bin" / "bo-forge-app"), "--help"],
        cwd=probe_dir,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )


def _install_api_extra_and_probe(
    *,
    wheel: Path,
    probe_root: Path,
    env: dict[str, str],
) -> None:
    venv_dir = probe_root / "venv"
    probe_dir = probe_root / "probe"
    probe_dir.mkdir(parents=True)
    subprocess.run(
        [sys.executable, "-m", "venv", "--system-site-packages", str(venv_dir)],
        env=env,
        check=True,
        text=True,
    )
    python = venv_dir / "bin" / "python"
    pip = venv_dir / "bin" / "pip"
    subprocess.run(
        [str(pip), "install", "--no-deps", f"{wheel}[api]"],
        cwd=probe_dir,
        env=env,
        check=True,
        text=True,
    )
    source_root = str(PROJECT_ROOT.resolve())
    probe_env = dict(env)
    probe_env["PYTHONPATH"] = sysconfig.get_paths()["purelib"]
    script = f"""
from pathlib import Path
import fastapi
import uvicorn
import bo_forge_app.api

source_root = Path({source_root!r})
api_path = Path(bo_forge_app.api.__file__).resolve()
assert source_root not in api_path.parents, api_path
assert fastapi.__version__
assert uvicorn.__version__
"""
    subprocess.run(
        [str(python), "-c", script],
        cwd=probe_dir,
        env=probe_env,
        check=True,
        text=True,
    )
    subprocess.run(
        [str(venv_dir / "bin" / "bo-forge-api"), "--help"],
        cwd=probe_dir,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )


def _install_core_only_app_missing_streamlit_probe(
    *,
    wheel: Path,
    probe_root: Path,
    env: dict[str, str],
) -> None:
    venv_dir = probe_root / "venv"
    probe_dir = probe_root / "probe"
    probe_dir.mkdir(parents=True)
    subprocess.run(
        [sys.executable, "-m", "venv", str(venv_dir)],
        env=env,
        check=True,
        text=True,
    )
    subprocess.run(
        [str(venv_dir / "bin" / "pip"), "install", "--no-deps", str(wheel)],
        cwd=probe_dir,
        env=env,
        check=True,
        text=True,
    )
    for command in [
        [str(venv_dir / "bin" / "bo-forge-app")],
        [str(venv_dir / "bin" / "python"), "-m", "bo_forge_app"],
    ]:
        completed = subprocess.run(
            command,
            cwd=probe_dir,
            env=env,
            check=False,
            text=True,
            capture_output=True,
        )

        assert completed.returncode == 1
        assert 'pip install "bo-forge[app]"' in completed.stderr
        assert "Traceback" not in completed.stderr
    completed = subprocess.run(
        [str(venv_dir / "bin" / "bo-forge-api")],
        cwd=probe_dir,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 1
    assert 'pip install "bo-forge[api]"' in completed.stderr
    assert "Traceback" not in completed.stderr
