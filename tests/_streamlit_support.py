import shutil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from matplotlib import pyplot as plt

import bo_forge.suggestions as suggestions_module
from bo_forge.config import BOConfig, CampaignConfig, CostConfig, ObjectiveConfig, VariableConfig
from bo_forge.errors import LogBusyError, LogConflictError
from bo_forge.io import empty_campaign_log
from bo_forge.session import CampaignSession
from bo_forge.validation import canonical_columns
from bo_forge_app import streamlit_app, streamlit_helpers, streamlit_style
from bo_forge_app.streamlit_helpers import (
    active_variables_display,
    append_disabled_reason,
    available_plot_kinds,
    build_campaign_yaml_text,
    campaign_report_text,
    compact_dataframe,
    create_campaign_files,
    dataframe_fingerprint,
    default_export_path,
    default_new_campaign_paths,
    drop_all_blank_columns,
    empty_state_message,
    export_staged_suggestions_csv,
    extract_matplotlib_figure,
    feature_flags,
    file_fingerprint,
    format_dataframe_for_display,
    format_number_for_display,
    humanize_campaign_status,
    humanize_next_action,
    load_campaign_session,
    make_staged_suggestion_bundle,
    observable_row_options,
    observable_rows,
    parse_campaign_config_text,
    parse_categorical_values_text,
    parse_discrete_values_text,
    resolve_path_input,
    select_display_columns,
    staged_bundle_invalidation_reason,
    staged_bundle_is_appendable,
    staged_suggestions_from_bundle,
    status_tone,
    structured_stage_config_table,
    structured_stage_options,
)
from bo_forge_app.streamlit_style import FORGE_SUITE_CSS, forge_action_label, forge_status_label

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def copy_example_log(tmp_path: Path, name: str) -> Path:
    source = Path("examples") / name
    destination = tmp_path / name
    shutil.copyfile(source, destination)
    return destination


def simple_suggestions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "row_id": "suggested_1",
                "iteration": 1,
                "status": "suggested",
                "source": "sobol",
                "x": 0.5,
                "y": "",
                "predicted_mean": "",
                "predicted_std": "",
                "acquisition": "",
            }
        ]
    )

__all__ = [
    'BOConfig',
    'CampaignConfig',
    'CampaignSession',
    'CostConfig',
    'FORGE_SUITE_CSS',
    'LogBusyError',
    'LogConflictError',
    'ObjectiveConfig',
    'Path',
    'PROJECT_ROOT',
    'SimpleNamespace',
    'VariableConfig',
    'active_variables_display',
    'append_disabled_reason',
    'available_plot_kinds',
    'build_campaign_yaml_text',
    'campaign_report_text',
    'canonical_columns',
    'compact_dataframe',
    'copy_example_log',
    'create_campaign_files',
    'dataframe_fingerprint',
    'default_export_path',
    'default_new_campaign_paths',
    'drop_all_blank_columns',
    'empty_campaign_log',
    'empty_state_message',
    'export_staged_suggestions_csv',
    'extract_matplotlib_figure',
    'feature_flags',
    'file_fingerprint',
    'forge_action_label',
    'forge_status_label',
    'format_dataframe_for_display',
    'format_number_for_display',
    'humanize_campaign_status',
    'humanize_next_action',
    'load_campaign_session',
    'make_staged_suggestion_bundle',
    'observable_row_options',
    'observable_rows',
    'parse_campaign_config_text',
    'parse_categorical_values_text',
    'parse_discrete_values_text',
    'pd',
    'plt',
    'pytest',
    'resolve_path_input',
    'select_display_columns',
    'shutil',
    'simple_suggestions',
    'staged_bundle_invalidation_reason',
    'staged_bundle_is_appendable',
    'staged_suggestions_from_bundle',
    'status_tone',
    'streamlit_app',
    'streamlit_helpers',
    'streamlit_style',
    'structured_stage_config_table',
    'structured_stage_options',
    'suggestions_module',
    'torch',
]
