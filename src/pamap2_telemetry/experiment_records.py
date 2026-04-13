from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd


def _safe_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def write_selected_model_records(repo_root: Path) -> list[Path]:
    """Write lightweight experiment records for selected regression and classification models."""
    metrics_dir = repo_root / "artifacts" / "metrics"
    records_dirs = [
        repo_root / "artifacts" / "models" / "metadata",
        repo_root / "docs" / "model_records",
    ]
    for records_dir in records_dirs:
        records_dir.mkdir(parents=True, exist_ok=True)

    selected_df = pd.read_csv(metrics_dir / "grouped_cv_selected_model_summary.csv")
    preferred_setup_df = pd.read_csv(metrics_dir / "grouped_cv_preferred_setup_summary.csv")
    conformal_df = pd.read_csv(metrics_dir / "grouped_cv_conformal_summary.csv")

    preferred_setup = preferred_setup_df.iloc[0].to_dict()
    if "preferred_interval_variant" in conformal_df.columns:
        preferred_rows = conformal_df[conformal_df["preferred_interval_variant"] == 1]
        conformal_row = (
            preferred_rows.iloc[0].to_dict() if not preferred_rows.empty else conformal_df.iloc[0].to_dict()
        )
    else:
        conformal_row = conformal_df.iloc[0].to_dict()

    output_paths: list[Path] = []

    for _, row in selected_df.iterrows():
        task = str(row["task"])
        model_name = str(row["selected_model"])

        if task == "regression":
            metrics_summary = {
                "selected_mean_mae": _safe_float(row.get("selected_mean_mae")),
                "selected_std_mae": _safe_float(row.get("selected_std_mae")),
                "runner_up_mean_mae": _safe_float(row.get("runner_up_mean_mae")),
                "winner_margin": _safe_float(row.get("winner_margin")),
            }
            target_definition = str(preferred_setup.get("preferred_target_col", "hr_target_next30s_mean"))
        else:
            metrics_summary = {
                "selected_mean_macro_f1": _safe_float(row.get("selected_mean_macro_f1")),
                "selected_std_macro_f1": _safe_float(row.get("selected_std_macro_f1")),
                "runner_up_mean_macro_f1": _safe_float(row.get("runner_up_mean_macro_f1")),
                "winner_margin": _safe_float(row.get("winner_margin")),
            }
            target_definition = "activity_target"

        record = {
            "record_version": "v1",
            "record_date": str(date.today()),
            "task": task,
            "model_name": model_name,
            "feature_set": str(preferred_setup.get("preferred_feature_set", "upgraded")),
            "target_definition": target_definition,
            "fill_strategy": str(preferred_setup.get("preferred_fill_strategy", "current_ffill")),
            "split_strategy": "leave-one-subject-out grouped CV",
            "selection_rule": str(row.get("selection_rule", "")),
            "hyperparameters": "See model object saved in artifacts/models and grouped_cv_* summaries.",
            "metrics_summary": metrics_summary,
            "uncertainty_notes": {
                "method": "split conformal",
                "preferred_interval_variant": str(conformal_row.get("calibration_variant", "global")),
                "row_level_empirical_coverage": _safe_float(conformal_row.get("row_level_empirical_coverage")),
                "row_level_average_interval_width": _safe_float(
                    conformal_row.get("row_level_average_interval_width")
                ),
            },
            "notes": [
                "Designed to stay lean and interview-defensible.",
                "All temporal operations are subject-local and sorted by timestamp.",
                "Preprocessing and model are packaged in sklearn Pipeline objects for reproducible inference.",
            ],
        }

        for records_dir in records_dirs:
            output_path = records_dir / f"{task}_selected_model_record.json"
            output_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            output_paths.append(output_path)

    return output_paths


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    written_paths = write_selected_model_records(repo_root=repo_root)
    print("Wrote experiment records:")
    for path in written_paths:
        print(path)
