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


def _require_csv(path: Path, description: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {description}: {path}. "
            "Run scripts/compact_ablation_study.py and scripts/grouped_evaluation.py before "
            "scripts/write_experiment_records.py."
        )
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{description} is empty: {path}")
    return df


def _select_preferred_conformal_row(conformal_df: pd.DataFrame) -> dict[str, Any]:
    if "preferred_interval_variant" in conformal_df.columns:
        preferred_rows = conformal_df[conformal_df["preferred_interval_variant"] == 1]
        if not preferred_rows.empty:
            return preferred_rows.iloc[0].to_dict()
    return conformal_df.iloc[0].to_dict()


def _select_abstention_reference_row(abstention_df: pd.DataFrame) -> dict[str, Any]:
    if "confidence_threshold" not in abstention_df.columns:
        return abstention_df.iloc[0].to_dict()

    thresholds = pd.to_numeric(abstention_df["confidence_threshold"], errors="coerce")
    rows_with_threshold = abstention_df.assign(confidence_threshold_numeric=thresholds).dropna(
        subset=["confidence_threshold_numeric"]
    )
    if rows_with_threshold.empty:
        return abstention_df.iloc[0].to_dict()

    target_threshold = 0.80
    exact_match = rows_with_threshold[rows_with_threshold["confidence_threshold_numeric"] == target_threshold]
    if not exact_match.empty:
        return exact_match.iloc[0].drop(labels=["confidence_threshold_numeric"]).to_dict()

    nearest_idx = (rows_with_threshold["confidence_threshold_numeric"] - target_threshold).abs().idxmin()
    return rows_with_threshold.loc[nearest_idx].drop(labels=["confidence_threshold_numeric"]).to_dict()


def _build_abstention_note(abstention_row: dict[str, Any]) -> str:
    threshold = _safe_float(abstention_row.get("confidence_threshold"))
    retained_fraction = _safe_float(abstention_row.get("retained_fraction"))
    retained_accuracy = _safe_float(abstention_row.get("retained_accuracy"))
    error_capture_rate = _safe_float(abstention_row.get("error_capture_rate"))

    if (
        threshold is None
        or retained_fraction is None
        or retained_accuracy is None
        or error_capture_rate is None
    ):
        return "Abstention diagnostics were generated, but summary fields were incomplete in this run."

    return (
        f"At confidence threshold {threshold:.2f}, the model retained {retained_fraction:.1%} of rows "
        f"with retained accuracy {retained_accuracy:.3f}, and abstained rows captured "
        f"{error_capture_rate:.1%} of total classification errors."
    )


def write_selected_model_records(repo_root: Path) -> list[Path]:
    """Write lightweight experiment records for selected regression and classification models."""
    metrics_dir = repo_root / "artifacts" / "metrics"
    records_dirs = [
        repo_root / "artifacts" / "models" / "metadata",
        repo_root / "docs" / "model_records",
    ]
    for records_dir in records_dirs:
        records_dir.mkdir(parents=True, exist_ok=True)

    selected_df = _require_csv(
        metrics_dir / "grouped_cv_selected_model_summary.csv",
        "selected model summary",
    )
    preferred_setup_df = _require_csv(
        metrics_dir / "grouped_cv_preferred_setup_summary.csv",
        "preferred setup summary",
    )
    conformal_df = _require_csv(
        metrics_dir / "grouped_cv_conformal_summary.csv",
        "conformal summary",
    )
    regression_summary_df = _require_csv(
        metrics_dir / "grouped_cv_regression_summary.csv",
        "regression grouped summary",
    )
    classification_summary_df = _require_csv(
        metrics_dir / "grouped_cv_classification_summary.csv",
        "classification grouped summary",
    )
    classification_calibration_df = _require_csv(
        metrics_dir / "grouped_cv_classification_calibration_summary.csv",
        "classification calibration summary",
    )
    classification_abstention_df = _require_csv(
        metrics_dir / "grouped_cv_classification_abstention_summary.csv",
        "classification abstention summary",
    )

    preferred_setup = preferred_setup_df.iloc[0].to_dict()
    preferred_target_col = str(preferred_setup.get("preferred_target_col", "")).strip()
    if not preferred_target_col:
        raise ValueError("preferred_target_col is missing from grouped_cv_preferred_setup_summary.csv")

    if "regression_target_col" in selected_df.columns:
        selected_targets = selected_df["regression_target_col"].dropna().astype(str).unique().tolist()
        mismatched_targets = [target for target in selected_targets if target != preferred_target_col]
        if mismatched_targets:
            raise ValueError(
                "Selected model summary target does not match preferred setup target. "
                f"preferred_target_col={preferred_target_col}, summary_targets={mismatched_targets}. "
                "Rerun scripts/grouped_evaluation.py."
            )

    conformal_row = _select_preferred_conformal_row(conformal_df)
    calibration_row = classification_calibration_df.iloc[0].to_dict()
    abstention_reference_row = _select_abstention_reference_row(classification_abstention_df)
    abstention_note = _build_abstention_note(abstention_reference_row)

    output_paths: list[Path] = []

    for _, row in selected_df.iterrows():
        task = str(row["task"])
        model_name = str(row["selected_model"])

        base_record = {
            "record_version": "v1",
            "record_date": str(date.today()),
            "task": task,
            "model_name": model_name,
            "feature_set": str(preferred_setup.get("preferred_feature_set", "upgraded")),
            "fill_strategy": str(preferred_setup.get("preferred_fill_strategy", "current_ffill")),
            "split_strategy": "leave-one-subject-out grouped CV",
            "selection_rule": str(row.get("selection_rule", "")),
            "hyperparameters": "See model object saved in artifacts/models and grouped_cv_* summaries.",
            "notes": [
                "Designed to stay lean and interview-defensible.",
                "All temporal operations are subject-local and sorted by timestamp.",
                "Preprocessing and model are packaged in sklearn Pipeline objects for reproducible inference.",
            ],
        }

        if task == "regression":
            regression_model_rows = regression_summary_df[regression_summary_df["model"] == model_name]
            if regression_model_rows.empty:
                raise ValueError(
                    "Selected regression model was not found in grouped_cv_regression_summary.csv: "
                    f"{model_name}"
                )
            regression_summary_row = regression_model_rows.iloc[0]

            metrics_summary = {
                "selected_mean_mae": _safe_float(row.get("selected_mean_mae")),
                "selected_std_mae": _safe_float(row.get("selected_std_mae")),
                "selected_mean_rmse": _safe_float(regression_summary_row.get("mean_rmse")),
                "selected_mean_r2": _safe_float(regression_summary_row.get("mean_r2")),
                "runner_up_mean_mae": _safe_float(row.get("runner_up_mean_mae")),
                "winner_margin": _safe_float(row.get("winner_margin")),
            }
            record = {
                **base_record,
                "target_definition": preferred_target_col,
                "metrics_summary": metrics_summary,
                "uncertainty_notes": {
                    "method": "split conformal",
                    "preferred_interval_variant": str(conformal_row.get("calibration_variant", "global")),
                    "selection_reason": str(conformal_row.get("selection_reason", "")),
                    "row_level_empirical_coverage": _safe_float(conformal_row.get("row_level_empirical_coverage")),
                    "row_level_average_interval_width": _safe_float(
                        conformal_row.get("row_level_average_interval_width")
                    ),
                },
            }
        elif task == "classification":
            classification_model_rows = classification_summary_df[classification_summary_df["model"] == model_name]
            if classification_model_rows.empty:
                raise ValueError(
                    "Selected classification model was not found in grouped_cv_classification_summary.csv: "
                    f"{model_name}"
                )
            classification_summary_row = classification_model_rows.iloc[0]

            metrics_summary = {
                "selected_mean_accuracy": _safe_float(classification_summary_row.get("mean_accuracy")),
                "selected_std_accuracy": _safe_float(classification_summary_row.get("std_accuracy")),
                "selected_mean_macro_f1": _safe_float(row.get("selected_mean_macro_f1")),
                "selected_std_macro_f1": _safe_float(row.get("selected_std_macro_f1")),
                "runner_up_mean_macro_f1": _safe_float(row.get("runner_up_mean_macro_f1")),
                "winner_margin": _safe_float(row.get("winner_margin")),
            }
            record = {
                **base_record,
                "target_definition": "activity_target",
                "metrics_summary": metrics_summary,
                "confidence_diagnostics": {
                    "accuracy": _safe_float(calibration_row.get("accuracy")),
                    "macro_f1": _safe_float(calibration_row.get("macro_f1")),
                    "ece_10": _safe_float(calibration_row.get("ece_10")),
                    "multiclass_brier_score": _safe_float(calibration_row.get("multiclass_brier_score")),
                    "mean_confidence": _safe_float(calibration_row.get("mean_confidence")),
                    "overconfidence_gap": _safe_float(calibration_row.get("overconfidence_gap")),
                },
                "confidence_notes": [
                    "Classification confidence is summarized with ECE and multiclass Brier score.",
                    abstention_note,
                ],
            }
        else:
            raise ValueError(f"Unsupported task in selected summary: {task}")

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
