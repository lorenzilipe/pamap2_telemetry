from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .evaluate import (
    ALPHA,
    RANDOM_SEED,
    build_regression_model_specs,
    run_grouped_evaluation,
    run_grouped_regression_cv,
)


RAW_REQUIRED_COLUMNS = [
    "timestamp_s",
    "activity_id",
    "heart_rate_bpm",
    "hand_acc_16g_x",
    "hand_acc_16g_y",
    "hand_acc_16g_z",
    "hand_gyro_x",
    "hand_gyro_y",
    "hand_gyro_z",
    "chest_acc_16g_x",
    "chest_acc_16g_y",
    "chest_acc_16g_z",
    "chest_gyro_x",
    "chest_gyro_y",
    "chest_gyro_z",
    "ankle_acc_16g_x",
    "ankle_acc_16g_y",
    "ankle_acc_16g_z",
    "ankle_gyro_x",
    "ankle_gyro_y",
    "ankle_gyro_z",
]

BASE_SIGNAL_COLUMNS = [
    "heart_rate_bpm",
    "hand_acc_16g_mag",
    "chest_acc_16g_mag",
    "ankle_acc_16g_mag",
    "hand_gyro_mag",
    "chest_gyro_mag",
    "ankle_gyro_mag",
]

ACC_MAG_COLUMNS = ["hand_acc_16g_mag", "chest_acc_16g_mag", "ankle_acc_16g_mag"]
GYRO_MAG_COLUMNS = ["hand_gyro_mag", "chest_gyro_mag", "ankle_gyro_mag"]

FILL_STRATEGY_ORDER = [
    "current_ffill",
    "limited_ffill_5s",
    "strict_observed_only",
]

FILL_STRATEGY_LABELS = {
    "current_ffill": "current_ffill_unbounded",
    "limited_ffill_5s": "limited_ffill_5s",
    "strict_observed_only": "strict_observed_only",
}

TARGET_VARIANTS = {
    "direct_30s": "hr_target_30s",
    "mean_next_30s": "hr_target_next30s_mean",
    "direct_15s": "hr_target_15s",
}


def _default_paths(repo_root: Path) -> dict[str, Path]:
    raw_root = repo_root / "data" / "raw" / "pamap2+physical+activity+monitoring"
    return {
        "repo_root": repo_root,
        "raw_root": raw_root,
        "raw_dataset_root": raw_root / "PAMAP2_Dataset",
        "protocol_dir": raw_root / "PAMAP2_Dataset" / "Protocol",
        "metadata_dir": raw_root / "metadata",
        "metrics_dir": repo_root / "artifacts" / "metrics",
        "figures_dir": repo_root / "artifacts" / "figures",
        "models_dir": repo_root / "artifacts" / "models",
        "processed_regression_path": repo_root / "data" / "processed" / "pamap2_model_table_regression.parquet",
        "processed_classification_path": repo_root
        / "data"
        / "processed"
        / "pamap2_model_table_classification.parquet",
    }


def _extract_subject_id(file_path: Path) -> int:
    return int(file_path.stem.replace("subject", ""))


def _load_activity_metadata(metadata_dir: Path, metrics_dir: Path) -> tuple[dict[int, str], list[int]]:
    labels_path = metadata_dir / "activity_labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing activity labels metadata file: {labels_path}")

    labels_df = pd.read_csv(labels_path)
    labels_df["activity_id"] = pd.to_numeric(labels_df["activity_id"], errors="raise").astype(int)
    labels_df["activity_label"] = labels_df["activity_label"].astype(str)
    activity_map = dict(zip(labels_df["activity_id"], labels_df["activity_label"]))

    kept_path = metrics_dir / "phase1_protocol_kept_activity_summary.csv"
    if kept_path.exists():
        kept_df = pd.read_csv(kept_path)
        kept_activity_ids = sorted(pd.to_numeric(kept_df["activity_id"], errors="raise").astype(int).unique().tolist())
    else:
        # Fallback stays aligned with the documented v1 activity scope.
        kept_activity_ids = [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17]

    return activity_map, kept_activity_ids


def _load_schema_index_map(metadata_dir: Path) -> dict[str, int]:
    schema_path = metadata_dir / "schema_columns.csv"
    if not schema_path.exists():
        raise FileNotFoundError(f"Missing schema metadata file: {schema_path}")

    schema_df = pd.read_csv(schema_path)
    schema_pairs = schema_df[["column_name", "column_index"]].to_records(index=False)
    return {str(column_name): int(column_index) - 1 for column_name, column_index in schema_pairs}


def _load_protocol_per_second_prefill(paths: dict[str, Path]) -> pd.DataFrame:
    protocol_dir = paths["protocol_dir"]
    metadata_dir = paths["metadata_dir"]
    metrics_dir = paths["metrics_dir"]

    if not protocol_dir.exists():
        raise FileNotFoundError(f"Missing protocol directory: {protocol_dir}")

    activity_map, kept_activity_ids = _load_activity_metadata(metadata_dir, metrics_dir)
    column_index_map = _load_schema_index_map(metadata_dir)

    missing_required = [column for column in RAW_REQUIRED_COLUMNS if column not in column_index_map]
    if missing_required:
        raise ValueError(f"Missing required raw columns in schema metadata: {missing_required}")

    usecols = sorted(column_index_map[name] for name in RAW_REQUIRED_COLUMNS)
    rename_map = {column_index_map[name]: name for name in RAW_REQUIRED_COLUMNS}

    protocol_files = sorted(protocol_dir.glob("subject*.dat"))
    if not protocol_files:
        raise FileNotFoundError(f"No protocol files found under: {protocol_dir}")

    per_second_frames: list[pd.DataFrame] = []

    for protocol_path in protocol_files:
        subject_id = _extract_subject_id(protocol_path)
        subject_df = pd.read_csv(
            protocol_path,
            sep=r"\s+",
            header=None,
            usecols=usecols,
            na_values="NaN",
        ).rename(columns=rename_map)

        subject_df = subject_df.sort_values("timestamp_s").reset_index(drop=True)

        for location in ["hand", "chest", "ankle"]:
            subject_df[f"{location}_acc_16g_mag"] = np.sqrt(
                (subject_df[f"{location}_acc_16g_x"] ** 2)
                + (subject_df[f"{location}_acc_16g_y"] ** 2)
                + (subject_df[f"{location}_acc_16g_z"] ** 2)
            )
            subject_df[f"{location}_gyro_mag"] = np.sqrt(
                (subject_df[f"{location}_gyro_x"] ** 2)
                + (subject_df[f"{location}_gyro_y"] ** 2)
                + (subject_df[f"{location}_gyro_z"] ** 2)
            )

        # Keep only two compact raw-axis summaries as optional upgraded signals.
        subject_df["hand_acc_axis_absmean"] = (
            subject_df[["hand_acc_16g_x", "hand_acc_16g_y", "hand_acc_16g_z"]].abs().mean(axis=1)
        )
        subject_df["chest_acc_axis_absmean"] = (
            subject_df[["chest_acc_16g_x", "chest_acc_16g_y", "chest_acc_16g_z"]].abs().mean(axis=1)
        )

        subject_df["timestamp_s_bin"] = np.floor(subject_df["timestamp_s"]).astype("int64")

        per_second_df = (
            subject_df.groupby("timestamp_s_bin", as_index=False)
            .agg(
                activity_id=("activity_id", "last"),
                heart_rate_bpm=("heart_rate_bpm", "mean"),
                hand_acc_16g_mag=("hand_acc_16g_mag", "mean"),
                chest_acc_16g_mag=("chest_acc_16g_mag", "mean"),
                ankle_acc_16g_mag=("ankle_acc_16g_mag", "mean"),
                hand_gyro_mag=("hand_gyro_mag", "mean"),
                chest_gyro_mag=("chest_gyro_mag", "mean"),
                ankle_gyro_mag=("ankle_gyro_mag", "mean"),
                hand_acc_axis_absmean=("hand_acc_axis_absmean", "mean"),
                chest_acc_axis_absmean=("chest_acc_axis_absmean", "mean"),
            )
            .rename(columns={"timestamp_s_bin": "timestamp_s"})
        )

        per_second_df["subject_id"] = subject_id
        per_second_df["session"] = "protocol"
        per_second_frames.append(per_second_df)

    protocol_per_second_df = pd.concat(per_second_frames, ignore_index=True)
    protocol_per_second_df = protocol_per_second_df.sort_values(["subject_id", "timestamp_s"]).reset_index(drop=True)

    protocol_per_second_df["activity_id"] = pd.to_numeric(protocol_per_second_df["activity_id"], errors="coerce")
    protocol_per_second_df = protocol_per_second_df[protocol_per_second_df["activity_id"].isin(kept_activity_ids)].copy()
    protocol_per_second_df["activity_id"] = protocol_per_second_df["activity_id"].astype(int)
    protocol_per_second_df["activity_label"] = protocol_per_second_df["activity_id"].map(activity_map)

    protocol_per_second_df["heart_rate_observed_flag"] = protocol_per_second_df["heart_rate_bpm"].notna().astype(int)

    ordered_columns = [
        "subject_id",
        "session",
        "timestamp_s",
        "activity_id",
        "activity_label",
        "heart_rate_bpm",
        "heart_rate_observed_flag",
        "hand_acc_16g_mag",
        "chest_acc_16g_mag",
        "ankle_acc_16g_mag",
        "hand_gyro_mag",
        "chest_gyro_mag",
        "ankle_gyro_mag",
        "hand_acc_axis_absmean",
        "chest_acc_axis_absmean",
    ]
    return protocol_per_second_df[ordered_columns].reset_index(drop=True)


def _apply_fill_strategy(prefill_df: pd.DataFrame, fill_strategy: str) -> pd.DataFrame:
    if fill_strategy not in FILL_STRATEGY_ORDER:
        raise ValueError(f"Unsupported fill strategy: {fill_strategy}")

    filled_df = prefill_df.copy()

    if fill_strategy == "current_ffill":
        filled_df["heart_rate_bpm"] = (
            filled_df.groupby("subject_id", sort=False)["heart_rate_bpm"].transform(lambda s: s.ffill())
        )
    elif fill_strategy == "limited_ffill_5s":
        filled_df["heart_rate_bpm"] = (
            filled_df.groupby("subject_id", sort=False)["heart_rate_bpm"].transform(lambda s: s.ffill(limit=5))
        )
    elif fill_strategy == "strict_observed_only":
        # Keep the observed 1-second HR values only.
        pass

    filled_df["heart_rate_fill_strategy"] = fill_strategy
    return filled_df


def _future_window_mean(series: pd.Series, horizon: int) -> pd.Series:
    shifted = series.shift(-1)
    return shifted.iloc[::-1].rolling(window=horizon, min_periods=horizon).mean().iloc[::-1]


def _build_features_and_targets(per_second_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    model_df = per_second_df.sort_values(["subject_id", "timestamp_s"]).reset_index(drop=True).copy()

    baseline_derived_cols: list[str] = []

    for col in BASE_SIGNAL_COLUMNS:
        grouped = model_df.groupby("subject_id", sort=False)[col]
        model_df[f"{col}_lag_1"] = grouped.shift(1)
        model_df[f"{col}_lag_5"] = grouped.shift(5)

        rollmean_5 = grouped.transform(lambda s: s.rolling(window=5, min_periods=5).mean())
        rollstd_5 = grouped.transform(lambda s: s.rolling(window=5, min_periods=5).std())
        rollmean_10 = grouped.transform(lambda s: s.rolling(window=10, min_periods=10).mean())
        rollstd_10 = grouped.transform(lambda s: s.rolling(window=10, min_periods=10).std())

        model_df[f"{col}_rollmean_5"] = rollmean_5
        model_df[f"{col}_rollstd_5"] = rollstd_5
        model_df[f"{col}_rollmean_10"] = rollmean_10
        model_df[f"{col}_rollstd_10"] = rollstd_10
        model_df[f"{col}_delta_from_rollmean_5"] = model_df[col] - rollmean_5

        baseline_derived_cols.extend(
            [
                f"{col}_lag_1",
                f"{col}_lag_5",
                f"{col}_rollmean_5",
                f"{col}_rollstd_5",
                f"{col}_rollmean_10",
                f"{col}_rollstd_10",
                f"{col}_delta_from_rollmean_5",
            ]
        )

    grouped_hr = model_df.groupby("subject_id", sort=False)["heart_rate_bpm"]
    model_df["heart_rate_bpm_rollmin_10"] = grouped_hr.transform(lambda s: s.rolling(window=10, min_periods=10).min())
    model_df["heart_rate_bpm_rollmax_10"] = grouped_hr.transform(lambda s: s.rolling(window=10, min_periods=10).max())
    model_df["heart_rate_bpm_rollmedian_10"] = grouped_hr.transform(lambda s: s.rolling(window=10, min_periods=10).median())
    model_df["heart_rate_bpm_rollq25_10"] = grouped_hr.transform(
        lambda s: s.rolling(window=10, min_periods=10).quantile(0.25)
    )
    model_df["heart_rate_bpm_rollq75_10"] = grouped_hr.transform(
        lambda s: s.rolling(window=10, min_periods=10).quantile(0.75)
    )
    model_df["heart_rate_bpm_recent_change_5"] = model_df["heart_rate_bpm"] - grouped_hr.shift(5)
    model_df["heart_rate_bpm_rollmean_60"] = grouped_hr.transform(lambda s: s.rolling(window=60, min_periods=60).mean())
    model_df["heart_rate_bpm_vs_rollmean_60"] = model_df["heart_rate_bpm"] - model_df["heart_rate_bpm_rollmean_60"]

    model_df["motion_intensity_mean"] = model_df[ACC_MAG_COLUMNS + GYRO_MAG_COLUMNS].mean(axis=1)
    grouped_motion = model_df.groupby("subject_id", sort=False)["motion_intensity_mean"]
    model_df["motion_abs_change_1"] = grouped_motion.diff().abs()
    model_df["motion_abs_change_5"] = (model_df["motion_intensity_mean"] - grouped_motion.shift(5)).abs()
    model_df["motion_rollstd_5"] = grouped_motion.transform(lambda s: s.rolling(window=5, min_periods=5).std())
    model_df["motion_rollstd_20"] = grouped_motion.transform(lambda s: s.rolling(window=20, min_periods=20).std())
    model_df["motion_variance_burst_ratio"] = model_df["motion_rollstd_5"] / (
        model_df["motion_rollstd_20"] + 1e-6
    )
    model_df["acc_location_dispersion"] = model_df[ACC_MAG_COLUMNS].std(axis=1)

    grouped_hand_axis = model_df.groupby("subject_id", sort=False)["hand_acc_axis_absmean"]
    grouped_chest_axis = model_df.groupby("subject_id", sort=False)["chest_acc_axis_absmean"]
    model_df["hand_acc_axis_absmean_rollmean_5"] = grouped_hand_axis.transform(
        lambda s: s.rolling(window=5, min_periods=5).mean()
    )
    model_df["chest_acc_axis_absmean_rollmean_5"] = grouped_chest_axis.transform(
        lambda s: s.rolling(window=5, min_periods=5).mean()
    )

    upgraded_extra_cols = [
        "heart_rate_bpm_rollmin_10",
        "heart_rate_bpm_rollmax_10",
        "heart_rate_bpm_rollmedian_10",
        "heart_rate_bpm_rollq25_10",
        "heart_rate_bpm_rollq75_10",
        "heart_rate_bpm_recent_change_5",
        "heart_rate_bpm_rollmean_60",
        "heart_rate_bpm_vs_rollmean_60",
        "motion_intensity_mean",
        "motion_abs_change_1",
        "motion_abs_change_5",
        "motion_rollstd_5",
        "motion_rollstd_20",
        "motion_variance_burst_ratio",
        "acc_location_dispersion",
        "hand_acc_axis_absmean",
        "chest_acc_axis_absmean",
        "hand_acc_axis_absmean_rollmean_5",
        "chest_acc_axis_absmean_rollmean_5",
    ]

    model_df["hr_target_30s"] = grouped_hr.shift(-30)
    model_df["hr_target_15s"] = grouped_hr.shift(-15)
    model_df["hr_target_next30s_mean"] = grouped_hr.transform(lambda s: _future_window_mean(s, horizon=30))
    model_df["activity_target"] = model_df["activity_id"].astype(int)

    baseline_feature_cols = [*BASE_SIGNAL_COLUMNS, *baseline_derived_cols]
    upgraded_feature_cols = [*baseline_feature_cols, *upgraded_extra_cols]

    return model_df, baseline_feature_cols, upgraded_feature_cols


def _evaluate_regression_setup(
    setup_id: str,
    ablation_group: str,
    model_df: pd.DataFrame,
    feature_cols: list[str],
    target_variant: str,
    target_col: str,
    fill_strategy: str,
    feature_set: str,
    random_seed: int,
    regression_specs: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    eval_df = model_df.dropna(subset=[*feature_cols, target_col]).reset_index(drop=True)
    if eval_df.empty:
        raise ValueError(f"No rows left after dropna for setup {setup_id}")

    regression_results = run_grouped_regression_cv(
        model_df=eval_df,
        feature_cols=feature_cols,
        target_col=target_col,
        random_seed=random_seed,
        model_specs=regression_specs,
    )

    summary_df = regression_results["summary"].copy()
    fold_df = regression_results["fold"].copy()

    best_row = (
        summary_df.sort_values(["mean_mae", "std_mae", "mean_rmse"], ascending=[True, True, True])
        .reset_index(drop=True)
        .iloc[0]
    )
    persistence_row = summary_df[summary_df["model"] == "persistence_current_hr"].iloc[0]

    output_row = {
        "setup_id": setup_id,
        "ablation_group": ablation_group,
        "fill_strategy": fill_strategy,
        "fill_strategy_label": FILL_STRATEGY_LABELS[fill_strategy],
        "feature_set": feature_set,
        "target_variant": target_variant,
        "target_col": target_col,
        "rows_used": int(len(eval_df)),
        "subjects_used": int(eval_df["subject_id"].nunique()),
        "feature_count": int(len(feature_cols)),
        "best_model": str(best_row["model"]),
        "best_mean_mae": float(best_row["mean_mae"]),
        "best_mean_rmse": float(best_row["mean_rmse"]),
        "best_mean_r2": float(best_row["mean_r2"]),
        "persistence_mean_mae": float(persistence_row["mean_mae"]),
        "mae_gain_vs_persistence": float(persistence_row["mean_mae"] - best_row["mean_mae"]),
    }

    fold_df["setup_id"] = setup_id
    fold_df["ablation_group"] = ablation_group
    fold_df["fill_strategy"] = fill_strategy
    fold_df["feature_set"] = feature_set
    fold_df["target_variant"] = target_variant
    fold_df["target_col"] = target_col

    summary_df["setup_id"] = setup_id
    summary_df["ablation_group"] = ablation_group
    summary_df["fill_strategy"] = fill_strategy
    summary_df["feature_set"] = feature_set
    summary_df["target_variant"] = target_variant
    summary_df["target_col"] = target_col

    return output_row, fold_df, summary_df


def _choose_preferred_setup(
    feature_ablation_df: pd.DataFrame,
    target_comparison_df: pd.DataFrame,
    fill_sensitivity_df: pd.DataFrame,
) -> tuple[dict[str, str], pd.DataFrame]:
    feature_baseline = feature_ablation_df[feature_ablation_df["feature_set"] == "baseline"].iloc[0]
    feature_upgraded = feature_ablation_df[feature_ablation_df["feature_set"] == "upgraded"].iloc[0]

    feature_improvement = float(feature_baseline["best_mean_mae"] - feature_upgraded["best_mean_mae"])
    if feature_improvement >= 0.10:
        preferred_feature_set = "upgraded"
        feature_reason = (
            f"Upgraded features improved mean MAE by {feature_improvement:.3f} bpm over baseline on the default setup."
        )
    else:
        preferred_feature_set = "baseline"
        feature_reason = (
            "Upgraded feature gains were too small to justify extra complexity; baseline remained the preferred set."
        )

    direct_target_row = target_comparison_df[target_comparison_df["target_variant"] == "direct_30s"].iloc[0]
    best_target_row = target_comparison_df.sort_values("best_mean_mae", ascending=True).iloc[0]
    target_gap_vs_best = float(direct_target_row["best_mean_mae"] - best_target_row["best_mean_mae"])

    if best_target_row["target_variant"] != "direct_30s" and target_gap_vs_best >= 0.75:
        preferred_target_variant = str(best_target_row["target_variant"])
        preferred_target_col = str(best_target_row["target_col"])
        target_reason = (
            "Alternative target materially outperformed direct 30s."
            f" MAE improvement={target_gap_vs_best:.3f} bpm."
        )
    else:
        preferred_target_variant = "direct_30s"
        preferred_target_col = "hr_target_30s"
        target_reason = (
            "Direct 30s target remained preferred for task realism and interview defensibility"
            f" (gap to best alternative={target_gap_vs_best:.3f} bpm)."
        )

    current_fill_row = fill_sensitivity_df[fill_sensitivity_df["fill_strategy"] == "current_ffill"].iloc[0]
    best_fill_row = fill_sensitivity_df.sort_values("best_mean_mae", ascending=True).iloc[0]

    fill_gain = float(current_fill_row["best_mean_mae"] - best_fill_row["best_mean_mae"])
    row_retention_vs_current = float(best_fill_row["rows_used"] / current_fill_row["rows_used"])

    if (
        best_fill_row["fill_strategy"] != "current_ffill"
        and fill_gain >= 0.30
        and row_retention_vs_current >= 0.90
    ):
        preferred_fill_strategy = str(best_fill_row["fill_strategy"])
        fill_reason = (
            "Alternative fill strategy provided a meaningful MAE gain"
            f" ({fill_gain:.3f} bpm) while retaining {row_retention_vs_current:.1%} of rows."
        )
    else:
        preferred_fill_strategy = "current_ffill"
        fill_reason = (
            "Current fill strategy remained preferred because alternatives did not deliver"
            " enough MAE gain after row-retention tradeoffs."
        )

    preferred_setup = {
        "preferred_feature_set": preferred_feature_set,
        "preferred_target_variant": preferred_target_variant,
        "preferred_target_col": preferred_target_col,
        "preferred_fill_strategy": preferred_fill_strategy,
    }

    preference_summary_df = pd.DataFrame(
        [
            {
                **preferred_setup,
                "feature_reason": feature_reason,
                "target_reason": target_reason,
                "fill_reason": fill_reason,
            }
        ]
    )

    return preferred_setup, preference_summary_df


def _build_feature_inventory(feature_cols: list[str], preferred_feature_set: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature_name in feature_cols:
        if feature_name in BASE_SIGNAL_COLUMNS:
            feature_group = "core_signal"
            is_upgraded = False
        elif feature_name.endswith("_lag_1") or feature_name.endswith("_lag_5"):
            feature_group = "lag"
            is_upgraded = False
        elif "rollmean" in feature_name or "rollstd" in feature_name or "delta_from_rollmean" in feature_name:
            feature_group = "rolling_baseline"
            is_upgraded = False
        elif feature_name.startswith("heart_rate_bpm_rollmin") or feature_name.startswith("heart_rate_bpm_rollmax"):
            feature_group = "hr_extreme"
            is_upgraded = True
        elif "rollmedian" in feature_name or "rollq" in feature_name:
            feature_group = "hr_quantile"
            is_upgraded = True
        elif feature_name.startswith("heart_rate_bpm_recent_change") or feature_name.startswith("heart_rate_bpm_vs_rollmean"):
            feature_group = "hr_relative"
            is_upgraded = True
        elif feature_name.startswith("motion_"):
            feature_group = "transition_motion"
            is_upgraded = True
        elif feature_name.startswith("acc_location_dispersion"):
            feature_group = "cross_location_motion"
            is_upgraded = True
        elif feature_name.startswith("hand_acc_axis") or feature_name.startswith("chest_acc_axis"):
            feature_group = "raw_axis_summary"
            is_upgraded = True
        else:
            feature_group = "other"
            is_upgraded = preferred_feature_set == "upgraded"

        rows.append(
            {
                "feature_name": feature_name,
                "feature_group": feature_group,
                "is_upgraded_feature": int(is_upgraded),
                "preferred_feature_set": preferred_feature_set,
            }
        )

    return pd.DataFrame(rows)


def _save_ablation_plots(
    feature_ablation_df: pd.DataFrame,
    target_comparison_df: pd.DataFrame,
    fill_sensitivity_df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=feature_ablation_df, x="feature_set", y="best_mean_mae", color="tab:blue", ax=ax)
    ax.set_title("Feature ablation: mean MAE")
    ax.set_xlabel("feature set")
    ax.set_ylabel("best-model mean MAE")
    fig.tight_layout()
    fig.savefig(figures_dir / "grouped_cv_feature_ablation_mae.png", dpi=140)
    plt.close(fig)

    target_plot_df = target_comparison_df.copy()
    target_plot_df["target_label"] = target_plot_df["target_variant"].map(
        {
            "direct_30s": "direct t+30s",
            "mean_next_30s": "mean next 30s",
            "direct_15s": "direct t+15s",
        }
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=target_plot_df, x="target_label", y="best_mean_mae", color="tab:green", ax=ax)
    ax.set_title("Target comparison: mean MAE")
    ax.set_xlabel("target variant")
    ax.set_ylabel("best-model mean MAE")
    ax.tick_params(axis="x", rotation=10)
    fig.tight_layout()
    fig.savefig(figures_dir / "grouped_cv_target_comparison_mae.png", dpi=140)
    plt.close(fig)

    fill_plot_df = fill_sensitivity_df.copy()
    fill_plot_df["fill_label"] = fill_plot_df["fill_strategy"].map(FILL_STRATEGY_LABELS)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=fill_plot_df, x="fill_label", y="best_mean_mae", color="tab:orange", ax=ax)
    ax.set_title("Fill strategy sensitivity: mean MAE")
    ax.set_xlabel("fill strategy")
    ax.set_ylabel("best-model mean MAE")
    ax.tick_params(axis="x", rotation=10)
    fig.tight_layout()
    fig.savefig(figures_dir / "grouped_cv_fill_sensitivity_mae.png", dpi=140)
    plt.close(fig)


def run_compact_ablation_study(
    repo_root: Path,
    random_seed: int = RANDOM_SEED,
    alpha: float = ALPHA,
) -> dict[str, Any]:
    paths = _default_paths(repo_root)
    metrics_dir = paths["metrics_dir"]
    figures_dir = paths["figures_dir"]
    models_dir = paths["models_dir"]
    processed_regression_path = paths["processed_regression_path"]
    processed_classification_path = paths["processed_classification_path"]

    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    processed_regression_path.parent.mkdir(parents=True, exist_ok=True)

    prefill_df = _load_protocol_per_second_prefill(paths)

    fill_strategy_tables: dict[str, pd.DataFrame] = {}
    baseline_features: list[str] = []
    upgraded_features: list[str] = []

    for fill_strategy in FILL_STRATEGY_ORDER:
        filled_df = _apply_fill_strategy(prefill_df, fill_strategy)
        table_df, baseline_cols, upgraded_cols = _build_features_and_targets(filled_df)
        fill_strategy_tables[fill_strategy] = table_df

        if not baseline_features:
            baseline_features = baseline_cols
        if not upgraded_features:
            upgraded_features = upgraded_cols

    regression_specs = build_regression_model_specs(random_seed=random_seed)

    setup_rows: list[dict[str, Any]] = []
    all_fold_rows: list[pd.DataFrame] = []
    all_model_summaries: list[pd.DataFrame] = []

    feature_ablation_setups = [
        {
            "setup_id": "feature_baseline_current_target30_fill_current",
            "ablation_group": "feature_ablation",
            "feature_set": "baseline",
            "feature_cols": baseline_features,
            "target_variant": "direct_30s",
            "target_col": TARGET_VARIANTS["direct_30s"],
            "fill_strategy": "current_ffill",
        },
        {
            "setup_id": "feature_upgraded_current_target30_fill_current",
            "ablation_group": "feature_ablation",
            "feature_set": "upgraded",
            "feature_cols": upgraded_features,
            "target_variant": "direct_30s",
            "target_col": TARGET_VARIANTS["direct_30s"],
            "fill_strategy": "current_ffill",
        },
    ]

    target_comparison_setups = [
        {
            "setup_id": "target_direct30_upgraded_fill_current",
            "ablation_group": "target_comparison",
            "feature_set": "upgraded",
            "feature_cols": upgraded_features,
            "target_variant": "direct_30s",
            "target_col": TARGET_VARIANTS["direct_30s"],
            "fill_strategy": "current_ffill",
        },
        {
            "setup_id": "target_mean_next30_upgraded_fill_current",
            "ablation_group": "target_comparison",
            "feature_set": "upgraded",
            "feature_cols": upgraded_features,
            "target_variant": "mean_next_30s",
            "target_col": TARGET_VARIANTS["mean_next_30s"],
            "fill_strategy": "current_ffill",
        },
        {
            "setup_id": "target_direct15_upgraded_fill_current",
            "ablation_group": "target_comparison",
            "feature_set": "upgraded",
            "feature_cols": upgraded_features,
            "target_variant": "direct_15s",
            "target_col": TARGET_VARIANTS["direct_15s"],
            "fill_strategy": "current_ffill",
        },
    ]

    fill_sensitivity_setups = [
        {
            "setup_id": "fill_current_upgraded_target30",
            "ablation_group": "fill_sensitivity",
            "feature_set": "upgraded",
            "feature_cols": upgraded_features,
            "target_variant": "direct_30s",
            "target_col": TARGET_VARIANTS["direct_30s"],
            "fill_strategy": "current_ffill",
        },
        {
            "setup_id": "fill_limited5_upgraded_target30",
            "ablation_group": "fill_sensitivity",
            "feature_set": "upgraded",
            "feature_cols": upgraded_features,
            "target_variant": "direct_30s",
            "target_col": TARGET_VARIANTS["direct_30s"],
            "fill_strategy": "limited_ffill_5s",
        },
        {
            "setup_id": "fill_strict_observed_upgraded_target30",
            "ablation_group": "fill_sensitivity",
            "feature_set": "upgraded",
            "feature_cols": upgraded_features,
            "target_variant": "direct_30s",
            "target_col": TARGET_VARIANTS["direct_30s"],
            "fill_strategy": "strict_observed_only",
        },
    ]

    all_setups = feature_ablation_setups + target_comparison_setups + fill_sensitivity_setups

    for setup in all_setups:
        setup_df = fill_strategy_tables[setup["fill_strategy"]]

        result_row, fold_df, model_summary_df = _evaluate_regression_setup(
            setup_id=setup["setup_id"],
            ablation_group=setup["ablation_group"],
            model_df=setup_df,
            feature_cols=setup["feature_cols"],
            target_variant=setup["target_variant"],
            target_col=setup["target_col"],
            fill_strategy=setup["fill_strategy"],
            feature_set=setup["feature_set"],
            random_seed=random_seed,
            regression_specs=regression_specs,
        )

        setup_rows.append(result_row)
        all_fold_rows.append(fold_df)
        all_model_summaries.append(model_summary_df)

    all_setup_df = pd.DataFrame(setup_rows)
    ablation_fold_df = pd.concat(all_fold_rows, ignore_index=True)
    ablation_model_summary_df = pd.concat(all_model_summaries, ignore_index=True)

    feature_ablation_df = (
        all_setup_df[all_setup_df["ablation_group"] == "feature_ablation"]
        .sort_values("feature_set")
        .reset_index(drop=True)
    )
    target_comparison_df = (
        all_setup_df[all_setup_df["ablation_group"] == "target_comparison"]
        .sort_values("best_mean_mae")
        .reset_index(drop=True)
    )
    fill_sensitivity_df = (
        all_setup_df[all_setup_df["ablation_group"] == "fill_sensitivity"]
        .sort_values("best_mean_mae")
        .reset_index(drop=True)
    )

    preferred_setup, preferred_setup_df = _choose_preferred_setup(
        feature_ablation_df=feature_ablation_df,
        target_comparison_df=target_comparison_df,
        fill_sensitivity_df=fill_sensitivity_df,
    )

    preferred_fill = preferred_setup["preferred_fill_strategy"]
    preferred_feature_set = preferred_setup["preferred_feature_set"]
    preferred_target_col = preferred_setup["preferred_target_col"]

    preferred_feature_cols = baseline_features if preferred_feature_set == "baseline" else upgraded_features
    preferred_table_raw = fill_strategy_tables[preferred_fill].copy()
    preferred_table = preferred_table_raw.copy()

    output_columns = [
        "subject_id",
        "session",
        "timestamp_s",
        "activity_id",
        "activity_label",
        "heart_rate_bpm",
        "heart_rate_observed_flag",
        "heart_rate_fill_strategy",
        *preferred_feature_cols,
        "hr_target_30s",
        "hr_target_15s",
        "hr_target_next30s_mean",
        "activity_target",
    ]
    deduped_output_columns: list[str] = []
    seen_columns: set[str] = set()
    for column_name in output_columns:
        if column_name in preferred_table.columns and column_name not in seen_columns:
            deduped_output_columns.append(column_name)
            seen_columns.add(column_name)
    preferred_table = preferred_table[deduped_output_columns].copy()

    duplicate_count = int(preferred_table.duplicated(subset=["subject_id", "timestamp_s"]).sum())
    if duplicate_count > 0:
        raise ValueError(f"Preferred table has duplicate subject-second rows: {duplicate_count}")

    regression_required = [*preferred_feature_cols, preferred_target_col]
    classification_required = [*preferred_feature_cols, "activity_target"]

    regression_table = preferred_table.dropna(subset=regression_required).reset_index(drop=True)
    classification_table = preferred_table.dropna(subset=classification_required).reset_index(drop=True)

    if regression_table.empty:
        raise ValueError("Regression-ready table is empty after preferred setup filtering.")
    if classification_table.empty:
        raise ValueError("Classification-ready table is empty after preferred setup filtering.")

    regression_table.to_parquet(processed_regression_path, index=False)
    classification_table.to_parquet(processed_classification_path, index=False)

    task_table_summary_df = pd.DataFrame(
        [
            {
                "preferred_feature_set": preferred_feature_set,
                "preferred_fill_strategy": preferred_fill,
                "preferred_target_col": preferred_target_col,
                "regression_rows": int(len(regression_table)),
                "classification_rows": int(len(classification_table)),
                "classification_row_gain_vs_regression": int(len(classification_table) - len(regression_table)),
                "classification_row_gain_pct_vs_regression": float(
                    (len(classification_table) - len(regression_table)) / max(len(regression_table), 1)
                ),
            }
        ]
    )

    feature_inventory_df = _build_feature_inventory(preferred_feature_cols, preferred_feature_set)

    feature_ablation_df.to_csv(metrics_dir / "grouped_cv_feature_ablation_summary.csv", index=False)
    target_comparison_df.to_csv(metrics_dir / "grouped_cv_target_comparison_summary.csv", index=False)
    fill_sensitivity_df.to_csv(metrics_dir / "grouped_cv_fill_sensitivity_summary.csv", index=False)
    ablation_fold_df.to_csv(metrics_dir / "grouped_cv_ablation_fold_metrics.csv", index=False)
    ablation_model_summary_df.to_csv(metrics_dir / "grouped_cv_ablation_model_summaries.csv", index=False)
    feature_inventory_df.to_csv(metrics_dir / "grouped_cv_final_feature_summary.csv", index=False)
    preferred_setup_df.to_csv(metrics_dir / "grouped_cv_preferred_setup_summary.csv", index=False)
    task_table_summary_df.to_csv(metrics_dir / "grouped_cv_task_table_row_summary.csv", index=False)

    _save_ablation_plots(
        feature_ablation_df=feature_ablation_df,
        target_comparison_df=target_comparison_df,
        fill_sensitivity_df=fill_sensitivity_df,
        figures_dir=figures_dir,
    )

    grouped_results = run_grouped_evaluation(
        regression_processed_path=processed_regression_path,
        classification_processed_path=processed_classification_path,
        metrics_dir=metrics_dir,
        figures_dir=figures_dir,
        models_dir=models_dir,
        random_seed=random_seed,
        alpha=alpha,
        regression_target_col=preferred_target_col,
    )

    return {
        "feature_ablation": feature_ablation_df,
        "target_comparison": target_comparison_df,
        "fill_sensitivity": fill_sensitivity_df,
        "preferred_setup": preferred_setup_df,
        "preferred_target_col": preferred_target_col,
        "preferred_feature_set": preferred_feature_set,
        "preferred_fill_strategy": preferred_fill,
        "task_table_row_summary": task_table_summary_df,
        "grouped_results": grouped_results,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    results = run_compact_ablation_study(
        repo_root=repo_root,
        random_seed=RANDOM_SEED,
        alpha=ALPHA,
    )

    preferred_df = results["preferred_setup"]
    print("Compact ablation study complete.")
    print(preferred_df.to_string(index=False))


if __name__ == "__main__":
    main()
