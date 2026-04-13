from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42
ALPHA = 0.10


def _regression_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse_value = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse_value)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _classification_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _build_regression_model_specs(random_seed: int) -> dict[str, Any]:
    return {
        "persistence_current_hr": None,
        "linear_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=8,
            min_samples_leaf=30,
            max_leaf_nodes=31,
            random_state=random_seed,
        ),
    }


def _build_classification_model_specs(random_seed: int) -> dict[str, Any]:
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2500,
                        random_state=random_seed,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=16,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=-1,
            random_state=random_seed,
        ),
    }


def _predict_regression(
    model_name: str,
    fitted_model: Any,
    x_df: pd.DataFrame,
) -> np.ndarray:
    if model_name == "persistence_current_hr":
        return x_df["heart_rate_bpm"].to_numpy()
    return np.asarray(fitted_model.predict(x_df))


def _fit_if_needed(model_name: str, model_obj: Any, x_df: pd.DataFrame, y: pd.Series) -> Any:
    if model_name == "persistence_current_hr":
        return None
    fitted = clone(model_obj)
    fitted.fit(x_df, y)
    return fitted


def _run_grouped_cv_regression(
    x_df: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    meta_df: pd.DataFrame,
    model_specs: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    logo = LeaveOneGroupOut()
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    group_arr = groups.to_numpy()

    for fold_id, (train_idx, test_idx) in enumerate(logo.split(x_df, y, groups=group_arr), start=1):
        train_subjects = sorted(pd.unique(group_arr[train_idx]).tolist())
        test_subjects = sorted(pd.unique(group_arr[test_idx]).tolist())
        if len(test_subjects) != 1:
            raise ValueError("Leave-one-subject-out should produce exactly one test subject per fold.")
        if set(train_subjects).intersection(set(test_subjects)):
            raise ValueError(f"Grouped CV leakage check failed in regression fold {fold_id}.")
        test_subject = int(test_subjects[0])

        for model_name, model_obj in model_specs.items():
            fitted_model = _fit_if_needed(model_name, model_obj, x_df.iloc[train_idx], y.iloc[train_idx])
            y_pred = _predict_regression(model_name, fitted_model, x_df.iloc[test_idx])
            y_true = y.iloc[test_idx].to_numpy()
            scores = _regression_scores(y_true, y_pred)

            fold_rows.append(
                {
                    "task": "regression",
                    "model": model_name,
                    "fold": fold_id,
                    "test_subject_id": test_subject,
                    "train_subject_count": len(train_subjects),
                    **scores,
                }
            )

            fold_meta = meta_df.iloc[test_idx].copy()
            fold_meta["task"] = "regression"
            fold_meta["model"] = model_name
            fold_meta["fold"] = fold_id
            fold_meta["y_true"] = y_true
            fold_meta["y_pred"] = y_pred
            prediction_rows.extend(fold_meta.to_dict(orient="records"))

    return pd.DataFrame(fold_rows), pd.DataFrame(prediction_rows)


def _run_grouped_cv_classification(
    x_df: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    meta_df: pd.DataFrame,
    model_specs: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    logo = LeaveOneGroupOut()
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    group_arr = groups.to_numpy()

    for fold_id, (train_idx, test_idx) in enumerate(logo.split(x_df, y, groups=group_arr), start=1):
        train_subjects = sorted(pd.unique(group_arr[train_idx]).tolist())
        test_subjects = sorted(pd.unique(group_arr[test_idx]).tolist())
        if len(test_subjects) != 1:
            raise ValueError("Leave-one-subject-out should produce exactly one test subject per fold.")
        if set(train_subjects).intersection(set(test_subjects)):
            raise ValueError(f"Grouped CV leakage check failed in classification fold {fold_id}.")
        test_subject = int(test_subjects[0])

        for model_name, model_obj in model_specs.items():
            fitted_model = clone(model_obj)
            fitted_model.fit(x_df.iloc[train_idx], y.iloc[train_idx])
            y_pred = np.asarray(fitted_model.predict(x_df.iloc[test_idx]))
            y_true = y.iloc[test_idx].to_numpy()
            scores = _classification_scores(y_true, y_pred)

            fold_rows.append(
                {
                    "task": "classification",
                    "model": model_name,
                    "fold": fold_id,
                    "test_subject_id": test_subject,
                    "train_subject_count": len(train_subjects),
                    **scores,
                }
            )

            fold_meta = meta_df.iloc[test_idx].copy()
            fold_meta["task"] = "classification"
            fold_meta["model"] = model_name
            fold_meta["fold"] = fold_id
            fold_meta["y_true"] = y_true
            fold_meta["y_pred"] = y_pred
            prediction_rows.extend(fold_meta.to_dict(orient="records"))

    return pd.DataFrame(fold_rows), pd.DataFrame(prediction_rows)


def _summarize_regression_fold_metrics(fold_df: pd.DataFrame) -> pd.DataFrame:
    summary_df = (
        fold_df.groupby(["task", "model"], as_index=False)
        .agg(
            fold_count=("fold", "nunique"),
            mean_mae=("mae", "mean"),
            std_mae=("mae", "std"),
            min_mae=("mae", "min"),
            max_mae=("mae", "max"),
            mean_rmse=("rmse", "mean"),
            std_rmse=("rmse", "std"),
            min_rmse=("rmse", "min"),
            max_rmse=("rmse", "max"),
            mean_r2=("r2", "mean"),
            std_r2=("r2", "std"),
            min_r2=("r2", "min"),
            max_r2=("r2", "max"),
        )
        .sort_values(["mean_mae", "std_mae", "mean_rmse"])
        .reset_index(drop=True)
    )
    summary_df["rank"] = np.arange(1, len(summary_df) + 1)
    best_mae = float(summary_df["mean_mae"].min())
    summary_df["delta_mean_mae_vs_best"] = summary_df["mean_mae"] - best_mae
    return summary_df


def _summarize_classification_fold_metrics(fold_df: pd.DataFrame) -> pd.DataFrame:
    summary_df = (
        fold_df.groupby(["task", "model"], as_index=False)
        .agg(
            fold_count=("fold", "nunique"),
            mean_macro_f1=("macro_f1", "mean"),
            std_macro_f1=("macro_f1", "std"),
            min_macro_f1=("macro_f1", "min"),
            max_macro_f1=("macro_f1", "max"),
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            min_accuracy=("accuracy", "min"),
            max_accuracy=("accuracy", "max"),
        )
        .sort_values(["mean_macro_f1", "std_macro_f1", "mean_accuracy"], ascending=[False, True, False])
        .reset_index(drop=True)
    )
    summary_df["rank"] = np.arange(1, len(summary_df) + 1)
    best_macro_f1 = float(summary_df["mean_macro_f1"].max())
    summary_df["delta_mean_macro_f1_vs_best"] = best_macro_f1 - summary_df["mean_macro_f1"]
    return summary_df


def _select_final_models(
    regression_summary_df: pd.DataFrame,
    classification_summary_df: pd.DataFrame,
) -> tuple[str, str, pd.DataFrame]:
    reg_sorted = regression_summary_df.sort_values(["mean_mae", "std_mae", "mean_rmse"]).reset_index(drop=True)
    cls_sorted = classification_summary_df.sort_values(
        ["mean_macro_f1", "std_macro_f1", "mean_accuracy"],
        ascending=[False, True, False],
    ).reset_index(drop=True)

    selected_regression_model = str(reg_sorted.iloc[0]["model"])
    selected_classification_model = str(cls_sorted.iloc[0]["model"])

    selected_summary_df = pd.DataFrame(
        [
            {
                "task": "regression",
                "selected_model": selected_regression_model,
                "selection_rule": "lowest mean MAE across LOSO folds, tie-break by lower MAE std then lower mean RMSE",
                "selected_mean_mae": float(reg_sorted.iloc[0]["mean_mae"]),
                "selected_std_mae": float(reg_sorted.iloc[0]["std_mae"]),
                "runner_up_model": str(reg_sorted.iloc[1]["model"]),
                "runner_up_mean_mae": float(reg_sorted.iloc[1]["mean_mae"]),
                "winner_margin": float(reg_sorted.iloc[1]["mean_mae"] - reg_sorted.iloc[0]["mean_mae"]),
            },
            {
                "task": "classification",
                "selected_model": selected_classification_model,
                "selection_rule": "highest mean macro F1 across LOSO folds, tie-break by lower macro F1 std then higher mean accuracy",
                "selected_mean_macro_f1": float(cls_sorted.iloc[0]["mean_macro_f1"]),
                "selected_std_macro_f1": float(cls_sorted.iloc[0]["std_macro_f1"]),
                "runner_up_model": str(cls_sorted.iloc[1]["model"]),
                "runner_up_mean_macro_f1": float(cls_sorted.iloc[1]["mean_macro_f1"]),
                "winner_margin": float(cls_sorted.iloc[0]["mean_macro_f1"] - cls_sorted.iloc[1]["mean_macro_f1"]),
            },
        ]
    )

    return selected_regression_model, selected_classification_model, selected_summary_df


def _regression_breakdown(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = df.groupby(group_cols, as_index=False)

    for keys, part in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = {col: value for col, value in zip(group_cols, keys)}
        row["rows"] = int(len(part))

        y_true = part["y_true"].to_numpy()
        y_pred = part["y_pred"].to_numpy()
        row.update(_regression_scores(y_true, y_pred))
        row["mean_abs_error"] = float(np.mean(np.abs(y_true - y_pred)))

        rows.append(row)

    return pd.DataFrame(rows).sort_values("rows", ascending=False).reset_index(drop=True)


def _classification_breakdown(df: pd.DataFrame, group_cols: list[str], labels: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = df.groupby(group_cols, as_index=False)

    for keys, part in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = {col: value for col, value in zip(group_cols, keys)}
        row["rows"] = int(len(part))

        y_true = part["y_true"].to_numpy()
        y_pred = part["y_pred"].to_numpy()

        row["accuracy"] = float(accuracy_score(y_true, y_pred))
        row["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))

        rows.append(row)

    return pd.DataFrame(rows).sort_values("rows", ascending=False).reset_index(drop=True)


def _classification_per_class(df: pd.DataFrame, label_lookup: dict[int, str]) -> pd.DataFrame:
    labels = sorted(label_lookup)
    precision, recall, f1_values, support = precision_recall_fscore_support(
        df["y_true"].to_numpy(),
        df["y_pred"].to_numpy(),
        labels=labels,
        zero_division=0,
    )

    rows = []
    for idx, class_id in enumerate(labels):
        rows.append(
            {
                "activity_target": int(class_id),
                "activity_label": label_lookup[class_id],
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1_values[idx]),
                "support": int(support[idx]),
            }
        )

    return pd.DataFrame(rows).sort_values("support", ascending=False).reset_index(drop=True)


def _run_grouped_conformal(
    x_df: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    meta_df: pd.DataFrame,
    selected_regression_model: str,
    model_specs: dict[str, Any],
    alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logo = LeaveOneGroupOut()
    target_coverage = 1.0 - alpha
    group_arr = groups.to_numpy()

    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for fold_id, (train_idx, test_idx) in enumerate(logo.split(x_df, y, groups=group_arr), start=1):
        train_subjects = sorted(pd.unique(group_arr[train_idx]).tolist())
        test_subjects = sorted(pd.unique(group_arr[test_idx]).tolist())
        if len(test_subjects) != 1:
            raise ValueError("Leave-one-subject-out should produce exactly one test subject per fold.")

        calibration_subject = int(train_subjects[-1])
        train_groups = group_arr[train_idx]
        calibration_mask = train_groups == calibration_subject

        calibration_idx = train_idx[calibration_mask]
        proper_train_idx = train_idx[~calibration_mask]

        if len(calibration_idx) == 0 or len(proper_train_idx) == 0:
            raise ValueError("Conformal split failed: missing proper train or calibration rows.")

        proper_train_subjects = sorted(pd.unique(group_arr[proper_train_idx]).tolist())
        overlap = set(proper_train_subjects).intersection({calibration_subject, int(test_subjects[0])})
        if overlap:
            raise ValueError(f"Conformal leakage check failed for fold {fold_id}: overlap {overlap}")

        model_obj = model_specs[selected_regression_model]
        fitted_model = _fit_if_needed(
            selected_regression_model,
            model_obj,
            x_df.iloc[proper_train_idx],
            y.iloc[proper_train_idx],
        )

        calibration_pred = _predict_regression(
            selected_regression_model,
            fitted_model,
            x_df.iloc[calibration_idx],
        )
        calibration_true = y.iloc[calibration_idx].to_numpy()
        abs_residuals = np.abs(calibration_true - calibration_pred)

        n_cal = len(abs_residuals)
        quantile_level = min(np.ceil((n_cal + 1) * target_coverage) / n_cal, 1.0)
        q_hat = float(np.quantile(abs_residuals, quantile_level, method="higher"))

        test_pred = _predict_regression(selected_regression_model, fitted_model, x_df.iloc[test_idx])
        test_true = y.iloc[test_idx].to_numpy()
        lower = test_pred - q_hat
        upper = test_pred + q_hat
        covered = (test_true >= lower) & (test_true <= upper)

        fold_rows.append(
            {
                "fold": fold_id,
                "test_subject_id": int(test_subjects[0]),
                "calibration_subject_id": calibration_subject,
                "proper_train_subject_count": len(proper_train_subjects),
                "n_proper_train": int(len(proper_train_idx)),
                "n_calibration": int(len(calibration_idx)),
                "n_test": int(len(test_idx)),
                "alpha": alpha,
                "target_coverage": target_coverage,
                "q_hat": q_hat,
                "empirical_coverage": float(covered.mean()),
                "average_interval_width": float(np.mean(upper - lower)),
            }
        )

        fold_meta = meta_df.iloc[test_idx].copy()
        fold_meta["fold"] = fold_id
        fold_meta["y_true"] = test_true
        fold_meta["y_pred"] = test_pred
        fold_meta["lower"] = lower
        fold_meta["upper"] = upper
        fold_meta["covered"] = covered.astype(int)
        fold_meta["interval_width"] = fold_meta["upper"] - fold_meta["lower"]
        prediction_rows.extend(fold_meta.to_dict(orient="records"))

    conformal_fold_df = pd.DataFrame(fold_rows)
    conformal_predictions_df = pd.DataFrame(prediction_rows)

    conformal_summary_df = pd.DataFrame(
        [
            {
                "selected_regression_model": selected_regression_model,
                "alpha": alpha,
                "target_coverage": target_coverage,
                "mean_fold_coverage": float(conformal_fold_df["empirical_coverage"].mean()),
                "std_fold_coverage": float(conformal_fold_df["empirical_coverage"].std(ddof=1)),
                "min_fold_coverage": float(conformal_fold_df["empirical_coverage"].min()),
                "max_fold_coverage": float(conformal_fold_df["empirical_coverage"].max()),
                "mean_fold_interval_width": float(conformal_fold_df["average_interval_width"].mean()),
                "std_fold_interval_width": float(conformal_fold_df["average_interval_width"].std(ddof=1)),
                "row_level_empirical_coverage": float(conformal_predictions_df["covered"].mean()),
                "row_level_average_interval_width": float(conformal_predictions_df["interval_width"].mean()),
                "fold_count": int(conformal_fold_df["fold"].nunique()),
                "n_predictions": int(len(conformal_predictions_df)),
            }
        ]
    )

    by_subject_df = (
        conformal_predictions_df.groupby("subject_id", as_index=False)
        .agg(
            rows=("covered", "size"),
            empirical_coverage=("covered", "mean"),
            average_interval_width=("interval_width", "mean"),
        )
        .sort_values("subject_id")
    )

    by_activity_df = (
        conformal_predictions_df.groupby(["activity_target", "activity_label"], as_index=False)
        .agg(
            rows=("covered", "size"),
            empirical_coverage=("covered", "mean"),
            average_interval_width=("interval_width", "mean"),
        )
        .sort_values("rows", ascending=False)
    )

    return conformal_fold_df, conformal_predictions_df, conformal_summary_df, by_subject_df, by_activity_df


def _save_plots(
    regression_fold_df: pd.DataFrame,
    classification_fold_df: pd.DataFrame,
    regression_by_activity_df: pd.DataFrame,
    classification_per_class_df: pd.DataFrame,
    conformal_by_activity_df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.boxplot(data=regression_fold_df, x="model", y="mae", ax=ax)
    sns.stripplot(data=regression_fold_df, x="model", y="mae", ax=ax, color="black", alpha=0.5)
    ax.set_title("Regression LOSO fold MAE by model")
    ax.set_xlabel("model")
    ax.set_ylabel("MAE")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(figures_dir / "grouped_cv_regression_mae_by_model.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=classification_fold_df, x="model", y="macro_f1", ax=ax)
    sns.stripplot(data=classification_fold_df, x="model", y="macro_f1", ax=ax, color="black", alpha=0.5)
    ax.set_title("Classification LOSO fold macro F1 by model")
    ax.set_xlabel("model")
    ax.set_ylabel("macro F1")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(figures_dir / "grouped_cv_classification_macro_f1_by_model.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    top_reg_activity = regression_by_activity_df.sort_values("rows", ascending=False).head(12)
    sns.barplot(data=top_reg_activity, x="activity_label", y="mae", ax=ax, color="tab:blue")
    ax.set_title("Selected regression model MAE by activity")
    ax.set_xlabel("activity")
    ax.set_ylabel("MAE")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(figures_dir / "grouped_cv_regression_mae_by_activity.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    per_class_sorted = classification_per_class_df.sort_values("support", ascending=False)
    sns.barplot(data=per_class_sorted, x="activity_label", y="f1", ax=ax, color="tab:green")
    ax.set_title("Selected classification model per-class F1")
    ax.set_xlabel("activity")
    ax.set_ylabel("F1")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(figures_dir / "grouped_cv_classification_per_class_f1.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    coverage_plot_df = conformal_by_activity_df.sort_values("rows", ascending=False).head(12)
    sns.barplot(data=coverage_plot_df, x="activity_label", y="empirical_coverage", ax=ax, color="tab:orange")
    target_coverage = 1.0 - ALPHA
    ax.axhline(target_coverage, color="black", linestyle="--", linewidth=1.2, label="target coverage")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Grouped conformal coverage by activity")
    ax.set_xlabel("activity")
    ax.set_ylabel("coverage")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "grouped_cv_conformal_coverage_by_activity.png", dpi=140)
    plt.close(fig)


def run_grouped_evaluation(
    processed_path: Path,
    metrics_dir: Path,
    figures_dir: Path,
    models_dir: Path,
    random_seed: int = RANDOM_SEED,
    alpha: float = ALPHA,
) -> dict[str, pd.DataFrame]:
    if not processed_path.exists():
        raise FileNotFoundError(f"Missing processed model table: {processed_path}")

    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    model_df = pd.read_parquet(processed_path).copy()
    model_df = model_df.sort_values(["subject_id", "timestamp_s"]).reset_index(drop=True)

    required_columns = [
        "subject_id",
        "timestamp_s",
        "activity_target",
        "activity_label",
        "heart_rate_bpm",
        "hr_target_30s",
    ]
    missing_required = [column for column in required_columns if column not in model_df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns in processed table: {missing_required}")

    duplicate_count = int(model_df.duplicated(subset=["subject_id", "timestamp_s"]).sum())
    if duplicate_count > 0:
        raise ValueError(f"Found duplicate subject-second rows: {duplicate_count}")

    non_feature_cols = {
        "subject_id",
        "session",
        "timestamp_s",
        "activity_id",
        "activity_label",
        "activity_target",
        "hr_target_30s",
    }
    feature_cols = [column for column in model_df.columns if column not in non_feature_cols]

    x_df = model_df[feature_cols]
    y_reg = model_df["hr_target_30s"]
    y_cls = model_df["activity_target"].astype(int)
    groups = model_df["subject_id"].astype(int)

    meta_df = model_df[["subject_id", "timestamp_s", "activity_target", "activity_label"]].copy()
    label_lookup = (
        model_df[["activity_target", "activity_label"]]
        .drop_duplicates()
        .set_index("activity_target")["activity_label"]
        .to_dict()
    )

    regression_specs = _build_regression_model_specs(random_seed)
    classification_specs = _build_classification_model_specs(random_seed)

    regression_fold_df, regression_pred_df = _run_grouped_cv_regression(
        x_df=x_df,
        y=y_reg,
        groups=groups,
        meta_df=meta_df,
        model_specs=regression_specs,
    )
    classification_fold_df, classification_pred_df = _run_grouped_cv_classification(
        x_df=x_df,
        y=y_cls,
        groups=groups,
        meta_df=meta_df,
        model_specs=classification_specs,
    )

    regression_summary_df = _summarize_regression_fold_metrics(regression_fold_df)
    classification_summary_df = _summarize_classification_fold_metrics(classification_fold_df)

    selected_regression_model, selected_classification_model, selected_models_df = _select_final_models(
        regression_summary_df=regression_summary_df,
        classification_summary_df=classification_summary_df,
    )

    selected_regression_preds_df = regression_pred_df[regression_pred_df["model"] == selected_regression_model].copy()
    selected_classification_preds_df = classification_pred_df[
        classification_pred_df["model"] == selected_classification_model
    ].copy()

    class_labels = sorted(label_lookup)
    regression_by_subject_df = _regression_breakdown(selected_regression_preds_df, ["subject_id"])
    regression_by_activity_df = _regression_breakdown(
        selected_regression_preds_df,
        ["activity_target", "activity_label"],
    )
    classification_by_subject_df = _classification_breakdown(
        selected_classification_preds_df,
        ["subject_id"],
        labels=class_labels,
    )
    classification_by_activity_df = _classification_breakdown(
        selected_classification_preds_df,
        ["activity_target", "activity_label"],
        labels=class_labels,
    )
    classification_per_class_df = _classification_per_class(selected_classification_preds_df, label_lookup)

    conformal_fold_df, conformal_predictions_df, conformal_summary_df, conformal_by_subject_df, conformal_by_activity_df = (
        _run_grouped_conformal(
            x_df=x_df,
            y=y_reg,
            groups=groups,
            meta_df=meta_df,
            selected_regression_model=selected_regression_model,
            model_specs=regression_specs,
            alpha=alpha,
        )
    )

    # Train selected models on full data for optional reuse outside this notebook workflow.
    reg_model_to_save = _fit_if_needed(
        selected_regression_model,
        regression_specs[selected_regression_model],
        x_df,
        y_reg,
    )
    cls_model_to_save = clone(classification_specs[selected_classification_model])
    cls_model_to_save.fit(x_df, y_cls)

    if reg_model_to_save is not None:
        joblib.dump(reg_model_to_save, models_dir / "grouped_cv_selected_regression_model.joblib")
    joblib.dump(cls_model_to_save, models_dir / "grouped_cv_selected_classification_model.joblib")

    regression_fold_df.to_csv(metrics_dir / "grouped_cv_regression_fold_metrics.csv", index=False)
    regression_summary_df.to_csv(metrics_dir / "grouped_cv_regression_summary.csv", index=False)
    classification_fold_df.to_csv(metrics_dir / "grouped_cv_classification_fold_metrics.csv", index=False)
    classification_summary_df.to_csv(metrics_dir / "grouped_cv_classification_summary.csv", index=False)
    selected_models_df.to_csv(metrics_dir / "grouped_cv_selected_model_summary.csv", index=False)

    regression_by_subject_df.to_csv(metrics_dir / "grouped_cv_regression_selected_by_subject.csv", index=False)
    regression_by_activity_df.to_csv(metrics_dir / "grouped_cv_regression_selected_by_activity.csv", index=False)
    classification_by_subject_df.to_csv(metrics_dir / "grouped_cv_classification_selected_by_subject.csv", index=False)
    classification_by_activity_df.to_csv(metrics_dir / "grouped_cv_classification_selected_by_activity.csv", index=False)
    classification_per_class_df.to_csv(metrics_dir / "grouped_cv_classification_selected_per_class.csv", index=False)

    regression_pred_df.to_csv(metrics_dir / "grouped_cv_regression_predictions_all_models.csv", index=False)
    classification_pred_df.to_csv(metrics_dir / "grouped_cv_classification_predictions_all_models.csv", index=False)

    conformal_fold_df.to_csv(metrics_dir / "grouped_cv_conformal_fold_summary.csv", index=False)
    conformal_predictions_df.to_csv(metrics_dir / "grouped_cv_conformal_predictions.csv", index=False)
    conformal_summary_df.to_csv(metrics_dir / "grouped_cv_conformal_summary.csv", index=False)
    conformal_by_subject_df.to_csv(metrics_dir / "grouped_cv_conformal_by_subject.csv", index=False)
    conformal_by_activity_df.to_csv(metrics_dir / "grouped_cv_conformal_by_activity.csv", index=False)

    _save_plots(
        regression_fold_df=regression_fold_df,
        classification_fold_df=classification_fold_df,
        regression_by_activity_df=regression_by_activity_df,
        classification_per_class_df=classification_per_class_df,
        conformal_by_activity_df=conformal_by_activity_df,
        figures_dir=figures_dir,
    )

    return {
        "regression_fold": regression_fold_df,
        "regression_summary": regression_summary_df,
        "classification_fold": classification_fold_df,
        "classification_summary": classification_summary_df,
        "selected_models": selected_models_df,
        "regression_by_subject": regression_by_subject_df,
        "regression_by_activity": regression_by_activity_df,
        "classification_by_subject": classification_by_subject_df,
        "classification_by_activity": classification_by_activity_df,
        "classification_per_class": classification_per_class_df,
        "conformal_fold": conformal_fold_df,
        "conformal_summary": conformal_summary_df,
        "conformal_by_subject": conformal_by_subject_df,
        "conformal_by_activity": conformal_by_activity_df,
    }


def _default_paths() -> tuple[Path, Path, Path, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    processed_path = repo_root / "data" / "processed" / "pamap2_model_table.parquet"
    metrics_dir = repo_root / "artifacts" / "metrics"
    figures_dir = repo_root / "artifacts" / "figures"
    models_dir = repo_root / "artifacts" / "models"
    return processed_path, metrics_dir, figures_dir, models_dir


def main() -> None:
    processed_path, metrics_dir, figures_dir, models_dir = _default_paths()
    results = run_grouped_evaluation(
        processed_path=processed_path,
        metrics_dir=metrics_dir,
        figures_dir=figures_dir,
        models_dir=models_dir,
        random_seed=RANDOM_SEED,
        alpha=ALPHA,
    )

    selected_models_df = results["selected_models"]
    print("Grouped evaluation complete.")
    print(selected_models_df.to_string(index=False))


if __name__ == "__main__":
    main()
