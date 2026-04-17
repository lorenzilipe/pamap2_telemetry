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
from sklearn.impute import SimpleImputer
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
MIN_ACTIVITY_CALIBRATION_ROWS = 40
LARGE_ERROR_QUANTILE = 0.90
ABSTENTION_THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90]


def _read_preferred_target_col(metrics_dir: Path) -> str | None:
    preferred_setup_path = metrics_dir / "grouped_cv_preferred_setup_summary.csv"
    if not preferred_setup_path.exists():
        return None

    preferred_setup_df = pd.read_csv(preferred_setup_path)
    if preferred_setup_df.empty:
        raise ValueError(f"Preferred setup artifact is empty: {preferred_setup_path}")
    if "preferred_target_col" not in preferred_setup_df.columns:
        raise ValueError(
            "Preferred setup artifact is missing preferred_target_col: "
            f"{preferred_setup_path}"
        )

    preferred_target_col = str(preferred_setup_df.iloc[0]["preferred_target_col"]).strip()
    if not preferred_target_col:
        raise ValueError(
            "Preferred setup artifact contains an empty preferred_target_col: "
            f"{preferred_setup_path}"
        )
    return preferred_target_col


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
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        learning_rate=0.05,
                        max_depth=8,
                        min_samples_leaf=30,
                        max_leaf_nodes=31,
                        random_state=random_seed,
                    ),
                ),
            ]
        ),
    }


def build_regression_model_specs(random_seed: int = RANDOM_SEED) -> dict[str, Any]:
    """Public wrapper used by compact ablation workflows."""
    return _build_regression_model_specs(random_seed)


def _build_classification_model_specs(random_seed: int) -> dict[str, Any]:
    return {
        "logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
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
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=16,
                        min_samples_leaf=2,
                        max_features="sqrt",
                        n_jobs=-1,
                        random_state=random_seed,
                    ),
                ),
            ]
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


def run_grouped_regression_cv(
    model_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    random_seed: int = RANDOM_SEED,
    model_specs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run LOSO grouped CV for regression on an in-memory table."""
    required_cols = {"subject_id", "timestamp_s", target_col, *feature_cols}
    missing_required = sorted(required_cols.difference(set(model_df.columns)))
    if missing_required:
        raise ValueError(f"Missing required columns for grouped regression CV: {missing_required}")

    eval_subset = model_df[["subject_id", "timestamp_s", *feature_cols, target_col]].copy()
    valid_mask = eval_subset[[*feature_cols, target_col]].notna().all(axis=1)
    eval_df = eval_subset.loc[valid_mask].reset_index(drop=True)

    if eval_df.empty:
        raise ValueError("Grouped regression CV has no rows after dropping missing feature or target values.")

    duplicate_count = int(eval_df.duplicated(subset=["subject_id", "timestamp_s"]).sum())
    if duplicate_count > 0:
        raise ValueError(f"Found duplicate subject-second rows in grouped regression CV input: {duplicate_count}")

    x_df = eval_df[feature_cols]
    y = eval_df[target_col]
    groups = eval_df["subject_id"].astype(int)

    meta_cols = ["subject_id", "timestamp_s"]
    optional_meta = ["activity_target", "activity_label"]
    for col in optional_meta:
        if col in model_df.columns and col not in meta_cols:
            meta_cols.append(col)
    meta_df = model_df.loc[valid_mask, meta_cols].reset_index(drop=True)

    specs = model_specs if model_specs is not None else _build_regression_model_specs(random_seed)

    fold_df, prediction_df = _run_grouped_cv_regression(
        x_df=x_df,
        y=y,
        groups=groups,
        meta_df=meta_df,
        model_specs=specs,
    )
    summary_df = _summarize_regression_fold_metrics(fold_df)

    return {
        "fold": fold_df,
        "summary": summary_df,
        "predictions": prediction_df,
    }


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


def _align_class_probabilities(
    fitted_model: Any,
    x_df: pd.DataFrame,
    class_labels: list[int],
) -> tuple[np.ndarray, list[str]]:
    if not hasattr(fitted_model, "predict_proba"):
        raise ValueError("Classification model must support predict_proba for confidence diagnostics.")

    raw_proba = np.asarray(fitted_model.predict_proba(x_df))
    model_classes = [int(value) for value in np.asarray(fitted_model.classes_).tolist()]

    aligned_proba = np.zeros((len(x_df), len(class_labels)), dtype=float)
    class_to_col = {class_id: idx for idx, class_id in enumerate(class_labels)}

    for raw_idx, class_id in enumerate(model_classes):
        if class_id in class_to_col:
            aligned_proba[:, class_to_col[class_id]] = raw_proba[:, raw_idx]

    probability_columns = [f"proba_class_{class_id}" for class_id in class_labels]
    return aligned_proba, probability_columns


def _run_grouped_cv_classification(
    x_df: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    meta_df: pd.DataFrame,
    model_specs: dict[str, Any],
    class_labels: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    logo = LeaveOneGroupOut()
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    group_arr = groups.to_numpy()
    class_to_idx = {class_id: idx for idx, class_id in enumerate(class_labels)}

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
            aligned_proba, probability_columns = _align_class_probabilities(
                fitted_model=fitted_model,
                x_df=x_df.iloc[test_idx],
                class_labels=class_labels,
            )

            true_idx = np.array([class_to_idx[int(label)] for label in y_true], dtype=int)
            pred_idx = np.array([class_to_idx[int(label)] for label in y_pred], dtype=int)
            true_class_probability = aligned_proba[np.arange(len(y_true)), true_idx]
            predicted_class_probability = aligned_proba[np.arange(len(y_pred)), pred_idx]
            confidence = np.max(aligned_proba, axis=1)
            is_correct = (y_pred == y_true).astype(int)

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
            fold_meta["is_correct"] = is_correct
            fold_meta["confidence"] = confidence
            fold_meta["predicted_class_probability"] = predicted_class_probability
            fold_meta["true_class_probability"] = true_class_probability

            for class_idx, col_name in enumerate(probability_columns):
                fold_meta[col_name] = aligned_proba[:, class_idx]

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


def _conformal_quantile(abs_residuals: np.ndarray, target_coverage: float) -> float:
    n_cal = len(abs_residuals)
    if n_cal == 0:
        raise ValueError("Conformal quantile requires at least one calibration residual.")
    quantile_level = min(np.ceil((n_cal + 1) * target_coverage) / n_cal, 1.0)
    return float(np.quantile(abs_residuals, quantile_level, method="higher"))


def _run_grouped_conformal(
    x_df: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    meta_df: pd.DataFrame,
    selected_regression_model: str,
    model_specs: dict[str, Any],
    alpha: float,
    min_activity_calibration_rows: int = MIN_ACTIVITY_CALIBRATION_ROWS,
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
        global_q_hat = _conformal_quantile(abs_residuals, target_coverage)

        calibration_meta = meta_df.iloc[calibration_idx].copy()
        calibration_meta["abs_residual"] = abs_residuals
        activity_q_hats: dict[int, float] = {}

        for activity_id, part in calibration_meta.groupby("activity_target"):
            if len(part) >= min_activity_calibration_rows:
                activity_q_hats[int(activity_id)] = _conformal_quantile(
                    part["abs_residual"].to_numpy(),
                    target_coverage,
                )

        test_pred = _predict_regression(selected_regression_model, fitted_model, x_df.iloc[test_idx])
        test_true = y.iloc[test_idx].to_numpy()
        test_activity = meta_df.iloc[test_idx]["activity_target"].astype(int).to_numpy()

        for calibration_variant in ["global", "activity_conditioned"]:
            if calibration_variant == "global":
                q_hat_used = np.full(len(test_idx), global_q_hat, dtype=float)
                fallback_to_global = np.zeros(len(test_idx), dtype=int)
            else:
                q_hat_used = np.array(
                    [activity_q_hats.get(int(activity_id), global_q_hat) for activity_id in test_activity],
                    dtype=float,
                )
                fallback_to_global = np.array(
                    [0 if int(activity_id) in activity_q_hats else 1 for activity_id in test_activity],
                    dtype=int,
                )

            lower = test_pred - q_hat_used
            upper = test_pred + q_hat_used
            covered = (test_true >= lower) & (test_true <= upper)

            fold_rows.append(
                {
                    "fold": fold_id,
                    "calibration_variant": calibration_variant,
                    "test_subject_id": int(test_subjects[0]),
                    "calibration_subject_id": calibration_subject,
                    "proper_train_subject_count": len(proper_train_subjects),
                    "n_proper_train": int(len(proper_train_idx)),
                    "n_calibration": int(len(calibration_idx)),
                    "n_test": int(len(test_idx)),
                    "alpha": alpha,
                    "target_coverage": target_coverage,
                    "global_q_hat": global_q_hat,
                    "mean_q_hat_used": float(np.mean(q_hat_used)),
                    "activity_q_hat_count": int(len(activity_q_hats)),
                    "fallback_to_global_rate": float(fallback_to_global.mean()),
                    "empirical_coverage": float(covered.mean()),
                    "average_interval_width": float(np.mean(upper - lower)),
                }
            )

            fold_meta = meta_df.iloc[test_idx].copy()
            fold_meta["fold"] = fold_id
            fold_meta["calibration_variant"] = calibration_variant
            fold_meta["y_true"] = test_true
            fold_meta["y_pred"] = test_pred
            fold_meta["q_hat_used"] = q_hat_used
            fold_meta["used_global_fallback"] = fallback_to_global
            fold_meta["lower"] = lower
            fold_meta["upper"] = upper
            fold_meta["covered"] = covered.astype(int)
            fold_meta["interval_width"] = fold_meta["upper"] - fold_meta["lower"]
            prediction_rows.extend(fold_meta.to_dict(orient="records"))

    conformal_fold_df = pd.DataFrame(fold_rows)
    conformal_predictions_df = pd.DataFrame(prediction_rows)

    summary_rows: list[dict[str, Any]] = []
    for calibration_variant in sorted(conformal_fold_df["calibration_variant"].unique()):
        fold_part = conformal_fold_df[conformal_fold_df["calibration_variant"] == calibration_variant]
        prediction_part = conformal_predictions_df[conformal_predictions_df["calibration_variant"] == calibration_variant]

        summary_rows.append(
            {
                "selected_regression_model": selected_regression_model,
                "calibration_variant": calibration_variant,
                "alpha": alpha,
                "target_coverage": target_coverage,
                "mean_fold_coverage": float(fold_part["empirical_coverage"].mean()),
                "std_fold_coverage": float(fold_part["empirical_coverage"].std(ddof=1)),
                "min_fold_coverage": float(fold_part["empirical_coverage"].min()),
                "max_fold_coverage": float(fold_part["empirical_coverage"].max()),
                "mean_fold_interval_width": float(fold_part["average_interval_width"].mean()),
                "std_fold_interval_width": float(fold_part["average_interval_width"].std(ddof=1)),
                "row_level_empirical_coverage": float(prediction_part["covered"].mean()),
                "row_level_average_interval_width": float(prediction_part["interval_width"].mean()),
                "mean_fallback_to_global_rate": float(fold_part["fallback_to_global_rate"].mean()),
                "fold_count": int(fold_part["fold"].nunique()),
                "n_predictions": int(len(prediction_part)),
            }
        )

    conformal_summary_df = pd.DataFrame(summary_rows)

    by_subject_df = (
        conformal_predictions_df.groupby(["calibration_variant", "subject_id"], as_index=False)
        .agg(
            rows=("covered", "size"),
            empirical_coverage=("covered", "mean"),
            average_interval_width=("interval_width", "mean"),
        )
        .sort_values(["calibration_variant", "subject_id"])
    )
    by_subject_df["interval_failure_rate"] = 1.0 - by_subject_df["empirical_coverage"]

    by_activity_df = (
        conformal_predictions_df.groupby(["calibration_variant", "activity_target", "activity_label"], as_index=False)
        .agg(
            rows=("covered", "size"),
            empirical_coverage=("covered", "mean"),
            average_interval_width=("interval_width", "mean"),
        )
        .sort_values(["calibration_variant", "rows"], ascending=[True, False])
    )
    by_activity_df["interval_failure_rate"] = 1.0 - by_activity_df["empirical_coverage"]
    by_activity_df["coverage_gap_vs_target"] = by_activity_df["empirical_coverage"] - target_coverage
    by_activity_df["abs_coverage_gap"] = by_activity_df["coverage_gap_vs_target"].abs()

    return conformal_fold_df, conformal_predictions_df, conformal_summary_df, by_subject_df, by_activity_df


def _select_preferred_conformal_variant(
    conformal_summary_df: pd.DataFrame,
    conformal_by_activity_df: pd.DataFrame,
) -> tuple[str, pd.DataFrame]:
    target_coverage = float(conformal_summary_df["target_coverage"].iloc[0])

    comparison_rows: list[dict[str, Any]] = []
    for _, summary_row in conformal_summary_df.iterrows():
        calibration_variant = str(summary_row["calibration_variant"])
        activity_part = conformal_by_activity_df[
            conformal_by_activity_df["calibration_variant"] == calibration_variant
        ].copy()

        if activity_part.empty:
            weighted_abs_gap = np.nan
            activities_below_target = 0
            worst_activity_label = ""
            worst_activity_coverage = np.nan
        else:
            weighted_abs_gap = float(np.average(activity_part["abs_coverage_gap"], weights=activity_part["rows"]))
            activities_below_target = int((activity_part["empirical_coverage"] < target_coverage).sum())
            worst_idx = activity_part["empirical_coverage"].idxmin()
            worst_activity_label = str(activity_part.loc[worst_idx, "activity_label"])
            worst_activity_coverage = float(activity_part.loc[worst_idx, "empirical_coverage"])

        comparison_rows.append(
            {
                "calibration_variant": calibration_variant,
                "row_level_empirical_coverage": float(summary_row["row_level_empirical_coverage"]),
                "row_level_average_interval_width": float(summary_row["row_level_average_interval_width"]),
                "weighted_activity_abs_coverage_gap": weighted_abs_gap,
                "activities_below_target": activities_below_target,
                "worst_activity_label": worst_activity_label,
                "worst_activity_coverage": worst_activity_coverage,
                "mean_fallback_to_global_rate": float(summary_row["mean_fallback_to_global_rate"]),
                "target_coverage": target_coverage,
            }
        )

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        ["weighted_activity_abs_coverage_gap", "row_level_average_interval_width"],
        na_position="last",
    )

    preferred_variant = "global"
    selection_reason = "Global split conformal retained for simplicity and stable interval width."

    available_variants = set(comparison_df["calibration_variant"].tolist())
    if {"global", "activity_conditioned"}.issubset(available_variants):
        global_row = comparison_df[comparison_df["calibration_variant"] == "global"].iloc[0]
        conditioned_row = comparison_df[comparison_df["calibration_variant"] == "activity_conditioned"].iloc[0]

        gap_improvement = float(
            global_row["weighted_activity_abs_coverage_gap"] - conditioned_row["weighted_activity_abs_coverage_gap"]
        )
        width_ratio = float(
            conditioned_row["row_level_average_interval_width"]
            / max(global_row["row_level_average_interval_width"], 1e-9)
        )

        if gap_improvement >= 0.01 and width_ratio <= 1.15:
            preferred_variant = "activity_conditioned"
            selection_reason = (
                "Activity-conditioned conformal selected: improved activity-level coverage stability "
                f"(weighted abs gap improvement={gap_improvement:.4f}) with acceptable width tradeoff "
                f"(width ratio={width_ratio:.3f})."
            )
        else:
            selection_reason = (
                "Global split conformal retained: conditioning did not improve activity-level coverage enough "
                f"after width tradeoff (gap improvement={gap_improvement:.4f}, width ratio={width_ratio:.3f})."
            )

    comparison_df = comparison_df.reset_index(drop=True)
    comparison_df["preferred_interval_variant"] = (
        comparison_df["calibration_variant"] == preferred_variant
    ).astype(int)
    comparison_df["selection_reason"] = np.where(
        comparison_df["calibration_variant"] == preferred_variant,
        selection_reason,
        "Alternative not selected after coverage/width tradeoff check.",
    )

    return preferred_variant, comparison_df


def _build_uncertainty_failure_diagnostics(
    conformal_predictions_df: pd.DataFrame,
    target_coverage: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    diagnostics_df = conformal_predictions_df.copy()
    diagnostics_df["residual"] = diagnostics_df["y_true"] - diagnostics_df["y_pred"]
    diagnostics_df["abs_error"] = diagnostics_df["residual"].abs()

    large_error_threshold = float(diagnostics_df["abs_error"].quantile(LARGE_ERROR_QUANTILE))
    diagnostics_df["large_error_flag"] = (diagnostics_df["abs_error"] >= large_error_threshold).astype(int)
    diagnostics_df["interval_failure"] = (1 - diagnostics_df["covered"]).astype(int)
    diagnostics_df["failure_and_large_error"] = (
        (diagnostics_df["interval_failure"] == 1) & (diagnostics_df["large_error_flag"] == 1)
    ).astype(int)

    uncertainty_by_activity_df = (
        diagnostics_df.groupby(["activity_target", "activity_label"], as_index=False)
        .agg(
            rows=("abs_error", "size"),
            empirical_coverage=("covered", "mean"),
            interval_failure_rate=("interval_failure", "mean"),
            mean_interval_width=("interval_width", "mean"),
            mean_abs_error=("abs_error", "mean"),
            p90_abs_error=("abs_error", lambda values: float(np.quantile(values, 0.90))),
            large_error_rate=("large_error_flag", "mean"),
            failure_and_large_error_rate=("failure_and_large_error", "mean"),
        )
        .sort_values(["interval_failure_rate", "mean_abs_error"], ascending=[False, False])
        .reset_index(drop=True)
    )
    uncertainty_by_activity_df["coverage_gap_vs_target"] = (
        uncertainty_by_activity_df["empirical_coverage"] - target_coverage
    )

    uncertainty_by_subject_df = (
        diagnostics_df.groupby("subject_id", as_index=False)
        .agg(
            rows=("abs_error", "size"),
            empirical_coverage=("covered", "mean"),
            interval_failure_rate=("interval_failure", "mean"),
            mean_interval_width=("interval_width", "mean"),
            mean_abs_error=("abs_error", "mean"),
            p90_abs_error=("abs_error", lambda values: float(np.quantile(values, 0.90))),
            large_error_rate=("large_error_flag", "mean"),
            failure_and_large_error_rate=("failure_and_large_error", "mean"),
        )
        .sort_values("subject_id")
        .reset_index(drop=True)
    )
    uncertainty_by_subject_df["coverage_gap_vs_target"] = (
        uncertainty_by_subject_df["empirical_coverage"] - target_coverage
    )

    residual_summary_df = pd.DataFrame(
        [
            {
                "rows": int(len(diagnostics_df)),
                "target_coverage": target_coverage,
                "row_level_coverage": float(diagnostics_df["covered"].mean()),
                "interval_failure_rate": float(diagnostics_df["interval_failure"].mean()),
                "residual_mean": float(diagnostics_df["residual"].mean()),
                "residual_std": float(diagnostics_df["residual"].std(ddof=1)),
                "residual_median": float(diagnostics_df["residual"].median()),
                "residual_q05": float(diagnostics_df["residual"].quantile(0.05)),
                "residual_q95": float(diagnostics_df["residual"].quantile(0.95)),
                "mean_abs_error": float(diagnostics_df["abs_error"].mean()),
                "p90_abs_error": float(diagnostics_df["abs_error"].quantile(0.90)),
                "p95_abs_error": float(diagnostics_df["abs_error"].quantile(0.95)),
                "large_error_threshold": large_error_threshold,
            }
        ]
    )

    overall_abs_error = float(diagnostics_df["abs_error"].mean())
    overall_interval_width = float(diagnostics_df["interval_width"].mean())

    operating_envelope_df = uncertainty_by_activity_df.copy()
    operating_envelope_df["abs_error_ratio_vs_overall"] = (
        operating_envelope_df["mean_abs_error"] / max(overall_abs_error, 1e-9)
    )
    operating_envelope_df["interval_width_ratio_vs_overall"] = (
        operating_envelope_df["mean_interval_width"] / max(overall_interval_width, 1e-9)
    )
    operating_envelope_df["within_operating_envelope"] = (
        (operating_envelope_df["empirical_coverage"] >= (target_coverage - 0.03))
        & (operating_envelope_df["abs_error_ratio_vs_overall"] <= 1.25)
    ).astype(int)
    operating_envelope_df["envelope_status"] = np.where(
        operating_envelope_df["within_operating_envelope"] == 1,
        "supported",
        "fragile",
    )

    return (
        diagnostics_df,
        uncertainty_by_activity_df,
        uncertainty_by_subject_df,
        residual_summary_df,
        operating_envelope_df,
    )


def _multiclass_brier_score(classification_df: pd.DataFrame, class_labels: list[int]) -> float:
    probability_cols = [f"proba_class_{class_id}" for class_id in class_labels]
    probabilities = classification_df[probability_cols].to_numpy(dtype=float)

    y_true = classification_df["y_true"].astype(int).to_numpy()
    class_to_idx = {class_id: idx for idx, class_id in enumerate(class_labels)}
    y_idx = np.array([class_to_idx[int(value)] for value in y_true], dtype=int)

    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(y_idx)), y_idx] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def _expected_calibration_error(
    confidence: np.ndarray,
    is_correct: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, pd.DataFrame]:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(confidence, bin_edges[1:-1], right=True)

    rows: list[dict[str, Any]] = []
    ece_value = 0.0

    for bin_id in range(n_bins):
        mask = bin_indices == bin_id
        count = int(mask.sum())
        if count == 0:
            continue

        bin_confidence = float(confidence[mask].mean())
        bin_accuracy = float(is_correct[mask].mean())
        bin_gap = abs(bin_accuracy - bin_confidence)

        ece_value += (count / len(confidence)) * bin_gap

        rows.append(
            {
                "bin_id": bin_id + 1,
                "bin_lower": float(bin_edges[bin_id]),
                "bin_upper": float(bin_edges[bin_id + 1]),
                "rows": count,
                "mean_confidence": bin_confidence,
                "empirical_accuracy": bin_accuracy,
                "calibration_gap": bin_gap,
            }
        )

    reliability_df = pd.DataFrame(rows)
    return float(ece_value), reliability_df


def _classification_confidence_diagnostics(
    classification_df: pd.DataFrame,
    class_labels: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y_true = classification_df["y_true"].to_numpy()
    y_pred = classification_df["y_pred"].to_numpy()
    is_correct = classification_df["is_correct"].to_numpy(dtype=int)
    confidence = classification_df["confidence"].to_numpy(dtype=float)

    full_accuracy = float(accuracy_score(y_true, y_pred))
    full_macro_f1 = float(f1_score(y_true, y_pred, average="macro", labels=class_labels, zero_division=0))
    mean_confidence = float(np.mean(confidence))
    mean_true_class_probability = float(classification_df["true_class_probability"].mean())

    multiclass_brier = _multiclass_brier_score(classification_df, class_labels)
    ece_10, reliability_df = _expected_calibration_error(confidence, is_correct, n_bins=10)

    total_errors = int((is_correct == 0).sum())
    abstention_rows: list[dict[str, Any]] = []

    for threshold in ABSTENTION_THRESHOLDS:
        retained_mask = confidence >= threshold
        abstained_mask = ~retained_mask

        retained_rows = int(retained_mask.sum())
        abstained_rows = int(abstained_mask.sum())

        if retained_rows > 0:
            retained_accuracy = float(accuracy_score(y_true[retained_mask], y_pred[retained_mask]))
            retained_macro_f1 = float(
                f1_score(
                    y_true[retained_mask],
                    y_pred[retained_mask],
                    average="macro",
                    labels=class_labels,
                    zero_division=0,
                )
            )
        else:
            retained_accuracy = np.nan
            retained_macro_f1 = np.nan

        if abstained_rows > 0:
            abstained_error_rate = float(1.0 - accuracy_score(y_true[abstained_mask], y_pred[abstained_mask]))
            errors_captured = int((is_correct[abstained_mask] == 0).sum())
        else:
            abstained_error_rate = np.nan
            errors_captured = 0

        error_capture_rate = (errors_captured / total_errors) if total_errors > 0 else np.nan

        abstention_rows.append(
            {
                "confidence_threshold": threshold,
                "retained_rows": retained_rows,
                "retained_fraction": float(retained_rows / len(classification_df)),
                "abstained_rows": abstained_rows,
                "abstained_fraction": float(abstained_rows / len(classification_df)),
                "retained_accuracy": retained_accuracy,
                "retained_macro_f1": retained_macro_f1,
                "delta_accuracy_vs_full": (
                    retained_accuracy - full_accuracy if not np.isnan(retained_accuracy) else np.nan
                ),
                "abstained_error_rate": abstained_error_rate,
                "error_capture_rate": error_capture_rate,
            }
        )

    abstention_df = pd.DataFrame(abstention_rows)

    calibration_summary_df = pd.DataFrame(
        [
            {
                "rows": int(len(classification_df)),
                "accuracy": full_accuracy,
                "macro_f1": full_macro_f1,
                "multiclass_brier_score": multiclass_brier,
                "ece_10": ece_10,
                "mean_confidence": mean_confidence,
                "mean_true_class_probability": mean_true_class_probability,
                "overconfidence_gap": mean_confidence - full_accuracy,
            }
        ]
    )

    return calibration_summary_df, reliability_df, abstention_df


def _save_plots(
    regression_fold_df: pd.DataFrame,
    classification_fold_df: pd.DataFrame,
    regression_by_activity_df: pd.DataFrame,
    classification_per_class_df: pd.DataFrame,
    conformal_by_activity_df: pd.DataFrame,
    conformal_variant_comparison_df: pd.DataFrame,
    residual_diagnostics_df: pd.DataFrame,
    classification_reliability_df: pd.DataFrame,
    classification_abstention_df: pd.DataFrame,
    preferred_conformal_variant: str,
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
    ax.set_title(f"Conformal coverage by activity ({preferred_conformal_variant})")
    ax.set_xlabel("activity")
    ax.set_ylabel("coverage")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "grouped_cv_conformal_coverage_by_activity.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    failure_plot_df = conformal_by_activity_df.sort_values("rows", ascending=False).head(12)
    sns.barplot(data=failure_plot_df, x="activity_label", y="interval_failure_rate", ax=ax, color="tab:red")
    ax.set_title(f"Conformal interval failure rate by activity ({preferred_conformal_variant})")
    ax.set_xlabel("activity")
    ax.set_ylabel("interval failure rate")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(figures_dir / "grouped_cv_conformal_interval_failure_by_activity.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.histplot(residual_diagnostics_df["residual"], bins=50, kde=True, ax=ax, color="tab:gray")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2)
    ax.set_title("Residual distribution for selected regression model")
    ax.set_xlabel("residual (y_true - y_pred)")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(figures_dir / "grouped_cv_regression_residual_distribution.png", dpi=140)
    plt.close(fig)

    if len(conformal_variant_comparison_df) > 1:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(
            data=conformal_variant_comparison_df,
            x="calibration_variant",
            y="weighted_activity_abs_coverage_gap",
            ax=ax,
            color="tab:cyan",
        )
        ax.set_title("Conformal variant comparison: activity-level coverage stability")
        ax.set_xlabel("calibration variant")
        ax.set_ylabel("weighted abs coverage gap")
        fig.tight_layout()
        fig.savefig(figures_dir / "grouped_cv_conformal_variant_activity_gap.png", dpi=140)
        plt.close(fig)

    if not classification_reliability_df.empty:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1.0, label="perfect calibration")
        ax.plot(
            classification_reliability_df["mean_confidence"],
            classification_reliability_df["empirical_accuracy"],
            marker="o",
            color="tab:blue",
            label="model",
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_title("Classification reliability curve")
        ax.set_xlabel("mean confidence")
        ax.set_ylabel("empirical accuracy")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / "grouped_cv_classification_reliability_curve.png", dpi=140)
        plt.close(fig)

    if not classification_abstention_df.empty:
        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(
            classification_abstention_df["confidence_threshold"],
            classification_abstention_df["retained_accuracy"],
            marker="o",
            color="tab:green",
            label="retained accuracy",
        )
        ax1.set_xlabel("confidence threshold")
        ax1.set_ylabel("retained accuracy", color="tab:green")
        ax1.tick_params(axis="y", labelcolor="tab:green")
        ax1.set_ylim(0.0, 1.05)

        ax2 = ax1.twinx()
        ax2.plot(
            classification_abstention_df["confidence_threshold"],
            classification_abstention_df["retained_fraction"],
            marker="s",
            color="tab:orange",
            label="retained fraction",
        )
        ax2.set_ylabel("retained fraction", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
        ax2.set_ylim(0.0, 1.05)

        ax1.set_title("Abstention tradeoff by confidence threshold")
        fig.tight_layout()
        fig.savefig(figures_dir / "grouped_cv_classification_abstention_tradeoff.png", dpi=140)
        plt.close(fig)


def _load_and_validate_model_table(
    table_path: Path,
    required_columns: list[str],
    task_name: str,
) -> pd.DataFrame:
    if not table_path.exists():
        raise FileNotFoundError(f"Missing {task_name} processed model table: {table_path}")

    model_df = pd.read_parquet(table_path).copy()
    model_df = model_df.sort_values(["subject_id", "timestamp_s"]).reset_index(drop=True)

    missing_required = [column for column in required_columns if column not in model_df.columns]
    if missing_required:
        raise ValueError(
            f"Missing required columns in {task_name} processed table ({table_path}): {missing_required}"
        )

    duplicate_count = int(model_df.duplicated(subset=["subject_id", "timestamp_s"]).sum())
    if duplicate_count > 0:
        raise ValueError(f"Found duplicate subject-second rows in {task_name} table ({table_path}): {duplicate_count}")

    return model_df


def _infer_feature_columns(model_df: pd.DataFrame) -> list[str]:
    non_feature_cols = {
        "subject_id",
        "session",
        "timestamp_s",
        "activity_id",
        "activity_label",
        "activity_target",
        "heart_rate_observed_flag",
        "heart_rate_fill_strategy",
    }
    return [
        column
        for column in model_df.columns
        if column not in non_feature_cols and not column.startswith("hr_target_")
    ]


def run_grouped_evaluation(
    regression_processed_path: Path,
    classification_processed_path: Path,
    metrics_dir: Path,
    figures_dir: Path,
    models_dir: Path,
    random_seed: int = RANDOM_SEED,
    alpha: float = ALPHA,
    regression_target_col: str | None = None,
) -> dict[str, Any]:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    preferred_target_col = _read_preferred_target_col(metrics_dir)
    if regression_target_col is None:
        if preferred_target_col is None:
            preferred_setup_path = metrics_dir / "grouped_cv_preferred_setup_summary.csv"
            raise FileNotFoundError(
                "No regression_target_col was provided and preferred setup artifact is missing. "
                "Run scripts/compact_ablation_study.py first, or pass regression_target_col explicitly: "
                f"{preferred_setup_path}"
            )
        regression_target_col = preferred_target_col

    regression_target_col = str(regression_target_col).strip()
    if not regression_target_col:
        raise ValueError("regression_target_col must be a non-empty column name.")

    if preferred_target_col is not None and regression_target_col != preferred_target_col:
        preferred_setup_path = metrics_dir / "grouped_cv_preferred_setup_summary.csv"
        raise ValueError(
            "Requested regression_target_col does not match preferred_target_col from the preferred setup "
            f"artifact ({preferred_target_col}). Requested: {regression_target_col}. Artifact: {preferred_setup_path}"
        )

    regression_required_columns = [
        "subject_id",
        "timestamp_s",
        "activity_target",
        "activity_label",
        "heart_rate_bpm",
        regression_target_col,
    ]
    classification_required_columns = [
        "subject_id",
        "timestamp_s",
        "activity_target",
        "activity_label",
        "heart_rate_bpm",
    ]

    regression_df = _load_and_validate_model_table(
        table_path=regression_processed_path,
        required_columns=regression_required_columns,
        task_name="regression",
    )
    classification_df = _load_and_validate_model_table(
        table_path=classification_processed_path,
        required_columns=classification_required_columns,
        task_name="classification",
    )

    regression_feature_cols = _infer_feature_columns(regression_df)
    classification_feature_cols = _infer_feature_columns(classification_df)

    if not regression_feature_cols:
        raise ValueError("No feature columns detected in regression processed table.")
    if not classification_feature_cols:
        raise ValueError("No feature columns detected in classification processed table.")

    regression_feature_set = set(regression_feature_cols)
    classification_feature_set = set(classification_feature_cols)
    if regression_feature_set != classification_feature_set:
        missing_in_classification = sorted(regression_feature_set.difference(classification_feature_set))
        missing_in_regression = sorted(classification_feature_set.difference(regression_feature_set))
        raise ValueError(
            "Regression and classification feature columns must match for grouped evaluation. "
            f"Missing in classification={missing_in_classification}. "
            f"Missing in regression={missing_in_regression}."
        )

    feature_cols = regression_feature_cols

    x_reg_df = regression_df[feature_cols]
    y_reg = regression_df[regression_target_col]
    groups_reg = regression_df["subject_id"].astype(int)
    meta_reg_df = regression_df[["subject_id", "timestamp_s", "activity_target", "activity_label"]].copy()

    x_cls_df = classification_df[feature_cols]
    y_cls = classification_df["activity_target"].astype(int)
    groups_cls = classification_df["subject_id"].astype(int)
    meta_cls_df = classification_df[["subject_id", "timestamp_s", "activity_target", "activity_label"]].copy()

    label_lookup = (
        classification_df[["activity_target", "activity_label"]]
        .drop_duplicates()
        .set_index("activity_target")["activity_label"]
        .to_dict()
    )
    class_labels = sorted(int(label) for label in label_lookup)

    regression_specs = _build_regression_model_specs(random_seed)
    classification_specs = _build_classification_model_specs(random_seed)

    regression_fold_df, regression_pred_df = _run_grouped_cv_regression(
        x_df=x_reg_df,
        y=y_reg,
        groups=groups_reg,
        meta_df=meta_reg_df,
        model_specs=regression_specs,
    )
    classification_fold_df, classification_pred_df = _run_grouped_cv_classification(
        x_df=x_cls_df,
        y=y_cls,
        groups=groups_cls,
        meta_df=meta_cls_df,
        model_specs=classification_specs,
        class_labels=class_labels,
    )

    regression_summary_df = _summarize_regression_fold_metrics(regression_fold_df)
    classification_summary_df = _summarize_classification_fold_metrics(classification_fold_df)

    regression_fold_df["regression_target_col"] = regression_target_col
    regression_summary_df["regression_target_col"] = regression_target_col

    selected_regression_model, selected_classification_model, selected_models_df = _select_final_models(
        regression_summary_df=regression_summary_df,
        classification_summary_df=classification_summary_df,
    )
    selected_models_df["regression_target_col"] = regression_target_col

    selected_regression_preds_df = regression_pred_df[regression_pred_df["model"] == selected_regression_model].copy()
    selected_classification_preds_df = classification_pred_df[
        classification_pred_df["model"] == selected_classification_model
    ].copy()

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

    (
        conformal_fold_all_df,
        conformal_predictions_all_df,
        conformal_summary_all_df,
        conformal_by_subject_all_df,
        conformal_by_activity_all_df,
    ) = (
        _run_grouped_conformal(
            x_df=x_reg_df,
            y=y_reg,
            groups=groups_reg,
            meta_df=meta_reg_df,
            selected_regression_model=selected_regression_model,
            model_specs=regression_specs,
            alpha=alpha,
        )
    )

    preferred_conformal_variant, conformal_variant_comparison_df = _select_preferred_conformal_variant(
        conformal_summary_df=conformal_summary_all_df,
        conformal_by_activity_df=conformal_by_activity_all_df,
    )

    conformal_fold_df = conformal_fold_all_df[
        conformal_fold_all_df["calibration_variant"] == preferred_conformal_variant
    ].copy()
    conformal_predictions_df = conformal_predictions_all_df[
        conformal_predictions_all_df["calibration_variant"] == preferred_conformal_variant
    ].copy()
    conformal_summary_df = conformal_summary_all_df[
        conformal_summary_all_df["calibration_variant"] == preferred_conformal_variant
    ].copy()
    conformal_by_subject_df = conformal_by_subject_all_df[
        conformal_by_subject_all_df["calibration_variant"] == preferred_conformal_variant
    ].copy()
    conformal_by_activity_df = conformal_by_activity_all_df[
        conformal_by_activity_all_df["calibration_variant"] == preferred_conformal_variant
    ].copy()

    target_coverage = float(conformal_summary_df.iloc[0]["target_coverage"])
    (
        conformal_predictions_df,
        uncertainty_failure_by_activity_df,
        uncertainty_failure_by_subject_df,
        regression_residual_summary_df,
        operating_envelope_by_activity_df,
    ) = _build_uncertainty_failure_diagnostics(
        conformal_predictions_df=conformal_predictions_df,
        target_coverage=target_coverage,
    )

    (
        classification_calibration_summary_df,
        classification_reliability_df,
        classification_abstention_df,
    ) = _classification_confidence_diagnostics(
        classification_df=selected_classification_preds_df,
        class_labels=class_labels,
    )

    outputs_with_target = [
        regression_fold_df,
        regression_summary_df,
        selected_models_df,
        conformal_fold_df,
        conformal_fold_all_df,
        conformal_predictions_df,
        conformal_predictions_all_df,
        conformal_summary_df,
        conformal_summary_all_df,
        conformal_by_subject_df,
        conformal_by_subject_all_df,
        conformal_by_activity_df,
        conformal_by_activity_all_df,
        conformal_variant_comparison_df,
        uncertainty_failure_by_activity_df,
        uncertainty_failure_by_subject_df,
        regression_residual_summary_df,
        operating_envelope_by_activity_df,
        classification_calibration_summary_df,
        classification_reliability_df,
        classification_abstention_df,
    ]
    for output_df in outputs_with_target:
        output_df["regression_target_col"] = regression_target_col

    selected_models_df["preferred_conformal_variant"] = preferred_conformal_variant
    selected_models_df["regression_rows_used"] = int(len(regression_df))
    selected_models_df["classification_rows_used"] = int(len(classification_df))
    conformal_summary_df["preferred_interval_variant"] = 1
    conformal_summary_all_df["preferred_interval_variant"] = (
        conformal_summary_all_df["calibration_variant"] == preferred_conformal_variant
    ).astype(int)

    # Train selected models on full data for optional reuse outside this notebook workflow.
    reg_model_to_save = _fit_if_needed(
        selected_regression_model,
        regression_specs[selected_regression_model],
        x_reg_df,
        y_reg,
    )
    cls_model_to_save = clone(classification_specs[selected_classification_model])
    cls_model_to_save.fit(x_cls_df, y_cls)

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
    conformal_summary_all_df.to_csv(metrics_dir / "grouped_cv_conformal_summary_all_variants.csv", index=False)
    conformal_variant_comparison_df.to_csv(metrics_dir / "grouped_cv_conformal_summary.csv", index=False)
    conformal_variant_comparison_df.to_csv(metrics_dir / "grouped_cv_conformal_variant_comparison.csv", index=False)
    conformal_by_subject_df.to_csv(metrics_dir / "grouped_cv_conformal_by_subject.csv", index=False)
    conformal_by_activity_df.to_csv(metrics_dir / "grouped_cv_conformal_by_activity.csv", index=False)
    conformal_fold_all_df.to_csv(metrics_dir / "grouped_cv_conformal_fold_summary_all_variants.csv", index=False)
    conformal_predictions_all_df.to_csv(metrics_dir / "grouped_cv_conformal_predictions_all_variants.csv", index=False)
    conformal_by_subject_all_df.to_csv(metrics_dir / "grouped_cv_conformal_by_subject_all_variants.csv", index=False)
    conformal_by_activity_all_df.to_csv(metrics_dir / "grouped_cv_conformal_by_activity_all_variants.csv", index=False)

    uncertainty_failure_by_activity_df.to_csv(metrics_dir / "grouped_cv_uncertainty_failure_by_activity.csv", index=False)
    uncertainty_failure_by_subject_df.to_csv(metrics_dir / "grouped_cv_uncertainty_failure_by_subject.csv", index=False)
    regression_residual_summary_df.to_csv(metrics_dir / "grouped_cv_regression_residual_summary.csv", index=False)
    operating_envelope_by_activity_df.to_csv(
        metrics_dir / "grouped_cv_uncertainty_operating_envelope_by_activity.csv",
        index=False,
    )

    classification_calibration_summary_df.to_csv(
        metrics_dir / "grouped_cv_classification_calibration_summary.csv",
        index=False,
    )
    classification_reliability_df.to_csv(
        metrics_dir / "grouped_cv_classification_reliability_by_bin.csv",
        index=False,
    )
    classification_abstention_df.to_csv(
        metrics_dir / "grouped_cv_classification_abstention_summary.csv",
        index=False,
    )

    _save_plots(
        regression_fold_df=regression_fold_df,
        classification_fold_df=classification_fold_df,
        regression_by_activity_df=regression_by_activity_df,
        classification_per_class_df=classification_per_class_df,
        conformal_by_activity_df=conformal_by_activity_df,
        conformal_variant_comparison_df=conformal_variant_comparison_df,
        residual_diagnostics_df=conformal_predictions_df,
        classification_reliability_df=classification_reliability_df,
        classification_abstention_df=classification_abstention_df,
        preferred_conformal_variant=preferred_conformal_variant,
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
        "classification_calibration_summary": classification_calibration_summary_df,
        "classification_reliability": classification_reliability_df,
        "classification_abstention": classification_abstention_df,
        "conformal_fold": conformal_fold_df,
        "conformal_fold_all_variants": conformal_fold_all_df,
        "conformal_predictions": conformal_predictions_df,
        "conformal_predictions_all_variants": conformal_predictions_all_df,
        "conformal_summary": conformal_summary_df,
        "conformal_summary_all_variants": conformal_summary_all_df,
        "conformal_variant_comparison": conformal_variant_comparison_df,
        "conformal_by_subject": conformal_by_subject_df,
        "conformal_by_subject_all_variants": conformal_by_subject_all_df,
        "conformal_by_activity": conformal_by_activity_df,
        "conformal_by_activity_all_variants": conformal_by_activity_all_df,
        "preferred_conformal_variant": preferred_conformal_variant,
        "uncertainty_failure_by_activity": uncertainty_failure_by_activity_df,
        "uncertainty_failure_by_subject": uncertainty_failure_by_subject_df,
        "regression_residual_summary": regression_residual_summary_df,
        "uncertainty_operating_envelope": operating_envelope_by_activity_df,
        "regression_rows_used": int(len(regression_df)),
        "classification_rows_used": int(len(classification_df)),
    }


def _default_paths() -> tuple[Path, Path, Path, Path, Path]:
    repo_root = Path(__file__).resolve().parents[2]
    regression_processed_path = repo_root / "data" / "processed" / "pamap2_model_table_regression.parquet"
    classification_processed_path = repo_root / "data" / "processed" / "pamap2_model_table_classification.parquet"
    metrics_dir = repo_root / "artifacts" / "metrics"
    figures_dir = repo_root / "artifacts" / "figures"
    models_dir = repo_root / "artifacts" / "models"
    return regression_processed_path, classification_processed_path, metrics_dir, figures_dir, models_dir


def main() -> None:
    (
        regression_processed_path,
        classification_processed_path,
        metrics_dir,
        figures_dir,
        models_dir,
    ) = _default_paths()
    results = run_grouped_evaluation(
        regression_processed_path=regression_processed_path,
        classification_processed_path=classification_processed_path,
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
