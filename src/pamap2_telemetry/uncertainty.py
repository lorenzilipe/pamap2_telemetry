from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .evaluate import (
    MIN_ACTIVITY_CALIBRATION_ROWS,
    _build_uncertainty_failure_diagnostics,
    _conformal_quantile,
    _run_grouped_conformal,
    _select_preferred_conformal_variant,
)


def conformal_quantile(abs_residuals: np.ndarray, target_coverage: float) -> float:
    """Compute the split conformal absolute-residual quantile."""
    return _conformal_quantile(abs_residuals, target_coverage)


def run_grouped_conformal(
    x_df: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    meta_df: pd.DataFrame,
    selected_regression_model: str,
    model_specs: dict[str, Any],
    alpha: float,
    min_activity_calibration_rows: int = MIN_ACTIVITY_CALIBRATION_ROWS,
):
    """Run grouped split conformal with optional activity-conditioned margins."""
    return _run_grouped_conformal(
        x_df=x_df,
        y=y,
        groups=groups,
        meta_df=meta_df,
        selected_regression_model=selected_regression_model,
        model_specs=model_specs,
        alpha=alpha,
        min_activity_calibration_rows=min_activity_calibration_rows,
    )


def select_preferred_conformal_variant(
    conformal_summary_df: pd.DataFrame,
    conformal_by_activity_df: pd.DataFrame,
) -> tuple[str, pd.DataFrame]:
    """Select the preferred conformal variant by coverage-width tradeoff."""
    return _select_preferred_conformal_variant(conformal_summary_df, conformal_by_activity_df)


def build_uncertainty_failure_diagnostics(
    conformal_predictions_df: pd.DataFrame,
    target_coverage: float,
):
    """Compute residual and interval failure diagnostics tables."""
    return _build_uncertainty_failure_diagnostics(conformal_predictions_df, target_coverage)


__all__ = [
    "build_uncertainty_failure_diagnostics",
    "conformal_quantile",
    "run_grouped_conformal",
    "select_preferred_conformal_variant",
]
