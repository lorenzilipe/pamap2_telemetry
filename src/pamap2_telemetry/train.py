from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .evaluate import (
    RANDOM_SEED,
    _align_class_probabilities,
    _build_classification_model_specs,
    _build_regression_model_specs,
    _fit_if_needed,
    _predict_regression,
)


def build_regression_model_specs(random_seed: int = RANDOM_SEED) -> dict[str, Any]:
    """Return regression model specs with preprocessing + model pipelines."""
    return _build_regression_model_specs(random_seed)


def build_classification_model_specs(random_seed: int = RANDOM_SEED) -> dict[str, Any]:
    """Return classification model specs with preprocessing + model pipelines."""
    return _build_classification_model_specs(random_seed)


def fit_if_needed(model_name: str, model_obj: Any, x_df: pd.DataFrame, y: pd.Series) -> Any:
    """Fit a model unless it is a no-fit baseline."""
    return _fit_if_needed(model_name, model_obj, x_df, y)


def predict_regression(model_name: str, fitted_model: Any, x_df: pd.DataFrame) -> np.ndarray:
    """Predict for regression tasks, including persistence baseline behavior."""
    return _predict_regression(model_name, fitted_model, x_df)


def align_class_probabilities(
    fitted_model: Any,
    x_df: pd.DataFrame,
    class_labels: list[int],
) -> tuple[np.ndarray, list[str]]:
    """Align class probabilities to a stable class label ordering."""
    return _align_class_probabilities(fitted_model, x_df, class_labels)


__all__ = [
    "align_class_probabilities",
    "build_classification_model_specs",
    "build_regression_model_specs",
    "fit_if_needed",
    "predict_regression",
]
