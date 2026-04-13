from __future__ import annotations

import pandas as pd

from .ablation import (
    ACC_MAG_COLUMNS,
    BASE_SIGNAL_COLUMNS,
    GYRO_MAG_COLUMNS,
    TARGET_VARIANTS,
    _build_feature_inventory,
    _build_features_and_targets,
)


def build_features_and_targets(per_second_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Create lag/rolling features and the three regression target variants."""
    return _build_features_and_targets(per_second_df)


def build_feature_inventory(feature_cols: list[str], preferred_feature_set: str) -> pd.DataFrame:
    """Return a compact feature catalog for reporting and experiment records."""
    return _build_feature_inventory(feature_cols, preferred_feature_set)


__all__ = [
    "ACC_MAG_COLUMNS",
    "BASE_SIGNAL_COLUMNS",
    "GYRO_MAG_COLUMNS",
    "TARGET_VARIANTS",
    "build_feature_inventory",
    "build_features_and_targets",
]
