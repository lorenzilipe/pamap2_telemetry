from __future__ import annotations

from pathlib import Path

import pandas as pd

from .ablation import (
    FILL_STRATEGY_LABELS,
    FILL_STRATEGY_ORDER,
    RAW_REQUIRED_COLUMNS,
    _apply_fill_strategy,
    _default_paths,
    _load_protocol_per_second_prefill,
)


def default_paths(repo_root: Path) -> dict[str, Path]:
    """Build canonical repository paths used by the lean pipeline."""
    return _default_paths(repo_root)


def load_protocol_per_second_prefill(paths: dict[str, Path]) -> pd.DataFrame:
    """Load and aggregate PAMAP2 Protocol data to 1-second rows before HR fill."""
    return _load_protocol_per_second_prefill(paths)


def apply_fill_strategy(prefill_df: pd.DataFrame, fill_strategy: str) -> pd.DataFrame:
    """Apply a documented heart-rate fill strategy subject-locally."""
    return _apply_fill_strategy(prefill_df, fill_strategy)


__all__ = [
    "FILL_STRATEGY_LABELS",
    "FILL_STRATEGY_ORDER",
    "RAW_REQUIRED_COLUMNS",
    "apply_fill_strategy",
    "default_paths",
    "load_protocol_per_second_prefill",
]
