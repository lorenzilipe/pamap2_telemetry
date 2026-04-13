from __future__ import annotations

import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut


def validate_subject_second_uniqueness(df: pd.DataFrame) -> None:
    """Raise if duplicated subject-second rows are found."""
    duplicate_count = int(df.duplicated(subset=["subject_id", "timestamp_s"]).sum())
    if duplicate_count > 0:
        raise ValueError(f"Found duplicate subject-second rows: {duplicate_count}")


def iter_leave_one_subject_out(groups: pd.Series):
    """Yield LOSO split indices for subject-grouped evaluation."""
    logo = LeaveOneGroupOut()
    dummy = groups.to_numpy()
    for train_idx, test_idx in logo.split(dummy, dummy, groups=dummy):
        yield train_idx, test_idx


__all__ = [
    "iter_leave_one_subject_out",
    "validate_subject_second_uniqueness",
]
