from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pamap2_telemetry.evaluate import (  # noqa: E402
    ALPHA,
    RANDOM_SEED,
    build_regression_model_specs,
    run_grouped_evaluation,
    run_grouped_regression_cv,
)


def _default_paths() -> tuple[Path, Path, Path, Path, Path]:
    regression_processed_path = REPO_ROOT / "data" / "processed" / "pamap2_model_table_regression.parquet"
    classification_processed_path = (
        REPO_ROOT / "data" / "processed" / "pamap2_model_table_classification.parquet"
    )
    metrics_dir = REPO_ROOT / "artifacts" / "metrics"
    figures_dir = REPO_ROOT / "artifacts" / "figures"
    models_dir = REPO_ROOT / "artifacts" / "models"
    return regression_processed_path, classification_processed_path, metrics_dir, figures_dir, models_dir


def _load_preferred_regression_target(metrics_dir: Path) -> str:
    preferred_setup_path = metrics_dir / "grouped_cv_preferred_setup_summary.csv"
    if not preferred_setup_path.exists():
        raise FileNotFoundError(
            "Missing preferred setup artifact. Run scripts/compact_ablation_study.py first: "
            f"{preferred_setup_path}"
        )

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


def main() -> None:
    (
        regression_processed_path,
        classification_processed_path,
        metrics_dir,
        figures_dir,
        models_dir,
    ) = _default_paths()
    preferred_target_col = _load_preferred_regression_target(metrics_dir)

    results = run_grouped_evaluation(
        regression_processed_path=regression_processed_path,
        classification_processed_path=classification_processed_path,
        metrics_dir=metrics_dir,
        figures_dir=figures_dir,
        models_dir=models_dir,
        random_seed=RANDOM_SEED,
        alpha=ALPHA,
        regression_target_col=preferred_target_col,
    )

    selected_models_df = results["selected_models"]
    print("Grouped evaluation complete.")
    print(f"Regression target: {preferred_target_col}")
    print(selected_models_df.to_string(index=False))


if __name__ == "__main__":
    main()


__all__ = [
    "ALPHA",
    "RANDOM_SEED",
    "build_regression_model_specs",
    "run_grouped_evaluation",
    "run_grouped_regression_cv",
]
