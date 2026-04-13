from __future__ import annotations

from pathlib import Path
import sys

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


def _default_paths() -> tuple[Path, Path, Path, Path]:
    processed_path = REPO_ROOT / "data" / "processed" / "pamap2_model_table.parquet"
    metrics_dir = REPO_ROOT / "artifacts" / "metrics"
    figures_dir = REPO_ROOT / "artifacts" / "figures"
    models_dir = REPO_ROOT / "artifacts" / "models"
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


__all__ = [
    "ALPHA",
    "RANDOM_SEED",
    "build_regression_model_specs",
    "run_grouped_evaluation",
    "run_grouped_regression_cv",
]
