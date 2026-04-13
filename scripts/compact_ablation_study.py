from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pamap2_telemetry.ablation import run_compact_ablation_study  # noqa: E402
from pamap2_telemetry.evaluate import ALPHA, RANDOM_SEED  # noqa: E402


def main() -> None:
    results = run_compact_ablation_study(
        repo_root=REPO_ROOT,
        random_seed=RANDOM_SEED,
        alpha=ALPHA,
    )

    preferred_df = results["preferred_setup"]
    print("Compact ablation study complete.")
    print(preferred_df.to_string(index=False))


if __name__ == "__main__":
    main()
