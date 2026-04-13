from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pamap2_telemetry.experiment_records import write_selected_model_records  # noqa: E402


def main() -> None:
    written_paths = write_selected_model_records(repo_root=REPO_ROOT)
    print("Wrote experiment records:")
    for path in written_paths:
        print(path.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
