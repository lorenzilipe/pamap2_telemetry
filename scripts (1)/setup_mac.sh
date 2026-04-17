#!/usr/bin/env bash
set -euo pipefail

ENV_PATH="${1:-$HOME/.venvs/pamap2_telemetry}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REQUIREMENTS_PATH="$REPO_ROOT/requirements.txt"
REPO_VENV_PATH="$REPO_ROOT/.venv"
WIN_BACKUP_PATH="$REPO_ROOT/.venv_win"

if [[ ! -f "$REQUIREMENTS_PATH" ]]; then
  echo "requirements.txt not found at $REQUIREMENTS_PATH"
  exit 1
fi

# If a synced Windows environment is present as .venv, rename it once so macOS does not use it.
if [[ -d "$REPO_VENV_PATH" ]]; then
  PYVENV_CFG="$REPO_VENV_PATH/pyvenv.cfg"
  LOOKS_WINDOWS=false
  HAS_WINDOWS_LAYOUT=false

  if [[ -f "$PYVENV_CFG" ]]; then
    CFG_TEXT="$(cat "$PYVENV_CFG")"
    if [[ "$CFG_TEXT" == *"\\Scripts\\python.exe"* || "$CFG_TEXT" == *":\\"* ]]; then
      LOOKS_WINDOWS=true
    fi
  fi

  if [[ -d "$REPO_VENV_PATH/Scripts" ]]; then
    HAS_WINDOWS_LAYOUT=true
  fi

  if { [[ "$LOOKS_WINDOWS" == true ]] || [[ "$HAS_WINDOWS_LAYOUT" == true ]]; } && [[ ! -e "$WIN_BACKUP_PATH" ]]; then
    mv "$REPO_VENV_PATH" "$WIN_BACKUP_PATH"
    echo "Renamed synced Windows venv: .venv -> .venv_win"
  fi
fi

mkdir -p "$(dirname "$ENV_PATH")"

if [[ ! -d "$ENV_PATH" ]]; then
  if command -v python3.11 >/dev/null 2>&1; then
    python3.11 -m venv "$ENV_PATH"
  else
    python3 -m venv "$ENV_PATH"
  fi
  echo "Created virtual environment at: $ENV_PATH"
else
  echo "Using existing virtual environment at: $ENV_PATH"
fi

PY_EXE="$ENV_PATH/bin/python"
if [[ ! -x "$PY_EXE" ]]; then
  echo "Python executable not found in environment: $PY_EXE"
  exit 1
fi

"$PY_EXE" -m pip install --upgrade pip
"$PY_EXE" -m pip install -r "$REQUIREMENTS_PATH"

# Register a stable kernel name used across both operating systems.
"$PY_EXE" -m ipykernel install --user --name "pamap2-telemetry" --display-name "Python (pamap2-telemetry)"

echo ""
echo "Setup complete."
echo "1) In VS Code, open a notebook."
echo "2) Select kernel: Python (pamap2-telemetry)."
echo "3) Keep using this same kernel name on both Windows and macOS."
