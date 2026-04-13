Param(
    [string]$EnvPath = "$env:USERPROFILE\.venvs\pamap2_telemetry"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Resolve repository root from script location.
$RepoRoot = Split-Path -Parent $PSScriptRoot
$RequirementsPath = Join-Path $RepoRoot "requirements.txt"
$RepoVenvPath = Join-Path $RepoRoot ".venv"
$MacBackupPath = Join-Path $RepoRoot ".venv_mac"

if (-not (Test-Path $RequirementsPath)) {
    throw "requirements.txt not found at $RequirementsPath"
}

# If a synced macOS environment is present as .venv, rename it once so Windows does not use it.
if (Test-Path $RepoVenvPath) {
    $PyVenvCfg = Join-Path $RepoVenvPath "pyvenv.cfg"
    $LooksMac = $false

    if (Test-Path $PyVenvCfg) {
        $CfgText = Get-Content $PyVenvCfg -Raw
        if ($CfgText -match "/Library/Frameworks/Python.framework" -or $CfgText -match "/usr/local/bin/python3") {
            $LooksMac = $true
        }
    }

    $HasMacLayout = Test-Path (Join-Path $RepoVenvPath "bin")

    if (($LooksMac -or $HasMacLayout) -and -not (Test-Path $MacBackupPath)) {
        Rename-Item -Path $RepoVenvPath -NewName ".venv_mac"
        Write-Host "Renamed synced macOS venv: .venv -> .venv_mac"
    }
}

# Create or reuse a local machine-specific environment outside the synced repo.
$EnvParent = Split-Path -Parent $EnvPath
if (-not (Test-Path $EnvParent)) {
    New-Item -ItemType Directory -Path $EnvParent -Force | Out-Null
}

if (-not (Test-Path $EnvPath)) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        py -3.11 -m venv "$EnvPath"
    } else {
        python -m venv "$EnvPath"
    }
    Write-Host "Created virtual environment at: $EnvPath"
} else {
    Write-Host "Using existing virtual environment at: $EnvPath"
}

$PyExe = Join-Path $EnvPath "Scripts\python.exe"
if (-not (Test-Path $PyExe)) {
    throw "Python executable not found in environment: $PyExe"
}

& $PyExe -m pip install --upgrade pip
& $PyExe -m pip install -r $RequirementsPath

# Register a stable kernel name used across both operating systems.
& $PyExe -m ipykernel install --user --name "pamap2-telemetry" --display-name "Python (pamap2-telemetry)"

Write-Host ""
Write-Host "Setup complete."
Write-Host "1) In VS Code, open a notebook."
Write-Host "2) Select kernel: Python (pamap2-telemetry)."
Write-Host "3) Keep using this same kernel name on both Windows and macOS."
