param(
    [string]$EvtxRoot = "evtx_samples",
    [string]$DatasetOut = "results/threat_agent_data.json",
    [int]$Seed = 42,
    [int]$TabularEpisodes = 3000,
    [int]$DeepEpisodes = 2000,
    [int]$EvalEpisodes = 100
)

$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RootDir

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Command
    )

    & $Command[0] $Command[1..($Command.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $($Command -join ' ')"
    }
}

Write-Host "[prep] Ensure output folders"
New-Item -ItemType Directory -Force -Path "results" | Out-Null
New-Item -ItemType Directory -Force -Path "checkpoints" | Out-Null

$env:PIPENV_VENV_IN_PROJECT = "1"

if (-not (Test-Path ".venv")) {
    Write-Host "[prep] .venv not found. Running setup_pipenv.ps1 first."
    & "$RootDir\setup_pipenv.ps1"
    if ($LASTEXITCODE -ne 0) {
        throw "setup_pipenv.ps1 failed. Please fix environment setup first."
    }
}

Write-Host "[1/2] Build dataset -> $DatasetOut"
Invoke-Checked @(
    "python", "-m", "pipenv", "run", "python", "threat_agent/build_dataset.py",
    "--evtx-root", "$EvtxRoot",
    "--evtx-lib-dir", "$EvtxRoot/EVTX_ATT&CK_Metadata",
    "-o", "$DatasetOut"
)

Write-Host "[2/2] Run all comparison experiments"
Invoke-Checked @(
    "python", "-m", "pipenv", "run", "python", "-m", "threat_agent.experiment_compare",
    "--dataset", "$DatasetOut",
    "--seed", "$Seed",
    "--tabular-episodes", "$TabularEpisodes",
    "--deep-episodes", "$DeepEpisodes",
    "--eval-episodes", "$EvalEpisodes"
)

Write-Host "Done."
Write-Host "Summary JSON: results/compare_summary.json"
Write-Host "Summary CSV : results/compare_summary.csv"
