param(
    [string]$InputEvents = "events.ndjson",
    [string]$WeakLabeledOut = "results/events_weak_labeled.ndjson",
    [string]$StreamOut = "results/stream_events.ndjson",
    [ValidateSet("source", "event")][string]$SplitMode = "source",
    [string]$SplitRatio = "0.7,0.15,0.15",
    [int]$Seed = 42,
    [int]$TabularEpisodes = 3000,
    [int]$DeepEpisodes = 1500,
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

Write-Host "[1/3] Weak label events -> $WeakLabeledOut"
Invoke-Checked @(
    "python", "-m", "pipenv", "run", "python", "-m", "threat_agent.stream_labeler",
    "--input", "$InputEvents",
    "--output", "$WeakLabeledOut",
    "--summary-json", "results/events_weak_label_summary.json"
)

Write-Host "[2/3] Build stream episodes -> $StreamOut"
Invoke-Checked @(
    "python", "-m", "pipenv", "run", "python", "-m", "threat_agent.stream_builder",
    "--input", "$WeakLabeledOut",
    "--output", "$StreamOut",
    "--summary-json", "results/stream_summary.json",
    "--split-mode", "$SplitMode",
    "--split-ratio", "$SplitRatio",
    "--seed", "$Seed"
)

Write-Host "[3/3] Run stream comparison experiments"
Invoke-Checked @(
    "python", "-m", "pipenv", "run", "python", "-m", "threat_agent.stream_experiment_compare",
    "--stream-data", "$StreamOut",
    "--seed", "$Seed",
    "--tabular-episodes", "$TabularEpisodes",
    "--deep-episodes", "$DeepEpisodes",
    "--eval-episodes", "$EvalEpisodes"
)

Write-Host "Done."
Write-Host "Summary JSON: results/stream_compare_summary.json"
Write-Host "Summary CSV : results/stream_compare_summary.csv"
