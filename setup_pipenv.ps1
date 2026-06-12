$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RootDir

# Use module invocation so PATH does not need pipenv.exe.
$PipenvCmd = @("python", "-m", "pipenv")
try {
    & $PipenvCmd[0] $PipenvCmd[1] $PipenvCmd[2] --version | Out-Null
} catch {
    Write-Host "pipenv module is not available in current Python."
    Write-Host "Install once: python -m pip install --user pipenv"
    exit 1
}

$env:PIPENV_VENV_IN_PROJECT = "1"
$VenvPath = Join-Path $RootDir ".venv"
$VirtualenvCache = Join-Path $env:LOCALAPPDATA "pypa\virtualenv"
$DistutilsPthCandidates = @(
    (Join-Path $RootDir ".venv\Lib\site-packages\distutils-precedence.pth"),
    (Join-Path $RootDir ".venv\lib\site-packages\distutils-precedence.pth")
)

if (Test-Path $VenvPath) {
    Write-Host "[prep] Removing existing .venv (clean rebuild)"
    Remove-Item -Recurse -Force $VenvPath
}

Write-Host "[1/2] Creating pipenv virtual environment"
try {
    & $PipenvCmd[0] $PipenvCmd[1] $PipenvCmd[2] --clear | Out-Null
    & $PipenvCmd[0] $PipenvCmd[1] $PipenvCmd[2] --python 3.10 | Out-Null
} catch {
    Write-Host "[warn] Virtualenv creation failed once. Clearing virtualenv wheel cache and retrying."
    if (Test-Path $VirtualenvCache) {
        Remove-Item -Recurse -Force $VirtualenvCache
    }
    & $PipenvCmd[0] $PipenvCmd[1] $PipenvCmd[2] --clear | Out-Null
    & $PipenvCmd[0] $PipenvCmd[1] $PipenvCmd[2] --python 3.10 | Out-Null
}

Write-Host "[2/2] Installing dependencies from Pipfile"
& $PipenvCmd[0] $PipenvCmd[1] $PipenvCmd[2] install

foreach ($pth in $DistutilsPthCandidates) {
    if (Test-Path $pth) {
        Remove-Item $pth -Force
    }
}

Write-Host "[repair] Installing setuptools cleanly"
& $PipenvCmd[0] $PipenvCmd[1] $PipenvCmd[2] run python -m pip install --upgrade setuptools | Out-Null

Write-Host "Done. Virtual env location: $RootDir\.venv"
& $PipenvCmd[0] $PipenvCmd[1] $PipenvCmd[2] run python -c "import sys, setuptools, _distutils_hack; print('Venv Python:', sys.executable); print('setuptools:', setuptools.__version__)"
