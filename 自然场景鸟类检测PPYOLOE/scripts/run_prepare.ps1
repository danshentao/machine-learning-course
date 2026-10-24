param(
    [switch]$SkipCrops
)
$ErrorActionPreference = 'Stop'
$Project = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Project '.venv\Scripts\python.exe'
Set-Location $Project
if ($SkipCrops) {
    & $Python scripts\audit_data.py --skip-crops
} else {
    & $Python scripts\audit_data.py
}
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python scripts\prepare_data.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python scripts\make_review_galleries.py
exit $LASTEXITCODE
