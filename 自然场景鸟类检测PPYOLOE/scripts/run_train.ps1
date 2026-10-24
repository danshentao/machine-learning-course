param(
    [ValidateSet('smoke', 'baseline')]
    [string]$Stage = 'smoke'
)
$ErrorActionPreference = 'Stop'
$Project = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Project '.venv\Scripts\python.exe'
$NvidiaRuntime = Join-Path $Project '.venv\Lib\site-packages\nvidia'
$RuntimeBins = Get-ChildItem $NvidiaRuntime -Directory | ForEach-Object {
    Get-ChildItem $_.FullName -Directory -Filter 'bin' | ForEach-Object FullName
}
$env:PATH = ($RuntimeBins -join ';') + ';' + $env:PATH
if (!(Test-Path (Join-Path $Project 'data\prepared\annotations\instances_train.json'))) {
    throw '未找到派生训练数据。请先运行 scripts\run_prepare.ps1。'
}
Set-Location $Project
if ($Stage -eq 'smoke') {
    $Config = 'configs\ppyoloe_s_bird_smoke.yml'
    $Run = 'runs\smoke_640'
} else {
    $Config = 'configs\ppyoloe_s_bird_640.yml'
    $Run = 'runs\baseline_640'
}
New-Item -ItemType Directory -Force -Path $Run | Out-Null
$Log = Join-Path $Run 'training_console.log'
$TrainArgs = @('third_party\PaddleDetection\tools\train.py', '-c', $Config, '--eval', '-o', "save_dir=$Run")
& $Python @TrainArgs *>&1 | Tee-Object -FilePath $Log
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
