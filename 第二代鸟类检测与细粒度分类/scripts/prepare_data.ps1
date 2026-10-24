$ErrorActionPreference = "Stop"
$root = "E:\机器学习课设\第二代鸟类检测与细粒度分类"
$python = "E:\机器学习课设\自然场景鸟类检测PPYOLOE\.venv\Scripts\python.exe"
. "$root\scripts\set_runtime_env.ps1"
Set-Location $root
& $python "$root\scripts\prepare_data.py"
if ($LASTEXITCODE -ne 0) { throw "Data preparation failed: exit=$LASTEXITCODE" }
