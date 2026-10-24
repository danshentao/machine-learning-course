param([string]$Config = "configs/ppyoloe_plus_bird_640.yml", [string]$Weights = "", [string]$Split = "val")
$ErrorActionPreference = "Stop"
$root = "E:\机器学习课设\第二代鸟类检测与细粒度分类"
$python = "E:\机器学习课设\自然场景鸟类检测PPYOLOE\.venv\Scripts\python.exe"
$repo = "E:\机器学习课设\自然场景鸟类检测PPYOLOE\third_party\PaddleDetection"
$site = "E:\机器学习课设\自然场景鸟类检测PPYOLOE\.venv\Lib\site-packages"
$env:PATH = "$site\nvidia\cudnn\bin;$site\nvidia\cublas\bin;$site\nvidia\cuda_nvrtc\bin;$site\paddle\libs;$env:PATH"
$env:PADDLE_HOME = "$root\.cache\paddle"
$evalDir = "$root\runs\eval_$Split"
New-Item -ItemType Directory -Force -Path $evalDir | Out-Null
if ($Weights -eq "") { $Weights = "$root\runs\detector\model_final" }
Set-Location $repo
$out = "$evalDir\stdout.log"; $err = "$evalDir\stderr.log"
$configPath = if ([System.IO.Path]::IsPathRooted($Config)) { $Config } else { Join-Path $root $Config }
$argsList = @("tools/eval.py", "-c", $configPath, "--classwise", "--output_eval", $evalDir, "-o", "use_gpu=True", "weights=$Weights")
$proc = Start-Process -FilePath $python -ArgumentList $argsList -WorkingDirectory $repo -Wait -PassThru -WindowStyle Hidden -RedirectStandardOutput $out -RedirectStandardError $err
Get-Content $out,$err -ErrorAction SilentlyContinue | Set-Content -Encoding UTF8 "$evalDir\eval.log"
if ($proc.ExitCode -ne 0) { throw "PP-YOLOE+ evaluation failed: exit=$($proc.ExitCode); see $evalDir\eval.log" }
