param([string]$Config = "configs/ppyoloe_plus_bird_640.yml", [string]$Device = "gpu")
$ErrorActionPreference = "Stop"
$root = "E:\机器学习课设\第二代鸟类检测与细粒度分类"
$python = "E:\机器学习课设\自然场景鸟类检测PPYOLOE\.venv\Scripts\python.exe"
$repo = "E:\机器学习课设\自然场景鸟类检测PPYOLOE\third_party\PaddleDetection"
$site = "E:\机器学习课设\自然场景鸟类检测PPYOLOE\.venv\Lib\site-packages"
$env:PATH = "$site\nvidia\cudnn\bin;$site\nvidia\cublas\bin;$site\nvidia\cuda_nvrtc\bin;$site\paddle\libs;$env:PATH"
$env:PADDLE_HOME = "$root\.cache\paddle"
$env:XDG_CACHE_HOME = "$root\.cache"
New-Item -ItemType Directory -Force -Path "$root\.cache\paddle","$root\runs\detector" | Out-Null
Set-Location $repo
if ($Device -eq "gpu") { $deviceOption = "use_gpu=True" } else { $deviceOption = "use_gpu=False" }
$stdout = "$root\runs\detector\train.stdout.log"
$stderr = "$root\runs\detector\train.stderr.log"
$argsList = @("tools/train.py", "-c", "$root\$Config", "--enable_ce=True", "-o", $deviceOption)
$process = Start-Process -FilePath $python -ArgumentList $argsList -WorkingDirectory $repo -Wait -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr
Get-Content $stdout,$stderr -ErrorAction SilentlyContinue | Set-Content -Encoding UTF8 "$root\runs\detector\train.log"
if ($process.ExitCode -ne 0) { throw "PP-YOLOE+ training failed: exit=$($process.ExitCode); see runs/detector/train.log" }
