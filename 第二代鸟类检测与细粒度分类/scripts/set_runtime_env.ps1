$root = "E:\机器学习课设\第二代鸟类检测与细粒度分类"
$venv = "E:\机器学习课设\自然场景鸟类检测PPYOLOE\.venv"
$site = "$venv\Lib\site-packages"
$dllDirs = @("$site\nvidia\cudnn\bin", "$site\nvidia\cublas\bin", "$site\nvidia\cuda_nvrtc\bin", "$site\paddle\libs")
$env:PATH = (($dllDirs -join ";") + ";" + $env:PATH)
$env:PADDLE_HOME = "$root\.cache\paddle"
$env:XDG_CACHE_HOME = "$root\.cache"
New-Item -ItemType Directory -Force -Path "$root\.cache\paddle" | Out-Null
