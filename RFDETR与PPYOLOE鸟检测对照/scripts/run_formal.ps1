$ErrorActionPreference = 'Stop'
$root = 'E:\机器学习课设\RFDETR与PPYOLOE鸟检测对照'
$env:PYTHONPATH = "$root\runtime\local_packages;$root\runtime\shareddeps"
$env:TEMP = "$root\runtime\tmp"
$env:TMP = $env:TEMP
$env:HF_HOME = "$root\cache\huggingface"
$env:RF_HOME = "$root\weights\cache"
$env:ROBOFLOW_HOME = "$root\weights\cache"
New-Item -ItemType Directory -Force -Path $env:TEMP,$env:HF_HOME,$env:RF_HOME | Out-Null
$python = "$root\runtime\python311-embed\python.exe"
& $python "$root\scripts\train_rfdetr.py" *> "$root\logs\formal_train.log"
if ($LASTEXITCODE -ne 0) { throw "RF-DETR训练失败，查看 logs\formal_train.log" }

