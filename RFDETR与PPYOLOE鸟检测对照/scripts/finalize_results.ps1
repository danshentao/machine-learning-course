$ErrorActionPreference = 'Stop'
$root = 'E:\机器学习课设\RFDETR与PPYOLOE鸟检测对照'
$rfpy = "$root\runtime\python311-embed\python.exe"
$paddlePy = "$root\runtime\python311-embed\python.exe"
if (-not (Test-Path -LiteralPath $paddlePy)) { throw "未找到本实验E盘内嵌Python：$paddlePy" }
$env:PYTHONPATH = "$root\runtime\local_packages;$root\runtime\shareddeps;E:\机器学习课设\自然场景鸟类检测PPYOLOE\.venv\Lib\site-packages;E:\机器学习课设\自然场景鸟类检测PPYOLOE\third_party\PaddleDetection"
$env:PATH = "E:\机器学习课设\自然场景鸟类检测PPYOLOE\.venv\Scripts;E:\机器学习课设\自然场景鸟类检测PPYOLOE\.venv\Lib\site-packages\paddle\libs;$env:PATH"
$env:TEMP = "$root\runtime\tmp"; $env:TMP = $env:TEMP
$env:HF_HOME = "$root\cache\huggingface"; $env:RF_HOME = "$root\weights\cache"; $env:ROBOFLOW_HOME = "$root\weights\cache"
$best = "$root\runs\formal\checkpoint_best_total.pth"
$final = "$root\runs\formal\last_ema.pth"
if (-not (Test-Path -LiteralPath $best)) { throw "没有训练最佳权重：$best" }
if (-not (Test-Path -LiteralPath $final)) { throw "没有训练最终EMA权重：$final" }
Copy-Item -LiteralPath $best -Destination "$root\weights\rfdetr_small_bird_best.pth" -Force
Copy-Item -LiteralPath $final -Destination "$root\weights\rfdetr_small_bird_final_ema.pth" -Force
$env:PADDLE_HOME = "$root\cache\paddle"
& $paddlePy "$root\scripts\infer_ppyoloe.py" --split fixed_val --device cpu *> "$root\logs\ppyoloe_fixed_val.log"
if ($LASTEXITCODE -ne 0) { throw 'PP-YOLOE+固定验证推理失败' }
& $paddlePy "$root\scripts\infer_ppyoloe.py" --split external80 --device cpu *> "$root\logs\ppyoloe_external80.log"
if ($LASTEXITCODE -ne 0) { throw 'PP-YOLOE+外来图推理失败' }
& $rfpy "$root\scripts\infer_rfdetr.py" --split fixed_val --checkpoint "$root\weights\rfdetr_small_bird_best.pth" *> "$root\logs\rfdetr_fixed_val.log"
if ($LASTEXITCODE -ne 0) { throw 'RF-DETR固定验证推理失败' }
& $rfpy "$root\scripts\infer_rfdetr.py" --split external80 --checkpoint "$root\weights\rfdetr_small_bird_best.pth" *> "$root\logs\rfdetr_external80.log"
if ($LASTEXITCODE -ne 0) { throw 'RF-DETR外来图推理失败' }
& $rfpy "$root\scripts\evaluate_and_build_reports.py" *> "$root\logs\evaluation.log"
if ($LASTEXITCODE -ne 0) { throw '评估及图册生成失败' }
& $rfpy "$root\scripts\write_provenance.py"
if ($LASTEXITCODE -ne 0) { throw '实验来源清单生成失败' }
Write-Host "评估完成；打开 $root\reports\index.html"


