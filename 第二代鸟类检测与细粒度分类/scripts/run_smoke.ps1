$ErrorActionPreference = "Stop"
$root = "E:\机器学习课设\第二代鸟类检测与细粒度分类"
$python = "E:\机器学习课设\自然场景鸟类检测PPYOLOE\.venv\Scripts\python.exe"
. "$root\scripts\set_runtime_env.ps1"
& $python "$root\scripts\prepare_smoke_annotations.py"
& $python "$root\scripts\download_weights.py"
if ($LASTEXITCODE -ne 0) { throw "pretrained checkpoint preparation failed" }
powershell -ExecutionPolicy Bypass -File "$root\scripts\train_detector.ps1" -Config "configs/ppyoloe_plus_bird_smoke.yml" -Device gpu
if ($LASTEXITCODE -ne 0) { throw "PP-YOLOE+ smoke training failed" }
powershell -ExecutionPolicy Bypass -File "$root\scripts\eval_detector.ps1" -Config "configs/ppyoloe_plus_bird_smoke.yml" -Weights "$root\runs\smoke_detector\model_final" -Split "smoke"
if ($LASTEXITCODE -ne 0) { throw "PP-YOLOE+ smoke validation failed" }
& $python "$root\scripts\train_classifier.py" --model resnet18 --epochs 1 --batch-size 2 --max-steps 2 --device cpu --run-name classifier_resnet_smoke
if ($LASTEXITCODE -ne 0) { throw "CNN classifier smoke failed" }
& $python "$root\scripts\train_classifier.py" --model bilinear --epochs 1 --batch-size 1 --max-steps 1 --device cpu --run-name classifier_bilinear_smoke
if ($LASTEXITCODE -ne 0) { throw "Bilinear classifier smoke failed" }
& $python "$root\scripts\infer_two_stage.py" --image-dir "$root\data\smoke_infer_inputs" --detector-config "$root\configs\ppyoloe_plus_bird_smoke.yml" --detector-weights "$root\runs\smoke_detector\model_final" --classifier-weights "$root\runs\classifier_resnet_smoke\model_final.pdparams" --classifier resnet18 --output-dir "$root\runs\two_stage_smoke" --device gpu
if ($LASTEXITCODE -ne 0) { throw "Two-stage inference smoke failed" }
& $python "$root\scripts\evaluate_two_stage.py" --predictions "$root\runs\two_stage_smoke\predictions.json" --split val --output-dir "$root\runs\two_stage_smoke\eval"
if ($LASTEXITCODE -ne 0) { throw "Two-stage evaluation smoke failed" }
