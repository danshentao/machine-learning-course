$ErrorActionPreference = "Stop"
$root = "E:\机器学习课设\第二代鸟类检测与细粒度分类"
$first = "E:\机器学习课设\自然场景鸟类检测PPYOLOE"
$python = "E:\机器学习课设\自然场景鸟类检测PPYOLOE\.venv\Scripts\python.exe"
. "$root\scripts\set_runtime_env.ps1"
$valImages = "$first\data\prepared\images\val"
$detector = "$root\runs\detector\best_model_ap50"
$config = "$root\configs\ppyoloe_plus_bird_640.yml"
$resnetDir = "$root\runs\two_stage_resnet_fixed_val"
$bilinearDir = "$root\runs\two_stage_bilinear_fixed_val"
if (-not (Test-Path -LiteralPath "$detector.pdparams")) { throw "Best detector checkpoint missing: $detector.pdparams" }
& $python "$root\scripts\infer_two_stage.py" --image-dir $valImages --detector-config $config --detector-weights $detector --classifier-weights "$root\runs\classifier_resnet_v2\model_final.pdparams" --classifier resnet18 --output-dir $resnetDir --device gpu
if ($LASTEXITCODE -ne 0) { throw "ResNet end-to-end inference failed" }
& $python "$root\scripts\infer_two_stage.py" --image-dir $valImages --detector-config $config --detector-weights $detector --classifier-weights "$root\runs\classifier_bilinear_v2\model_final.pdparams" --classifier bilinear --proposals-json "$resnetDir\proposals.json" --output-dir $bilinearDir --device gpu
if ($LASTEXITCODE -ne 0) { throw "Bilinear end-to-end inference failed" }
& $python "$root\scripts\evaluate_two_stage.py" --predictions "$resnetDir\predictions.json" --split val --output-dir "$resnetDir\eval"
if ($LASTEXITCODE -ne 0) { throw "ResNet end-to-end evaluation failed" }
& $python "$root\scripts\evaluate_two_stage.py" --predictions "$bilinearDir\predictions.json" --split val --output-dir "$bilinearDir\eval"
if ($LASTEXITCODE -ne 0) { throw "Bilinear end-to-end evaluation failed" }
& $python "$root\scripts\build_comparison_gallery.py" --gen2-resnet "$resnetDir\predictions.json" --gen2-bilinear "$bilinearDir\predictions.json" --output "$root\reports\fixed_val_comparison_gallery.html" --title "固定验证集：第一代与第二代端到端对照"
if ($LASTEXITCODE -ne 0) { throw "Comparison gallery generation failed" }
& $python "$root\scripts\write_experiment_status.py"
if ($LASTEXITCODE -ne 0) { throw "Experiment status update failed" }
