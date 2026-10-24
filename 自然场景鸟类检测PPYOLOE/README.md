# 自然场景鸟类检测与识别（PP-YOLOE-s）

这是课程项目的可运行工程目录，使用 PaddleDetection 官方 `release/2.9` PP-YOLOE-s 和 COCO 标注格式。原始图片和两个 ZIP 保留在工作区原位置；审计表、裁剪页和转换数据都在本目录。

## 当前实际进度

- 已核实原始标注 ZIP：968 张图片、966 份 XML、1065 个鸟实例、65 张多鸟图；缺失 XML 的是 `meishanque_012.jpg` 和 `meishanque_101.jpg`。
- 已核实亮度 ZIP：1934 张图片、1932 份 XML，其中 966 张带亮度变化的图。原始 ZIP 和亮度 ZIP 内的 968 张原图，与工作区原图逐文件哈希完全一致。
- 已规范化“眉山雀”为“煤山雀”，并保存类别原名、标准名及固定编号映射。名称映射不证明每个实例的物种标签正确。
- 已按课程示例清单排除 8 张教师示例图。另隔离 2 张缺 XML 图和 `dashanque_119.jpg`（目录前缀与 XML 逐实例标签冲突）待人工确认。结构检查通过的 957 张独立原图按种子42划分为训练785张、验证172张；77对近似图候选保守放在同一集合，最终验证集仍覆盖八类。
- 亮度图与原图尺寸和逐实例框一致。以“白位≥250灰度像素比例不超过20%、黑位≤5比例不超过5%”做可复现的保守筛查，训练集纳入684张亮度图。该阈值是像素异常筛查线，不是物种细节已无损的人工结论；对应原图保持为一个独立样本。
- 已安装独立虚拟环境，PaddlePaddle 2.6.2 GPU、cuDNN 8.9.5、Python 3.11.9、PaddleDetection 提交 `b25522a0f4bde8c80603f3ba5e3472059972e3b5`。单卡 `paddle.utils.run_check()` 已报告 GPU 运行正常。
- 训练和评估配置已加载检查，确认 PP-YOLOE-s、8类、batch size 4、640×640、官方 ImageNet 归一化、官方 PP-YOLOE-s COCO 检测器初始化（8类不匹配分类头由加载逻辑跳过）、固定拉伸缩放、水平翻转概率0.5、NMS IoU 0.7；继承后的训练增强列表中没有色彩扰动、扩展或随机裁剪。
- **smoke训练已完成**：2轮、batch 4、640输入。官方验证日志第2轮 bbox AP50=0.567；对同一172张验证图另行运行批量推理和项目评估器，得到 mAP@0.5=0.5666、micro P/R/F1=0.2157/0.8480/0.3439（IoU=0.5、P/R/F1分数阈值0.25）。这只是 smoke 管线诊断值，不能当作完整基线结果。日志、最佳/最终权重、逐图预测、可视化、指标表及混淆矩阵在 `runs/smoke_640/`。
- **完整基线训练已完成**：官方 PP-YOLOE-s 完整训练100轮，batch=4，640输入；实际用时1小时42分46秒（2026-10-02 19:23:28 至21:06:14，Asia/Shanghai），训练日志和每轮checkpoint在 `runs/baseline_640/`。官方选择器按 COCO AP@[.50:.95] 得到 best_model，指标0.630；课程指定关注AP@0.5，因此另按完整日志中 AP50 最高轮次（第13轮，官方打印到3位小数为0.861）保存 `best_model_ap50.pdparams`，并在固定172张验证图重新批量预测评估。该选择规则和两种权重用途已分开记录。
- **课程口径验证结果已运行**：AP50所选权重的自定义 mAP@0.5=0.8615，micro Precision=0.4564、Recall=0.8725、F1=0.5993（score=0.25、同类一对一IoU=0.5；NMS IoU=0.7）。八类明细、混淆矩阵、错误清单、逐图原图坐标预测和可视化位于 `runs/baseline_640/evaluation_ap50_epoch_13/` 与 `runs/baseline_640/ap50_epoch_13_predictions/`。该验证集只有172张图，且XML物种标签尚未逐一人工核验，结果不等于已证明每个物种真值正确。
- **针对性比较已运行**：基线错误中存在大量分数阈值以上的重叠跨类别候选；以同权重、同候选最低分、同验证集和同指标阈值，仅将类别内NMS改为类别无关贪心NMS（IoU=0.7）。mAP50从0.8615变为0.7942，micro P/R/F1从0.4564/0.8725/0.5993变为0.6709/0.7696/0.7169；TP/FP/FN由178/212/26变为157/77/47。Precision和F1上升，但Recall和mAP50下降，说明抑制跨类重复框会同时压掉一些正确类别框。汇总表、5个代表性实例及并排图在 `runs/baseline_640/nms_class_agnostic/`，没有将结果写成单向“提升”。
- **固定八图输入变化检查已运行并修正汇总**：每类按固定规则从验证集选一张，同一权重先用正常输入，再把图像双线性缩小约50%后走同一640预处理。原汇总误将低分辨率图片坐标与原图真值比较，错误得到0/9。现使用 `comparison.json` 中已还原到原图坐标的框重新汇总，原图与低分辨率输入均为9/9（score=0.25、同类IoU=0.5）。更正结果见 `runs/baseline_640/input_change_check/corrected_summary/`；旧汇总仅保留为历史记录，不能引用其中的0/9结论。这只是八个固定样例上的观察，不是总体指标，也没有据此调参。

需人工查看：

- `reports/manual_review.csv`：2张无 XML 图片已由用户确认因看不到鸟而排除，不需补标，也不作负样本；本次全量推理同样排除。
- `reports/label_folder_mismatch_review.csv` 与 `reports/held_out_manual_review.csv`：`dashanque_119.jpg` 的逐实例类别与文件夹前缀不同，当前隔离且没有自动改成目录类别。
- `reports/near_duplicate_candidates.csv` 和 `reports/review_galleries/near_duplicate_candidates/`：77对感知哈希近似图。配对页显示同一场景相似图，划分时保守保持同组；人工确认后可更新记录。
- `reports/brightness_audit.csv` 和 `reports/review_galleries/brightness_clipping_examples/`：逐图亮度差异和高白位样例；色羽纹理是否损失仍应结合原图判断。
- `data/review_crops/`、`reports/crop_manifest.csv` 和 `reports/review_galleries/`：按八类排列的原图框、留少量背景的单实例裁剪、文件路径、实例编号和 XML 原名。

## Windows 运行

项目专用 Python 在 `.venv` 中，不改动系统 Python 或其他环境。首次克隆/复制工程后，可按以下顺序运行：

```powershell
Set-Location 'E:\机器学习课设\自然场景鸟类检测PPYOLOE'
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-runtime.txt
```

`PaddleDetection` 官方源码已放在 `third_party/PaddleDetection`。若从头准备数据，运行：

```powershell
.\scripts\run_prepare.ps1
```

默认审计会生成鸟框裁剪文件。之后若只更新审计表并沿用现有裁剪页，可运行 `scripts\run_prepare.ps1 -SkipCrops`。

先做小规模 smoke，再做完整基线：

```powershell
.\scripts\run_train.ps1 -Stage smoke
.\scripts\run_train.ps1 -Stage baseline
```

基线是100轮，最佳权重按训练过程中固定验证集的官方 COCO bbox mAP 保存。smoke 仅为数据与训练管线检查，不能作为最终实验结果。

任意图片目录批量推理（递归读取图片；类别由模型输出，不读文件夹名或标注）：

```powershell
$env:PYTHONUTF8 = '1'
& .\.venv\Scripts\python.exe .\scripts\batch_infer.py `
  --input 'D:\待推理图片' `
  --weights '.\runs\baseline_640\best_model_ap50.pdparams' `
  --output '.\runs\unknown_test'
```

结果写入 `predictions.json`，框是原图坐标，类别为标准中文名；可视化图在 `visualizations`。未知测试图只有预测，没有虚构的真值或指标。

训练完成后，对固定验证集预测并计算课程口径指标：

```powershell
$env:PYTHONUTF8 = '1'
& .\.venv\Scripts\python.exe .\scripts\batch_infer.py `
  --input '.\data\prepared\images\val' `
  --weights '.\runs\baseline_640\best_model_ap50.pdparams' `
  --output '.\runs\baseline_640\ap50_epoch_13_predictions' `
  --nms-mode classwise
& .\.venv\Scripts\python.exe .\scripts\evaluate_predictions.py `
  --predictions '.\runs\baseline_640\ap50_epoch_13_predictions\predictions.json' `
  --output '.\runs\baseline_640\evaluation_ap50_epoch_13'
```

P/R/F1 使用分数阈值0.25、IoU阈值0.5、同类别逐图一对一匹配；总体值为 micro。AP 使用分数至少0.001的完整排序预测，按每类101点插值计算 AP@0.5 并求八类均值。混淆矩阵中，类别错误按“真实类→预测类”记录，漏检和误检分别进入背景列和背景行。

八图输入变化检查在同一模型权重上将图像先按双线性插值缩小至约一半，再走相同640预处理；输出框按实际宽高比例映射回原图坐标：

```powershell
& .\.venv\Scripts\python.exe .\scripts\input_change_check.py `
  --weights '.\runs\baseline_640\best_model_ap50.pdparams' `
  --output '.\runs\baseline_640\input_change_check'
```

类别无关NMS结果可通过 `scripts\class_agnostic_nms.py` 从基线完整预测生成，再调用 `scripts\evaluate_predictions.py` 计算同口径指标。`scripts\build_comparison_artifacts.py` 生成指标对照表、代表实例表和并排可视化；`scripts\summarize_input_change.py` 汇总八图目标类别匹配情况。

## 完整基线结果（AP50选择权重）

以下值来自 `runs/baseline_640/evaluation_ap50_epoch_13/metrics.json`，P/R/F1的分数阈值为0.25，类别正确、一对一IoU≥0.5；AP50使用分数≥0.001候选、101点插值。整体值使用micro汇总。

| 类别 | AP@0.5 | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| 树麻雀 | 0.5295 | 0.3514 | 0.5909 | 0.4407 |
| 家麻雀 | 0.7410 | 0.3243 | 0.8889 | 0.4752 |
| 山麻雀 | 0.8905 | 0.3830 | 0.9000 | 0.5373 |
| 白头鹎 | 0.9900 | 0.4889 | 1.0000 | 0.6567 |
| 红耳鹎 | 0.9058 | 0.6970 | 0.9200 | 0.7931 |
| 黑喉红臀鹎 | 0.9863 | 0.6452 | 1.0000 | 0.7843 |
| 大山雀 | 0.8977 | 0.5610 | 0.9583 | 0.7077 |
| 煤山雀 | 0.9509 | 0.4889 | 1.0000 | 0.6567 |
| 整体micro | **mAP=0.8615** | **0.4564** | **0.8725** | **0.5993** |

类别无关NMS的逐类数据以及混淆矩阵放在 `runs/baseline_640/nms_class_agnostic/evaluation/`。本次没有另行训练针对性比较模型，因此无需在这里声称类别或光照策略带来学习能力变化；比较只支持对当前固定模型后处理行为的观察。

## 复现限制与状态

- 已运行验证：数据审计、COCO转换、训练集/验证集数据加载、2轮smoke、100轮完整训练、AP50权重选择、验证集批量推理/指标/混淆矩阵、类别无关NMS比较、八图输入变化检查、关键脚本编译。
- 已准备但未运行：960输入对照、检测器加分类器方案、超分或额外增强实验；本次按真实基线错误选择了一个后处理比较，未扩展成多组实验。
- 需要人工复核：缺标注两图、`dashanque_119.jpg`冲突标签、77对近重复候选、亮度样例、物种标签真实性及5个比较示例。教师未公开29张图片未参与任何过程。
- 数据划分随机种子固定为42；PaddleDetection 2.9标准训练CLI没有在本次运行显式固定训练随机种子。训练可能不能逐位复现，详见 `runs/baseline_640/run_metadata.json`。
- 官方 `best_model`以AP@[.50:.95]选择；`best_model_ap50.pdparams`以本项目记录的最高AP50轮次选择。课程验证和演示建议使用后者，并在答辩中说明选择口径。

## 关键口径

- PP-YOLOE 官方 reader 使用 `Resize(640, 640, keep_ratio=false)`，即将任意长宽比图像拉伸至640×640。训练和预测都用同一预处理，模型后处理按官方 `im_shape/scale_factor` 还原原图框。对 XML 转 COCO 采用 `bbox=[xmin,ymin,xmax-xmin,ymax-ymin]`，保存端点数值，不擅自加减像素；原图叠框复核页可供检查。
- 未核验的类别没有用目录或文件名前缀替代。多鸟图逐实例写入 COCO，预测也逐框导出。
- 固定验证清单在 `reports/split_manifest.csv`。任何后续比较都必须使用相同的172张原图验证集，brightness 衍生图不进入验证集。
- 随机性记录：数据划分种子固定为42；本次已启动的官方 PaddleDetection 2.9 标准训练入口没有显式设训练随机种子（其内部仅在测试用 `enable_ce` 开关下设种子0），因此训练随机状态未固定，属于本次运行的复现限制，见 `runs/baseline_640/run_metadata.json`。不把 split seed 误称为训练 seed。
- `reports/` 中的内容是审计数据、运行配置和实验输出，不是代写的课程提交报告。物种标注准确性、疑似错误解释以及真实训练结论需要依据复核和模型运行记录填写。


## 全量原图置信度导出与核查

已重新加载 `best_model_ap50.pdparams` 对966张原图逐张推理，排除用户确认无效的 `meishanque_012.jpg` 和 `meishanque_101.jpg`。输出含785张训练图、172张验证图、9张未参与训练或验证的图。只以固定172张验证图复算性能，与既有指标一致：mAP50=0.8614555945，micro P/R/F1=0.4564102564/0.8725490196/0.5993265993。

新增 `scripts/export_class_confidences.py`：逐图保存八类NMS前最高候选得分，并用NMS返回的候选索引提取每个保留框同一位置的八类得分，逐框核验所选类别得分与官方输出一致。得分不是整图类别概率，各类最大值可能来自不同候选位置，不归一化为总和1。低分结果保存在JSON，Excel明细仅列出分数≥0.25的1417个候选框，可能包含同一鸟的跨类别重复框。

本次输出：`runs/confidence_audit/all_originals/class_confidences.json`。复算记录：`runs/confidence_audit/all_originals/verification.json`。`scripts/verify_confidence_run.py` 可复算固定验证集指标。八图旧汇总0/9由坐标错误造成，已修正为9/9，历史文件不应继续作为结论引用。

运行示例（项目根目录）：

```powershell
$env:PYTHONUTF8 = '1'
.\.venv\Scripts\python.exe .\scripts\export_class_confidences.py --input 'E:\机器学习课设\考核任务数据\考核任务数据\自然场景鸟类识别数据\数据集' --output '.\runs\confidence_audit\all_originals'
.\.venv\Scripts\python.exe .\scripts\verify_confidence_run.py
```


## 模型与人工标注一致性审查

复用966张原图推理，直接对原XML中1065个鸟实例按score≥0.25、同类一对一IoU≥0.5核对。有210张图片存在多余框、未匹配标注或主预测错类，其中训练110张、验证94张、其他6张。33个实例的最高分空间匹配候选类别与标注不同，11个错类候选score≥0.8。28个标注实例没有同类正确匹配；380个多余框含276个额外跨类框、56个同类重复框、32个定位不足候选和16个背景误检或漏标候选。诊断计数可重叠，不能相加当成总错误数。

`dashanque_119.jpg`需优先核验标签：XML为黑喉红臀鹎，模型为大山雀（0.9864，IoU=0.9301）；`jiamaque_094.jpg`目视可见右下方第三只鸟而XML只有两个框，需检查漏标。以上只标记复核，不直接确定物种或修改标签。

新增 `scripts/audit_annotation_matches.py` 可生成核对数据与人工/模型并排图册，接收 `--output` 输出目录。核对记录保存在 `runs/confidence_audit/annotation_matches/annotation_match_audit.json`。原始标注、划分、权重均未修改。


## 单鸟最高分显示与隐藏候选

`batch_infer.py`命令行现默认采用跨类别显示去重：按置信度降序，每个IoU≥0.5的重叠组保留最高分原框和类别。不同组各保留一框，避免将整张多鸟图片仅保留一个框。其他候选保存到 `candidate_groups`，未跨类别去重的候选保存在 `raw_predictions.json`；可视化只画最高分代表框，默认显示阈值0.25。

这属于后处理与显示规则变化，没有重新训练，也没有修正物种分类能力。IoU分组不能保证识别真实个体，邻近鸟或差异较大的重复框仍须目视检查。用户已确认人工标注可能漏鸟，不将额外模型预测自动当作误检或新标签。

内部 `run_predict()` 默认保持原基线classwise模式，避免改变已有输入变化检查调用；命令行需要复现原课程基线时显式使用 `--nms-mode classwise`。通常图片演示使用新的默认模式，可用 `--display-nms-iou`调整重叠阈值。

已生成966张原图的单鸟最高分图册，隐藏335个原本达到0.25的候选框，显示框由1417变为1082。固定验证集按现有标注核对，去重后TP/FP/FN为156/50/48，去重前178/212/26。这反映按最大分保留类别的取舍，不能宣称检测能力提高。结果和全部隐藏候选见 `runs/unique_bird_display/predictions.json`，摘要见 `display_summary.json`。原权重、人工标注、划分未改变。

`scripts/build_unique_birds_gallery.py --output <html路径>` 可从已有逐图置信度结果重新生成自包含HTML。图册有全部966张原图、人工标注开关、最高分显示框、同候选八类得分、隐藏候选数据与图片筛选。


## 当前显示阈值与报告位置

按用户要求，命令行推理默认仅显示置信度严格大于0.5的跨类别去重框；等于0.5不显示。原始候选和隐藏候选仍完整保留，不改动训练权重、人工标签、数据划分和历史评估口径。HTML生成器增加 `--score-threshold`，默认0.5，采用严格大于比较。

50%新版图册位于 `E:/机器学习课设/模型检查报告/鸟类预测_置信度超过50.html`。之前在C盘生成的Excel和标注复核图册已迁移至 `E:/机器学习课设/模型检查报告/`，旧25%图册保存在该目录的 `历史25阈值/`。新检查脚本、截图和推理数据在E盘项目目录中。

966张图现在显示1042个框。固定验证集在同一跨类别去重规则下，25%阈值TP/FP/FN=156/50/48；严格大于50%后为152/29/52。按现有标注，精确率由75.73%升至83.98%，召回率由76.47%降至74.51%；存在漏标，不能把所有多余预测直接认定为误检。更高阈值没有解决高置信度的物种错分类。

建议后续先人工补齐漏标和复核类别冲突，保存新版本标注，再比较八类检测器与“检测定位＋鸟框八类分类器”。训练分类器用训练集鸟框，正式验证用检测器预测框并评估整图流程，验证图不进入训练。当前只完成显示改动，没有启动新训练或自动修改标签。
