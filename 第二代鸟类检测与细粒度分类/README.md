# 第二代：单类鸟检测 + 八类细粒度分类

## 当前状态

- 数据转换、裁剪、模型加载、小规模检查和正式训练均已在E盘完成。
- 用户已确认7张训练图片原标注正确；确认与数据划分记录分别见 `data/annotation_corrections_v2.json`、`reports/manual_review_confirmation_log.csv`、`reports/user_confirmation_split_check.json`。这些确认只保留现有类别和框，没有修改原始XML。
- `shanmaque_090.jpg`仍未检查并隔离；`dashanque_119.jpg`继续永久排除。验证集固定172张，确认训练图及其brightness衍生图留在训练集。
- 未读取、使用或调参教师未公开的29张测试图。
- PP-YOLOE+ s 是官方 PP-YOLOE 的增强版本，检测头增加任务对齐学习（TAL）相关设计并改进训练策略；本项目使用 PaddleDetection 2.9 官方 `ppyoloe_plus_crn_s_80e_coco.yml` 继承结构，不替换成其他YOLO。
- 第一代归档在 `E:\机器学习课设\第一代结果归档`；第二代逐次运行证据和待复核入口见 `E:\机器学习课设\模型检查报告\第二代实验\运行记录与复核入口.md`。

## 目录与入口

- `configs/paths.json`：路径配置。
- `configs/class_mapping.json`：标准类别名及固定编号；XML“眉山雀”规范化为“煤山雀”。
- `scripts/prepare_data.py`：固定沿用第一代图片划分，生成单类鸟COCO标注及逐实例分类裁剪清单。
- `scripts/train_detector.ps1`：PP-YOLOE+官方训练入口。
- `scripts/train_classifier.py`：普通ResNet分类与共享骨干Bilinear CNN训练入口。
- `scripts/infer_two_stage.py`：任意图片目录批量检测、裁剪、分类和结果导出。
- `scripts/evaluate_two_stage.py`：按预测框评估端到端八类结果；人工框裁剪准确率单独输出。
- `scripts/run_smoke.ps1`：小数据短跑，用于核查加载与管线，不代表正式训练。
- `scripts/eval_detector_snapshots.ps1`：对训练快照和最终权重逐个跑固定验证集官方COCO评估，按AP@0.5选择权重。
- `scripts/run_fixed_validation_comparison.ps1`：对固定验证集运行同检测候选上的ResNet18/Bilinear CNN比较、评估并生成HTML图册。
- `scripts/verify_user_confirmation_split.py`：检查用户确认图片的训练/增强归属及固定验证集是否变化。
- `scripts/write_experiment_status.py`：按实际权重、日志和元数据生成哈希清单及进度快照。

模型权重放入 `weights/`，数据、缓存、日志和结果均位于本工程或已存在的第一代E盘目录。当前复用第一代E盘Python环境与PaddleDetection源码，避免重复建立环境和复制完整仓库。

## 分类模型

- 普通CNN基线：PaddlePaddle ImageNet预训练ResNet18，最后一层改为8类。
- Bilinear CNN：共享ResNet18骨干，取末端空间特征图，计算通道两两外积并在空间维求平均；第一版采用完整通道Gram矩阵，signed-square-root与L2归一化后接线性分类头。不声称其为某篇论文的逐行复现。
- 训练图片只取固定划分的训练图鸟框及其合格brightness衍生图。验证裁剪只用于独立诊断；正式端到端评估始终用检测器预测框。

## 运行

```powershell
cd E:\机器学习课设\第二代鸟类检测与细粒度分类
.\scripts\prepare_data.ps1
.\scripts\run_smoke.ps1
```

本次正式训练和评估已实际完成。正式配置是100轮PP-YOLOE+ s（batch=2、640、seed=0），以及各60轮/12000步ResNet18和Bilinear CNN（batch=8、224、seed=123）。当前结果和哈希清单见 `E:\机器学习课设\模型检查报告\第二代实验\运行记录与复核入口.md`、其CSV/JSON附件。

人工确认只录入 `data/annotation_corrections_v2.json`。每张图片必须填写 `confirmed_by`、`note` 和 `current_labels_confirmed` 或 `corrected` 状态；类别/框更改另填逐实例 `changes`，实例编号按原XML从1开始。修改会输出 `reports/annotation_change_log.csv` 和 `reports/annotation_version.json`；源ZIP与XML保持只读。没有确认的复核图继续隔离。

数据准备后，可先执行：

```powershell
.\scripts\prepare_data.ps1
```

需要从头复现实验时，先执行数据准备和复核划分检查，再依次训练和评估：

```powershell
.\scripts\prepare_data.ps1
& 'E:\机器学习课设\自然场景鸟类检测PPYOLOE\.venv\Scripts\python.exe' .\scripts\verify_user_confirmation_split.py
.\scripts\train_detector.ps1
& 'E:\机器学习课设\自然场景鸟类检测PPYOLOE\.venv\Scripts\python.exe' .\scripts\train_classifier.py --model resnet18 --epochs 60 --batch-size 8 --seed 123 --size 224 --lr 0.0003 --run-name classifier_resnet_v2 --device gpu
& 'E:\机器学习课设\自然场景鸟类检测PPYOLOE\.venv\Scripts\python.exe' .\scripts\train_classifier.py --model bilinear --epochs 60 --batch-size 8 --seed 123 --size 224 --lr 0.0003 --run-name classifier_bilinear_v2 --device gpu
.\scripts\eval_detector_snapshots.ps1
.\scripts\run_fixed_validation_comparison.ps1
```

快照评估会覆盖 `best_model_ap50.pdparams`，完整重跑前需先备份想保留的权重。教师29张未公开图未参与训练、阈值或模型选择。

当前共享运行环境来自第一代E盘独立虚拟环境，记录的 PaddlePaddle 为2.6.2，PaddleDetection commit 为 `b25522a0f4bde8c80603f3ba5e3472059972e3b5`。官方模型配置参照 [PP-YOLOE+ s COCO 配置](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/configs/ppyoloe/ppyoloe_plus_crn_s_80e_coco.yml)，COCO权重来源为 `https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_s_80e_coco.pdparams`；权重SHA-256及实际运行记录见实验清单。
