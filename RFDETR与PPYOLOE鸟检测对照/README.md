# RF-DETR 与网站正式 PP-YOLOE+ 单类检测对照

本目录是独立实验。原网站和历史工程、模型、XML及预测均未修改。实验只检测单类 `bird`，不运行物种分类头。

## 模型与来源

- 基准：网站当前正式部署的 PP-YOLOE+ s 单类 bird 权重，来源路径记录于 `configs/experiment.json`。网站第二代运行配置与归档配置 SHA-256 相同；网站 worker 从第二代工程配置生成运行配置。权重 SHA-256、配置哈希及推理设置写入预测 JSON 元数据。
- 对照：Roboflow 官方 `rfdetr==1.11.2` 的 `RFDETRSmall`，配置中的编码器为 `dinov2_windowed_small`，patch size 16，单类 bird，基础输入512；官方训练配置实际启用了 per-batch 多尺度与 scale jitter，最终推理输入512。它是 DINOv2 系列骨干的检测器，初始化使用官方 COCO 检测器 checkpoint，不是把 DINOv2 分类器改成检测器。许可证及官方版本信息见 `reports/model_provenance.json`。
- 预训练权重下载源：官方包引用的 `https://storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth`。本地文件位于 `weights/cache/rf-detr-small.pth`，已记录哈希和官方 MD5 校验。

## 数据与可比性

固定验证集按现有官方 COCO 清单原样复用：172张、204个鸟框。训练集沿用既定划分和有效标签；`jiamaque_087.jpg` 及其 brightness 衍生图从新训练副本排除，避免待补标数据参与训练。其余排除、映射和源文件对应关系见 `reports/data_audit.json` 与 `data/dataset/`。图片和原标注未改动。PP-YOLOE+ 是既有部署基准，历史训练集曾包含上述样本，因此本轮不是严格的“只改变架构”控制实验。

验证图此前已反复用于探索与权重选择，不是独立测试集。外来80图只做冻结权重推理与人工检查，不参与训练、阈值选择或权重选择；目录名仅在结果页作为事后备注。

## 预处理与评估口径

- PP-YOLOE+：640×640 方形直接缩放，官方配置归一化，官方 NMS；本轮因遗留环境缺少匹配的 cuDNN 8，在 CPU 上推理，报告会标明速度不可与 RF-DETR GPU 速度直接比较；仅 CPU 建层时绕过 cuDNN 版本探测，卷积仍由 CPU 算子执行。旧venv启动器指向已不存在的 C 盘 Python，因此复用其 E 盘site-packages并记录NumPy/Paddle版本。
- RF-DETR-S：官方推理 API，512方形输入、官方 RGB/归一化及框坐标还原。
- 两者预测先以原生低阈值候选（score floor 0.001；每图最多保留300个结果）保存，再做网站原有的非传递、分数优先 IoU≥0.5 抑制，最后以分数严格大于0.30生成展示框。抑制候选单独保留。AP与PR曲线从未按0.30截断的候选计算；AP采用COCO IoU 0.50:0.95与AP50。不同模型分数可能未校准，故阈值结果与PR/AP一起解读。
- 定点指标按检测分数优先逐图一对一匹配，IoU≥0.5算TP。未匹配框/真值只表示相对现有XML，不直接等于背景误检/真实漏检。所有类别统一为bird。

## 训练与恢复

训练随机种子123，RF-DETR-S 官方 COCO 检测权重初始化，基础输入512；训练时按官方默认开启 per-batch 多尺度（expanded_scales=true）和 scale jitter，30轮预算，batch size 1、梯度累积8、AdamW、主学习率1e-4、编码器学习率1.5e-5、weight decay 1e-4、fp16、EMA、gradient checkpointing。最佳权重由官方验证 mAP50:95 规则选取；训练期间保存 `runs/formal/last.ckpt` 与间隔检查点、EMA/最终权重及 `metrics.csv`。实际日志、资源占用和完成轮次以本目录文件为准。

环境在本目录 `runtime/`，PyTorch CUDA 共用的是E盘第三代实验环境依赖目录，未改动旧环境；新增 RF-DETR 及相关包保存在本目录。CUDA、缓存、临时目录均设置到E盘。正式复跑命令见 `scripts/run_formal.ps1`；中断后可使用日志中最新的完整 checkpoint，通过 `runtime/python311-embed/python.exe scripts/train_rfdetr.py --resume <checkpoint>` 恢复。不要同时启动多个训练。

## 复现入口

1. `scripts/prepare_data.py` 审计并准备固定数据副本。
2. `scripts/train_rfdetr.py` 完成正式微调（短跑使用 `--smoke`）。
3. `scripts/infer_ppyoloe.py --split fixed_val --device cpu` 和 `--split external80 --device cpu` 生成部署基准候选。
4. `scripts/infer_rfdetr.py --split fixed_val` 和 `--split external80` 生成RF-DETR候选。
5. `scripts/evaluate_and_build_reports.py` 生成AP/PR/逐图指标与图册。

正式结果完成后优先打开 `reports/index.html`；分集页面为固定验证集与外来80图的独立页面。外来图片新增框可在浏览器本地记录“确实是鸟/背景/同鸟重复框/暂不能确定”，并导出复核文件。

## 局限

这是单一RF-DETR-S基线对照，不是架构控制变量严格实验：官方模型的输入尺寸、预处理、后处理、预训练任务与训练集边界均不同。固定验证集被重复使用。外来图无逐实例真值，只能做候选复核，不能报告准确率或mAP。未匹配预测、框数变化不能自动视作真实错误或性能改进。




