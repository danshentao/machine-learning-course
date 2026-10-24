# 第三代检测器训练改进实验

本实验对当前PP-YOLOE+ s单类bird检测器做A/B/C三组同预算微调。冻结BioCLIP 2编码器与最佳八类头，不重训分类器。D困难样本重采样需等训练集复核页面上的人工确认，本轮不会自动把未匹配框当背景负样本。

## 固定数据和起点

- 初始检测器：`第二代结果归档\20261003_211135\工程\runs\detector\best_model_ap50.pdparams`；SHA-256记录在`configs/training_design.json`。
- PP-Detection提交、Paddle/PyTorch环境完整依赖冻结、GPU和驱动信息写入`reports/environment_and_source_versions.json`。
- COCO单类bird训练标注：1467张含合格brightness衍生图记录，1594实例；784个来源原图组。固定验证：172张原图、204实例。沿用已有划分，来源组跨集合数为0。
- 缺XML图、dashanque_119.jpg、shanmaque_090.jpg、meishanque_012.jpg、meishanque_101.jpg均未进入派生训练/验证JSON。
- brightness衍生图仅属于来源训练组，不增加独立原图数；验证集仅使用原图。
- 训练随机种子123；划分未重新抽样。每组batch 2、优化器和EMA沿用官方PP-YOLOE+配置、base LR 1e-4、总计40轮，初始权重相同且新建优化器；前30轮后接续10轮。

## A/B/C定义

- A：固定640、官方水平翻转及brightness来源数据，控制训练时长。
- B：前30轮轻量颜色扰动、概率0.35的框感知随机裁剪、480–768/32多尺度；末10轮关闭这些变化并固定640。
- C：与B相同，前30轮另加官方`RandomErasingImage`，每个GT框独立p=0.15，矩形面积为框面积5%–15%；末10轮关闭擦除。
- 三组训练输入使用PP-YOLOE+官方推理一致的原始像素0–255口径。核对发现旧第二代配置曾在训练reader使用ImageNet归一化，而其评估/推理reader使用原始像素；A/B/C将这一历史不一致作为共同修正，因此相对旧基准的变化不能只解释为增强策略。
- 不使用Mosaic、MixUp、强旋转、翻转外的裁切堆叠、超分或DINOv3。多尺度阶段结束后回到640。

## 运行与恢复

使用第一代Paddle环境及官方PaddleDetection仓库（第三方源码未修改）。独立配置位于`configs`，运行缓存与日志位于本目录。后台正式进程由`start_training.ps1`启动，实际PID及stdout/stderr记录在`reports/training_process.json`，各阶段状态及训练子进程PID在`reports/formal_training_status.json`。

如中断，检查阶段日志和`runs/<组>/N.pdparams`、`N.pdopt`、`N.pdema`（官方文件名N从0起，例如`4.*`表示完成第5轮）；在PowerShell运行`start_training.ps1 -Resume`，阶段一从该组最新完整快照恢复。阶段二先从阶段一的`model_final`开始；若阶段二已经保存>30轮完整快照，则从最新阶段二快照恢复。每5轮保存检查点；检查点含模型、优化器及EMA状态。若阶段已完成会跳过，不会重复从初始权重训练。当前另有watchdog只对已观察到的Windows原生0xC0000005退出、且存在完整模型/EMA/优化器快照时自动尝试最多3次恢复；事件写入`reports/training_recovery_attempts.jsonl`。其他错误不盲目重试。

烟测仅用固定16张训练原图、每组1轮，用来检查加载、loss、GPU内存和保存，不进入正式报告指标。运行入口为`python scripts/build_smoke_configs.py`后按`configs/smoke_A.yml`、`smoke_B.yml`、`smoke_C.yml`调用`scripts/train_seeded_official.py`。正式训练已由独立后台脚本顺序执行A/B/C。

训练完成后，按顺序运行：

1. Paddle环境：`python scripts/evaluate_checkpoints.py`，官方COCO evaluator评估5轮快照，取验证AP50最高权重（并列取更早轮），复制为每组`best_model_ap50.pdparams`，最终权重另存`model_final.pdparams`。
2. Paddle环境：`python scripts/predict_val_candidates.py --group A`、B、C，保存同一640推理reader的低阈值候选。
3. 第三代PyTorch环境：`python scripts/compare_end_to_end.py`，重用冻结BioCLIP 2与固定去重/显示规则，输出逐图、逐框及固定验证对照。
4. 训练集复核页面仅在GPU空闲时运行`scripts/generate_hard_training_review.py`（脚本使用初始检测器）。人工复核确认前D组保持未运行。

正式启动时另有后台`finalize_after_training.py`等待器，会自动执行以上评估、逐图推理、对照和困难样本页，并将阶段PID/日志写入`reports/post_training_closeout.json`。若收尾阶段报错，修复后运行`python scripts/finalize_after_training.py --no-wait`；已完成阶段会跳过。

## 口径

官方AP50/AP50:95由PP-YOLOE+单类COCO指标给出，不受显示阈值0.30限制。两阶段整体P/R/F1按预测类别正确且IoU≥0.5一对一匹配；另单独报告忽略类别的bird定位P/R/F1。预测框跨类别按检测分数由高到低、IoU≥0.5与已保留框分组，只保留最高分框，显示条件为检测分数严格大于0.30。错类框导致对应类别FP/FN。未匹配框只称“未匹配显示框/待复核”，不直接称背景误检。无框图和目录类别分歧另行统计。

最终排序分数为检测分数乘分类器top softmax分数，仅用于排序，不保证是校准后的正确概率。所有验证结果都用于选权重/探索，不是独立测试成绩；教师测试图不参与。
