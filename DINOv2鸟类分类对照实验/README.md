# PP-YOLOE+ + DINOv2 鸟类分类对照

本实验使用现有正式 PP-YOLOE+ 检测器和冻结的官方 DINOv2 ViT-B/14，只训练独立的八类线性分类头。网站默认方案仍是 BioCLIP 2；没有修改网站、原图、XML、既有模型或人工复核记录。

## 复现顺序

在 PowerShell 中运行以下命令。Python 运行时、依赖联接、模型、缓存和产物均在 E 盘工程内。`train_linear_head.py` 使用已有可恢复检查点；需从头训练时，先把该次实验复制到新的 runs 子目录并记录初始化状态，不要删除已交付权重。

```powershell
$root = 'E:\机器学习课设\DINOv2鸟类分类对照实验'
$python = Join-Path $root 'runtime\python311-embed\python.exe'
Set-Location $root
& $python scripts\prepare_data.py
& $python scripts\smoke_dinov2.py
& $python scripts\extract_train_features.py
& $python scripts\train_linear_head.py
& $python scripts\evaluate_and_gallery.py
node scripts\check_pages.mjs
```

继续训练记录位于 `runs\dinov2\last_checkpoint.pt`，包含分类头、优化器、调度器和随机状态。最佳头是 `runs\dinov2\best_head.pt`，最后一轮是 `runs\dinov2\final_head.pt`。训练逐轮日志在 `runs\dinov2\training.jsonl` 和 `logs\training.log`。更改配置后需重新抽取特征并明确生成新运行目录，避免覆盖本次结果。

## 本次设置

- 官方源码：[`facebookresearch/dinov2`](https://github.com/facebookresearch/dinov2)，修订 `7764ea0f912e53c92e82eb78a2a1631e92725fc8`。
- 官方 ViT-B/14 权重：[dinov2_vitb14_pretrain.pth](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth)。本地文件哈希见 `reports\environment_and_model_provenance.json`。
- 冻结骨干输出官方 `x_norm_clstoken`，768维、L2归一化；分类头单独训练。
- 使用第二代归档固定裁剪清单：鸟框外扩15%，灰色114正方形补边后双三次缩放到224，ImageNet均值/标准差归一化。训练/推理同预处理。
- 随机种子123，AdamW，学习率0.02，权重衰减0.01，batch 128，CosineAnnealingLR，60轮。用固定裁剪验证集准确率选头，最佳第4轮。
- 正式整图比较共享归档PP-YOLOE+ s候选：检测分数严格大于0.30，按分数降序贪心、IoU≥0.5非传递去重，每框裁剪外扩15%。两分类器用完全相同候选。
- 类别顺序读取归档映射：树麻雀、家麻雀、山麻雀、白头鹎、红耳鹎、黑喉红臀鹎、大山雀、煤山雀。未使用目录或文件名决定模型预测。

## 数据范围与限制

沿用第二代归档固定训练/验证来源归属。可用训练实例裁剪1586张，验证实例裁剪146张（140张来源图）；端到端固定验证集172图、204个人工目标。验证复核待定实例不计入纯分类；训练中排除了仍待补标的 `jiamaque_087.jpg`（8个裁剪），另遵守 `dashanque_119.jpg`、`shanmaque_090.jpg` 等排除/隔离状态。亮度衍生图沿来源划分；未改 XML。数据哈希和审计见 `reports\data_audit.json`。

BioCLIP 2正式预测曾尝试独立特征重算，但分数最大差异0.320995且4个顶类变化，因此不将该次不一致重算用于比较。核验后复用正式冻结特征缓存和原最佳头：保存分数可精确复现，固定验证候选框与正式报告0处不一致，检测器权重哈希吻合。详情见 `reports\run_audit.json`。

固定验证集反复用于历史探索和选型，不是独立测试。80张外来图没有逐实例人工真值；目录类别仅事后查看，不是识别正确率，也没有用来训练、选头或调阈值。没有使用教师测试图。本实验只换分类器，检测器AP不会因分类头变化而提高；无检测框时分类器无输入。BioCLIP纯分类分数来自既有验证缓存，且本次DINO按未解决的训练图隔离规则比旧BioCLIP头少8个训练裁剪，比较有此数据差异。

## 查看结果

- 首页：`reports\index.html`
- 固定验证集逐图对照：`reports\fixed_validation_comparison.html`
- 80张外来图回放：`reports\external80_comparison.html`
- 汇总指标、逐类结果、混淆矩阵、逐图/逐框CSV、训练曲线及验收记录均在 `reports`。
