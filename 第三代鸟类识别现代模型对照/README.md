# 第三代：现代鸟类分类模型对照

检测继续使用第二代 PP-YOLOE+ s 最佳权重。第三代比较 BioCLIP 2 冻结编码器 + 八类线性头、FET-FGVC 作者 Swin-B 全局/局部/成对交互实现。按用户2026-10-04的决定，DINOv3保留为第三代效果不足时的第四代候选，第三代不自动运行DINOv3。DINOv3接口代码仍保存供后续复用。

第二代已复制封存到 `E:/机器学习课设/第二代结果归档`，归档位置见 `reports/第二代归档位置.json`。原工程、环境和标注保持原样。第三代代码、环境、模型、下载缓存和临时文件全部位于本目录。

固定使用第二代划分：1594个训练裁剪（含亮度衍生，不能当作1594张独立原图）、146个已通过复核的验证裁剪用于权重选择。统一在172张验证图、177个最终显示检测框上比较。框由检测器分数优先、IoU≥0.5贪心去重，并要求检测分数严格大于0.30；裁剪外扩15%。类别预测不能读取目录类别，目录只用于结果审核。无框图继续算漏检，不能用分类模型补出检测框。

第二代比较基准为最佳 Bilinear CNN（epoch47）：TP159、FP18、FN45、F1约0.83465，显示类别与所属目录类别不一致13张，无显示框5张。所有结果属于反复用于探索和选权重的验证集，不能写成独立测试成绩。未计算全部候选框的AP，不能将这里的结果与旧版AP直接对比。教师测试图片不参与。

BioCLIP 2 与 DINOv3 首先训练60轮冻结特征分类头；FET-FGVC训练60轮，8GB显存采用4张一批、混合精度、2类×2实例平衡采样，保留全局、局部GCN、PFI和动态剪枝，关闭可选教师蒸馏。使用固定种子123。本项目进行了数据适配和图批次索引修正，并非论文公开数据集的完整复现；适配记录在reports中。

## 运行

使用本目录 `.venv/Scripts/python.exe`：

```
python scripts/frozen_encoder.py --model bioclip2
python scripts/fet_classifier.py --smoke
python scripts/fet_classifier.py --epochs 60
python scripts/build_gallery.py
```

DINOv3 权重须用户自行在官方模型页面获得访问授权，下载 model.safetensors、config.json、preprocessor_config.json 到上述目录；本项目不绕过访问控制或替换成其他模型。

官方来源：

- BioCLIP 2：https://github.com/Imageomics/bioclip-2 ，https://huggingface.co/imageomics/bioclip-2
- DINOv3：https://github.com/facebookresearch/dinov3 ，https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m
- FET-FGVC：https://github.com/NICE-FUTURE/FET-FGVC ，源码提交 b8e20b2846875fb6ea189477500be2ff1c3d37b7

实际状态、日志、权重和指标以各runs及reports文件为准，准备好代码不代表已训练。

## 640与960检测推理对照

已完成172张验证图的相同权重对照，保持30%阈值和相同BioCLIP 2分类器。640复跑重现原结果；960只找回原5张中的家麻雀045，却新增18张无框图，总无框22张、F1降至0.7487。保留640方案。完整报告和HTML位于reports/640与960检测输入对照。此结果只涉及现有权重改变推理尺寸，没有进行960重新训练。
