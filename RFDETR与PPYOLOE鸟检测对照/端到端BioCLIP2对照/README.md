# RF-DETR＋BioCLIP 2 端到端对照

本目录复用已训练的 RF-DETR Small、网站当前 PP-YOLOE+ 正式检测结果，以及现有冻结 BioCLIP 2 编码器和八类线性分类头；没有训练或修改这些模型。

## 查看结果

- `reports/index.html`：总览入口。
- `reports/fixed_val_end_to_end.html`：固定172张验证图逐图对照与筛选。
- `reports/external80_end_to_end.html`：外来80张逐图对照、用户确认状态和待复核候选。
- `reports/实验说明与结论.md`：本轮结果解释、边界和优先复核项。
- `reports/end_to_end_results.json`：指标、逐图完整结构化结果、分数、来源哈希和计时。
- `reports/*per*.csv`：逐图、逐框及逐类别表；`fixed_val_confusion_matrix.csv` 为混淆矩阵。

## 复跑

在 Windows PowerShell 中运行：

```powershell
E:\机器学习课设\RFDETR与PPYOLOE鸟检测对照\runtime\python311-embed\python.exe E:\机器学习课设\RFDETR与PPYOLOE鸟检测对照\端到端BioCLIP2对照\scripts\run_end_to_end.py
E:\机器学习课设\RFDETR与PPYOLOE鸟检测对照\runtime\python311-embed\python.exe E:\机器学习课设\RFDETR与PPYOLOE鸟检测对照\端到端BioCLIP2对照\scripts\build_gallery.py
E:\机器学习课设\RFDETR与PPYOLOE鸟检测对照\runtime\python311-embed\python.exe E:\机器学习课设\RFDETR与PPYOLOE鸟检测对照\端到端BioCLIP2对照\scripts\browser_qa.py
```

推理脚本按配置读取历史检测/验证结果和冻结模型，运行 BioCLIP 2 分类及来源一致性审计。缓存、临时文件及输出路径集中在本目录。运行日志见 `logs/`，浏览器验收记录和截图见 `reports/browser_qa/`。若依赖环境或源文件变化，脚本会在哈希/数据校验处失败；请先核对 `configs/experiment.json` 和结果记录，勿用变化后的数据覆盖本轮归档结果。

## 固定口径

检测分数严格大于0.30；使用原有分数优先、非传递 IoU≥0.5 去重；分类裁剪在检测框四周外扩15%，裁到图像边界后灰色补边并按 BioCLIP 配置预处理。检测分数和八类分类分数分开保存。两者乘积仅用于排序参考，不是校准概率。外来图目录名只作事后预期展示。
