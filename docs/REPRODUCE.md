# 复现范围与依赖

本仓库保存实验源码，不含训练数据、权重、整套虚拟环境或运行缓存。原始文件按项目目录保留，各项目README给出原运行入口。

## 上游代码

- PaddleDetection：https://github.com/PaddlePaddle/PaddleDetection ，原实验commit `b25522a0f4bde8c80603f3ba5e3472059972e3b5`，放到 `自然场景鸟类检测PPYOLOE/third_party/PaddleDetection`。
- FET-FGVC：https://github.com/NICE-FUTURE/FET-FGVC ，原commit `b8e20b2846875fb6ea189477500be2ff1c3d37b7`，放到 `第三代鸟类识别现代模型对照/third_party/FET-FGVC`。作者代码及本地适配记录需结合阅读。
- DINOv2：https://github.com/facebookresearch/dinov2 ，版本与权重见对照工程配置。DINOv3未运行，不是最终模型依赖。
- RF-DETR：https://github.com/roboflow/rf-detr ，本项目使用版本1.11.2、RF-DETR Small、DINOv2 windowed small骨干。

## 环境

PP-YOLOE实验基于PaddlePaddle GPU 2.6.2（CUDA11.8，cuDNN8）；BioCLIP 2 / RF-DETR运行环境使用PyTorch2.7.1+cu126。Paddle与Torch以独立进程运行，避免动态库与NumPy版本混用。GPU推理需要匹配的NVIDIA驱动及CUDA依赖，现有网站默认GPU优先。

本地第四代运行器曾使用位于RF项目的Python3.11嵌入式环境；该运行器及site-packages没有随代码发布。重新部署时请创建环境并修改网站settings.py中的Python解释器、依赖目录与模型路径。不要复制用户机器上的二进制环境当作源码。

## 数据

自行准备有授权的课程图片与XML，使用各代prepare_data入口。保持原图与brightness衍生的来源分组，使用固定划分与人工复核排除规则。验证集172图204实例；未读取教师未公开测试图。

## 核心入口

1. 第一代：`自然场景鸟类检测PPYOLOE/scripts/run_prepare.ps1`、`run_train.ps1`、`evaluate_predictions.py`。
2. 第二代：`第二代鸟类检测与细粒度分类/scripts/prepare_data.ps1`、`train_detector.ps1`、`train_classifier.py`、`infer_two_stage.py`。
3. 第三代：`第三代鸟类识别现代模型对照/scripts/frozen_encoder.py`、`fet_classifier.py`。两CNN是独立对照；Bilinear采用共享骨干与双线性池化，不是两个分类器投票融合。
4. 第四代：`RFDETR与PPYOLOE鸟检测对照/scripts/train_rfdetr.py`，端到端：`端到端BioCLIP2对照/scripts/run_end_to_end.py`，mAP补算：`compute_species_map.py`。
5. 网站：`本地鸟类识别网站/server.py`、`worker.py`、`settings.py`。各代使用不同环境，先修改路径，再启动。

原始数据、上游代码、环境及权重未准备齐时，不能仅凭源码仓库完成即开即用。没有对新机器上的完整重训作验证。
