# 模型权重与来源

权重不在Git仓库中，请按原实验记录准备。以下为课程微调权重哈希，用于核对版本，并非官方原始权重哈希。

|权重|原位置|SHA-256|
|---|---|---|
|第一代PP-YOLOE|第一代结果归档/模型与训练/best_model_ap50.pdparams|701ca5b989f8227dbad0c80825885a0e9f14583b5b0f96c04f0042501ca59fbf|
|二/三代单类PP-YOLOE+|第二代结果归档/20261003_211135/工程/runs/detector/best_model_ap50.pdparams|dc394290449ddf40fef31c144320acb8b13d0fb57b6d35f39419c3aeadfca539|
|BioCLIP 2编码器|第三代鸟类识别现代模型对照/weights/bioclip2/open_clip_model.safetensors|b7b2bf6fbc95799e42630e394cf95803892ab447c1a8ab629dbc82fbeaf7dfef|
|BioCLIP 2八类头|第三代鸟类识别现代模型对照/runs/bioclip2/best_head.pt|7d199f954bf7a125fe50d1f11f632268bbfa2d0c7b1bfa1bcbdf8a681b4b4ece|
|第四代RF-DETR|RFDETR与PPYOLOE鸟检测对照/weights/rfdetr_small_bird_best.pth|b2acea3b6cfccc011b9b0bd1237114e1f881c5e7fbed306870fd695a58d3ed16|

官方预训练来源：
- PP-YOLOE / PP-YOLOE+：https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.9/configs/ppyoloe
- BioCLIP 2：https://huggingface.co/imageomics/bioclip-2
- DINOv2：https://github.com/facebookresearch/dinov2
- RF-DETR：https://github.com/roboflow/rf-detr

ResNet18/Bilinear课程分类头和FET-FGVC适配权重的版本请结合各代原日志与配置核对。未上传这些权重时，仓库链接不能替代它们的下载地址。
