"""通过官方PaddleDetection Trainer对任意图片目录批量推理并导出标准类别JSON。"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "third_party" / "PaddleDetection"
MAPPING = json.loads((ROOT / "configs" / "class_mapping.json").read_text(encoding="utf-8"))
CLASS_NAMES = [x["standard_name"] for x in MAPPING["classes"]]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def add_cuda_runtime_dll_paths():
    """NVIDIA运行库保持在项目虚拟环境，不写入系统级PATH。"""
    site = Path(sys.executable).resolve().parent.parent / "Lib" / "site-packages" / "nvidia"
    if site.exists():
        for path in site.glob("*/bin"):
            os.environ["PATH"] = str(path) + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(str(path))


def detector(config_path: Path, weights_path: Path, device="gpu"):
    add_cuda_runtime_dll_paths()
    sys.path.insert(0, str(REPO))
    import cv2
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    import paddle
    from ppdet.core.workspace import load_config
    from ppdet.engine import Trainer
    cfg = load_config(str(config_path))
    # 数据集配置使用项目相对路径；从其他工作目录启动时也应正确解析。
    for key in ('TrainDataset', 'EvalDataset', 'TestDataset'):
        dataset = cfg.get(key)
        if dataset and dataset.get('dataset_dir'):
            directory = Path(dataset['dataset_dir'])
            if not directory.is_absolute():
                dataset['dataset_dir'] = str((ROOT / directory).resolve())
    cfg.weights = str(weights_path)
    cfg.use_gpu = (device == "gpu")
    paddle.set_device(device)
    model = Trainer(cfg, mode="test")
    model.load_weights(str(weights_path))
    return model


def collect_images(path: Path):
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)


def run_predict(images, config_path, weights_path, output_dir, device="gpu", draw_threshold=0.25,
                nms_mode="classwise", display_nms_iou=0.5):
    from PIL import Image, ImageOps
    images = [Path(x).resolve() for x in images]
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    visual_dir = output_dir / "visualizations"
    model = detector(Path(config_path).resolve(), Path(weights_path).resolve(), device)
    started = time.perf_counter()
    raw = model.predict([str(p) for p in images], draw_threshold=draw_threshold,
                        output_dir=str(visual_dir), save_results=False, visualize=(nms_mode == 'classwise'),
                        save_threshold=0.001, do_eval=False)
    duration = time.perf_counter() - started
    id_to_path = model.dataset.get_imid2path()
    records = []
    for batch in raw:
        image_id = int(batch["im_id"].reshape(-1)[0])
        image_path = Path(id_to_path[image_id]).resolve()
        with Image.open(image_path) as im:
            im = ImageOps.exif_transpose(im)
            width, height = im.size
        boxes = batch.get("bbox")
        if boxes is None:
            boxes = []
        detections = []
        for row in boxes:
            cls = int(row[0]); score = float(row[1])
            if cls < 0 or cls >= len(CLASS_NAMES):
                continue
            detections.append({"bbox_xyxy": [float(x) for x in row[2:6]],
                               "class_id": cls, "class_name": CLASS_NAMES[cls], "score": score})
        records.append({"image": str(image_path), "width": width, "height": height,
                        "detections": detections})
    payload = {"metadata": {"model": "PaddleDetection PP-YOLOE-s", "config": str(Path(config_path).resolve()),
                            "weights": str(Path(weights_path).resolve()), "device": device,
                            "input_size": 640, "resize": "official Resize 640x640 keep_ratio=false; outputs restored by PPDetection",
                            "bbox_policy": "框已还原到原图坐标；保留模型输出，不裁剪到图像边界，低分候选可能越界",
                            "nms_iou": 0.7, "minimum_score_from_model": 0.001,
                            "inference_seconds_total": duration, "image_count": len(records)},
               "images": records}
    if nms_mode == 'class_agnostic':
        from detection_display import group_detections, render_detections
        # 先保存未跨类去重数据，便于查看其他类别和复现原始评估。
        (output_dir / 'raw_predictions.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
        for rec in records:
            groups=group_detections(rec['detections'],display_nms_iou)
            rec['candidate_groups']=groups
            rec['detections']=[g['winner'] for g in groups]
            render_detections(rec['image'],rec['detections'],visual_dir/Path(rec['image']).name,draw_threshold)
        payload['metadata'].update({'nms_mode':'class_agnostic_greedy_on_exported_candidates',
            'candidate_nms_iou':0.7,'nms_iou':display_nms_iou,
            'display_rule':'每个IoU重叠组仅显示原候选最高分框；其余候选在candidate_groups与raw_predictions.json保留'})
    (output_dir / "predictions.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["metadata"], ensure_ascii=False, indent=2))
    return payload


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path, help="图片文件或任意图片目录，递归搜索")
    ap.add_argument("--weights", required=True, type=Path, help="PP-YOLOE训练权重路径")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "ppyoloe_s_bird_640.yml")
    ap.add_argument("--output", type=Path, default=ROOT / "runs" / "batch_infer")
    ap.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--draw-threshold", type=float, default=0.5,
                    help='默认仅画置信度严格大于0.5的去重框，其他候选仍保留为数据')
    ap.add_argument('--nms-mode', choices=['class_agnostic','classwise'],default='class_agnostic',
                    help='默认跨类别去重；classwise用于复现原基线候选与课程评估')
    ap.add_argument('--display-nms-iou',type=float,default=0.5)
    args = ap.parse_args()
    images = collect_images(args.input)
    if not images:
        raise SystemExit(f"找不到支持的图片：{args.input}")
    run_predict(images, args.config, args.weights, args.output, args.device, args.draw_threshold,
                args.nms_mode,args.display_nms_iou)


if __name__ == "__main__":
    main()
