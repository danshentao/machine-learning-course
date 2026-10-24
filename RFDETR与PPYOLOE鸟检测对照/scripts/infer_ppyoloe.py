"""重跑网站正式PP-YOLOE+ s，保留官方NMS后的完整低分候选（上限300/图）。"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs/experiment.json").read_text(encoding="utf-8"))
PADDLE_DET = Path("E:/机器学习课设/自然场景鸟类检测PPYOLOE/third_party/PaddleDetection")
PADDLE_SITE = Path("E:/机器学习课设/自然场景鸟类检测PPYOLOE/.venv/Lib/site-packages")
# 旧venv的python启动器指向已不存在的C盘基础解释器；由E盘内嵌Python运行，显式复用原venv包目录。
# RF-DETR环境的NumPy 2依赖不能混入旧Paddle环境，避免cv2/scikit-learn/pyarrow ABI冲突。
_rf_paths = {(ROOT / "runtime/local_packages").resolve(), (ROOT / "runtime/shareddeps").resolve()}
sys.path = [p for p in sys.path if Path(p).resolve() not in _rf_paths]
sys.path.insert(0, str(PADDLE_SITE))
sys.path.insert(0, str(PADDLE_DET))
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from ppdet.core.workspace import load_config  # noqa: E402
from ppdet.engine import Trainer  # noqa: E402


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def clip(box: list[float], width: int, height: int) -> list[float] | None:
    out = [min(max(float(x), 0.0), float(lim)) for x, lim in zip(box, [width, height, width, height])]
    return out if out[2] > out[0] and out[3] > out[1] else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["fixed_val", "external80"], required=True)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu",
                        help="当前部署权重环境缺少匹配的cuDNN 8时使用CPU；GPU需要该环境依赖已就绪")
    args = parser.parse_args()
    source_config = Path(CFG["deployed_detector_config"])
    weights = Path(CFG["deployed_detector_weights"])
    if not weights.is_file():
        raise FileNotFoundError(weights)
    try:
        text = source_config.read_text(encoding="utf-8")
        encoding = "utf-8"
    except UnicodeDecodeError:
        text = source_config.read_text(encoding="gb18030")
        encoding = "gb18030"
    import yaml

    config_doc = yaml.safe_load(text)
    config_doc["_BASE_"] = [str((source_config.parent / x).resolve()) for x in config_doc["_BASE_"]]
    outdir = ROOT / "runs/baseline_ppyoloe"
    outdir.mkdir(parents=True, exist_ok=True)
    local_config = outdir / "resolved_deployed_config.yml"
    local_config.write_text(yaml.safe_dump(config_doc, allow_unicode=False), encoding="utf-8")
    cfg = load_config(str(local_config))
    cfg.weights = str(weights)
    cfg.worker_num = 0
    cfg.TestReader["inputs_def"]["image_shape"] = [3, 640, 640]
    for transform in cfg.TestReader["sample_transforms"]:
        if "Resize" in transform:
            transform["Resize"]["target_size"] = [640, 640]
            transform["Resize"]["keep_ratio"] = False
    nms = cfg.PPYOLOEHead["nms"]
    nms["score_threshold"] = 0.001
    nms["nms_threshold"] = 0.7
    nms["nms_top_k"] = 1000
    nms["keep_top_k"] = 300
    cfg.TestReader.update({"batch_size": 1, "worker_num": 0})

    if args.split == "fixed_val":
        ann = json.loads(Path(CFG["fixed_validation_json"]).read_text(encoding="utf-8"))
        image_paths = [Path(CFG["gen1_prepared_images"]) / "val" / row["file_name"] for row in ann["images"]]
    else:
        image_root = Path(CFG["external_image_dir"])
        # 外来图按递归清单读取；子目录只提供文件位置，不进入模型或类别决策。
        image_paths = sorted([p for p in image_root.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}], key=lambda p: p.name.casefold())
        if len(image_paths) != 80:
            raise RuntimeError(f"外来图片数应为80，实际{len(image_paths)}: {image_root}")
        if len({p.name for p in image_paths}) != len(image_paths):
            raise RuntimeError("外来图存在重复文件名，不能安全关联现有同源审查标记")
    missing = [str(p) for p in image_paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(missing[:5])

    import paddle

    if args.device == "gpu" and not paddle.is_compiled_with_cuda():
        raise RuntimeError("网站正式PP-YOLOE基准环境没有CUDA")
    paddle.set_device(args.device)
    cudnn_query_skipped = False
    if args.device == "cpu":
        # 该旧GPU wheel在CPU建层时也查询cuDNN 8。仅在明确CPU设备时跳过这个版本查询；
        # 卷积实际仍由Paddle CPU算子执行，不改模型、权重或预测后处理。
        import paddle.nn.layer.conv as paddle_conv

        paddle_conv.get_cudnn_version = lambda: 0
        cudnn_query_skipped = True
    trainer = Trainer(cfg, mode="test")
    trainer.load_weights(str(weights))
    input_shapes: list[list[int]] = []
    forward = trainer.model.forward

    def record(inputs, *args, **kwargs):
        shape = list(inputs["image"].shape)
        if shape not in input_shapes:
            input_shapes.append(shape)
        return forward(inputs, *args, **kwargs)

    trainer.model.forward = record
    start = time.perf_counter()
    outputs = trainer.predict([str(p) for p in image_paths], output_dir=str(outdir / f"predict_{args.split}"), save_results=False, visualize=False)
    image_id_to_path = trainer.dataset.get_imid2path()
    rows_by_name: dict[str, dict] = {}
    for batch in outputs:
        boxes = np.asarray(batch["bbox"])
        image_ids = np.asarray(batch["im_id"]).reshape(-1)
        counts = np.asarray(batch["bbox_num"]).reshape(-1)
        offset = 0
        for image_id, count in zip(image_ids, counts):
            path = Path(image_id_to_path[int(image_id)])
            part = boxes[offset : offset + int(count)]
            offset += int(count)
            with Image.open(path) as im:
                width, height = im.size
            candidates = []
            # 保留官方NMS输出的低分候选；NMS前算子按正式配置，额外展示去重在后续共同处理。
            for ix, value in enumerate(sorted(part, key=lambda x: -float(x[1]))[:300]):
                raw = list(map(float, value[2:6]))
                clipped = clip(raw, width, height)
                if clipped is None:
                    continue
                candidates.append({
                    "candidate_index": ix,
                    "bbox_xyxy": clipped,
                    "raw_bbox_xyxy": raw,
                    "detector_score": float(value[1]),
                    "class_id": int(value[0]),
                    "class_name": "bird",
                    "box_clipped": clipped != raw,
                })
            rows_by_name[path.name] = {
                "filename": path.name,
                "image_path": str(path),
                "width": width,
                "height": height,
                "native_candidates": candidates,
            }
    if len(rows_by_name) != len(image_paths):
        raise RuntimeError(f"输出图片数不匹配: {len(rows_by_name)} != {len(image_paths)}")
    metadata = {
        "model": "PaddleDetection PP-YOLOE+ s single class bird",
        "paddle_version": paddle.__version__,
        "numpy_version": np.__version__,
        "split": args.split,
        "weights": str(weights),
        "weights_sha256": sha(weights),
        "config": str(source_config),
        "config_sha256": sha(source_config),
        "config_encoding": encoding,
        "input_shapes": input_shapes,
        "input_size": [640, 640],
        "inference_device": args.device,
        "cudnn_version_query_skipped_for_cpu": cudnn_query_skipped,
        "resize": "square stretch; keep_ratio=false; original aspect ratio not preserved",
        "preprocess_from_resolved_test_reader": cfg.TestReader["sample_transforms"],
        "postprocess_from_model": dict(nms),
        "native_outputs": "PaddleDetection model postprocess/NMS, score floor .001, at most 300 per image",
        "image_count": len(image_paths),
        "elapsed_seconds": time.perf_counter() - start,
        "coordinates": "clipped xyxy in original-image pixel coordinates",
    }
    ordered = [rows_by_name[p.name] for p in image_paths]
    target = outdir / f"{args.split}_raw.json"
    target.write_text(json.dumps({"metadata": metadata, "images": ordered}, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"split": args.split, "images": len(ordered), "candidates": sum(len(x["native_candidates"]) for x in ordered),
                      "seconds": metadata["elapsed_seconds"], "input_shapes": input_shapes, "output": str(target)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
