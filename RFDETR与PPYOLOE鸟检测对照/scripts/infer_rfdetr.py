"""使用官方RF-DETR API推理固定验证或外来80图，保存原生低分候选。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime/local_packages"))
from rfdetr import RFDETRSmall  # noqa: E402

CFG = json.loads((ROOT / "configs/experiment.json").read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["fixed_val", "external80"], required=True)
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "runs/formal/checkpoint_best_total.pth")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    checkpoint = a.checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    os.environ["RF_HOME"] = str(ROOT / "weights/cache")
    os.environ["ROBOFLOW_HOME"] = str(ROOT / "weights/cache")
    os.environ["HF_HOME"] = str(ROOT / "cache/huggingface")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(ROOT / "cache/huggingface/hub")
    if a.smoke:
        image_paths = [next((ROOT / "data/smoke/valid").glob("*.jpg"))]
    elif a.split == "fixed_val":
        ann = json.loads(Path(CFG["fixed_validation_json"]).read_text(encoding="utf-8"))
        image_paths = [ROOT / "data/dataset/valid" / row["file_name"] for row in ann["images"]]
    else:
        image_root = Path(CFG["external_image_dir"])
        # 外来图递归枚举；类别目录不参与预测，只为找到图片文件。
        image_paths = sorted([p for p in image_root.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}], key=lambda p: p.name.casefold())
        if len(image_paths) != 80:
            raise RuntimeError(f"外来图片数应为80，实际{len(image_paths)}")
        if len({p.name for p in image_paths}) != len(image_paths):
            raise RuntimeError("外来图存在重复文件名，不能安全关联既有同源审查标记")
    if any(not p.is_file() for p in image_paths):
        raise FileNotFoundError(next(str(p) for p in image_paths if not p.is_file()))
    model = RFDETRSmall.from_checkpoint(str(checkpoint), device="cuda")
    rows = []
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for path in image_paths:
        pred = model.predict(str(path), threshold=0.001)
        det = pred[0] if isinstance(pred, list) else pred
        boxes = np.asarray(det.xyxy, dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(det.confidence, dtype=np.float32).reshape(-1)
        classes = np.asarray(det.class_id, dtype=np.int64).reshape(-1)
        if not (len(boxes) == len(scores) == len(classes)):
            raise RuntimeError(f"预测数量不匹配: {path.name}")
        with Image.open(path) as im:
            width, height = im.size
        candidates = []
        for i, (box, score, class_id) in enumerate(zip(boxes, scores, classes)):
            raw = [float(x) for x in box]
            clipped = [min(max(raw[j], 0.0), float([width, height, width, height][j])) for j in range(4)]
            if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                continue
            candidates.append({
                "candidate_index": i,
                "bbox_xyxy": clipped,
                "raw_bbox_xyxy": raw,
                "detector_score": float(score),
                "class_id": int(class_id),
                "class_name": "bird",
                "box_clipped": clipped != raw,
            })
        rows.append({"filename": path.name, "image_path": str(path), "width": width, "height": height,
                     "native_candidates": candidates})
    elapsed = time.perf_counter() - start
    metadata = {
        "model": "RF-DETR Small, one class bird",
        "rfdetr_version": "1.11.2",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha(checkpoint),
        "source_pretrained_sha256": sha(ROOT / "weights/cache/rf-detr-small.pth"),
        "split": "smoke" if a.smoke else a.split,
        "input_size": model.model_config.resolution,
        "backbone": model.model_config.encoder,
        "prediction_threshold": 0.001,
        "native_outputs": "official RF-DETR postprocess; top query detections, before common IoU grouping",
        "image_count": len(rows),
        "seconds": elapsed,
        "seconds_per_image": elapsed / max(len(rows), 1),
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "coordinates": "prediction boxes returned in original-image pixel coordinates and clipped to image bounds",
    }
    outdir = ROOT / "runs/formal"
    outdir.mkdir(parents=True, exist_ok=True)
    filename = "smoke_predictions.json" if a.smoke else f"rfdetr_{a.split}_raw.json"
    target = outdir / filename
    target.write_text(json.dumps({"metadata": metadata, "images": rows}, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"split": metadata["split"], "images": len(rows), "candidates": sum(len(r["native_candidates"]) for r in rows),
                      "seconds": elapsed, "peak_cuda_memory_gib": metadata["peak_cuda_memory_gib"], "output": str(target)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
