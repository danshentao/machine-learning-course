"""短跑/正式训练入口。正式训练默认每轮保存检查点，以RF-DETR验证mAP50:95选最佳。"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime/local_packages"))
from rfdetr import RFDETRSmall  # noqa: E402

CFG = json.loads((ROOT / "configs/experiment.json").read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--epochs", type=int)
    args = ap.parse_args()
    random.seed(CFG["seed"])
    np.random.seed(CFG["seed"])
    torch.manual_seed(CFG["seed"])
    torch.cuda.manual_seed_all(CFG["seed"])
    torch.set_num_threads(4)
    out = ROOT / ("runs/smoke" if args.smoke else "runs/formal")
    out.mkdir(parents=True, exist_ok=True)
    dataset = ROOT / ("data/smoke" if args.smoke else "data/dataset")
    epochs = 1 if args.smoke else (args.epochs or CFG["epochs"])
    batch = 1
    grad_accum = 2 if args.smoke else CFG["grad_accum_steps"]
    pretrain_cache = ROOT / "weights/cache"
    os.environ["RF_HOME"] = str(pretrain_cache)
    os.environ["ROBOFLOW_HOME"] = str(pretrain_cache)
    os.environ["HF_HOME"] = str(ROOT / "cache/huggingface")
    model = RFDETRSmall(
        num_classes=1,
        device="cuda",
        resolution=CFG["resolution"],
        gradient_checkpointing=True,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("RF-DETR正式训练要求本机CUDA；未切换CPU训练")
    # 先用真实图做一次推理，检查官方预处理、分数和原图像素坐标范围。
    probe = next((dataset / "valid").glob("*.jpg"))
    before = torch.cuda.max_memory_allocated()
    pred_start = time.perf_counter()
    pred = model.predict(str(probe), threshold=0.001)
    pred_elapsed = time.perf_counter() - pred_start
    det = pred[0] if isinstance(pred, list) else pred
    if hasattr(det, "xyxy"):
        boxes = np.asarray(det.xyxy, dtype=float).reshape(-1, 4)
        scores = np.asarray(det.confidence, dtype=float).reshape(-1)
        from PIL import Image

        with Image.open(probe) as im:
            w, h = im.size
        if len(boxes) and (np.any(boxes[:, 0] < -1) or np.any(boxes[:, 1] < -1)
                           or np.any(boxes[:, 2] > w + 1) or np.any(boxes[:, 3] > h + 1)):
            raise AssertionError("RF-DETR预测框未映射回原图坐标")
    else:
        boxes = np.empty((0, 4))
        scores = np.empty((0,))
    train_kw = {
        "dataset_dir": str(dataset),
        "epochs": epochs,
        "batch_size": batch,
        "grad_accum_steps": grad_accum,
        "lr": 1e-4 if args.smoke else CFG["learning_rate"],
        "lr_encoder": 1.5e-5 if args.smoke else 1.5e-5,
        "weight_decay": CFG["weight_decay"],
        "num_workers": CFG["num_workers"],
        "eval_batch_size": 1,
        "amp_dtype": CFG["amp_dtype"],
        "use_ema": True,
        "checkpoint_interval": 1 if args.smoke else CFG["checkpoint_interval"],
        "eval_interval": 1,
        "early_stopping": False,
        "best_model_metric": "map",
        "seed": CFG["seed"],
        "output_dir": str(out),
        "tensorboard": False,
    }
    if args.resume:
        train_kw["resume"] = str(args.resume)
    report = {
        "run": "smoke" if args.smoke else "formal",
        "status": "starting",
        "model": "RFDETRSmall",
        "encoder": model.model_config.encoder,
        "resolution": model.model_config.resolution,
        "train_args": train_kw,
        "probe_image": str(probe),
        "probe_candidates_at_0.001": int(len(boxes)),
        "probe_seconds": pred_elapsed,
        "probe_peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
    }
    (out / "launcher_config.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    start = time.perf_counter()
    if args.smoke:
        torch.cuda.reset_peak_memory_stats()
    model.train(**train_kw)
    report["status"] = "completed"
    report["elapsed_seconds"] = time.perf_counter() - start
    report["peak_cuda_memory_gib"] = torch.cuda.max_memory_allocated() / 1024**3
    (out / "launcher_config.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
