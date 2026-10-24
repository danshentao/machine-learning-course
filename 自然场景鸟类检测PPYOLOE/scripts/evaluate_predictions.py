"""固定口径计算PP-YOLOE验证集的类别正确检测指标、AP及混淆矩阵。"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = [x["standard_name"] for x in json.loads((ROOT / "configs" / "class_mapping.json").read_text(encoding="utf-8"))["classes"]]


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0., x2-x1) * max(0., y2-y1)
    area_a = max(0., a[2]-a[0]) * max(0., a[3]-a[1])
    area_b = max(0., b[2]-b[0]) * max(0., b[3]-b[1])
    return inter / max(area_a + area_b - inter, 1e-12)


def read_truth(path):
    doc = json.loads(path.read_text(encoding="utf-8"))
    images = {x["file_name"]: {"width": x["width"], "height": x["height"], "gt": []} for x in doc["images"]}
    id_name = {x["id"]: x["file_name"] for x in doc["images"]}
    for ann in doc["annotations"]:
        x, y, w, h = ann["bbox"]
        images[id_name[ann["image_id"]]]["gt"].append({"class_id": ann["category_id"], "bbox": [x, y, x+w, y+h]})
    return images


def ap_101(gt_by_image, pred_by_image, cls, iou_threshold=0.5):
    total_gt = sum(sum(g["class_id"] == cls for g in vals) for vals in gt_by_image.values())
    if total_gt == 0:
        return 0.0
    preds = []
    for image, vals in pred_by_image.items():
        preds.extend((float(p["score"]), image, p) for p in vals if p["class_id"] == cls)
    preds.sort(key=lambda x: x[0], reverse=True)
    matched = defaultdict(set)
    tp, fp = [], []
    for score, image, pred in preds:
        options = [(iou(pred["bbox_xyxy"], gt["bbox"]), j) for j, gt in enumerate(gt_by_image[image])
                   if gt["class_id"] == cls and j not in matched[image]]
        best_iou, best_j = max(options, default=(0., -1))
        ok = best_iou >= iou_threshold
        if ok:
            matched[image].add(best_j)
        tp.append(1 if ok else 0); fp.append(0 if ok else 1)
    if not tp:
        return 0.0
    tp = np.cumsum(tp); fp = np.cumsum(fp)
    recall = tp / total_gt
    precision = tp / np.maximum(tp + fp, 1e-12)
    return float(np.mean([np.max(precision[recall >= r], initial=0.0) for r in np.linspace(0, 1, 101)]))


def class_correct_counts(gt_by_image, pred_by_image, threshold, iou_threshold):
    tp = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    pred_total = np.zeros_like(tp); gt_total = np.zeros_like(tp)
    for image in gt_by_image:
        gts = gt_by_image[image]
        preds = sorted([p for p in pred_by_image.get(image, []) if p["score"] >= threshold],
                       key=lambda p: p["score"], reverse=True)
        gt_total += np.bincount([g["class_id"] for g in gts], minlength=len(CLASS_NAMES))
        pred_total += np.bincount([p["class_id"] for p in preds], minlength=len(CLASS_NAMES))
        used = set()
        for p in preds:
            options = [(iou(p["bbox_xyxy"], g["bbox"]), j) for j, g in enumerate(gts)
                       if j not in used and g["class_id"] == p["class_id"]]
            score, j = max(options, default=(0., -1))
            if score >= iou_threshold:
                tp[p["class_id"]] += 1; used.add(j)
    fp = pred_total - tp
    fn = gt_total - tp
    return tp, fp, fn, pred_total, gt_total


def build_confusion(gt_by_image, pred_by_image, threshold, iou_threshold):
    n = len(CLASS_NAMES); bg = n
    matrix = np.zeros((n+1, n+1), dtype=np.int64)
    for image, vals in gt_by_image.items():
        gts = vals
        preds = sorted([p for p in pred_by_image.get(image, []) if p["score"] >= threshold],
                       key=lambda p: p["score"], reverse=True)
        candidates = []
        for pi, p in enumerate(preds):
            for gi, g in enumerate(gts):
                overlap = iou(p["bbox_xyxy"], g["bbox"])
                if overlap >= iou_threshold:
                    candidates.append((overlap, p["score"], pi, gi))
        candidates.sort(reverse=True)
        used_p, used_g = set(), set()
        for overlap, score, pi, gi in candidates:
            if pi not in used_p and gi not in used_g:
                matrix[gts[gi]["class_id"], preds[pi]["class_id"]] += 1
                used_p.add(pi); used_g.add(gi)
        for gi, g in enumerate(gts):
            if gi not in used_g: matrix[g["class_id"], bg] += 1
        for pi, p in enumerate(preds):
            if pi not in used_p: matrix[bg, p["class_id"]] += 1
    return matrix


def error_rows(gt_by_image, pred_by_image, threshold, iou_threshold):
    rows = []
    for image, gts in gt_by_image.items():
        preds = sorted([p for p in pred_by_image.get(image, []) if p["score"] >= threshold],
                       key=lambda p: p["score"], reverse=True)
        # 先记录类别正确匹配，便于把其余预测区分成错类/定位/背景误检。
        used_g, used_p = set(), set()
        for pi, p in enumerate(preds):
            candidates = [(iou(p["bbox_xyxy"], g["bbox"]), gi) for gi, g in enumerate(gts)
                          if gi not in used_g and g["class_id"] == p["class_id"]]
            ov, gi = max(candidates, default=(0., -1))
            if ov >= iou_threshold:
                used_g.add(gi); used_p.add(pi)
        for pi, p in enumerate(preds):
            if pi in used_p: continue
            overlaps = [(iou(p["bbox_xyxy"], g["bbox"]), gi) for gi, g in enumerate(gts) if gi not in used_g]
            ov, gi = max(overlaps, default=(0., -1))
            if gi >= 0 and ov >= iou_threshold:
                rows.append({"image": image, "error_type": "类别错误", "ground_truth": CLASS_NAMES[gts[gi]["class_id"]],
                    "predicted": p["class_name"], "score": p["score"], "iou": ov, "gt_instance": gi+1,
                    "pred_bbox_xyxy": json.dumps(p["bbox_xyxy"], ensure_ascii=False),
                    "note": "定位达到IoU阈值，但标准类别不同；物种真值仍需人工复核。"})
                used_g.add(gi); used_p.add(pi)
            elif gi >= 0 and ov > 0.1:
                rows.append({"image": image, "error_type": "定位疑似不准", "ground_truth": CLASS_NAMES[gts[gi]["class_id"]],
                    "predicted": p["class_name"], "score": p["score"], "iou": ov, "gt_instance": gi+1,
                    "pred_bbox_xyxy": json.dumps(p["bbox_xyxy"], ensure_ascii=False),
                    "note": "与未匹配真值部分重叠但IoU不足0.5。"})
            else:
                rows.append({"image": image, "error_type": "误检", "ground_truth": "", "predicted": p["class_name"],
                    "score": p["score"], "iou": ov, "gt_instance": "",
                    "pred_bbox_xyxy": json.dumps(p["bbox_xyxy"], ensure_ascii=False), "note": "没有与未匹配真值充分重叠。"})
        for gi, g in enumerate(gts):
            if gi not in used_g:
                rows.append({"image": image, "error_type": "漏检", "ground_truth": CLASS_NAMES[g["class_id"]],
                    "predicted": "", "score": "", "iou": "", "gt_instance": gi+1,
                    "pred_bbox_xyxy": "", "note": "未找到分数阈值以上的同类匹配框；建议人工看全图及裁剪。"})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--ground-truth", type=Path, default=ROOT / "data" / "prepared" / "annotations" / "instances_val.json")
    ap.add_argument("--output", type=Path, default=ROOT / "runs" / "evaluation")
    ap.add_argument("--score-threshold", type=float, default=0.25)
    ap.add_argument("--iou-threshold", type=float, default=0.5)
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    gt_doc = read_truth(args.ground_truth)
    gt_by_image = {str((ROOT / "data" / "prepared" / "images" / "val" / name).resolve()): v["gt"] for name, v in gt_doc.items()}
    pred_doc = json.loads(args.predictions.read_text(encoding="utf-8"))
    pred_by_image = {str(Path(x["image"]).resolve()): [dict(p, bbox_xyxy=p["bbox_xyxy"]) for p in x["detections"]]
                     for x in pred_doc["images"]}
    # 保证没有预测框的验证图片也参与漏检及整体召回统计。
    for image in gt_by_image: pred_by_image.setdefault(image, [])
    tp, fp, fn, pred_n, gt_n = class_correct_counts(gt_by_image, pred_by_image, args.score_threshold, args.iou_threshold)
    aps = np.asarray([ap_101(gt_by_image, pred_by_image, cid, args.iou_threshold) for cid in range(len(CLASS_NAMES))])
    p = tp / np.maximum(pred_n, 1); r = tp / np.maximum(gt_n, 1); f1 = 2*p*r / np.maximum(p+r, 1e-12)
    total_tp, total_pred, total_gt = int(tp.sum()), int(pred_n.sum()), int(gt_n.sum())
    micro_p = total_tp / max(total_pred, 1); micro_r = total_tp / max(total_gt, 1)
    matrix = build_confusion(gt_by_image, pred_by_image, args.score_threshold, args.iou_threshold)
    class_rows = [{"category_id": i, "class_name": name, "AP50_101point": float(aps[i]),
                   "precision_at_threshold": float(p[i]), "recall_at_threshold": float(r[i]),
                   "F1_at_threshold": float(f1[i]), "TP": int(tp[i]), "FP": int(fp[i]), "FN": int(fn[i]),
                   "ground_truth_instances": int(gt_n[i]), "predictions_at_threshold": int(pred_n[i])}
                  for i, name in enumerate(CLASS_NAMES)]
    metrics = {"mAP50": float(aps.mean()), "AP50_by_class": class_rows,
        "precision": float(micro_p), "recall": float(micro_r), "F1": float(2*micro_p*micro_r/max(micro_p+micro_r, 1e-12)),
        "macro_precision": float(p.mean()), "macro_recall": float(r.mean()), "macro_F1": float(f1.mean()),
        "matching": "类别正确的一对一贪心匹配，先按分数降序，同类IoU>=阈值；重复框为FP",
        "confusion_matching": "IoU>=阈值的一对一全类别空间匹配，错类进入真值类×预测类；未匹配真值/预测落入背景列/行",
        "AP_method": "逐类别按分数排序，IoU>=阈值，101个Recall点插值平均；含所有预测分数，预测文件的最小保留分数为0.001",
        "score_threshold_for_P_R_F1_confusion": args.score_threshold,
        "nms_iou_threshold": pred_doc.get("metadata", {}).get("nms_iou", "配置记录"),
        "iou_threshold": args.iou_threshold, "total_GT": total_gt, "total_predictions_at_threshold": total_pred,
        "TP": total_tp, "FP": int(fp.sum()), "FN": int(fn.sum()), "validation_images": len(gt_by_image),
        "source_predictions": str(args.predictions)}
    (args.output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output / "metrics_by_class.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(class_rows[0])); w.writeheader(); w.writerows(class_rows)
    labels = CLASS_NAMES + ["背景/未匹配"]
    with (args.output / "confusion_matrix.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["真值\\预测"] + labels)
        for label, row in zip(labels, matrix): w.writerow([label] + row.tolist())
    errors = error_rows(gt_by_image, pred_by_image, args.score_threshold, args.iou_threshold)
    if errors:
        with (args.output / "error_cases.csv").open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(errors[0])); w.writeheader(); w.writerows(errors)
    try:
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        # Windows 上优先用系统微软雅黑，避免八类中文名在矩阵图中显示为空白方框。
        for font_path in [Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\simhei.ttf")]:
            if font_path.exists():
                font_manager.fontManager.addfont(str(font_path))
                plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
                break
        fig, ax = plt.subplots(figsize=(11, 9)); im = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), labels); ax.set_xlabel("预测类别"); ax.set_ylabel("真实类别")
        ax.set_title("检测混淆矩阵（IoU≥0.5，一对一匹配）")
        for y in range(matrix.shape[0]):
            for x in range(matrix.shape[1]): ax.text(x, y, str(matrix[y, x]), ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(args.output / "confusion_matrix.png", dpi=180); plt.close(fig)
    except ImportError:
        pass
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
