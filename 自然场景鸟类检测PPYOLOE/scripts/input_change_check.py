"""同一权重对固定八张验证图做原图/50%分辨率输入检查，并映射回原图坐标。"""
import argparse
import csv
import json
from pathlib import Path

from PIL import Image

from batch_infer import ROOT, CLASS_NAMES, run_predict
from evaluate_predictions import iou


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ["empty"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "ppyoloe_s_bird_640.yml")
    ap.add_argument("--output", type=Path, default=ROOT / "runs" / "input_change_check")
    ap.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    args = ap.parse_args(); out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    gt = json.loads((ROOT / "data" / "prepared" / "annotations" / "instances_val.json").read_text(encoding="utf-8"))
    by_image = {x["id"]: [] for x in gt["images"]}
    for ann in gt["annotations"]: by_image[ann["image_id"]].append(ann)
    selected, used_images = [], set()
    for cid, name in enumerate(CLASS_NAMES):
        candidates = []
        for image in gt["images"]:
            if image["file_name"] in used_images: continue
            if any(a["category_id"] == cid for a in by_image[image["id"]]): candidates.append(image)
        if not candidates:
            raise RuntimeError(f"固定验证集无{ name }类别样本，无法执行八类输入变化检查")
        pick = min(candidates, key=lambda x: x["file_name"])
        used_images.add(pick["file_name"])
        selected.append({"target_class_id": cid, "target_class": name,
                         "filename": pick["file_name"], "path": str((ROOT / "data" / "prepared" / "images" / "val" / pick["file_name"]).resolve()),
                         "selection_rule": "按类别编号逐类选择验证集文件名排序最前且未被其他类别选中的实际含该类实例图片"})
    with (out / "selection.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(selected[0])); w.writeheader(); w.writerows(selected)
    originals = [Path(x["path"]) for x in selected]
    degraded_dir = out / "images_downsampled_50pct"; degraded_dir.mkdir(exist_ok=True)
    degraded_for = {}
    for row in selected:
        path = Path(row["path"])
        with Image.open(path) as im:
            src = im.convert("RGB"); low_size = (max(1, round(src.width*0.5)), max(1, round(src.height*0.5)))
            low = src.resize(low_size, Image.Resampling.BILINEAR)
            dest = degraded_dir / (path.stem + "_50pct.png"); low.save(dest)
            degraded_for[path] = {"path": dest, "scale_x": src.width/low.width, "scale_y": src.height/low.height,
                                  "original_size": src.size, "downsampled_size": low.size}
    pred_orig = run_predict(originals, args.config, args.weights, out / "original", args.device)
    low_paths = [degraded_for[p]["path"] for p in originals]
    pred_low = run_predict(low_paths, args.config, args.weights, out / "downsampled", args.device)
    original_by_path = {Path(x["image"]).resolve(): x for x in pred_orig["images"]}
    low_by_stem = {Path(x["image"]).stem.removesuffix("_50pct"): x for x in pred_low["images"]}
    details, image_summary = [], []
    for row, original_path in zip(selected, originals):
        a = original_by_path[original_path.resolve()]
        b = low_by_stem[original_path.stem]
        scale = degraded_for[original_path]
        low_dets = []
        for det in b["detections"]:
            box = det["bbox_xyxy"]
            mapped = [box[0]*scale["scale_x"], box[1]*scale["scale_y"], box[2]*scale["scale_x"], box[3]*scale["scale_y"]]
            low_dets.append({**det, "bbox_xyxy": mapped, "bbox_in_downsampled_coordinates": box})
        image_gt = next(i for i in gt["images"] if i["file_name"] == row["filename"])
        gts = by_image[image_gt["id"]]
        matched_a, matched_b = set(), set()
        for gi, truth in enumerate(gts):
            x, y, w, h = truth["bbox"]; truth_box = [x, y, x+w, y+h]
            row_out = {"image": row["filename"], "target_class": row["target_class"],
                       "gt_instance": gi+1, "gt_class": CLASS_NAMES[truth["category_id"]],
                       "gt_bbox_xyxy": json.dumps(truth_box, ensure_ascii=False)}
            for label, detections, used in [("original", a["detections"], matched_a), ("downsampled", low_dets, matched_b)]:
                options = [(iou(det["bbox_xyxy"], truth_box), j, det) for j, det in enumerate(detections) if j not in used]
                ov, j, det = max(options, default=(0., -1, None))
                if det:
                    used.add(j)
                    row_out.update({f"{label}_prediction_class": det["class_name"], f"{label}_score": det["score"],
                                    f"{label}_iou_to_gt": ov, f"{label}_bbox_original_xyxy": json.dumps(det["bbox_xyxy"], ensure_ascii=False),
                                    f"{label}_correct_class_iou50": det["class_id"] == truth["category_id"] and ov >= .5})
                else:
                    row_out.update({f"{label}_prediction_class": "", f"{label}_score": "", f"{label}_iou_to_gt": 0,
                                    f"{label}_bbox_original_xyxy": "", f"{label}_correct_class_iou50": False})
            details.append(row_out)
        image_summary.append({"image": row["filename"], "target_class": row["target_class"],
            "original_dimensions": scale["original_size"], "downsampled_dimensions": scale["downsampled_size"],
            "original_prediction_count": len(a["detections"]), "downsampled_prediction_count": len(low_dets),
            "original_predictions": a["detections"], "downsampled_predictions_mapped_to_original": low_dets})
    write_csv(out / "per_instance_comparison.csv", details)
    (out / "comparison.json").write_text(json.dumps({"method": "same trained weights; input downsampled with bilinear to ~50% then official 640x640 preprocessing; downsampled output boxes scaled back to source coordinates",
        "images": image_summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"八图输入变化检查已写入：{out}")


if __name__ == "__main__": main()
