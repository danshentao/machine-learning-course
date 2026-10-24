"""为类别无关NMS实验生成汇总表和可复核的代表图。"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "baseline_640"
OUT = RUN / "nms_class_agnostic"
FONT_PATH = Path("C:/Windows/Fonts/msyh.ttc")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1)*max(0, y2-y1)
    area_a = max(0, a[2]-a[0])*max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0])*max(0, b[3]-b[1])
    return inter / max(area_a+area_b-inter, 1e-12)


def add_box(draw, box, text, color, scale, font):
    xy = tuple(round(v*scale) for v in box)
    draw.rectangle(xy, outline=color, width=max(2, round(3*scale)))
    x, y = xy[0], max(0, xy[1]-round(24*scale))
    draw.text((x, y), text, fill=color, font=font, stroke_width=1, stroke_fill="black")


def main():
    base = read_json(RUN / "ap50_epoch_13_predictions" / "predictions.json")
    variant = read_json(OUT / "predictions.json")
    gt_doc = read_json(ROOT / "data" / "prepared" / "annotations" / "instances_val.json")
    mapping = read_json(ROOT / "configs" / "class_mapping.json")
    names = {x["category_id"]: x["standard_name"] for x in mapping["classes"]}
    image_by_name = {x["file_name"]: x for x in gt_doc["images"]}
    anns = {x["id"]: [] for x in gt_doc["images"]}
    for a in gt_doc["annotations"]:
        anns[a["image_id"]].append(a)
    base_by_name = {Path(x["image"]).name: x for x in base["images"]}
    var_by_name = {Path(x["image"]).name: x for x in variant["images"]}
    cases = [
        ("dashanque_076.jpg", "高分煤山雀框压过基线正确大山雀框"),
        ("heihouhongtunbei_005.jpg", "高分煤山雀框压过基线正确黑喉红臀鹎框"),
        ("jiamaque_003.jpg", "高分煤山雀框压过基线正确家麻雀框"),
        ("meishanque_068.jpg", "高分大山雀框压过基线正确煤山雀框"),
        ("shumaque_095.jpg", "跨类别抑制保留家麻雀预测，移除了树麻雀正确预测"),
    ]
    panel_dir = OUT / "representative_examples"
    panel_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        font = ImageFont.truetype(str(FONT_PATH), 20)
        title_font = ImageFont.truetype(str(FONT_PATH), 24)
    except OSError:
        font = title_font = ImageFont.load_default()

    for filename, explanation in cases:
        image = image_by_name[filename]
        gt_list = anns[image["id"]]
        gt_boxes = [{"class": names[a["category_id"]], "bbox": [a["bbox"][0], a["bbox"][1], a["bbox"][0]+a["bbox"][2], a["bbox"][1]+a["bbox"][3]]} for a in gt_list]
        bpred = [p for p in base_by_name[filename]["detections"] if p["score"] >= .25]
        vpred = [p for p in var_by_name[filename]["detections"] if p["score"] >= .25]
        source = ROOT / "data" / "prepared" / "images" / "val" / filename
        original = Image.open(source).convert("RGB")
        scale = min(1.0, 1000 / original.width)
        canvas_w, canvas_h = round(original.width*scale), round(original.height*scale)
        panels = []
        for label, preds, pred_color in [("GT", [], "#32d74b"), ("基线", bpred, "#45a1ff"), ("类别无关NMS", vpred, "#ff453a")]:
            panel = original.resize((canvas_w, canvas_h), Image.Resampling.BILINEAR).copy()
            draw = ImageDraw.Draw(panel)
            draw.text((8, 8), label, fill="white", font=title_font, stroke_width=3, stroke_fill="black")
            if label == "GT":
                for g in gt_boxes:
                    add_box(draw, g["bbox"], g["class"], pred_color, scale, font)
            else:
                for p in preds:
                    add_box(draw, p["bbox_xyxy"], f'{p["class_name"]} {p["score"]:.2f}', pred_color, scale, font)
            panels.append(panel)
        margin = 8
        composite = Image.new("RGB", (canvas_w*3+margin*4, canvas_h+margin*2+40), "#252525")
        d = ImageDraw.Draw(composite)
        d.text((margin, 4), f"{filename} | {explanation}", font=font, fill="white")
        for i, panel in enumerate(panels):
            composite.paste(panel, (margin+i*(canvas_w+margin), 40))
        composite_path = panel_dir / (Path(filename).stem + "_comparison.jpg")
        composite.save(composite_path, quality=92)

        # 用每个真值框的最高IoU候选展示差别，数值供人工核对而非自动判物种。
        for idx, g in enumerate(gt_boxes, 1):
            r = {"image": filename, "gt_instance": idx, "gt_class_from_xml": g["class"],
                 "gt_bbox_xyxy": json.dumps(g["bbox"]), "baseline_prediction_count_at_0.25": len(bpred),
                 "class_agnostic_prediction_count_at_0.25": len(vpred), "computed_case_note": explanation,
                 "source_image": str(source), "comparison_visual": str(composite_path)}
            for label, preds in [("baseline", bpred), ("class_agnostic", vpred)]:
                if preds:
                    p = max(preds, key=lambda p: iou(p["bbox_xyxy"], g["bbox"]))
                    r[f"{label}_class"] = p["class_name"]
                    r[f"{label}_score"] = p["score"]
                    r[f"{label}_iou_to_this_gt"] = iou(p["bbox_xyxy"], g["bbox"])
                    r[f"{label}_bbox_xyxy"] = json.dumps(p["bbox_xyxy"])
                else:
                    r[f"{label}_class"] = ""; r[f"{label}_score"] = ""; r[f"{label}_iou_to_this_gt"] = 0; r[f"{label}_bbox_xyxy"] = ""
                same_class = [p for p in preds if p["class_name"] == g["class"]]
                if same_class:
                    q = max(same_class, key=lambda p: (iou(p["bbox_xyxy"], g["bbox"]), p["score"]))
                    r[f"{label}_best_same_class_score"] = q["score"]
                    r[f"{label}_best_same_class_iou"] = iou(q["bbox_xyxy"], g["bbox"])
                    r[f"{label}_same_class_iou50_match"] = iou(q["bbox_xyxy"], g["bbox"]) >= .5
                else:
                    r[f"{label}_best_same_class_score"] = ""
                    r[f"{label}_best_same_class_iou"] = 0
                    r[f"{label}_same_class_iou50_match"] = False
            rows.append(r)
    write_csv(OUT / "representative_examples.csv", rows)

    bm = read_json(RUN / "evaluation_ap50_epoch_13" / "metrics.json")
    vm = read_json(OUT / "evaluation" / "metrics.json")
    summary = []
    for label, m in [("基线：640输入+类别内NMS", bm), ("比较：相同权重+类别无关NMS", vm)]:
        summary.append({"experiment": label, "mAP50": m["mAP50"], "micro_precision": m["precision"],
                        "micro_recall": m["recall"], "micro_F1": m["F1"], "TP": m["TP"], "FP": m["FP"], "FN": m["FN"],
                        "predictions_at_score_0.25": m["total_predictions_at_threshold"], "validation_images": m["validation_images"],
                        "score_threshold": m["score_threshold_for_P_R_F1_confusion"], "iou_threshold": m["iou_threshold"],
                        "NMS_IoU": m["nms_iou_threshold"], "weight": str(RUN / "best_model_ap50.pdparams"),
                        "note": "同一权重、固定验证集与评估口径；类别无关NMS改变跨类别抑制规则，属于后处理针对性比较。"})
    write_csv(OUT / "targeted_comparison_summary.csv", summary)
    (OUT / "targeted_comparison_summary.json").write_text(json.dumps({"question": "基线存在大量重叠跨类别预测；类别无关NMS能否减少这类重复误检？",
        "control": {"weights": str(RUN / "best_model_ap50.pdparams"), "validation_images": 172, "input_size": 640,
                    "candidate_minimum_score": 0.001, "metric_score_threshold": 0.25, "metric_iou_threshold": 0.5,
                    "NMS_IoU": 0.7, "same_training": True},
        "interpretation": "此实验只改变后处理的NMS类别约束，不重训模型。类别无关NMS提高micro Precision/F1，但降低Recall与mAP50；它压掉部分跨类重复框，也会让高分错误类别框压掉正确类别框。不能据此宣称总体检测能力提升。人工真值标签仍需复核。",
        "baseline": bm, "class_agnostic_nms": vm, "representative_cases_csv": str(OUT / "representative_examples.csv")}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"比较汇总、5个实例和可视化已写入：{OUT}")


if __name__ == "__main__":
    main()
