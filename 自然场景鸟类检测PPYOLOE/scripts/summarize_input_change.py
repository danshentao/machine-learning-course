"""用固定八图的真值及预测文件总结分辨率变化的同类IoU@0.5结果。"""
from __future__ import annotations

import csv
import argparse
import json
from collections import defaultdict
from pathlib import Path

from evaluate_predictions import iou
from build_comparison_artifacts import write_csv

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "baseline_640" / "input_change_check"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, default=OUT)
    ap.add_argument('--output', type=Path, default=OUT / 'corrected_summary')
    args = ap.parse_args()
    source = args.input.resolve()
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    gt_doc = json.loads((ROOT / "data" / "prepared" / "annotations" / "instances_val.json").read_text(encoding="utf-8"))
    id_to_name = {x["id"]: x["file_name"] for x in gt_doc["images"]}
    gts = defaultdict(list)
    for ann in gt_doc["annotations"]:
        x, y, w, h = ann["bbox"]
        gts[id_to_name[ann["image_id"]]].append({"class_id": ann["category_id"], "bbox": [x, y, x+w, y+h]})
    selection = list(csv.DictReader((source / "selection.csv").open(encoding="utf-8-sig")))
    # comparison.json已将低分辨率框还原到原图；不能用downsampled/predictions.json里的小图坐标匹配原图真值。
    comparison = json.loads((source / 'comparison.json').read_text(encoding='utf-8'))
    predsets = {
        'original': {Path(x['image']).stem: x['original_predictions'] for x in comparison['images']},
        'downsampled': {Path(x['image']).stem: x['downsampled_predictions_mapped_to_original'] for x in comparison['images']},
    }
    rows = []
    for sel in selection:
        filename = sel["filename"]
        cid = int(sel["target_class_id"])
        image_truth = [x for x in gts[filename] if x["class_id"] == cid]
        row = {"image": filename, "target_class": sel["target_class"], "target_class_gt_count": len(image_truth),
               "score_threshold": .25, "IoU_threshold": .5,
               "selection_rule": sel["selection_rule"], "source_path": sel["path"]}
        for key in ("original", "downsampled"):
            preds = predsets[key].get(Path(filename).stem, [])
            preds = [p for p in preds if p["score"] >= .25 and p["class_id"] == cid]
            preds.sort(key=lambda p: p["score"], reverse=True)
            used = set(); tp = 0
            for p in preds:
                candidates = [(iou(p["bbox_xyxy"], truth["bbox"]), j) for j, truth in enumerate(image_truth) if j not in used]
                ov, j = max(candidates, default=(0., -1))
                if ov >= .5:
                    used.add(j); tp += 1
            fn = len(image_truth)-tp; fp = len(preds)-tp
            row[f"{key}_target_class_predictions"] = len(preds)
            row[f"{key}_TP"] = tp; row[f"{key}_FP"] = fp; row[f"{key}_FN"] = fn
            row[f"{key}_precision"] = tp/max(tp+fp, 1)
            row[f"{key}_recall"] = tp/max(tp+fn, 1)
            row[f"{key}_F1"] = 2*row[f"{key}_precision"]*row[f"{key}_recall"]/max(row[f"{key}_precision"]+row[f"{key}_recall"], 1e-12)
        rows.append(row)
    write_csv(output / "input_change_summary.csv", rows)
    result = {"method": "固定验证集按类别逐类选图；同一权重、同一640官方预处理；低分辨率输入先双线性缩至约50%，预测框按实际宽高缩放回原图坐标。以下仅报告所选8张图的目标类别同类IoU>=0.5匹配，不混入整体验证指标或调参。",
              "metrics_thresholds": {"score": .25, "IoU": .5}, "images": rows}
    result['correction'] = '原汇总错误地用小图坐标与原图真值匹配，产生低分辨率TP=0；本次使用已还原到原图的坐标，原汇总保留仅作历史记录。'
    result['total_TP'] = {key: sum(r[f'{key}_TP'] for r in rows) for key in ('original', 'downsampled')}
    (output / "input_change_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result['total_TP'], ensure_ascii=False))
    print(f"八图逐图同类匹配汇总已写入：{output}")


if __name__ == "__main__":
    main()
