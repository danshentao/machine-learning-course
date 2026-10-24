"""在模型原始候选上比较类别无关NMS，模型权重和分数口径保持不变。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_predictions import iou


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.7)
    parser.add_argument("--score-threshold", type=float, default=0.001)
    parser.add_argument("--keep-top-k", type=int, default=300)
    args = parser.parse_args()

    source = json.loads(args.predictions.read_text(encoding="utf-8"))
    kept_total = 0
    for image in source["images"]:
        # 采用分数降序贪心NMS；抑制时不区分类别，只改变后处理规则。
        candidates = sorted((d for d in image["detections"] if d["score"] >= args.score_threshold),
                            key=lambda d: d["score"], reverse=True)
        kept = []
        for det in candidates:
            if all(iou(det["bbox_xyxy"], prev["bbox_xyxy"]) <= args.iou_threshold for prev in kept):
                kept.append(det)
                if len(kept) >= args.keep_top_k:
                    break
        image["detections"] = kept
        kept_total += len(kept)

    source["metadata"]["nms_mode"] = "class_agnostic_greedy_on_exported_candidates"
    source["metadata"]["nms_iou"] = args.iou_threshold
    source["metadata"]["minimum_score_from_model"] = args.score_threshold
    source["metadata"]["postprocess_note"] = "与基线使用同一模型权重和候选框，仅将NMS由类别内抑制改成跨类别贪心抑制；坐标不裁剪。"
    source["metadata"]["prediction_count_after_postprocess"] = kept_total
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(source, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"images": len(source["images"]), "detections_after_nms": kept_total,
                      "score_threshold": args.score_threshold, "nms_iou": args.iou_threshold,
                      "keep_top_k": args.keep_top_k}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
