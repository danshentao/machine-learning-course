"""从官方训练日志找出按验证集AP50排名最高的轮次检查点候选。"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--checkpoint-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    current_epoch = None
    rows = []
    epoch_re = re.compile(r"Epoch:\s*\[(\d+)\]")
    ap50_re = re.compile(r"Average Precision\s+\(AP\)\s+@\[\s*IoU=0\.50\s*\|")
    value_re = re.compile(r"=\s*([0-9]+(?:\.[0-9]+)?)\s*$")
    for line in args.log.read_text(encoding="utf-8", errors="replace").splitlines():
        match = epoch_re.search(line)
        if match:
            current_epoch = int(match.group(1))
        if current_epoch is not None and ap50_re.search(line):
            value = value_re.search(line)
            if value:
                rows.append({"epoch_index_zero_based": current_epoch,
                             "epoch_number_one_based": current_epoch + 1,
                             "official_coco_ap50_rounded": float(value.group(1))})
    if not rows:
        raise SystemExit("训练日志中没有解析到逐轮COCO AP50，请保留原始日志并人工检查格式。")
    maximum = max(row["official_coco_ap50_rounded"] for row in rows)
    candidates = [row for row in rows if row["official_coco_ap50_rounded"] == maximum]
    for row in candidates:
        checkpoint = args.checkpoint_dir / f"{row['epoch_index_zero_based']}.pdparams"
        row["checkpoint"] = str(checkpoint.resolve())
        row["checkpoint_exists"] = checkpoint.is_file()
    doc = {"selection_metric": "官方COCO验证集AP@0.5，日志显示精度为三位小数",
           "rounded_max_ap50": maximum, "candidate_epochs": candidates,
           "all_epoch_ap50": rows,
           "note": "候选轮次须再用固定验证清单和项目评估器核对未舍入AP50；最终权重单独保存，不覆盖官方mAP@[0.50:0.95] best_model。"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: doc[k] for k in ["rounded_max_ap50", "candidate_epochs"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
