"""构造仅用于检查加载/损失/坐标的小型子集，不参与正式训练或指标。"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/dataset"
DST = ROOT / "data/smoke"


def split(src_name: str, dst_name: str, count: int) -> None:
    src_dir, dst_dir = SRC / src_name, DST / src_name
    dst_dir.mkdir(parents=True, exist_ok=True)
    doc = json.loads((src_dir / "_annotations.coco.json").read_text(encoding="utf-8"))
    selected = doc["images"][:count]
    ids = {r["id"] for r in selected}
    doc["images"] = selected
    doc["annotations"] = [a for a in doc["annotations"] if a["image_id"] in ids]
    (dst_dir / "_annotations.coco.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    for row in selected:
        shutil.copy2(src_dir / row["file_name"], dst_dir / row["file_name"])


def main() -> None:
    split("train", "train", 12)
    split("valid", "valid", 4)
    print("smoke subset prepared: 12 train images, 4 validation images; metrics are diagnostic only", flush=True)


if __name__ == "__main__":
    main()
