"""从已归档的单类COCO清单重建隔离的数据副本，排除待补标训练来源。"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs/experiment.json").read_text(encoding="utf-8"))
SOURCE_IMAGES = Path(CFG["gen1_prepared_images"])
DEST = ROOT / "data/dataset"
EXCLUDED = set(CFG["excluded_train_sources"])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_split(split: str, source_json: Path, source_split: str) -> dict:
    source = json.loads(source_json.read_text(encoding="utf-8"))
    images = source["images"]
    anns = source["annotations"]
    if split == "train":
        excluded_names = {name for stem in EXCLUDED for name in (stem, stem[:-4] + "_brightness.jpg")}
        excluded_ids = {im["id"] for im in images if im["file_name"] in excluded_names}
        images = [im for im in images if im["id"] not in excluded_ids]
        anns = [a for a in anns if a["image_id"] not in excluded_ids]
        if any(im["file_name"] in excluded_names for im in images):
            raise RuntimeError("隔离图像仍出现在训练清单")
    dest_dir = DEST / ("train" if split == "train" else "valid")
    dest_dir.mkdir(parents=True, exist_ok=True)
    ids = {im["id"] for im in images}
    anns = [a for a in anns if a["image_id"] in ids]
    copied = []
    for row in images:
        src = SOURCE_IMAGES / source_split / row["file_name"]
        if not src.is_file():
            raise FileNotFoundError(src)
        dst = dest_dir / row["file_name"]
        if not dst.exists() or sha256(dst) != sha256(src):
            shutil.copy2(src, dst)
        if sha256(dst) != sha256(src):
            raise IOError(f"复制后哈希不一致: {row['file_name']}")
        copied.append({"file_name": row["file_name"], "sha256": sha256(src), "source": str(src)})
    # 类别统一为一个bird；所有bbox/area数值照原始固定COCO记录保留，不做坐标偏移。
    out = {k: v for k, v in source.items() if k not in {"images", "annotations", "categories"}}
    out["images"] = images
    out["annotations"] = anns
    out["categories"] = [{"id": 0, "name": "bird", "supercategory": "animal"}]
    target = dest_dir / "_annotations.coco.json"
    target.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    brightness_count = sum(row["file_name"].lower().endswith("_brightness.jpg") for row in images)
    return {
        "split": split,
        "source_json": str(source_json),
        "source_split": source_split,
        "source_image_rows": len(source["images"]),
        "source_annotations": len(source["annotations"]),
        "output_images": len(images),
        "independent_original_sources": len(images) - brightness_count,
        "brightness_derivatives": brightness_count,
        "output_annotations": len(anns),
        "output_json": str(target),
        "output_json_sha256": sha256(target),
        "excluded_image_ids": sorted(excluded_ids) if split == "train" else [],
        "image_files": copied,
    }


def main() -> None:
    archive = Path(CFG["archive"])
    train = build_split("train", Path(CFG["train_json"]), "train")
    valid = build_split("valid", Path(CFG["fixed_validation_json"]), "val")
    # 固定验证身份必须逐文件等于正式单类验证清单。
    source_val = json.loads(Path(CFG["fixed_validation_json"]).read_text(encoding="utf-8"))
    valid_names = [x["file_name"] for x in json.loads(Path(valid["output_json"]).read_text(encoding="utf-8"))["images"]]
    expected_names = [x["file_name"] for x in source_val["images"]]
    if valid_names != expected_names:
        raise RuntimeError("验证图身份/顺序与正式固定验证集不一致")
    report = {
        "status": "passed",
        "train": {k: v for k, v in train.items() if k != "image_files"},
        "validation": {k: v for k, v in valid.items() if k != "image_files"},
        "train_image_files_sha256_manifest": train["image_files"],
        "validation_image_files_sha256_manifest": valid["image_files"],
        "validation_identity_unchanged": True,
        "categories": ["bird"],
        "training_source_exclusions": sorted(EXCLUDED),
        "source_archive": str(archive),
        "note": "所有训练增强图只沿用归档训练清单；jiamaque_087.jpg及brightness衍生图从训练副本排除；未修改源图/XML/归档。",
    }
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/data_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"train_images": train["output_images"], "train_boxes": train["output_annotations"],
                      "train_independent_original_sources": train["independent_original_sources"],
                      "train_brightness_derivatives": train["brightness_derivatives"],
                      "valid_images": valid["output_images"], "valid_boxes": valid["output_annotations"],
                      "audit": str(ROOT / 'reports/data_audit.json')}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
