"""确定独立原图划分、规范化COCO标注并在训练集派生区放入合格brightness图。"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
SOURCE = WORKSPACE / "考核任务数据" / "考核任务数据" / "自然场景鸟类识别数据" / "数据集"
ORIGINAL_ZIP = WORKSPACE / "2804306_1790845704.zip"
BRIGHTNESS_ZIP = WORKSPACE / "2806021_1790929799.zip"
MAP_PATH = ROOT / "configs" / "class_mapping.json"
DATA = ROOT / "data" / "prepared"
REPORTS = ROOT / "reports"
SEED = 42
TEACHER_EXAMPLES = {
    "shumaque_019.jpg", "jiamaque_012.jpg", "shanmaque_012.jpg", "baitoubei_028.jpg",
    "hongerbei_046.jpg", "heihouhongtunbei_023.jpg", "dashanque_041.jpg", "meishanque_013.jpg",
}
# 目录与实例类别不一致的单例先隔离待复核，绝不由文件名前缀覆盖逐实例标签。
QUARANTINED_LABELS = {"dashanque_119.jpg"}


def csv_write(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def parse_xml(path: Path, names_to_id):
    root = ET.parse(path).getroot()
    size = root.find("size")
    width = int(size.findtext("width")); height = int(size.findtext("height"))
    objects = []
    for inst, item in enumerate(root.findall("object"), 1):
        raw = (item.findtext("name") or "").strip()
        name = "煤山雀" if raw == "眉山雀" else raw
        if name not in names_to_id:
            raise ValueError(f"未知标签，需人工复核: {path.name}, instance {inst}: {raw}")
        box = item.find("bndbox")
        xyxy = [float(box.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
        if not (xyxy[2] > xyxy[0] and xyxy[3] > xyxy[1]):
            raise ValueError(f"退化框，需人工复核: {path.name}, instance {inst}: {xyxy}")
        if xyxy[0] < 0 or xyxy[1] < 0 or xyxy[2] > width or xyxy[3] > height:
            raise ValueError(f"越界框，需人工复核: {path.name}, instance {inst}: {xyxy} vs {width}x{height}")
        # 保留XML四个端点的数值口径；COCO的宽高使用端点差，不人为加减1。
        objects.append({"id": None, "category_id": names_to_id[name],
                        "bbox": [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]],
                        "area": (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]), "iscrowd": 0,
                        "raw_label": raw, "standard_name": name, "xyxy": xyxy})
    return width, height, objects


def make_coco(records, image_dir, names, out_path):
    images, annotations = [], []
    ann_id = 1
    for image_id, item in enumerate(records, 1):
        images.append({"id": image_id, "file_name": item["prepared_name"],
                       "width": item["width"], "height": item["height"]})
        for obj in item["objects"]:
            ann = {k: obj[k] for k in ("category_id", "bbox", "area", "iscrowd")}
            ann.update({"id": ann_id, "image_id": image_id}); annotations.append(ann); ann_id += 1
    categories = [{"id": i, "name": name, "supercategory": "bird"} for i, name in enumerate(names)]
    out_path.write_text(json.dumps({"images": images, "annotations": annotations, "categories": categories},
                                   ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brightness-max-clipped", type=float, default=0.20,
                    help="仅用黑位/白位剪切比例不超过该值的增强图")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    mapping = json.loads(MAP_PATH.read_text(encoding="utf-8"))
    names = [x["standard_name"] for x in mapping["classes"]]
    names_to_id = {x: i for i, x in enumerate(names)}
    xmls = {p.stem + ".jpg": p for p in (ROOT / "data" / "annotations_raw").glob("*.xml")}
    eligible, held = [], []
    for folder in SOURCE.iterdir():
        if not folder.is_dir():
            continue
        for source_image in sorted(folder.glob("*.jpg")):
            name = source_image.name
            if name in TEACHER_EXAMPLES:
                held.append({"filename": name, "source": str(source_image), "reason": "教师提供示例，不计入学生标注"})
                continue
            if name not in xmls:
                held.append({"filename": name, "source": str(source_image), "reason": "缺失XML，不能视为负样本"})
                continue
            if name in QUARANTINED_LABELS:
                held.append({"filename": name, "source": str(source_image), "reason": "逐实例类别与目录前缀冲突，需人工复核"})
                continue
            width, height, objects = parse_xml(xmls[name], names_to_id)
            with Image.open(source_image) as im:
                actual = im.size
            if actual != (width, height):
                held.append({"filename": name, "source": str(source_image), "reason": f"XML尺寸{width}x{height}与图像{actual}不一致"})
                continue
            eligible.append({"filename": name, "source": source_image, "folder": folder.name,
                             "width": width, "height": height, "objects": objects})

    # 以独立原图和目录层的类别分层打乱，任何增强图都在此步骤之后加入。
    rng = random.Random(args.seed)
    by_folder = defaultdict(list)
    for item in eligible:
        by_folder[item["folder"]].append(item)
    train, val = [], []
    for folder in sorted(by_folder):
        group = list(by_folder[folder]); rng.shuffle(group)
        n_val = max(1, round(len(group) * 0.2))
        val.extend(group[:n_val]); train.extend(group[n_val:])
    train_names, val_names = {x["filename"] for x in train}, {x["filename"] for x in val}
    if train_names & val_names:
        raise RuntimeError("同一原图同时进入训练和验证")

    # 近重复感知哈希只作为候选筛查。未完成人工核实的候选保持在清单中，不自动改标签。
    near_path = REPORTS / "near_duplicate_candidates.csv"
    near = list(csv.DictReader(near_path.open(encoding="utf-8-sig"))) if near_path.exists() else []
    # 对dHash候选作保守分组：若候选边跨集合，将验证端整体移入训练集，防止近重复泄漏。
    # dHash候选仍只代表相似性线索；是否真为同一连拍/复刻，留在复核表由人工确认。
    moved_to_train = set()
    changed = True
    while changed:
        changed = False
        for pair in near:
            a, b = pair["image_a"], pair["image_b"]
            if a in train_names and b in val_names:
                val_names.remove(b); train_names.add(b); moved_to_train.add(b); changed = True
            elif b in train_names and a in val_names:
                val_names.remove(a); train_names.add(a); moved_to_train.add(a); changed = True
    train = [x for x in eligible if x["filename"] in train_names]
    val = [x for x in eligible if x["filename"] in val_names]
    near_cross_split = [r for r in near if (r["image_a"] in train_names and r["image_b"] in val_names)
                        or (r["image_a"] in val_names and r["image_b"] in train_names)]
    if near_cross_split:
        raise RuntimeError(f"疑似近重复仍跨集合: {near_cross_split[:3]}")

    # 写入派生数据区：验证集只有原图；训练集原图及通过像素异常筛查的brightness图。
    for part in ("train", "val"):
        (DATA / "images" / part).mkdir(parents=True, exist_ok=True)
    for item in train + val:
        part = "train" if item in train else "val"
        dest = DATA / "images" / part / item["filename"]
        shutil.copy2(item["source"], dest)
        item["prepared_name"] = item["filename"]
    train_anns = [{**x, "prepared_name": x["filename"]} for x in train]
    val_anns = [{**x, "prepared_name": x["filename"]} for x in val]

    audit_by_image = {r["brightness_image"].replace("_brightness.jpg", ".jpg"): r
                      for r in csv.DictReader((REPORTS / "brightness_audit.csv").open(encoding="utf-8-sig"))}
    brightness_manifest = []
    with zipfile.ZipFile(BRIGHTNESS_ZIP) as z:
        bright_members = {Path(n).name: n for n in z.namelist() if Path(n).suffix.lower() in {".jpg", ".jpeg", ".png"}}
        for item in train:
            bright_name = Path(item["filename"]).stem + "_brightness.jpg"
            audit = audit_by_image.get(item["filename"])
            accepted = bool(audit) and float(audit["fraction_luma_ge_250"]) <= args.brightness_max_clipped and \
                float(audit["fraction_luma_le_5"]) <= 0.05 and audit["matched_annotations"] == "True" and \
                audit["zip_original_matches_workspace_pixelwise"] == "True"
            if accepted and bright_name in bright_members:
                with z.open(bright_members[bright_name]) as src, (DATA / "images" / "train" / bright_name).open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                width, height, objects = parse_xml(xmls[item["filename"]], names_to_id)
                train_anns.append({"filename": bright_name, "prepared_name": bright_name, "source": bright_name,
                                   "folder": item["folder"], "width": width, "height": height,
                                   "objects": objects, "source_original": item["filename"], "source_type": "brightness"})
            brightness_manifest.append({"original": item["filename"], "brightness": bright_name,
                "included_in_training": accepted and bright_name in bright_members,
                "reason": "像素筛查通过且框逐实例一致" if accepted and bright_name in bright_members else "剪切率/匹配检查未通过，或增强文件缺失",
                "mean_luminance_change": audit["relative_mean_luminance_change"] if audit else "",
                "fraction_luma_ge_250": audit["fraction_luma_ge_250"] if audit else "",
                "fraction_luma_le_5": audit["fraction_luma_le_5"] if audit else ""})

    DATA.joinpath("annotations").mkdir(parents=True, exist_ok=True)
    make_coco(train_anns, "train/images", names, DATA / "annotations" / "instances_train.json")
    make_coco(val_anns, "val/images", names, DATA / "annotations" / "instances_val.json")
    # 留一份原图级映射，供课程答辩、对比实验和泄漏核验使用。
    # 候选连接分量共用一个来源组编号，增强图继续继承其原图组。
    parent = {x["filename"]: x["filename"] for x in eligible}
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    def union(a, b):
        if a in parent and b in parent:
            ra, rb = find(a), find(b)
            if ra != rb: parent[rb] = ra
    for pair in near:
        union(pair["image_a"], pair["image_b"])
    group_ids = {name: "near_group_" + find(name) for name in parent}
    split_rows = []
    for item in train:
        split_rows.append({"source_image": item["filename"], "source_path": str(item["source"]), "split": "train",
                           "source_group": group_ids[item["filename"]], "source_type": "original",
                           "moved_to_train_for_near_duplicate_guard": item["filename"] in moved_to_train,
                           "objects": len(item["objects"]), "standard_labels": "|".join(o["standard_name"] for o in item["objects"])})
    for item in val:
        split_rows.append({"source_image": item["filename"], "source_path": str(item["source"]), "split": "val",
                           "source_group": group_ids[item["filename"]], "source_type": "original",
                           "moved_to_train_for_near_duplicate_guard": False,
                           "objects": len(item["objects"]), "standard_labels": "|".join(o["standard_name"] for o in item["objects"])})
    csv_write(REPORTS / "split_manifest.csv", split_rows)
    csv_write(REPORTS / "held_out_manual_review.csv", held)
    csv_write(REPORTS / "brightness_training_manifest.csv", brightness_manifest)
    class_presence = {}
    for i, name in enumerate(names):
        class_presence[name] = {"train_images": sum(any(o["category_id"] == i for o in x["objects"]) for x in train),
                                "val_images": sum(any(o["category_id"] == i for o in x["objects"]) for x in val),
                                "train_instances": sum(sum(o["category_id"] == i for o in x["objects"]) for x in train),
                                "val_instances": sum(sum(o["category_id"] == i for o in x["objects"]) for x in val)}
    summary = {"seed": args.seed, "split_ratio": "约80/20，按原图所在类目录分层随机",
        "student_source_images_eligible": len(eligible), "train_original_images": len(train), "val_original_images": len(val),
        "brightness_train_images": sum(bool(r["included_in_training"]) for r in brightness_manifest),
        "train_image_count_after_brightness": len(train) + sum(bool(r["included_in_training"]) for r in brightness_manifest),
        "held_out_counts_by_reason": dict(Counter(x["reason"] for x in held)),
        "suspected_near_duplicate_pairs_crossing_split": near_cross_split,
        "near_duplicate_candidates_not_yet_human_confirmed": len(near),
        "validation_images_moved_to_train_for_conservative_near_duplicate_grouping": sorted(moved_to_train),
        "class_presence_by_actual_instance_label": class_presence,
        "coordinate_conversion": "VOC XML原始xyxy保留；COCO bbox=[xmin,ymin,xmax-xmin,ymax-ymin]，无+1/-1修正。需结合overlay_samples确认视觉对齐。",
        "augmentation_filter": {"max_fraction_luma_ge_250": args.brightness_max_clipped,
            "max_fraction_luma_le_5": 0.05,
            "meaning": "仅为可复现的保守像素筛查，不等于人工确认羽色细节无损；所有亮度图审计指标均在brightness_audit.csv。"}}
    (REPORTS / "split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
