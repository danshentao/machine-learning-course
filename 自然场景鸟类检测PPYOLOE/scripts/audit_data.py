"""审计原始VOC标注与brightness增强ZIP，保留来源信息并生成复核材料。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
SOURCE_IMAGES = WORKSPACE / "考核任务数据" / "考核任务数据" / "自然场景鸟类识别数据" / "数据集"
ORIGINAL_ZIP = WORKSPACE / "2804306_1790845704.zip"
BRIGHTNESS_ZIP = WORKSPACE / "2806021_1790929799.zip"
OUT = ROOT / "reports"
RAW_XML = ROOT / "data" / "annotations_raw"
CLASSES = [
    ("shumaque", "树麻雀"), ("jiamaque", "家麻雀"), ("shanmaque", "山麻雀"),
    ("baitoubei", "白头鹎"), ("hongerbei", "红耳鹎"), ("heihouhongtunbei", "黑喉红臀鹎"),
    ("dashanque", "大山雀"), ("meishanque", "煤山雀"),
]
FOLDER_TO_ID = {folder: i for i, (folder, _) in enumerate(CLASSES)}
STANDARD_TO_ID = {name: i for i, (_, name) in enumerate(CLASSES)}
ALIASES = {"眉山雀": "煤山雀", "煤山雀": "煤山雀"}


def json_dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_xml(data: bytes):
    root = ET.fromstring(data)
    size = root.find("size")
    objects = []
    for idx, obj in enumerate(root.findall("object"), 1):
        box = obj.find("bndbox")
        if box is None:
            coords = None
        else:
            coords = {k: float(box.findtext(k, "nan")) for k in ("xmin", "ymin", "xmax", "ymax")}
        raw_name = (obj.findtext("name") or "").strip()
        standard = ALIASES.get(raw_name, raw_name)
        objects.append({"instance": idx, "raw_label": raw_name, "standard_name": standard,
                        "category_id": STANDARD_TO_ID.get(standard), "box": coords,
                        "difficult": obj.findtext("difficult", "0")})
    return {
        "filename": (root.findtext("filename") or "").strip(),
        "width": int(size.findtext("width", "0")) if size is not None else 0,
        "height": int(size.findtext("height", "0")) if size is not None else 0,
        "objects": objects,
    }


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(im: Image.Image, size=8):
    im = ImageOps.grayscale(im).resize((size + 1, size))
    px = list(im.get_flattened_data())
    bits = 0
    for y in range(size):
        for x in range(size):
            bits = (bits << 1) | (px[y * (size + 1) + x] > px[y * (size + 1) + x + 1])
    return bits


def hamming(a, b):
    return (a ^ b).bit_count()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-crops", action="store_true", help="不生成鸟框裁剪对照图")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    RAW_XML.mkdir(parents=True, exist_ok=True)

    mapping = [{"category_id": i, "folder": folder, "standard_name": name,
                "accepted_xml_labels": [name] + (["眉山雀"] if folder == "meishanque" else [])}
               for i, (folder, name) in enumerate(CLASSES)]
    json_dump(ROOT / "configs" / "class_mapping.json", {
        "description": "课程确认的八类固定编号；名称规范化不代表逐只物种标签已验证。",
        "classes": mapping,
        "source_label_to_standard": {name: name for _, name in CLASSES} | {"眉山雀": "煤山雀"},
    })

    with zipfile.ZipFile(ORIGINAL_ZIP) as zorig, zipfile.ZipFile(BRIGHTNESS_ZIP) as zbright:
        bad1, bad2 = zorig.testzip(), zbright.testzip()
        onames, bnames = zorig.namelist(), zbright.namelist()
        oimgs = {Path(n).name: n for n in onames if n.lower().endswith((".jpg", ".jpeg", ".png"))}
        ox = {Path(n).name: n for n in onames if n.lower().endswith(".xml")}
        bimgs = {Path(n).name: n for n in bnames if n.lower().endswith((".jpg", ".jpeg", ".png"))}
        bx = {Path(n).name: n for n in bnames if n.lower().endswith(".xml")}
        # 增强ZIP仅取人工标注XML；图像仍从原始图片目录读取，不重复合并原图。
        annotations = {}
        for basename, member in ox.items():
            annotations[basename] = parse_xml(zorig.read(member))
            (RAW_XML / basename).write_bytes(zorig.read(member))

        source_files = {p.name: p for folder, _ in CLASSES for p in (SOURCE_IMAGES / folder).glob("*")
                        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
        original_zip_source_mismatches = []
        brightness_zip_original_mismatches = []
        for fn, original_member in oimgs.items():
            if fn not in source_files:
                original_zip_source_mismatches.append({"filename": fn, "reason": "工作区图片目录缺失"})
                continue
            h = hashlib.sha256()
            with zorig.open(original_member) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(chunk)
            if h.hexdigest() != sha256(source_files[fn]):
                original_zip_source_mismatches.append({"filename": fn, "reason": "与工作区图片文件字节哈希不同"})
        bright_original_members = {fn: member for fn, member in bimgs.items()
                                   if not Path(fn).stem.endswith("_brightness")}
        for fn, original_member in oimgs.items():
            member = bright_original_members.get(fn)
            if member is None:
                brightness_zip_original_mismatches.append({"filename": fn, "reason": "增强ZIP没有同名原图"})
                continue
            h = hashlib.sha256()
            with zbright.open(member) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(chunk)
            if h.hexdigest() != sha256(source_files[fn]):
                brightness_zip_original_mismatches.append({"filename": fn, "reason": "增强ZIP中的原图与工作区字节哈希不同"})
        rows, errors, crop_rows, box_rows = [], [], [], []
        by_folder = Counter(); by_label = Counter(); instance_counts = Counter()
        image_hashes, d_hashes = {}, {}
        xml_to_image = defaultdict(list)
        for fn, p in source_files.items():
            xml_to_image[fn].append(p)
        annotated_images = set()
        for xml_fn, xml_member in ox.items():
            ann = annotations[xml_fn]
            fn = ann["filename"] or (Path(xml_fn).stem + ".jpg")
            annotated_images.add(fn)
            image_path = source_files.get(fn)
            image = None
            if image_path is None:
                errors.append({"type": "xml_without_source_image", "filename": fn, "xml": xml_member})
            else:
                try:
                    image = Image.open(image_path).convert("RGB")
                except Exception as e:
                    errors.append({"type": "bad_image", "filename": fn, "path": str(image_path), "error": str(e)})
                if image:
                    width, height = image.size
                    if (width, height) != (ann["width"], ann["height"]):
                        errors.append({"type": "xml_image_size_mismatch", "filename": fn,
                                       "xml_size": [ann["width"], ann["height"]], "image_size": [width, height]})
                    image_hashes[fn] = sha256(image_path)
                    d_hashes[fn] = dhash(image)
            folder = fn.split("_")[0]
            folder_id = FOLDER_TO_ID.get(folder)
            by_folder[folder] += 1
            if not ann["objects"]:
                errors.append({"type": "empty_xml", "filename": fn, "xml": xml_member})
            for obj in ann["objects"]:
                by_label[obj["raw_label"]] += 1
                if obj["category_id"] is None:
                    errors.append({"type": "unknown_label", "filename": fn, "instance": obj["instance"],
                                   "raw_label": obj["raw_label"], "xml": xml_member})
                if obj["box"]:
                    b = obj["box"]
                    if not all(math.isfinite(v) for v in b.values()):
                        errors.append({"type": "invalid_box", "filename": fn, "instance": obj["instance"], "box": b})
                    else:
                        if b["xmax"] <= b["xmin"] or b["ymax"] <= b["ymin"]:
                            errors.append({"type": "degenerate_box", "filename": fn, "instance": obj["instance"], "box": b})
                        if image and (b["xmin"] < 0 or b["ymin"] < 0 or b["xmax"] > image.width or b["ymax"] > image.height):
                            errors.append({"type": "out_of_bounds_box", "filename": fn, "instance": obj["instance"],
                                           "box": b, "image_size": list(image.size)})
                    instance_counts[obj["standard_name"]] += 1
                    if image:
                        scale_x640 = 640 / image.width
                        scale_y640 = 640 / image.height
                        box_rows.append({"filename": fn, "instance": obj["instance"],
                            "raw_label": obj["raw_label"], "standard_name": obj["standard_name"],
                            "category_id": obj["category_id"], "image_width": image.width, "image_height": image.height,
                            "box_width_xml": b["xmax"] - b["xmin"], "box_height_xml": b["ymax"] - b["ymin"],
                            "box_area_fraction": ((b["xmax"] - b["xmin"]) * (b["ymax"] - b["ymin"])) / (image.width * image.height),
                            "box_width_at_640_stretched": (b["xmax"] - b["xmin"]) * scale_x640,
                            "box_height_at_640_stretched": (b["ymax"] - b["ymin"]) * scale_y640,
                            "box_xyxy_xml": json.dumps(b, ensure_ascii=False)})
                    if image and not args.skip_crops:
                        pad_x = max(4, round((b["xmax"] - b["xmin"]) * 0.08))
                        pad_y = max(4, round((b["ymax"] - b["ymin"]) * 0.08))
                        crop = image.crop((max(0, int(b["xmin"] - pad_x)), max(0, int(b["ymin"] - pad_y)),
                                           min(image.width, math.ceil(b["xmax"] + pad_x)),
                                           min(image.height, math.ceil(b["ymax"] + pad_y))))
                        out = ROOT / "data" / "review_crops" / f"{obj['category_id']:02d}_{fn[:-4]}_inst{obj['instance']:02d}.jpg"
                        out.parent.mkdir(parents=True, exist_ok=True)
                        crop.save(out, quality=92)
                        crop_rows.append({"category_id": obj["category_id"], "standard_name": obj["standard_name"],
                                          "image": str(image_path), "crop": str(out), "instance": obj["instance"],
                                          "raw_label": obj["raw_label"], "box_xyxy_xml": b})
            if image:
                rows.append({"filename": fn, "image": str(image_path), "xml_member": xml_member,
                             "folder": folder, "folder_category_id": folder_id,
                             "width": image.width, "height": image.height,
                             "xml_width": ann["width"], "xml_height": ann["height"],
                             "object_count": len(ann["objects"]),
                             "labels": "|".join(o["standard_name"] for o in ann["objects"]),
                             "exact_sha256": image_hashes[fn]})

        # 来源目录中没有XML的图片均为需人工判断的缺失标注，不作为负样本。
        for fn, p in source_files.items():
            if fn not in annotated_images:
                errors.append({"type": "image_without_xml", "filename": fn, "path": str(p),
                               "note": "不能假定为无鸟负样本；人工补标或确认排除。"})
        for xml_fn in ox:
            image_name = annotations[xml_fn]["filename"] or (Path(xml_fn).stem + ".jpg")
            if image_name not in source_files:
                errors.append({"type": "orphan_xml", "filename": image_name, "xml": ox[xml_fn]})

        # 精确重复由哈希确认；dHash 距离小的图仅列为疑似近重复，要求人工复核。
        exact_groups = defaultdict(list)
        for fn, h in image_hashes.items(): exact_groups[h].append(fn)
        exact_duplicates = [v for v in exact_groups.values() if len(v) > 1]
        near = []
        fns = list(d_hashes)
        for i, a in enumerate(fns):
            for b in fns[i + 1:]:
                dist = hamming(d_hashes[a], d_hashes[b])
                if dist <= 4 and image_hashes[a] != image_hashes[b]:
                    near.append({"image_a": a, "image_b": b, "dhash_hamming": dist, "review": True})

        # 对比两个ZIP中原图字节与工作区原图，分析增强图的几何及像素变化。
        brightness_rows, bright_issues = [], []
        bright_image_names = [n for n in bimgs if Path(n).stem.endswith("_brightness")]
        for bright_fn in bright_image_names:
            original_fn = Path(bright_fn).stem.removesuffix("_brightness") + Path(bright_fn).suffix
            if original_fn not in oimgs or original_fn not in source_files:
                bright_issues.append({"type": "brightness_unmatched", "filename": bright_fn, "original": original_fn})
                continue
            bxml_fn = Path(bright_fn).stem + ".xml"
            oxml_fn = Path(original_fn).stem + ".xml"
            try:
                b_ann = parse_xml(zbright.read(bx[bxml_fn]))
                o_ann = annotations[oxml_fn]
                with Image.open(zbright.open(bimgs[bright_fn])) as bi, Image.open(source_files[original_fn]) as oi:
                    bi = bi.convert("RGB"); oi = oi.convert("RGB")
                    if bi.size != oi.size:
                        bright_issues.append({"type": "brightness_size_mismatch", "filename": bright_fn,
                                              "brightness_size": list(bi.size), "original_size": list(oi.size)})
                    same_objects = b_ann["objects"] == o_ann["objects"]
                    if not same_objects:
                        bright_issues.append({"type": "brightness_annotation_mismatch", "filename": bright_fn})
                    # 跨ZIP相同原图哈希核对，brightness统计按共同分辨率图像计算。
                    raw_zip = Image.open(zorig.open(oimgs[original_fn])).convert("RGB")
                    identical_originals = raw_zip.size == oi.size and raw_zip.tobytes() == oi.tobytes()
                    lum = ImageOps.grayscale(bi)
                    hist = lum.histogram(); npx = bi.width * bi.height
                    mean_luma = sum(i * c for i, c in enumerate(hist)) / npx
                    clipped_dark = sum(hist[:6]) / npx
                    clipped_bright = sum(hist[250:]) / npx
                    diff = ImageChops_difference(raw_zip, bi) if raw_zip.size == bi.size else None
                    mean_abs_diff = diff
                    brightness_rows.append({"brightness_image": bright_fn, "original_image": original_fn,
                        "source_image": str(source_files[original_fn]), "size": list(bi.size),
                        "matched_annotations": same_objects, "zip_original_matches_workspace_pixelwise": identical_originals,
                        "mean_luminance": round(mean_luma, 3), "fraction_luma_le_5": round(clipped_dark, 6),
                        "fraction_luma_ge_250": round(clipped_bright, 6),
                        "mean_absolute_pixel_difference_vs_original": round(mean_abs_diff, 4) if mean_abs_diff is not None else None,
                        "relative_mean_luminance_change": round((mean_luma - sum(i*c for i,c in enumerate(ImageOps.grayscale(raw_zip).histogram()))/npx), 4)})
            except Exception as e:
                bright_issues.append({"type": "brightness_read_error", "filename": bright_fn, "error": str(e)})

    # CSV输出便于审计和后续人工筛查。
    for name, records in [("image_audit.csv", rows), ("box_audit.csv", box_rows),
                          ("manual_review.csv", errors), ("near_duplicate_candidates.csv", near),
                          ("brightness_audit.csv", brightness_rows), ("brightness_issues.csv", bright_issues),
                          ("crop_manifest.csv", crop_rows)]:
        if name == "crop_manifest.csv" and args.skip_crops:
            continue
        if records:
            with (OUT / name).open("w", newline="", encoding="utf-8-sig") as f:
                fields = list(dict.fromkeys(key for row in records for key in row))
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(records)
        else:
            (OUT / name).write_text("", encoding="utf-8-sig")
    label_folder_mismatches = []
    for fn, ann in annotations.items():
        folder = Path(fn).stem.split("_")[0]
        expected_id = FOLDER_TO_ID.get(folder)
        for obj in ann["objects"]:
            if expected_id is not None and obj["category_id"] != expected_id:
                label_folder_mismatches.append({"xml": fn, "filename": ann["filename"], "folder_category_id": expected_id,
                    "instance": obj["instance"], "raw_label": obj["raw_label"], "category_id": obj["category_id"],
                    "note": "仅列出目录前缀与逐实例标签不一致样本，可能是真实多鸟类别，也可能需要复核；不自动改标签。"})
    if label_folder_mismatches:
        with (OUT / "label_folder_mismatch_review.csv").open("w", newline="", encoding="utf-8-sig") as f:
            fields = list(label_folder_mismatches[0])
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(label_folder_mismatches)
    else:
        (OUT / "label_folder_mismatch_review.csv").write_text("", encoding="utf-8-sig")
    import numpy as np
    lum_deltas = [r["relative_mean_luminance_change"] for r in brightness_rows]
    abs_diffs = [r["mean_absolute_pixel_difference_vs_original"] for r in brightness_rows if r["mean_absolute_pixel_difference_vs_original"] is not None]
    images_by_standard_label = Counter()
    for ann in annotations.values():
        for name in {obj["standard_name"] for obj in ann["objects"]}:
            images_by_standard_label[name] += 1
    image_size_counts = Counter(f"{r['width']}x{r['height']}" for r in rows)
    summary = {
        "workspace": str(WORKSPACE), "source_image_dir": str(SOURCE_IMAGES),
        "original_zip": str(ORIGINAL_ZIP), "brightness_zip": str(BRIGHTNESS_ZIP),
        "zip_integrity": {"original_testzip": bad1, "brightness_testzip": bad2},
        "original_zip_counts": {"images": len(oimgs), "xml": len(ox)},
        "brightness_zip_counts": {"images": len(bimgs), "xml": len(bx), "brightness_images": len(bright_image_names)},
        "original_zip_images_binary_match_workspace": len(oimgs) - len(original_zip_source_mismatches),
        "brightness_zip_source_originals_binary_match_workspace": len(oimgs) - len(brightness_zip_original_mismatches),
        "zip_original_image_mismatches": original_zip_source_mismatches,
        "brightness_zip_original_mismatches": brightness_zip_original_mismatches,
        "workspace_images": len(source_files), "xml_object_count": sum(len(a["objects"]) for a in annotations.values()),
        "multi_object_images": sum(len(a["objects"]) > 1 for a in annotations.values()),
        "workspace_folder_image_counts": {folder: sum(1 for p in source_files.values() if p.parent.name == folder) for folder, _ in CLASSES},
        "annotated_images_by_filename_folder": dict(by_folder), "raw_label_instance_counts": dict(by_label),
        "standard_instance_counts": dict(instance_counts), "xml_by_image": len(ox),
        "annotated_images_by_actual_instance_label": dict(images_by_standard_label),
        "annotated_image_size_distribution": dict(image_size_counts),
        "exact_duplicate_groups": exact_duplicates, "near_duplicate_candidate_count_dhash_le_4": len(near),
        "label_folder_mismatch_instances_for_manual_review": len(label_folder_mismatches),
        "manual_review_issue_counts": dict(Counter(e["type"] for e in errors)),
        "brightness_issue_counts": dict(Counter(e["type"] for e in bright_issues)),
        "brightness_mean_luminance_change_quantiles": {str(q): float(np.quantile(lum_deltas, q)) for q in [0, .25, .5, .75, .9, 1]} if lum_deltas else {},
        "brightness_mean_absolute_pixel_difference_quantiles": {str(q): float(np.quantile(abs_diffs, q)) for q in [0, .25, .5, .75, .9, 1]} if abs_diffs else {},
        "brightness_images_with_luma_ge_250_fraction_over_20pct": sum(r["fraction_luma_ge_250"] > .20 for r in brightness_rows),
        "brightness_images_with_luma_le_5_fraction_over_5pct": sum(r["fraction_luma_le_5"] > .05 for r in brightness_rows),
        "brightness_annotation_matches": sum(bool(r["matched_annotations"]) for r in brightness_rows),
        "zip_original_pixelwise_matches_workspace": sum(bool(r["zip_original_matches_workspace_pixelwise"]) for r in brightness_rows),
        "box_size_quantiles": {key: {str(q): float(np.quantile([r[key] for r in box_rows], q)) for q in [0, .25, .5, .75, .9, 1]}
                               for key in ["box_width_xml", "box_height_xml", "box_width_at_640_stretched", "box_height_at_640_stretched", "box_area_fraction"]} if box_rows else {},
        "coordinate_policy": "XML来源未提供坐标口径说明；保留原始xyxy数值，COCO宽高用xmax-xmin/ymax-ymin，不加减像素。人工抽查叠框页与裁剪页位置对齐；端点是否按VOC像素中心/边界解释仍记为来源限制。",
        "structural_validation_counts": {"audited_xml_images": len(rows), "bad_images": sum(e["type"] == "bad_image" for e in errors), "xml_image_size_mismatch": sum(e["type"] == "xml_image_size_mismatch" for e in errors), "orphan_xml": sum(e["type"] == "orphan_xml" for e in errors), "images_without_xml": sum(e["type"] == "image_without_xml" for e in errors), "unknown_labels": sum(e["type"] == "unknown_label" for e in errors), "degenerate_boxes": sum(e["type"] == "degenerate_box" for e in errors), "out_of_bounds_boxes": sum(e["type"] == "out_of_bounds_box" for e in errors), "invalid_boxes": sum(e["type"] == "invalid_box" for e in errors), "valid_box_rows": len(box_rows)},
        "counts_note": "目录/文件名前缀不用于覆盖逐实例类别；所有标准名映射是命名规范化，不代表物种真值已复核。"
    }
    json_dump(OUT / "audit_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def ImageChops_difference(a: Image.Image, b: Image.Image):
    import numpy as np
    x = np.asarray(a, dtype=np.int16)
    y = np.asarray(b, dtype=np.int16)
    return float(abs(x - y).mean())


if __name__ == "__main__":
    main()
