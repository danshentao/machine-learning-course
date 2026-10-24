"""为人工复核生成按标准类别排列的原图/框裁剪页、近似图页和亮度对照页。"""
import csv
import ast
import json
import zipfile
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FONT = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 16)
SMALL = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 11)


def fit(im, size):
    im = im.copy(); im.thumbnail(size)
    canvas = Image.new("RGB", size, "#e8e8e8")
    canvas.paste(im, ((size[0] - im.width)//2, (size[1] - im.height)//2))
    return canvas


def main():
    class_data = json.loads((ROOT / "configs" / "class_mapping.json").read_text(encoding="utf-8"))["classes"]
    crop_rows = list(csv.DictReader((REPORTS / "crop_manifest.csv").open(encoding="utf-8-sig")))
    by_class = defaultdict(list)
    for row in crop_rows:
        by_class[int(row["category_id"])].append(row)
    gallery_root = REPORTS / "review_galleries"
    for cls in class_data:
        cid, name = cls["category_id"], cls["standard_name"]
        rows = sorted(by_class[cid], key=lambda x: (x["image"], int(x["instance"])))
        for page, start in enumerate(range(0, len(rows), 12), 1):
            page_rows = rows[start:start+12]
            sheet = Image.new("RGB", (1200, 6*188), "white")
            draw = ImageDraw.Draw(sheet)
            for i, row in enumerate(page_rows):
                col, r = i % 2, i // 2
                x, y = col * 600, r * 188
                with Image.open(row["image"]) as im:
                    scene = im.convert("RGB")
                    b = ast.literal_eval(row["box_xyxy_xml"])
                    ImageDraw.Draw(scene).rectangle((b["xmin"], b["ymin"], b["xmax"], b["ymax"]), outline="#ff2020", width=max(3, scene.width//500))
                    scene_thumb = fit(scene, (282, 132))
                with Image.open(row["crop"]) as crop:
                    crop_thumb = fit(crop.convert("RGB"), (282, 132))
                sheet.paste(scene_thumb, (x+8, y+6)); sheet.paste(crop_thumb, (x+304, y+6))
                label = f"{Path(row['image']).name}  instance {row['instance']}  原标签：{row['raw_label']}"
                draw.text((x+8, y+142), label[:78], font=SMALL, fill="black")
                draw.text((x+8, y+158), f"图像路径：{row['image']}", font=SMALL, fill="#333333")
                draw.text((x+304, y+142), f"裁剪：{Path(row['crop']).name}", font=SMALL, fill="#333333")
            out_dir = gallery_root / f"{cid}_{name}"
            out_dir.mkdir(parents=True, exist_ok=True)
            sheet.save(out_dir / f"page_{page:02d}.jpg", quality=90)

    # 近重复候选仅为感知哈希筛查信号；以原图配对页供人判定是否同源连拍。
    near = list(csv.DictReader((REPORTS / "near_duplicate_candidates.csv").open(encoding="utf-8-sig")))
    near_dir = gallery_root / "near_duplicate_candidates"
    near_dir.mkdir(parents=True, exist_ok=True)
    base = ROOT.parent / "考核任务数据" / "考核任务数据" / "自然场景鸟类识别数据" / "数据集"
    class_folders = {row["folder"]: row["folder"] for row in class_data}
    def path_for(filename): return base / filename.split("_")[0] / filename
    for page, start in enumerate(range(0, len(near), 10), 1):
        page_rows = near[start:start+10]
        sheet = Image.new("RGB", (820, len(page_rows)*145), "white")
        draw = ImageDraw.Draw(sheet)
        for j, pair in enumerate(page_rows):
            y = j*145
            for x, key in [(6, "image_a"), (414, "image_b")]:
                with Image.open(path_for(pair[key])) as im:
                    sheet.paste(fit(im.convert("RGB"), (395, 112)), (x, y))
                draw.text((x+4, y+115), pair[key], font=SMALL, fill="black")
            draw.text((8, y+132), f"候选相似：dHash距离={pair['dhash_hamming']}（未人工确认）", font=SMALL, fill="#8c2020")
        sheet.save(near_dir / f"pairs_{page:02d}.jpg", quality=90)

    # 随机挑出剪切最明显的增强样例，支持检查重要羽色/纹理是否丢失。
    bright = list(csv.DictReader((REPORTS / "brightness_audit.csv").open(encoding="utf-8-sig")))
    bright.sort(key=lambda x: float(x["fraction_luma_ge_250"]), reverse=True)
    bright_dir = gallery_root / "brightness_clipping_examples"
    bright_dir.mkdir(parents=True, exist_ok=True)
    original_dir = base
    with zipfile.ZipFile(ROOT.parent / "2806021_1790929799.zip") as z:
        members = {Path(n).name: n for n in z.namelist() if Path(n).suffix.lower() in {".jpg", ".jpeg", ".png"}}
        for page, start in enumerate(range(0, min(30, len(bright)), 6), 1):
            page_rows = bright[start:start+6]
            sheet = Image.new("RGB", (1000, len(page_rows)*160), "white")
            draw = ImageDraw.Draw(sheet)
            for j, row in enumerate(page_rows):
                y = j*160
                orig = original_dir / row["original_image"].split("_")[0] / row["original_image"]
                with Image.open(orig) as im: sheet.paste(fit(im.convert("RGB"), (310, 126)), (5, y))
                with z.open(members[row["brightness_image"]]) as stream, Image.open(stream) as im:
                    sheet.paste(fit(im.convert("RGB"), (310, 126)), (335, y))
                draw.text((650, y+16), f"增强图白位≥250比例: {row['fraction_luma_ge_250']}", font=SMALL, fill="black")
                draw.text((650, y+38), Path(row["original_image"]).name, font=SMALL, fill="black")
                draw.text((650, y+60), "左：原图  右：亮度增强图", font=SMALL, fill="#555555")
            sheet.save(bright_dir / f"brightness_{page:02d}.jpg", quality=90)
    print(f"复核图集已生成：{gallery_root}")


if __name__ == "__main__":
    main()
