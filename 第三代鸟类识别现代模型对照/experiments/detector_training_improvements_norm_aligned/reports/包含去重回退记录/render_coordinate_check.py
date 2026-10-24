"""从HTML内嵌数据制作静态坐标抽查图，便于目视核验框位置。"""
import base64, json, re
from io import BytesIO
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
html = (HERE / '两图IoU回退预览.html').read_text(encoding='utf-8')
payload = re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S).group(1)
items = json.loads(payload)['items']
font_path = Path(r'C:\Windows\Fonts\msyh.ttc')
font = ImageFont.truetype(str(font_path), 17) if font_path.exists() else ImageFont.load_default()
tiles = []
for item in items:
    im = Image.open(BytesIO(base64.b64decode(item['image'].split(',', 1)[1]))).convert('RGB')
    # 每行依次显示XML、基准IoU、C组IoU（恢复蓝框）、C组包含0.9。
    layers = [([], 'XML人工框'), (item['baseline_iou']['predictions'], '基准原IoU'),
              (item['C_iou']['predictions'], 'C组原IoU：蓝框恢复'),
              (item['C_contain09']['predictions'], 'C组包含0.9：历史对照')]
    row = Image.new('RGB', (4 * 420, 34 + 270), 'white')
    d = ImageDraw.Draw(row)
    for col, (preds, label) in enumerate(layers):
        d.text((col * 420 + 6, 4), label, fill='#172d49', font=font)
        scale = min(408 / im.width, 258 / im.height)
        nw, nh = round(im.width * scale), round(im.height * scale)
        tile = im.resize((nw, nh))
        x0 = col * 420 + (420 - nw) // 2
        y0 = 34 + (258 - nh) // 2
        row.paste(tile, (x0, y0))
        dr = ImageDraw.Draw(row)
        if col == 0:
            boxes = [(t['bbox_xyxy'], '#00b45a', f"XML{i+1}") for i, t in enumerate(item['truth'])]
        else:
            boxes = [(p['bbox_xyxy'], '#1769d2' if col < 3 else '#df2632',
                      f"{i+1} D{p['detector_score']:.3f} C{p['classifier_top_score']:.3f}")
                     for i, p in enumerate(preds)]
        for box, color, text in boxes:
            b = [x0 + box[0] * scale, y0 + box[1] * scale,
                 x0 + box[2] * scale, y0 + box[3] * scale]
            dr.rectangle(b, outline=color, width=3)
            dr.text((max(x0, b[0]), max(y0, b[1] - 20)), text, fill=color, font=font)
    tiles.append((item['filename'], row))
out = Image.new('RGB', (1680, len(tiles) * 304), '#eef2f6')
draw = ImageDraw.Draw(out)
for i, (name, row) in enumerate(tiles):
    draw.text((8, i * 304), name, fill='#111', font=font)
    out.paste(row, (0, i * 304 + 25))
path = HERE / '两图框坐标抽查.png'
out.save(path)
print(path)
