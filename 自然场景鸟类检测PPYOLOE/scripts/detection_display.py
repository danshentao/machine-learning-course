"""跨类别去重：每个重叠组显示最高分候选，其余候选保留为数据。"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from evaluate_predictions import iou

def group_detections(detections, iou_threshold=0.5):
    if not 0 < iou_threshold <= 1:
        raise ValueError('IoU阈值必须在(0,1]之间')
    groups=[]
    for det in sorted(detections,key=lambda d:d['score'],reverse=True):
        # 与最高分代表框直接比较，不进行传递合并，避免邻近两只鸟被链式合并。
        matches=[(iou(det['bbox_xyxy'],g['winner']['bbox_xyxy']),i) for i,g in enumerate(groups)]
        overlap,index=max(matches,default=(0,-1))
        if overlap>=iou_threshold:
            groups[index]['hidden_candidates'].append(det)
        else:
            groups.append({'winner':det,'hidden_candidates':[]})
    return groups

def render_detections(path, detections, destination, threshold=0.25):
    """只画保留框，不画被抑制候选；保留最高分代表框的原始位置与类别。"""
    with Image.open(path) as im:
        image=im.convert('RGB')
    draw=ImageDraw.Draw(image)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',max(16,round(image.width/55)))
    for det in detections:
        if det['score']<=threshold:continue
        box=det['bbox_xyxy'];draw.rectangle(box,outline='#E55353',width=max(2,round(image.width/300)))
        xy=(max(0,min(image.width-200,box[0])),max(0,min(image.height-24,box[1]-24)))
        draw.text(xy,f"{det['class_name']} {det['score']:.1%}",font=font,fill='white',stroke_width=2,stroke_fill='black')
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    image.save(destination,quality=90)
