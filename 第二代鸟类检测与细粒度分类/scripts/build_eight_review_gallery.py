"""生成八张隔离训练图人工复核页面，包含同源对照、标注裁剪和可复制意见。"""
import base64
import csv
import io
import json
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
WORKSPACE=ROOT.parent
SOURCE=WORKSPACE/'考核任务数据/考核任务数据/自然场景鸟类识别数据/数据集'
FIRST=WORKSPACE/'自然场景鸟类检测PPYOLOE'
OUTPUT=WORKSPACE/'模型检查报告/第二代实验/疑难训练图复核/八张疑难训练图复核.html'
NAMES=['dashanque_095.jpg','dashanque_106.jpg','heihouhongtunbei_003.jpg',
       'heihouhongtunbei_004.jpg','heihouhongtunbei_086.jpg','shanmaque_028.jpg',
       'shanmaque_090.jpg','shumaque_112.jpg']
PAIRS={'dashanque_095.jpg':'dashanque_119.jpg','heihouhongtunbei_004.jpg':'heihouhongtunbei_086.jpg',
       'heihouhongtunbei_086.jpg':'heihouhongtunbei_004.jpg','shanmaque_028.jpg':'shanmaque_090.jpg',
       'shanmaque_090.jpg':'shanmaque_028.jpg'}
QUESTIONS={
 'dashanque_095.jpg':('同源图类别冲突','095标为大山雀，右侧119标为黑喉红臀鹎；请确认同一只鸟的类别。119只供参考，仍保持排除。'),
 'dashanque_106.jpg':('未找到同类正确显示框','请确认大山雀类别及绿框范围是否正确，并查看图中是否还有漏标鸟。第一代没有同类正确匹配不代表标注错误。'),
 'heihouhongtunbei_003.jpg':('同位置预测与标注不同','人工标为黑喉红臀鹎，第一代提示白头鹎。请先看原图与鸟体裁剪确认；模型预测不是正确答案。'),
 'heihouhongtunbei_004.jpg':('同源图标注框差异','与086同源，类别相同；几何对齐后的人工框IoU约0.49158。请比较两框是否完整包含同一只鸟。'),
 'heihouhongtunbei_086.jpg':('同源图标注框差异','与004同源，类别相同；几何对齐后的人工框IoU约0.49158。请比较两框是否完整包含同一只鸟。'),
 'shanmaque_028.jpg':('同源图标注框差异','与090同源，类别相同；几何对齐后的人工框IoU约0.48479。请比较两框是否完整包含同一只鸟。'),
 'shanmaque_090.jpg':('同源图标注框差异','与028同源，类别相同；几何对齐后的人工框IoU约0.48479。请比较两框是否完整包含同一只鸟。'),
 'shumaque_112.jpg':('模型多检或人工漏标候选','第一代有一个显示框没有同类匹配标注。请查看是否漏了一只鸟、框偏差或模型误检；可打开红色模型框作定位提示。')}

def encode(image):
    buf=io.BytesIO();image.save(buf,format='JPEG',quality=91)
    return 'data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode('ascii')

def main():
    mapping=json.loads((FIRST/'configs/class_mapping.json').read_text(encoding='utf-8'))['source_label_to_standard']
    xmls={}
    with zipfile.ZipFile(WORKSPACE/'2804306_1790845704.zip') as z:
        for name in z.namelist():
            if not name.endswith('.xml'):continue
            filename=Path(name).stem+'.jpg'
            if filename not in set(NAMES)|set(PAIRS.values()):continue
            root=ET.fromstring(z.read(name));objs=[]
            for j,obj in enumerate(root.findall('object'),1):
                raw=obj.findtext('name');box=obj.find('bndbox')
                objs.append({'instance':j,'label':mapping.get(raw,raw),
                    'box':[float(box.findtext(k)) for k in ('xmin','ymin','xmax','ymax')]})
            xmls[filename]=objs
    old=json.loads((FIRST/'runs/unique_bird_display_gt50/predictions.json').read_text(encoding='utf-8'))
    preds={Path(r['image']).name:[d for d in r['detections'] if d['score']>.5] for r in old['images']}
    evidence=list(csv.DictReader((ROOT/'reports/manual_review_instances.csv').open(encoding='utf-8-sig')))
    images={p.name:p for p in SOURCE.rglob('*.jpg')}
    def make(filename):
        with Image.open(images[filename]) as im:
            image=im.convert('RGB');w,h=image.size
            crops=[]
            for obj in xmls[filename]:
                x1,y1,x2,y2=obj['box'];padx=(x2-x1)*.12;pady=(y2-y1)*.12
                region=(max(0,int(x1-padx)),max(0,int(y1-pady)),min(w,int(x2+padx)),min(h,int(y2+pady)))
                crop=image.crop(region)
                crops.append({'instance':obj['instance'],'label':obj['label'],'image':encode(crop)})
            return {'name':filename,'width':w,'height':h,'image':encode(image),'gt':xmls[filename],
                'pred':[{'label':d['class_name'],'score':d['score'],'box':d['bbox_xyxy']} for d in preds.get(filename,[])],
                'crops':crops}
    cards=[]
    for name in NAMES:
        title,question=QUESTIONS[name]
        cards.append({'name':name,'issue':title,'question':question,'source':make(name),
            'partner':make(PAIRS[name]) if name in PAIRS else None,
            'evidence':[r for r in evidence if r['filename']==name]})
    assert len(cards)==8 and len(set(c['name'] for c in cards))==8
    data={'cards':cards,'classes':list(dict.fromkeys(mapping.values()))}
    template=Path(__file__).with_name('eight_review_template.html').read_text(encoding='utf-8')
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    OUTPUT.write_text(template.replace('__DATA__',json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')),encoding='utf-8')
    print(json.dumps({'output':str(OUTPUT),'review_images':8,'reference_only':'dashanque_119.jpg','bytes':OUTPUT.stat().st_size},ensure_ascii=False))

if __name__=='__main__':main()
