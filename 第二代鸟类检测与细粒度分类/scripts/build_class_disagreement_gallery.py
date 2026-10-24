"""按原图所属类别筛选最终显示类别分歧，不比较框数、位置或实例匹配。"""
from pathlib import Path
import base64, io, json
from PIL import Image
from evaluate_two_stage import load_gt

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path('E:/机器学习课设/模型检查报告/第二代实验/最高置信度框对照')
OUT=SOURCE/'显示类别与所属类别不同'

def main():
    cfg=json.loads((ROOT/'configs/paths.json').read_text(encoding='utf-8'))
    mapping=json.loads((ROOT/'configs/class_mapping.json').read_text(encoding='utf-8'))
    originals=Path(cfg['source_images']);files={}
    for cls in mapping['classes']:
        folder=originals/cls['folder']
        if not folder.is_dir():raise FileNotFoundError(folder)
        for p in folder.rglob('*'):
            if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png'}:
                if p.name in files:raise ValueError('Ambiguous filename: '+p.name)
                files[p.name]={'folder':cls['folder'],'name':cls['name'],'path':str(p)}
    truth=load_gt();rows=[];summaries={};audit={}
    for model in ['resnet','bilinear']:
        doc=json.loads((SOURCE/f'{model}_postprocessed_predictions.json').read_text(encoding='utf-8'))
        flagged=[];no_display=[];count=0
        for r in doc['images']:
            own=files[r['filename']];display=r['after']
            assert all(d['detector_score']>.5 for d in display)
            mismatch=[d for d in display if d['class_name']!=own['name']]
            count+=len(display)
            if not display:no_display.append(r['filename'])
            if not mismatch:continue
            item={'model':model,'filename':r['filename'],'folder':own['folder'],'expected_class':own['name'],
                  'source_image':own['path'],'width':r['width'],'height':r['height'],
                  'displayed':display,'mismatches':mismatch,'truth':truth.get(r['filename'],[])}
            flagged.append(item);rows.append(item)
        summaries[model]={'checked_images':len(doc['images']),'displayed_boxes':count,'flagged_images':len(flagged),
                          'mismatched_displayed_boxes':sum(len(r['mismatches']) for r in flagged),'no_display_images':len(no_display)}
        audit[model]={'flagged_images':flagged,'no_display_images':no_display}
    OUT.mkdir(parents=True,exist_ok=True)
    rule={'comparison':'final displayed class_name vs actual original folder mapped to task category',
          'display_threshold':'detector_score > 0.5','dedup':'existing IoU>=0.5 winner by detector_score',
          'image_selection':'any displayed prediction has a different category',
          'box_count_and_position_used':False,'no_display_images_are_not_class_disagreements':True,
          'scope':'172 fixed validation images, not all 966 annotated images'}
    (OUT/'筛选依据与结果.json').write_text(json.dumps({'rule':rule,'summary':summaries,'results':audit},ensure_ascii=False,indent=2),encoding='utf-8')
    for model in ['resnet','bilinear']:
        lines=[f"{r['filename']} | 所属：{r['expected_class']} | 不同预测："+', '.join(sorted({d['class_name'] for d in r['mismatches']})) for r in audit[model]['flagged_images']]
        (OUT/f'{model}_类别分歧图片清单.txt').write_text('\n'.join(lines),encoding='utf-8')
    cache={}
    for r in rows:
        if r['source_image'] not in cache:
            with Image.open(r['source_image']) as im:
                im=im.convert('RGB');im.thumbnail((1600,1600));buf=io.BytesIO();im.save(buf,format='JPEG',quality=87)
            cache[r['source_image']]='data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode('ascii')
        r['image']=cache[r['source_image']]
    template=(ROOT/'scripts/class_disagreement_template.html').read_text(encoding='utf-8')
    payload=json.dumps({'rows':rows,'summary':summaries},ensure_ascii=False).replace('<','\\u003c')
    (OUT/'最终显示鸟类与所属类别不同.html').write_text(template.replace('__PAYLOAD__',payload),encoding='utf-8')
    print(json.dumps({'output':str(OUT),'summary':summaries},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
