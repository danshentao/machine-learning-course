"""把助手逐图视觉观察写成独立报告，不改动任何人工确认或标签。"""
from pathlib import Path
import base64, html, io, json
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OUT=Path('E:/机器学习课设/模型检查报告/第二代实验/13张人工视觉复核')
SOURCE=Path('E:/机器学习课设/模型检查报告/第二代实验/最终与最佳分类权重对照/最终与最佳权重比较结果.json')

def main():
    notes=json.loads((ROOT/'scripts/visual_review_notes.json').read_text(encoding='utf-8'));doc=json.loads(SOURCE.read_text(encoding='utf-8'))
    rows=[r for r in doc['rows'] if r['models']['bilinear']['best']['class_disagreement']]
    assert set(r['filename'] for r in rows)==set(notes['notes'])
    OUT.mkdir(parents=True,exist_ok=True);data=[]
    for r in rows:
        with Image.open(r['image_path']) as fp:
            im=fp.convert('RGB');im.thumbnail((1400,1400));buf=io.BytesIO();im.save(buf,format='JPEG',quality=88)
        data.append({'filename':r['filename'],'expected':r['expected'],'width':r['width'],'height':r['height'],
            'truth':r['truth'],'final':r['models']['bilinear']['final']['displayed'],'best':r['models']['bilinear']['best']['displayed'],
            'note':notes['notes'][r['filename']],'image':'data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode('ascii')})
    (OUT/'逐图辅助视觉复核意见.json').write_text(json.dumps(notes,ensure_ascii=False,indent=2),encoding='utf-8')
    md=['# 13张类别分歧图：辅助视觉复核','',notes['scope'],'',notes['review_type'],'']
    for r in data:
        n=r['note'];md.extend([f"## {r['filename']}（{r['expected']}）",'',*[f'{k}：{n[v]}' for k,v in [('问题','type'),('观察','observation'),('判断','judgment'),('年龄','age'),('建议','action')]],''])
    md.extend(['参考资料：',*[f"- [{s['title']}]({s['url']})" for s in notes['references']]])
    (OUT/'逐图辅助视觉复核意见.md').write_text('\n'.join(md),encoding='utf-8')
    template=(ROOT/'scripts/visual_review_template.html').read_text(encoding='utf-8')
    payload=json.dumps({'rows':data,'references':notes['references']},ensure_ascii=False).replace('<','\\u003c')
    (OUT/'13张类别分歧图_视觉复核意见.html').write_text(template.replace('__PAYLOAD__',payload),encoding='utf-8')
    print(json.dumps({'output':str(OUT),'images':len(data),'original_labels_modified':False},ensure_ascii=False))

if __name__=='__main__':main()
