from common import *
import io,base64,copy,html,csv
from PIL import Image

def main():
    b=baseline();rows=copy.deepcopy(b['rows']);models={'gen2_bilinear_best':'第二代最佳 Bilinear CNN'};summaries={'gen2_bilinear_best':b['summary']['bilinear']['best']}
    for r in rows:r['models']={'gen2_bilinear_best':r['models']['bilinear']['best']}
    for name,label in [('bioclip2','BioCLIP 2'),('dinov3','DINOv3'),('fet_fgvc','FET-FGVC')]:
        p=ROOT/'reports'/name/'固定验证集结果.json'
        if not p.exists():continue
        d=json.loads(p.read_text(encoding='utf-8'));models[name]=label;summaries[name]=d['summary'];assert len(d['rows'])==len(rows)
        for r,n in zip(rows,d['rows']):assert r['filename']==n['filename'];r['models'][name]=n['models'][name]
    out=ROOT/'reports/第三代对照';out.mkdir(parents=True,exist_ok=True)
    with (out/'逐图分类对照.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['图片','所属类别','模型','最终框序号','检测置信度','分类最高置信度','最终显示类别','与所属类别不同']+NAMES)
        for r in rows:
            for m,v in r['models'].items():
                if not v['displayed']:w.writerow([r['filename'],r['expected'],models[m],'无显示框','','','','']+['']*8)
                for i,p in enumerate(v['displayed'],1):w.writerow([r['filename'],r['expected'],models[m],i,p['detector_score'],p['classifier_top_score'],p['class_name'],p['different_from_folder']]+[p['classifier_scores'][n] for n in NAMES])
            with Image.open(r['image_path']) as im:
                im=im.convert('RGB');im.thumbnail((1000,1000));buf=io.BytesIO();im.save(buf,format='JPEG',quality=85);r['image']='data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode()
    state_path=ROOT/'reports/当前实验状态.json'
    state=json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {}
    deferred=['dinov3'] if state.get('dinov3_deferred_to_generation4') else []
    write(out/'汇总.json',{'models':models,'summary':summaries,'rule':'detector strictly >30%, identical177groupedboxes; classifiersoftmax is not calibrated correctness probability','pending':[m for m in ['bioclip2','dinov3','fet_fgvc'] if m not in models and m not in deferred],'deferred_to_generation4':deferred})
    template=(ROOT/'scripts/gallery.html').read_text(encoding='utf-8');data=json.dumps({'rows':rows,'models':models,'summary':summaries},ensure_ascii=False).replace('<','\\u003c')
    (out/'第三代分类模型逐图对照.html').write_text(template.replace('__DATA__',data),encoding='utf-8');print('Gallery:',str(out),models)
if __name__=='__main__':main()
