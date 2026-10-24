"""用PP-YOLOE+同一官方擦除算子生成可复核样图，输入仅来自训练原图。"""
from __future__ import annotations
import json,random,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
P=json.loads((ROOT/'paths.json').read_text(encoding='utf-8'))
ARCH=Path(P['generation2_archive_engineering']);PD=Path(P['paddledetection'])
OUT=ROOT/'reports'/'augmentation_preview';OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(PD))
from ppdet.data.transform.operators import RandomErasingImage

def main():
    coco=json.loads((ARCH/'data/annotations/instances_train_bird.json').read_text(encoding='utf-8'))
    byid={im['id']:[] for im in coco['images']}
    for a in coco['annotations']:byid[a['image_id']].append(a)
    # 多目标图优先，确保检查“不能整只遮掉”的约束；不使用验证图或brightness变体。
    pool=[im for im in coco['images'] if not Path(im['file_name']).stem.endswith('_brightness') and len(byid[im['id']])>=2]
    pool=sorted(pool,key=lambda im:(-len(byid[im['id']]),im['file_name']))[:12]
    rows=[];op=RandomErasingImage(prob=.15,lower=.05,higher=.15,aspect_ratio=.3)
    for ii,imeta in enumerate(pool):
        random.seed(123+ii);np.random.seed(123+ii)
        p=Path(P['prepared_images'])/'images'/'train'/imeta['file_name']
        orig=np.asarray(Image.open(p).convert('RGB'))
        anns=byid[imeta['id']]
        boxes=np.asarray([[a['bbox'][0],a['bbox'][1],a['bbox'][0]+a['bbox'][2],a['bbox'][1]+a['bbox'][3]] for a in anns],dtype=np.float32)
        aug=op.apply({'image':orig.copy(),'gt_bbox':boxes.copy()})['image']
        mask=np.any(aug!=orig,axis=2)
        # 虚拟着色只用于HTML检查；实际预览面板同时给出原图与算子输出。
        visual=Image.fromarray(aug).convert('RGBA');overlay=Image.new('RGBA',visual.size,(0,0,0,0));od=ImageDraw.Draw(overlay)
        yy,xx=np.where(mask)
        if len(xx):
            for x,y in zip(xx,yy):od.point((int(x),int(y)),fill=(255,0,220,100))
        visual=Image.alpha_composite(visual,overlay).convert('RGB');d=ImageDraw.Draw(visual)
        ratio=[]
        for j,(b,a) in enumerate(zip(boxes,anns),1):
            x1,y1,x2,y2=map(int,b);den=max(1,(x2-x1)*(y2-y1));r=float(mask[y1:y2,x1:x2].sum()/den);ratio.append(r)
            d.rectangle((x1,y1,x2,y2),outline=(0,255,0),width=max(2,orig.shape[1]//500))
            d.text((x1,max(0,y1-16)),f'GT#{j} erased={r:.1%}',fill=(255,255,0),stroke_width=1,stroke_fill=(0,0,0))
        stem=Path(imeta['file_name']).stem
        Image.fromarray(orig).save(OUT/f'{stem}_original.jpg',quality=92)
        visual.save(OUT/f'{stem}_erasing.jpg',quality=92)
        rows.append({'file_name':imeta['file_name'],'source_path':str(p),'instance_count':len(anns),
          'instance_erased_fraction':ratio,'any_full_instance_erased':any(x>=1 for x in ratio),
          'preview_original':f'{stem}_original.jpg','preview_erasing':f'{stem}_erasing.jpg'})
    html=['<!doctype html><meta charset="utf-8"><title>Random Erasing增强样图</title><style>body{font:15px sans-serif;background:#eee}.card{background:white;padding:12px;margin:14px}.pair{display:flex;gap:12px}.pair img{max-width:48%;height:auto;object-fit:contain}code{word-break:break-all}</style>',
          '<h1>官方 RandomErasingImage 样图检查</h1><p>绿色框为原始人工框；洋红色像素为图像与原图的逐像素差异。该算子对每个标注框以 p=0.15 独立抽样，擦除目标框面积约5%–15%，擦除矩形必须严格小于框的宽高；样图可能按种子未触发擦除。</p>']
    for r in rows:
        html.append(f"<section class='card'><h2>{r['file_name']} | {r['instance_count']} instances</h2><div class='pair'><figure><img src='{r['preview_original']}'><figcaption>原图</figcaption></figure><figure><img src='{r['preview_erasing']}'><figcaption>官方擦除算子输出，实际遮挡比例: {r['instance_erased_fraction']}</figcaption></figure></div><code>{r['source_path']}</code></section>")
    (OUT/'index.html').write_text('\n'.join(html),encoding='utf-8')
    (OUT/'samples.json').write_text(json.dumps({'parameters':{'operator':'PaddleDetection RandomErasingImage','prob':.15,'lower':.05,'higher':.15,'aspect_ratio':.3,'seed_base':123},'training_only':True,'rows':rows},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'preview':str(OUT/'index.html'),'samples':len(rows),'any_full_instance_erased':sum(r['any_full_instance_erased'] for r in rows),'erased_samples':sum(any(v>0 for v in r['instance_erased_fraction']) for r in rows)},ensure_ascii=False))
if __name__=='__main__':main()
