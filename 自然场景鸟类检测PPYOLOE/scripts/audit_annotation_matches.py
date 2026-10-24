"""核对每只人工标注鸟与模型预测，区分主预测错类和额外跨类别候选。"""
import argparse
import base64
import io
import json
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from batch_infer import ROOT, CLASS_NAMES, MAPPING
from evaluate_predictions import iou, class_correct_counts

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args(); out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    doc=json.loads((ROOT/'runs/confidence_audit/all_originals/class_confidences.json').read_text(encoding='utf-8'))
    records=doc['images']; source={}
    with zipfile.ZipFile(ROOT.parent/'2804306_1790845704.zip') as z:
        for n in z.namelist():
            if not n.endswith('.xml'):continue
            xml=ET.fromstring(z.read(n)); objects=[]
            for j,o in enumerate(xml.findall('object'),1):
                name=MAPPING['source_label_to_standard'][o.findtext('name')]
                b=o.find('bndbox')
                objects.append({'instance':j,'class_id':CLASS_NAMES.index(name),'class_name':name,
                  'box':[float(b.findtext(k)) for k in ('xmin','ymin','xmax','ymax')]})
            source[Path(n).stem+'.jpg']=objects
    image_rows=[]; instances=[]; extra_rows=[]; gallery=[]
    for record in records:
        filename=record['filename']; truth=source[filename]
        preds=sorted([d for d in record['detections'] if d['score']>=.25],key=lambda d:d['score'],reverse=True)
        matched_gt={}; used_p=set()
        # 与原P/R/F1口径一致，按分数对同类别真值一对一匹配。
        for pi,p in enumerate(preds):
            choices=[(iou(p['bbox_xyxy'],g['box']),gi) for gi,g in enumerate(truth)
                     if gi not in matched_gt and p['class_id']==g['class_id']]
            overlap,gi=max(choices,default=(0,-1))
            if overlap>=.5: matched_gt[gi]=pi;used_p.add(pi)
        primary_wrong=0; high_wrong=0; missing=0;locating=0
        for gi,g in enumerate(truth):
            spatial=[(pi,p,iou(p['bbox_xyxy'],g['box'])) for pi,p in enumerate(preds) if iou(p['bbox_xyxy'],g['box'])>=.5]
            primary=spatial[0] if spatial else None
            same=[(pi,p,ov) for pi,p,ov in spatial if p['class_id']==g['class_id']]
            same_best=same[0] if same else None
            wrong=primary is not None and primary[1]['class_id']!=g['class_id']
            primary_wrong+=int(wrong)
            strong=bool(wrong and primary[1]['score']>=.8)
            high_wrong+=int(strong)
            correct=gi in matched_gt
            if correct: status='有同类正确框'
            elif spatial: status='仅有错类框或匹配冲突'
            elif any(p['class_id']==g['class_id'] and .1<=iou(p['bbox_xyxy'],g['box'])<.5 for p in preds):
                status='同类框定位不足';locating+=1
            else: status='无同类正确检测';missing+=1
            instances.append({'image':filename,'split':record['split'],'instance':g['instance'],
                'gt_class':g['class_name'],'gt_box':g['box'],'status':status,'correct_match':correct,
                'primary_wrong':wrong,'high_confidence_wrong':strong,
                'primary_class':primary[1]['class_name'] if primary else '',
                'primary_score':primary[1]['score'] if primary else None,
                'primary_iou':primary[2] if primary else None,
                'primary_box':primary[1]['bbox_xyxy'] if primary else None,
                'best_same_class_score':same_best[1]['score'] if same_best else None,
                'primary_class_scores':primary[1]['class_scores'] if primary else None,
                'path':record['image']})
        kinds=Counter()
        for pi,p in enumerate(preds):
            if pi in used_p:continue
            overlap,gi=max([(iou(p['bbox_xyxy'],g['box']),gi) for gi,g in enumerate(truth)],default=(0,-1))
            if overlap>=.5:
                kind='同类重复框' if p['class_id']==truth[gi]['class_id'] else '额外跨类别框'
            elif overlap>=.1: kind='定位不足候选'
            else:kind='背景误检或漏标候选'
            kinds[kind]+=1
            extra_rows.append({'image':filename,'split':record['split'],'kind':kind,
                'pred_class':p['class_name'],'score':p['score'],'pred_box':p['bbox_xyxy'],
                'nearest_gt_class':truth[gi]['class_name'] if gi>=0 else '',
                'nearest_gt_instance':truth[gi]['instance'] if gi>=0 else None,
                'iou':overlap,'path':record['image']})
        row={'image':filename,'split':record['split'],'gt_count':len(truth),'pred_count':len(preds),
             'TP':len(matched_gt),'FP':len(preds)-len(used_p),'FN':len(truth)-len(matched_gt),
             'primary_wrong':primary_wrong,'high_confidence_wrong':high_wrong,
             **{k:kinds[k] for k in ('同类重复框','额外跨类别框','定位不足候选','背景误检或漏标候选')},
             'path':record['image']}
        row['has_mismatch']=bool(row['FP'] or row['FN'] or primary_wrong)
        image_rows.append(row)
        if row['has_mismatch']:gallery.append((row,truth,preds))
    summary={'image_count':len(records),'instance_count':len(instances),
      'mismatch_image_count':sum(r['has_mismatch'] for r in image_rows),
      'primary_wrong_instances':sum(r['primary_wrong'] for r in instances),
      'high_confidence_wrong_instances':sum(r['high_confidence_wrong'] for r in instances),
      'unmatched_gt_instances':sum(r['FN'] for r in image_rows),
      'extra_predictions_by_kind':dict(Counter(r['kind'] for r in extra_rows)),
      'by_split':{split:{'images':sum(r['split']==split for r in image_rows),
          'mismatch_images':sum(r['has_mismatch'] for r in image_rows if r['split']==split),
          'TP':sum(r['TP'] for r in image_rows if r['split']==split),
          'FP':sum(r['FP'] for r in image_rows if r['split']==split),
          'FN':sum(r['FN'] for r in image_rows if r['split']==split)} for split in sorted(set(r['split'] for r in image_rows))},
      'matching':'score>=0.25；同类IoU>=0.5按预测分数一对一匹配。另独立审查每个GT空间重叠候选中得分最高的类别，区分主预测错类与额外跨类框。',
      'interpretation':'不匹配是模型或标注需复核的证据，不能自动认定人工标错。训练图包含已学习样本；未重新训练或调参。'}
    assert sum(r['TP']+r['FN'] for r in image_rows)==len(instances)
    assert sum(r['FP'] for r in image_rows)==len(extra_rows)
    val=summary['by_split']['val']; assert (val['TP'],val['FP'],val['FN'])==(178,212,26)
    payload={'summary':summary,'images':image_rows,'instances':instances,'extra_predictions':extra_rows}
    (out/'annotation_match_audit.json').write_text(json.dumps(payload,ensure_ascii=False),encoding='utf-8')
    gallery.sort(key=lambda t:(t[0]['high_confidence_wrong'],t[0]['FN'],t[0]['primary_wrong'],t[0]['FP']),reverse=True)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    cards=[]
    for row,truth,preds in gallery:
        with Image.open(row['path']) as im:
            image=im.convert('RGB'); image.thumbnail((540,400))
            w,h=image.size; scale_x=w/im.width;scale_y=h/im.height
            canvas=Image.new('RGB',(w*2+12,h+35),'#FFFFFF')
            for panel_index in (0,1):
                panel=image.copy(); draw=ImageDraw.Draw(panel)
                objects=truth if panel_index==0 else preds
                for obj in objects:
                    box=obj['box'] if panel_index==0 else obj['bbox_xyxy']
                    xy=[box[0]*scale_x,box[1]*scale_y,box[2]*scale_x,box[3]*scale_y]
                    draw.rectangle(xy,outline='#23C174' if panel_index==0 else '#E55353',width=2)
                    text=obj['class_name']+(f" {obj['score']:.2f}" if panel_index else '')
                    tx=max(0,min(w-160,xy[0]));ty=max(0,min(h-20,xy[1]-20))
                    draw.text((tx,ty),text,font=font,fill='white',stroke_width=2,stroke_fill='black')
                canvas.paste(panel,(panel_index*(w+12),35))
            draw=ImageDraw.Draw(canvas); draw.text((5,5),'人工标注',font=font,fill='#087D45');draw.text((w+17,5),'模型输出 ≥0.25',font=font,fill='#BC3030')
            buffer=io.BytesIO();canvas.save(buffer,format='JPEG',quality=78)
            encoded=base64.b64encode(buffer.getvalue()).decode('ascii')
            import html
            attrs=f"{row['image']} {row['split']}"; key=html.escape(attrs,quote=True)
            cards.append(f'<article data-key="{key}"><h3>{html.escape(row["image"])}（{html.escape(row["split"])}）</h3><p>同类正确框 {row["TP"]}，多余框 {row["FP"]}，未正确匹配标注 {row["FN"]}，主预测错类 {row["primary_wrong"]}</p><img loading="lazy" src="data:image/jpeg;base64,{encoded}"></article>')
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>模型与人工标注不匹配复核</title><style>body{font-family:Microsoft YaHei,sans-serif;max-width:1200px;margin:32px auto;color:#22354a}header{position:sticky;top:0;background:white;padding:12px 0;border-bottom:1px solid #ccc}input{padding:10px;width:450px}article{padding:12px 0;border-bottom:1px solid #ddd}img{max-width:100%}p{line-height:1.7}</style><header><h1>模型与人工标注不匹配复核</h1><p>左侧绿色为人工标注，右侧红色为模型分数≥0.25的预测。优先排列高置信度主预测错类与未匹配标注。训练图不作为独立测试成绩。分歧不能直接证明人工标错。</p><input placeholder="输入文件名或train / val筛选" oninput="document.querySelectorAll('article').forEach(a=>a.hidden=!a.dataset.key.includes(this.value.trim()))"></header>'''+''.join(cards)+'</html>'
    (out/'模型标注不匹配图册.html').write_text(page,encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
