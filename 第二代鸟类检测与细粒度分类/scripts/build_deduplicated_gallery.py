"""仅后处理已保存的预测：保留最高检测分数重叠框，原始预测和权重只读。"""
from pathlib import Path
import base64, csv, io, json
from PIL import Image
from evaluate_two_stage import iou, load_gt, match

ROOT=Path(__file__).resolve().parents[1]
OUT=Path('E:/机器学习课设/模型检查报告/第二代实验/最高置信度框对照')
IOU=.5
THRESHOLD=.5

def group_candidates(candidates):
    groups=[]
    for idx,d in sorted(enumerate(candidates),key=lambda item:(-item[1]['detector_score'],item[0])):
        possible=[(iou(d['bbox_xyxy'],g['winner']['bbox_xyxy']),j) for j,g in enumerate(groups)]
        overlap,j=max(possible,default=(0,None))
        if overlap>=IOU:
            groups[j]['hidden'].append({'candidate_index':idx,'iou_with_winner':overlap,**d})
        else:
            groups.append({'winner':{'candidate_index':idx,**d},'hidden':[]})
    return groups

def metrics(records,gt,field):
    tp=fp=fn=0
    for r in records:
        preds=r[field];truth=gt[r['filename']]
        ug,up,m=match(truth,preds)
        tp+=len(m);fp+=len(preds)-len(up);fn+=len(truth)-len(ug)
    p=tp/(tp+fp) if tp+fp else 0;r=tp/(tp+fn) if tp+fn else 0
    return {'TP':tp,'FP':fp,'FN':fn,'precision':p,'recall':r,'F1':2*p*r/(p+r) if p+r else 0}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    gt=load_gt();by_name={};summaries={};csv_rows=[]
    for model,run in [('resnet','two_stage_resnet_fixed_val'),('bilinear','two_stage_bilinear_fixed_val')]:
        source=ROOT/'runs'/run/'predictions.json'
        doc=json.loads(source.read_text(encoding='utf-8'));records=[]
        for r in doc['images']:
            groups=group_candidates(r['candidates'])
            before=[d for d in r['candidates'] if d['detector_score']>THRESHOLD]
            after=[g['winner'] for g in groups if g['winner']['detector_score']>THRESHOLD]
            hidden=[d for g in groups for d in g['hidden'] if d['detector_score']>THRESHOLD]
            item={**r,'before':before,'after':after,'groups':groups,'hidden_high_score':hidden}
            records.append(item)
            row=by_name.setdefault(r['filename'],{'filename':r['filename'],'image_path':r['image'],'width':r['width'],'height':r['height'],'truth':gt[r['filename']]})
            row[model]={'before':before,'after':after,'hidden':hidden,'all_candidates':r['candidates']}
            csv_rows.append({'classifier':model,'filename':r['filename'],'before':len(before),'after':len(after),'hidden_duplicates':len(hidden)})
        summaries[model]={'images':len(records),'before_boxes':sum(len(r['before']) for r in records),'after_boxes':sum(len(r['after']) for r in records),
            'changed_images':sum(len(r['before'])!=len(r['after']) for r in records),'before_metrics':metrics(records,gt,'before'),'after_metrics':metrics(records,gt,'after')}
        (OUT/f'{model}_postprocessed_predictions.json').write_text(json.dumps({'metadata':{**doc['metadata'],'postprocess':{'group_iou':IOU,'rank':'detector_score descending','threshold_strictly_greater':THRESHOLD,'cross_class':True,'matching':'each candidate compares only to retained winners; no transitive grouping'},'source_predictions':str(source)},'images':records},ensure_ascii=False),encoding='utf-8')
    rows=list(by_name.values())
    for row in rows:
        with Image.open(row['image_path']) as im:
            im=im.convert('RGB');im.thumbnail((1600,1600));buffer=io.BytesIO();im.save(buffer,format='JPEG',quality=85)
        row['image']='data:image/jpeg;base64,'+base64.b64encode(buffer.getvalue()).decode('ascii')
    (OUT/'summary.json').write_text(json.dumps({'group_iou':IOU,'display_detector_threshold_strictly_greater':THRESHOLD,'models':summaries,'note':'Overlap is a duplicate heuristic, not proof of same bird; no training or label changes. P/R/F1 use original XML, sameclass IoU>=0.5.'},ensure_ascii=False,indent=2),encoding='utf-8')
    with (OUT/'逐图框数量对比.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(csv_rows[0]));w.writeheader();w.writerows(csv_rows)
    template=(ROOT/'scripts/deduplicated_gallery_template.html').read_text(encoding='utf-8')
    payload=json.dumps({'rows':rows,'summary':summaries},ensure_ascii=False).replace('<','\\u003c')
    (OUT/'第二代最高置信度框对照.html').write_text(template.replace('__PAYLOAD__',payload),encoding='utf-8')
    print(json.dumps({'output':str(OUT),'summary':summaries},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
