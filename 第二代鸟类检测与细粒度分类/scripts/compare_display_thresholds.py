"""同一固定验证集与已保存去重组，对比严格>30/40/50%显示规则。"""
from pathlib import Path
import base64, io, json
from PIL import Image
from evaluate_two_stage import load_gt, match

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path('E:/机器学习课设/模型检查报告/第二代实验/最高置信度框对照')
OUT=Path('E:/机器学习课设/模型检查报告/第二代实验/30_40_50阈值比较')
THRESHOLDS=(.3,.4,.5)

def inspect(truth,preds,expected):
    ug,up,pairs=match(truth,preds,True)
    sg,sp,spairs=match(truth,preds,False)
    spatial={pi:(gi,ov) for gi,pi,ov in spairs}
    annotated=[]
    for pi,d in enumerate(preds):
        item={**d,'same_class_correct_match':pi in up,'different_from_folder':d['class_name']!=expected}
        if pi not in spatial:
            item['spatial_status']='未匹配人工框（待确认误检或漏标等）'
        else:
            gi,ov=spatial[pi];item['matched_truth_class']=truth[gi]['class_name'];item['matched_truth_iou']=ov
            item['spatial_status']='位置匹配、类别一致' if d['class_id']==truth[gi]['class_id'] else '位置匹配、类别不同'
        annotated.append(item)
    return {'displayed':annotated,'TP':len(pairs),'FP':len(preds)-len(up),'FN':len(truth)-len(ug),
        'unmatched_annotation_boxes':len(preds)-len(sp),
        'spatially_matched_wrong_class':sum(preds[pi]['class_id']!=truth[gi]['class_id'] for gi,pi,ov in spairs),
        'class_disagreement':any(d['class_name']!=expected for d in preds),'no_display':not preds}

def main():
    cfg=json.loads((ROOT/'configs/paths.json').read_text(encoding='utf-8'))
    mapping=json.loads((ROOT/'configs/class_mapping.json').read_text(encoding='utf-8'))
    sources={}
    for c in mapping['classes']:
        for p in (Path(cfg['source_images'])/c['folder']).rglob('*.jpg'):
            if p.name in sources:raise ValueError('Duplicate filename '+p.name)
            sources[p.name]={'expected':c['name'],'folder':c['folder']}
    truth=load_gt();rows={};summaries={};source_paths={}
    for model in ['resnet','bilinear']:
        path=SOURCE/f'{model}_postprocessed_predictions.json';source_paths[model]=str(path)
        doc=json.loads(path.read_text(encoding='utf-8'))
        assert len(doc['images'])==172
        assert doc['metadata']['postprocess']['group_iou']==.5
        for r in doc['images']:
            winners=[g['winner'] for g in r['groups']]
            name=r['filename'];origin=sources[name]
            row=rows.setdefault(name,{'filename':name,'image_path':r['image'],'width':r['width'],'height':r['height'],
                'expected':origin['expected'],'folder':origin['folder'],'truth':truth[name],'models':{}})
            row['models'][model]={str(int(t*100)):inspect(truth[name],[d for d in winners if d['detector_score']>t],origin['expected']) for t in THRESHOLDS}
            assert [d['candidate_index'] for d in row['models'][model]['50']['displayed']]==[d['candidate_index'] for d in r['after']]
        summaries[model]={}
        for key in ['30','40','50']:
            arr=[r['models'][model][key] for r in rows.values()]
            tp=sum(r['TP'] for r in arr);fp=sum(r['FP'] for r in arr);fn=sum(r['FN'] for r in arr)
            p=tp/(tp+fp) if tp+fp else 0;rec=tp/(tp+fn) if tp+fn else 0
            summaries[model][key]={'images':172,'GT':tp+fn,'displayed_boxes':tp+fp,'TP':tp,'FP':fp,'FN':fn,
                'precision':p,'recall':rec,'F1':2*p*rec/(p+rec) if p+rec else 0,
                'no_display_images':sum(r['no_display'] for r in arr),
                'class_disagreement_images':sum(r['class_disagreement'] for r in arr),
                'unmatched_annotation_boxes':sum(r['unmatched_annotation_boxes'] for r in arr),
                'spatially_matched_wrong_class':sum(r['spatially_matched_wrong_class'] for r in arr)}
    rows=list(rows.values())
    assert len(rows)==172 and all(set(r['models'])=={'resnet','bilinear'} for r in rows)
    for r in rows:
        for key in ['30','40','50']:
            a=r['models']['resnet'][key]['displayed'];b=r['models']['bilinear'][key]['displayed']
            assert [(d['bbox_xyxy'],d['detector_score']) for d in a]==[(d['bbox_xyxy'],d['detector_score']) for d in b]
    recovery={}
    for model in ['resnet','bilinear']:
        empty=[r for r in rows if r['models'][model]['50']['no_display']]
        recovery[model]={key:{'original_empty_images':len(empty),
            'now_has_display':sum(not r['models'][model][key]['no_display'] for r in empty),
            'now_has_correct_match':sum(r['models'][model][key]['TP']>0 for r in empty)} for key in ['30','40','50']}
    OUT.mkdir(parents=True,exist_ok=True)
    report={'rule':{'thresholds':THRESHOLDS,'comparison':'detector_score strictly > threshold',
        'dedup':'reuse existing cross-class groups, IoU>=0.5, winner by detector_score',
        'scope':'fixed172 validation; same saved top100 candidates; no retraining',
        'PRF':'same class and IoU>=0.5 one-to-one matching to existing XML',
        'unmatched_annotation_warning':'未匹配人工框并不等于已确认背景误检，可能涉及漏标、重复框或框偏移。',
        'folder_disagreement':'any displayed class differs from original folder; ignores count/position',
        'model_selection_warning':'阈值比较是验证集上的探索，不能视为独立测试集成绩。'},
        'sources':source_paths,'summary':summaries,'empty_image_recovery':recovery,'rows':rows}
    (OUT/'阈值比较完整结果.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    for r in rows:
        with Image.open(r['image_path']) as im:
            im=im.convert('RGB');im.thumbnail((1400,1400));buf=io.BytesIO();im.save(buf,format='JPEG',quality=85)
        r['image']='data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode('ascii')
    template=(ROOT/'scripts/threshold_comparison_template.html').read_text(encoding='utf-8')
    payload=json.dumps({'rows':rows,'summary':summaries,'recovery':recovery},ensure_ascii=False).replace('<','\\u003c')
    (OUT/'30_40_50阈值对照.html').write_text(template.replace('__PAYLOAD__',payload),encoding='utf-8')
    (OUT/'阈值比较汇总.json').write_text(json.dumps({'summary':summaries,'empty_image_recovery':recovery},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(OUT),'summary':summaries,'empty_image_recovery':recovery},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
