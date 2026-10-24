"""生成单鸟最高分显示图册，其他类别及框仅出现在候选数据表。"""
import argparse
import base64
import io
import json
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
from batch_infer import ROOT, CLASS_NAMES, MAPPING
from detection_display import group_detections
from evaluate_predictions import class_correct_counts, iou

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--predictions',type=Path,default=ROOT/'runs/confidence_audit/all_originals/class_confidences.json')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--iou-threshold',type=float,default=.5)
    ap.add_argument('--score-threshold',type=float,default=.5)
    args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    if not 0 <= args.score_threshold < 1:ap.error('显示阈值必须在[0,1)之间')
    threshold=args.score_threshold
    run=ROOT/f'runs/unique_bird_display_gt{round(threshold*100)}';run.mkdir(parents=True,exist_ok=True)
    original=json.loads(args.predictions.read_text(encoding='utf-8'))
    truth={}
    with zipfile.ZipFile(ROOT.parent/'2804306_1790845704.zip') as z:
        for n in z.namelist():
            if not n.endswith('.xml'):continue
            root=ET.fromstring(z.read(n)); boxes=[]
            for o in root.findall('object'):
                name=MAPPING['source_label_to_standard'][o.findtext('name')];b=o.find('bndbox')
                boxes.append({'class_id':CLASS_NAMES.index(name),'class_name':name,
                  'bbox_xyxy':[float(b.findtext(k)) for k in ('xmin','ymin','xmax','ymax')]})
            truth[Path(n).stem+'.jpg']=boxes
    pages=[];records=[];before_count=0;after_count=0;changed_images=0
    for rec in original['images']:
        # 若输入已经经过显示分组，必须复用其完整 candidate_groups；
        # 只读 detections 会丢掉此前隐藏的候选，导致图册漏记重复框。
        groups=rec.get('candidate_groups') or group_detections(rec['detections'],args.iou_threshold)
        winners=[g['winner'] for g in groups]
        for g in groups:
            assert all(g['winner']['score']>=p['score'] for p in g['hidden_candidates'])
            assert all(iou(g['winner']['bbox_xyxy'],p['bbox_xyxy'])>=args.iou_threshold for p in g['hidden_candidates'])
        if not rec.get('candidate_groups'):
            assert sum(1+len(g['hidden_candidates']) for g in groups)==len(rec['detections'])
        visible=[p for p in winners if p['score']>threshold]
        before=sum(p['score']>threshold for g in groups for p in [g['winner'], *g.get('hidden_candidates',[])])
        before_count+=before;after_count+=len(visible);changed_images+=int(before>len(visible))
        for i,p in enumerate(visible):
            assert all(iou(p['bbox_xyxy'],q['bbox_xyxy'])<args.iou_threshold for q in visible[i+1:])
        filename=rec['filename']; gt=truth[filename]
        counts=class_correct_counts({filename:[{'class_id':g['class_id'],'bbox':g['bbox_xyxy']} for g in gt]},
                                    {filename:visible},threshold,.5)
        tp,fp,fn=[int(a.sum()) for a in counts[:3]]
        with Image.open(rec['image']) as im:
            width,height=im.size;thumb=im.convert('RGB');thumb.thumbnail((720,520))
            buf=io.BytesIO();thumb.save(buf,format='JPEG',quality=68)
        # 将全部候选保存到项目JSON；HTML保留隐藏候选的类别、分数、框和同候选八类得分。
        def compact(p):return {'c':p['class_id'],'s':p['score'],
            'b':[round(v,1) for v in p['bbox_xyxy']],
            'cs':[round(v,6) for v in p['class_scores']] if p.get('class_scores') else None}
        shown_groups=[g for g in groups if g['winner']['score']>threshold]
        low_groups=[g for g in groups if g['winner']['score']<=threshold]
        def html_group(g):
            high=[p for p in g['hidden_candidates'] if p['score']>threshold]
            low=[p for p in g['hidden_candidates'] if p['score']<=threshold]
            return {'winner':compact(g['winner']),
                'hidden':[compact(p) for p in high+low[:10]],
                'hidden_total':len(g['hidden_candidates'])}
        pages.append({'name':filename,'split':rec['split'],'width':width,'height':height,
            'thumbnail':'data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode('ascii'),
            'gt':[compact(dict(g,score=1)) for g in gt],
            'groups':[html_group(g) for g in shown_groups+low_groups[:10]],
            'low_group_total':len(low_groups),
            'before':before,'after':len(visible),'tp':tp,'fp':fp,'fn':fn,
            'different':bool(fp or fn)})
        records.append({**rec,'detections':winners,'candidate_groups':groups})
    # 独立检查验证集；不能把训练图表现当作验证成绩。
    val_records=[r for r in records if r['split']=='val']
    gt_val={r['filename']:[{'class_id':g['class_id'],'bbox':g['bbox_xyxy']} for g in truth[r['filename']]] for r in val_records}
    pred_val={r['filename']:[p for p in r['detections'] if p['score']>threshold] for r in val_records}
    tp,fp,fn,_,_=class_correct_counts(gt_val,pred_val,threshold,.5)
    tp,fp,fn=int(tp.sum()),int(fp.sum()),int(fn.sum())
    before_val={r['filename']:[p for p in r['detections'] if p['score']>threshold] for r in original['images'] if r['split']=='val'}
    bt,bf,bn,_,_=class_correct_counts(gt_val,before_val,threshold,.5)
    summary={'images':len(pages),'display_score_threshold':threshold,'threshold_comparison':'>','group_iou_threshold':args.iou_threshold,
        'visible_boxes_before':before_count,'visible_boxes_after':after_count,
        'hidden_boxes_at_display_threshold':before_count-after_count,'images_with_removed_duplicate_boxes':changed_images,
        'validation_counts_after':{'TP':tp,'FP':fp,'FN':fn},
        'validation_counts_before':{'TP':int(bt.sum()),'FP':int(bf.sum()),'FN':int(bn.sum())},
        'rule':'按置信度降序与已保留代表框比较，不区分类别；IoU>=0.5的候选隐藏，只显示最高分原框。不同重叠组各保留一框。',
        'annotation_note':'用户确认可能漏标；与标注不匹配不能自动判断模型错或人工错。未改标注、权重或划分。'}
    (run/'predictions.json').write_text(json.dumps({'metadata':{**original['metadata'],
        'candidate_nms_iou':original['metadata']['nms_iou'],'nms_iou':args.iou_threshold,
        'display_nms_iou':args.iou_threshold,'nms_mode':'class_agnostic_greedy'},'images':records},ensure_ascii=False),encoding='utf-8')
    (run/'display_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    template=Path(__file__).with_name('unique_birds_gallery_template.html').read_text(encoding='utf-8')
    payload=json.dumps({'images':pages,'summary':summary,'classes':CLASS_NAMES},ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    args.output.write_text(template.replace('__DATA_JSON__',payload),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
