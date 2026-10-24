"""诊断无最终显示框图片；只读已有top100预测，不调阈值、不训练。"""
from pathlib import Path
import base64, io, json
from PIL import Image
from evaluate_two_stage import load_gt, iou

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path('E:/机器学习课设/模型检查报告/第二代实验/最高置信度框对照')
OUT=Path('E:/机器学习课设/模型检查报告/第二代实验/无显示框诊断')

def main():
    cfg=json.loads((ROOT/'configs/paths.json').read_text(encoding='utf-8'))
    res=json.loads((SOURCE/'resnet_postprocessed_predictions.json').read_text(encoding='utf-8'))
    bil=json.loads((SOURCE/'bilinear_postprocessed_predictions.json').read_text(encoding='utf-8'))
    empty=[r for r in res['images'] if not r['after']]
    assert {r['filename'] for r in empty}=={r['filename'] for r in bil['images'] if not r['after']}
    firstdoc=json.loads((Path(cfg['first_generation_review_archive'])/'严格阈值全量逐图候选/逐图原始候选_隐藏候选_最终分组.json').read_text(encoding='utf-8'))
    first={r['filename']:r for r in firstdoc['images']};gt=load_gt();rows=[]
    for r in empty:
        candidates=sorted(r['candidates'],key=lambda d:d['detector_score'],reverse=True)
        diagnostics=[]
        selected=list(candidates[:5])
        for n,g in enumerate(gt[r['filename']],1):
            overlaps=[(iou(g['bbox_xyxy'],d['bbox_xyxy']),d) for d in candidates]
            best=max(overlaps,key=lambda t:t[0],default=(0,None))
            suitable=sorted((d for ov,d in overlaps if ov>=.5),key=lambda d:d['detector_score'],reverse=True)
            d=suitable[0] if suitable else best[1]
            if d is not None and d not in selected:selected.append(d)
            diagnostics.append({'instance':n,'truth':g,'has_iou_ge_05_candidate':bool(suitable),
                'reason':'有贴合人工框的候选，但检测分数未超过50%' if suitable else '已保存的前100候选中，没有IoU达到0.5的框',
                'candidate':d,'candidate_iou':iou(g['bbox_xyxy'],d['bbox_xyxy']) if d else None,
                'max_iou_in_saved_candidates':best[0]})
        assert all(d['detector_score']<=.5 for d in candidates)
        first_display=[g['winner'] for g in first[r['filename']]['candidate_groups'] if g['winner']['score']>.5]
        rows.append({'filename':r['filename'],'image_path':r['image'],'width':r['width'],'height':r['height'],
            'truth':gt[r['filename']],'first_display':first_display,'selected_candidates':selected,
            'candidates':candidates,'diagnostics':diagnostics,
            'all_instances_have_candidate':all(d['has_iou_ge_05_candidate'] for d in diagnostics),
            'max_detector_score':candidates[0]['detector_score'] if candidates else None})
    summary={'images':len(rows),'all_instances_have_low_score_candidate':sum(r['all_instances_have_candidate'] for r in rows),
        'at_least_one_instance_has_no_iou05_candidate':sum(not r['all_instances_have_candidate'] for r in rows),
        'first_generation_has_display':sum(bool(r['first_display']) for r in rows),
        'instances':sum(len(r['diagnostics']) for r in rows),
        'instances_with_low_score_candidate':sum(d['has_iou_ge_05_candidate'] for r in rows for d in r['diagnostics'])}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'无显示框诊断依据.json').write_text(json.dumps({'summary':summary,'scope':'fixed172 validation, 13 empty after display; original saved top100 candidates only; manual XML used for diagnostic not inference',
        'criterion':'IoU>=0.5 against current XML is an approximate location check, not proof of bird or correct species; scores<=0.5 filtered by display rule',
        'rows':rows},ensure_ascii=False,indent=2),encoding='utf-8')
    for r in rows:
        with Image.open(r['image_path']) as im:
            im=im.convert('RGB');im.thumbnail((1400,1400));buf=io.BytesIO();im.save(buf,format='JPEG',quality=88)
        r['image']='data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode('ascii')
    template=(ROOT/'scripts/no_display_diagnostics_template.html').read_text(encoding='utf-8')
    payload=json.dumps({'rows':rows,'summary':summary},ensure_ascii=False).replace('<','\\u003c')
    (OUT/'十三张无显示框诊断.html').write_text(template.replace('__PAYLOAD__',payload),encoding='utf-8')
    print(json.dumps({'output':str(OUT),'summary':summary},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
