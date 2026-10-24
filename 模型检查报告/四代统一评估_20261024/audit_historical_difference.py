"""Explain the first-generation AP change and audit frozen-model identities."""
import sys,json,copy,io,contextlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import unified_evaluation as u
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
gt=u.read(u.ROOT/'coco_truth.json');raw=u.read(u.FIRST)
source={Path(r['image']).name:r for r in raw['images']}
def ap(clip):
    records=[]
    for im in gt['images']:
        for p in source[im['file_name']]['detections']:
            box=p['bbox_xyxy']
            if clip:box=[min(max(v,0),lim) for v,lim in zip(box,[im['width'],im['height']]*2)]
            x,y,x2,y2=box
            if x2<=x or y2<=y:continue
            records.append({'image_id':im['id'],'category_id':p['class_id']+1,'bbox':[x,y,x2-x,y2-y],'score':p['score']})
    with contextlib.redirect_stdout(io.StringIO()):
        c=COCO();c.dataset=gt;c.createIndex();dt=c.loadRes(records);ev=COCOeval(c,dt,'bbox');ev.evaluate();ev.accumulate();ev.summarize()
    return {'mAP50':float(ev.stats[1]),'mAP50_95':float(ev.stats[0]),'predictions':len(records)}
native=ap(False);clipped=ap(True);results=u.read(u.ROOT/'统一评估结果.json')
cache=u.read(u.RF_CACHE)['provenance']
assert cache['encoder_sha256']==u.sha(u.cfg.BIO_WEIGHTS)
assert cache['head_sha256']==u.sha(u.cfg.BIO_HEAD)
assert cache['detector_weights_sha256']==u.sha(u.cfg.RF_CHECKPOINT)
assert abs(native['mAP50']-.86146)<.00001
audit={'gen1_original_official_NMS_only':native,'gen1_clip_to_image_official_NMS_only':clipped,'gen1_unified_clip_and_extra_iou_dedup':{k:results['results']['gen1'][k] for k in ['mAP50','mAP50_95','low_score_candidates']},'first_generation_native_AP_matches_historical':True,'fourth_generation_weights_match_historical_classification_cache':True,'final_evaluation_script_sha256':u.sha(u.ROOT/'unified_evaluation.py'),'explanation':'The old 86.15% used native class-wise official NMS outputs. Common extra class-agnostic score-priority IoU>=0.5 dedup, as used by the fourth-generation pipeline, changes which boxes are retained. Bounding clipping is also applied uniformly. All versions are reported without picking a rule based on labels.'}
u.save(u.ROOT/'历史差异核验.json',audit)
print(json.dumps(audit,ensure_ascii=False,indent=2))
with (u.ROOT/'统一指标说明.md').open('a',encoding='utf-8') as f:
    f.write(f"\n\n## 第一代78.30%与历史86.15%的来源\n\n使用本轮同一COCO真值核验：原生官方NMS候选AP50为{native['mAP50']:.6%}，只裁图像边界后为{clipped['mAP50']:.6%}，再加入四代统一的跨类别IoU≥0.5额外去重后为{results['results']['gen1']['mAP50']:.6%}。第一代历史mAP已复现，差异来自后处理定义变化，并非训练退化。额外去重会移除部分重叠但定位更好的框，因此本轮统一后的AP可能更低。\n")
