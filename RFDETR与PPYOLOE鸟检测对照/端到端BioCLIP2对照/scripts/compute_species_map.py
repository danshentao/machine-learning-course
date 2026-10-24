"""COCO species AP for the deployed RF-DETR/BioCLIP pipeline, without display cutoff."""
import os,sys,json,hashlib,time,copy,contextlib,io
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/八类端到端mAP_20261014'
OUT.mkdir(parents=True,exist_ok=True)
for k,p in {'TEMP':OUT/'tmp','TMP':OUT/'tmp','HF_HOME':OUT/'hf','TORCH_HOME':OUT/'torch'}.items():
    p.mkdir(exist_ok=True);os.environ[k]=str(p)
os.environ.update(PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
APP=ROOT.parent.parent/'本地鸟类识别网站'
sys.path.insert(0,str(APP))
import worker as w
import numpy as np
from PIL import Image
DETECTION=ROOT.parent/'reports/fixed_validation_comparison.json'
TRUTH=ROOT.parent.parent/'第三代鸟类识别现代模型对照/reports/bioclip2/固定验证集结果.json'
FORMAL=ROOT/'reports/end_to_end_results.json'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def inputs():
    d=read(DETECTION);rows=d['images'];truth={r['filename']:r for r in read(TRUTH)['rows']}
    assert len(rows)==len(truth)==172
    assert sum(len(r['truth']) for r in truth.values())==204
    prepared=[]
    for r in rows:
        t=truth[r['filename']]
        assert sorted(g['bbox_xyxy'] for g in r['truth'])==sorted(g['bbox_xyxy'] for g in t['truth'])
        # Only geometry and detector confidence enter inference; XML match annotations stripped.
        native=[{k:p[k] for k in ['candidate_index','bbox_xyxy','detector_score']} for p in r['models']['rfdetr']['native_candidates']]
        kept=[g['winner'] for g in w.grouped(native)]
        expected=[p['candidate_index'] for p in r['models']['rfdetr']['dedup_candidates']]
        assert [p['candidate_index'] for p in kept]==expected,'Historical dedup differs: '+r['filename']
        with Image.open(r['image_path']) as im:
            assert im.size==(r['width'],r['height'])
        for g in t['truth']:assert w.NAMES[int(g['class_id'])]==g['class_name']
        prepared.append({'filename':r['filename'],'image_path':r['image_path'],'width':r['width'],'height':r['height'],'items':kept,'truth':t['truth']})
    assert sum(len(r['items']) for r in prepared)==19114
    return prepared
def infer(rows,provenance):
    import torch,open_clip
    assert torch.cuda.is_available(),'Requires working GPU for 19k BioCLIP crops'
    constructor=open_clip.create_model_and_transforms;cached=[]
    def factory(*a,**kw):
        if not cached:cached.append(constructor(*a,**kw))
        return cached[0]
    open_clip.create_model_and_transforms=factory
    cache=OUT/'classifications';cache.mkdir(exist_ok=True)
    outputs=[];start=time.perf_counter();done=0
    for i,r in enumerate(rows):
        path=cache/(Path(r['filename']).stem+'.json')
        sig={'image_sha256':sha(r['image_path']),'items_sha256':hashlib.sha256(json.dumps(r['items'],sort_keys=True).encode()).hexdigest(),'encoder':provenance['encoder_sha256'],'head':provenance['head_sha256'],'worker':provenance['worker_sha256']}
        if path.exists():
            old=read(path);assert old['signature']==sig,'Cache changed';predictions=old['predictions']
        else:
            job={'model':'gen4','image':r['image_path']}
            predictions=w.classify_bio(job,{'displayed':r['items']})['rfdetr_bioclip2']
            assert len(predictions)==len(r['items'])
            save(path,{'signature':sig,'filename':r['filename'],'predictions':predictions})
        for p in predictions:
            assert np.isfinite(list(p['classifier_scores'].values())).all()
            assert abs(sum(p['classifier_scores'].values())-1)<1e-4
            assert w.NAMES[p['class_id']]==p['class_name']
        outputs.append({'filename':r['filename'],'predictions':predictions})
        done+=len(predictions)
        print(f'CLASSIFY {i+1}/172 crops {done}/19114 elapsed {time.perf_counter()-start:.1f}s',flush=True)
    save(OUT/'完整低分候选八类得分.json',{'provenance':provenance,'images':outputs,'classification_seconds':time.perf_counter()-start})
    return {r['filename']:r['predictions'] for r in outputs}
def coco_ground_truth(rows):
    annotations=[];images=[]
    for image_id,r in enumerate(rows,1):
        images.append({'id':image_id,'file_name':r['filename'],'width':r['width'],'height':r['height']})
        for g in r['truth']:
            x,y,x2,y2=g['bbox_xyxy'];bbox=[x,y,x2-x,y2-y]
            annotations.append({'id':len(annotations)+1,'image_id':image_id,'category_id':int(g['class_id'])+1,'bbox':bbox,'area':bbox[2]*bbox[3],'iscrowd':0})
    return {'info':{'description':'Frozen 172-image validation, 8 species, 204 instances'},'images':images,'annotations':annotations,'categories':[{'id':i+1,'name':n} for i,n in enumerate(w.NAMES)]}
def evaluate(gt,records,label):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    coco=COCO();coco.dataset=gt;coco.createIndex();dt=coco.loadRes(records);ev=COCOeval(coco,dt,'bbox')
    ev.params.imgIds=[r['id'] for r in gt['images']];ev.params.catIds=[r['id'] for r in gt['categories']]
    ev.evaluate();ev.accumulate();ev.summarize()
    precision=ev.eval['precision'];result={'mAP50':float(ev.stats[1]),'mAP50_95':float(ev.stats[0]),'AP75':float(ev.stats[2]),'AR100':float(ev.stats[8]),'predictions':len(records),'maxDets':list(ev.params.maxDets),'per_class':[]}
    pr=[]
    for k,cat in enumerate(gt['categories']):
        all_ap=precision[:,:,k,0,-1];a50=precision[0,:,k,0,-1]
        result['per_class'].append({'class_name':cat['name'],'GT':sum(x['category_id']==cat['id'] for x in gt['annotations']),'AP50':float(a50[a50>=0].mean()),'AP50_95':float(all_ap[all_ap>=0].mean())})
        pr.append({'class_name':cat['name'],'recall':ev.params.recThrs.tolist(),'precision_at_iou_0_50':[float(v) if v>=0 else None for v in a50]})
    save(OUT/(label+'_COCO指标.json'),result);save(OUT/(label+'_PR曲线数据.json'),pr)
    assert abs(np.mean([r['AP50'] for r in result['per_class']])-result['mAP50'])<1e-10
    return result
def report(rows,predictions,provenance):
    gt=coco_ground_truth(rows);save(OUT/'固定验证集八类COCO真值.json',gt)
    records=[];detection_rank=[];single=[]
    for iid,r in enumerate(rows,1):
        for p in predictions[r['filename']]:
            x,y,x2,y2=p['bbox_xyxy'];bbox=[x,y,x2-x,y2-y]
            record={'image_id':iid,'category_id':p['class_id']+1,'bbox':bbox,'score':float(p['detector_score']*p['classifier_top_score'])}
            records.append(record);detection_rank.append({**record,'score':p['detector_score']});single.append({**record,'category_id':1,'score':p['detector_score']})
    save(OUT/'八类完整候选_COCO预测.json',records)
    primary=evaluate(gt,records,'检测乘分类分数排序')
    sensitivity=evaluate(gt,detection_rank,'仅检测分数排序_敏感性对照')
    single_gt=copy.deepcopy(gt);single_gt['categories']=[{'id':1,'name':'bird'}]
    for g in single_gt['annotations']:g['category_id']=1
    single_result=evaluate(single_gt,single,'单类检测一致性核查')
    old=read(ROOT.parent/'reports/metrics_summary.json')['fixed_validation']['AP']['rfdetr']['common_iou_deduplicated_low_score']['AP50_COCO']
    assert abs(single_result['mAP50']-old)<1e-10,'COCO single-bird AP diverges from earlier benchmark'
    formal=read(FORMAL)['fixed_validation']['rows'];comparison=[]
    for oldrow in formal:
        previous=oldrow['models']['rfdetr']['display_candidates'];current={p['candidate_index']:p for p in predictions[oldrow['filename']] if p['detector_score']>.3}
        assert len(previous)==len(current)
        for oldp in previous:
            newp=current[oldp['candidate_index']]
            comparison.append({'filename':oldrow['filename'],'candidate_index':oldp['candidate_index'],'top1_agrees':newp['class_name']==oldp['class_name'],'score_max_delta':max(abs(newp['classifier_scores'][n]-oldp['classifier_scores'][n]) for n in w.NAMES)})
    verification={'displayed_boxes_compared':len(comparison),'top1_disagreements':[x for x in comparison if not x['top1_agrees']],'max_classifier_score_delta':max(x['score_max_delta'] for x in comparison),'single_class_AP50_reproduced':single_result['mAP50']}
    assert len(comparison)==196 and not verification['top1_disagreements']
    result={'primary':primary,'detector_score_ranking_sensitivity':sensitivity,'verification':verification,'provenance':provenance,'rules':{'images':172,'truth_instances':204,'species_count':8,'candidate_count':len(records),'detector_score_floor':.001,'display_cutoff_0_30_used':False,'dedup':'Original score-priority non-transitive IoU>=0.5, before classification','crop_expansion':.15,'prediction_label':'one top-1 species per kept box','AP_ranking_score':'detector_score * classifier_top_score; ranking only, not a calibrated probability','COCO_IoUs':np.arange(.5,.96,.05).round(2).tolist(),'COCO_recall_samples':101,'max_detections_per_class_per_image':100},'limitations':['Repeatedly explored validation set, not independent test performance.','Only current frozen XML/species truth; potential missing annotations remain.','Native detector output already has 0.001 confidence floor and top-k cap; no 0.30 display filtering.','No all-species multiple hypotheses per box; evaluates deployed top-1 species behavior.']}
    save(OUT/'最终八类端到端mAP.json',result)
    lines=['# 最终八类端到端 mAP','',f"RF-DETR Small＋BioCLIP 2；固定172张验证图，204只标注鸟，八类。",'',f"**mAP@0.5 = {primary['mAP50']:.8f}（{primary['mAP50']:.2%}）**",f"**mAP@0.5:0.95 = {primary['mAP50_95']:.8f}（{primary['mAP50_95']:.2%}）**",'', '|类别|标注实例|AP50|AP50:95|','|---|---:|---:|---:|']
    for c in primary['per_class']:lines.append(f"|{c['class_name']}|{c['GT']}|{c['AP50']:.4f}|{c['AP50_95']:.4f}|")
    lines+=['','## 计算口径','', '复用正式 RF-DETR 低分候选，在原分数优先 IoU≥0.5 去重后，对全部19114个保留框补跑八类分类。没有先按0.30过滤。每框只输出分类top-1，使用检测分数×该类别分类分数排序，依照 pycocotools COCOeval，101点插值、IoU0.50:0.05:0.95、每图每类最多100框。分类分数乘积只用于排序，不是校准概率。','',f"仅按检测分数排序的敏感性对照：mAP50 {sensitivity['mAP50']:.2%}，mAP50:95 {sensitivity['mAP50_95']:.2%}。两种口径均报告，主口径在计算结果前预先确定。",'', '## 核验','', f"单类 bird 的 AP50复现为 {single_result['mAP50']:.8f}，与历史相同。196个显示框的类别与历史端到端结果一致，八类分数最大差 {verification['max_classifier_score_delta']:.8f}。八类 AP50 的算术平均与整体mAP50一致。",'', '## 限制','', '这是反复用于探索的固定验证集成绩，不是独立测试成绩。评估沿用原真值，未改标注、权重、网站或历史报告。外来80/85张没有逐实例真值，未用于本次mAP计算。候选来自已保存原生输出，具有0.001分数下限及模型原生top-k限制。']
    (OUT/'八类mAP结果说明.md').write_text('\n'.join(lines),encoding='utf-8')
    print('FINAL',json.dumps({'mAP50':primary['mAP50'],'mAP50_95':primary['mAP50_95'],'per_class':primary['per_class'],'verification':verification},ensure_ascii=False),flush=True)
if __name__=='__main__':
    provenance={'date':'2026-10-14','detection_source':str(DETECTION),'detection_sha256':sha(DETECTION),'truth_source':str(TRUTH),'truth_sha256':sha(TRUTH),'formal_reference_sha256':sha(FORMAL),'encoder_sha256':sha(w.BIO_WEIGHTS),'head_sha256':sha(w.BIO_HEAD),'worker_sha256':sha(APP/'worker.py'),'detector_weights_sha256':sha(w.RF_CHECKPOINT),'evaluation_script_sha256':sha(__file__)}
    assert provenance['encoder_sha256']=='b7b2bf6fbc95799e42630e394cf95803892ab447c1a8ab629dbc82fbeaf7dfef'
    assert provenance['head_sha256']=='7d199f954bf7a125fe50d1f11f632268bbfa2d0c7b1bfa1bcbdf8a681b4b4ece'
    assert provenance['detector_weights_sha256']=='b2acea3b6cfccc011b9b0bd1237114e1f881c5e7fbed306870fd695a58d3ed16'
    save(OUT/'计算口径与来源.json',provenance)
    rows=inputs();pred=infer(rows,provenance);report(rows,pred,provenance)
