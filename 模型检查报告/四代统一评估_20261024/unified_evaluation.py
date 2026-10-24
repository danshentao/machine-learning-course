"""Frozen four-generation evaluation. Never trains or modifies historical files."""
import os,sys,json,hashlib,time,math,csv,copy,argparse,contextlib,io
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parents[1]
APP=BASE/'本地鸟类识别网站'
sys.path.insert(0,str(APP))
import settings as cfg
NAMES=cfg.NAMES
for key,folder in {'TEMP':'tmp','TMP':'tmp','HF_HOME':'hf','TORCH_HOME':'torch','MPLCONFIGDIR':'mpl'}.items():
    p=ROOT/'cache'/folder;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
os.environ.update(PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
RF_DIR=BASE/'RFDETR与PPYOLOE鸟检测对照'
RF_MAP=RF_DIR/'端到端BioCLIP2对照/reports/八类端到端mAP_20261014'
TRUTH=BASE/'第三代鸟类识别现代模型对照/reports/bioclip2/固定验证集结果.json'
FIRST=BASE/'自然场景鸟类检测PPYOLOE/runs/baseline_640/ap50_epoch_13_predictions/predictions.json'
DET=RF_DIR/'reports/fixed_validation_comparison.json'
RF_CACHE=RF_MAP/'完整低分候选八类得分.json'
MODELS={'gen1':'第一代 PP-YOLOE-s','gen2_resnet18':'第二代 PP-YOLOE+ + ResNet18','gen2_bilinear':'第二代 PP-YOLOE+ + Bilinear CNN','gen3':'第三代 PP-YOLOE+ + BioCLIP 2','gen4':'第四代 RF-DETR Small + BioCLIP 2'}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def stable(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def strip(p):return {k:p[k] for k in ['candidate_index','bbox_xyxy','detector_score']}
def prepare():
    from PIL import Image
    src=read(DET);truth={r['filename']:r for r in read(TRUTH)['rows']}
    first=read(FIRST);g1={Path(r['image']).name:r for r in first['images']}
    rf={r['filename']:r['predictions'] for r in read(RF_CACHE)['images']}
    assert len(src['images'])==len(truth)==len(g1)==len(rf)==172
    assert sha(cfg.FIRST_WEIGHTS)==sha(Path(first['metadata']['weights']+'.pdparams'))
    assert sha(cfg.DETECTOR_WEIGHTS)==read(RF_DIR/'runs/baseline_ppyoloe/fixed_val_raw.json')['metadata']['weights_sha256']
    provenance={'sources':{str(p):sha(p) for p in [DET,TRUTH,FIRST,RF_CACHE]},'weights':{str(p):sha(p) for p in [cfg.FIRST_WEIGHTS,cfg.DETECTOR_WEIGHTS,*cfg.CNN_WEIGHTS.values(),cfg.BIO_HEAD,cfg.RF_CHECKPOINT,cfg.BIO_WEIGHTS]},'script_sha256':sha(__file__),'website_worker_sha256':sha(APP/'worker.py')}
    rows=[];g1rows=[];g4rows=[]
    for r in src['images']:
        name=r['filename'];t=truth[name];f=g1[name]
        assert sorted(g['bbox_xyxy'] for g in r['truth'])==sorted(g['bbox_xyxy'] for g in t['truth'])
        assert (r['width'],r['height'])==(t['width'],t['height'])==(f['width'],f['height'])
        with Image.open(r['image_path']) as im,Image.open(t['image_path']) as old,Image.open(f['image']) as orig:
            pix=im.convert('RGB').tobytes();assert pix==old.convert('RGB').tobytes()==orig.convert('RGB').tobytes()
        for g in t['truth']:assert NAMES[g['class_id']]==g['class_name']
        ps=[]
        for idx,p in enumerate(f['detections']):
            box=[min(max(float(v),0),lim) for v,lim in zip(p['bbox_xyxy'],[r['width'],r['height']]*2)]
            if box[2]<=box[0] or box[3]<=box[1]:continue
            ps.append({'candidate_index':idx,'bbox_xyxy':box,'detector_score':p['score'],'class_id':p['class_id'],'class_name':p['class_name'],'classifier_top_score':None,'classifier_scores':{}})
        g1kept=[g['winner'] for g in cfg.grouped(ps)]
        ppnative=[strip(p) for p in r['models']['ppyoloe']['native_candidates']]
        ppkept=[g['winner'] for g in cfg.grouped(ppnative)]
        assert [p['candidate_index'] for p in ppkept]==[p['candidate_index'] for p in r['models']['ppyoloe']['dedup_candidates']]
        rfkept=[strip(g['winner']) for g in cfg.grouped([strip(p) for p in r['models']['rfdetr']['native_candidates']])]
        assert [strip(p) for p in rf[name]]==rfkept
        rows.append({'filename':name,'image_path':r['image_path'],'width':r['width'],'height':r['height'],'image_sha256':sha(r['image_path']),'truth':t['truth'],'items':ppkept})
        g1rows.append({'filename':name,'predictions':g1kept});g4rows.append({'filename':name,'predictions':rf[name]})
    assert sum(len(r['truth']) for r in rows)==204
    save(ROOT/'inputs.json',{'provenance':provenance,'rows':rows})
    save(ROOT/'predictions/gen1.json',{'images':g1rows});save(ROOT/'predictions/gen4.json',{'images':g4rows})
    save(ROOT/'rules.json',{'dataset':'Same frozen 172 validation images, 204 instances, 8 species; decoded pixels checked across sources','AP':'pycocotools COCOeval; 101 recall points; IoU 0.50 for mAP50 and 0.50:0.05:0.95 for mAP; maxDets=[1,10,100] per category','native_candidate_floor':.001,'native_candidate_cap':300,'additional_dedup':'All generations use original score-priority non-transitive class-agnostic IoU>=0.5 suppression, before classifier','AP_display_filter':False,'one_species_per_box':'top-1 only, matching deployed pipeline','ranking_score':{'gen1':'native joint localization/species score; no separate classifier','gen2_gen3_gen4':'detector_score * classifier_top_score, as fourth-generation primary result'},'crop':'Existing per-model preprocessing, 15% box expansion for all stage-two classifiers; CNN uses original integer truncation; BioCLIP floor/ceil','operating_point':'native detector score strictly >0.30, IoU>=0.5 matching','image_exact_accuracy':'All annotated birds correctly located and classified, no extra predicted boxes, divided by 172; assessed against frozen XML','matched_box_classification_accuracy':'Correct species among geometry-matched bird boxes / number of geometry-matched boxes; includes all 8 species and all 172 images, excludes unmatched boxes from denominator','limitations':['Reused validation set, not an independent test.','Annotation omissions can make correct detections disagree with XML.','Historical weights and training manifests differ across generations; metric standardization does not equal identical training budget.','Native model scores are not equally calibrated; threshold 0.30 is a shared operating point, not an equivalent probability.','Historical native outputs are retained; first and PP use 640, RF uses 512.']})
    print(json.dumps({'images':172,'GT':204,'gen1_kept':sum(len(r['predictions']) for r in g1rows),'pp_kept':sum(len(r['items']) for r in rows),'rf_kept':sum(len(r['predictions']) for r in g4rows)},ensure_ascii=False),flush=True)
def classify_cnn():
    import worker as w
    paddle,handles=w.paddle_setup()
    import numpy as np
    from PIL import Image
    sys.path.insert(0,str(cfg.SECOND/'scripts'))
    from train_classifier import BirdClassifier,pad_resize,MEAN,STD
    rows=read(ROOT/'inputs.json')['rows']
    models={key:BirdClassifier(key=='bilinear') for key in cfg.CNN_WEIGHTS}
    for k,m in models.items():m.set_state_dict(paddle.load(str(cfg.CNN_WEIGHTS[k])));m.eval()
    outputs={key:[] for key in models};start=time.perf_counter()
    model_sig={'weights':{k:sha(p) for k,p in cfg.CNN_WEIGHTS.items()},'worker':sha(APP/'worker.py')}
    for ri,r in enumerate(rows):
        path=ROOT/'cache/cnn'/(Path(r['filename']).stem+'.json')
        signature=stable({'image':r['image_sha256'],'items':r['items'],**model_sig})
        if path.exists():
            d=read(path);assert d['signature']==signature;result=d['outputs']
        else:
            arrays=[]
            with Image.open(r['image_path']) as fp:
                im=fp.convert('RGB')
                for p in r['items']:
                    x,y,x2,y2=p['bbox_xyxy'];dx=(x2-x)*.15;dy=(y2-y)*.15
                    patch=im.crop((max(0,int(x-dx)),max(0,int(y-dy)),min(im.width,int(x2+dx)),min(im.height,int(y2+dy))))
                    a=np.asarray(pad_resize(patch,224),dtype=np.float32)/255
                    arrays.append(((a-MEAN)/STD).transpose(2,0,1).copy())
            result={}
            with paddle.no_grad():
                for key,m in models.items():
                    scores=[]
                    for st in range(0,len(arrays),16):scores.extend(paddle.nn.functional.softmax(m(paddle.to_tensor(np.stack(arrays[st:st+16]))),axis=1).numpy().tolist())
                    result[key]=w.decorate(r['items'],scores)
            save(path,{'signature':signature,'outputs':result})
        for k in models:outputs[k].append({'filename':r['filename'],'predictions':result[k]})
        if (ri+1)%10==0 or ri==0:print(f'CNN {ri+1}/172 elapsed={time.perf_counter()-start:.1f}s',flush=True)
    for k,v in outputs.items():save(ROOT/f'predictions/gen2_{k}.json',{'images':v,'seconds':time.perf_counter()-start})
def classify_bio():
    import worker as w
    import torch,open_clip
    assert torch.cuda.is_available()
    original=open_clip.create_model_and_transforms;cache=[]
    def factory(*a,**kw):
        if not cache:cache.append(original(*a,**kw))
        return cache[0]
    open_clip.create_model_and_transforms=factory
    rows=read(ROOT/'inputs.json')['rows'];outputs=[];start=time.perf_counter();count=0
    model_sig={str(p):sha(p) for p in [cfg.BIO_WEIGHTS,cfg.BIO_HEAD,APP/'worker.py']}
    for ri,r in enumerate(rows):
        path=ROOT/'cache/bio'/(Path(r['filename']).stem+'.json')
        signature=stable({'image':r['image_sha256'],'items':r['items'],'model':model_sig})
        if path.exists():
            d=read(path);assert d['signature']==signature;pred=d['predictions']
        else:
            pred=w.classify_bio({'model':'gen3','image':r['image_path']},{'displayed':r['items']})['bioclip2']
            save(path,{'signature':signature,'predictions':pred})
        assert len(pred)==len(r['items']);count+=len(pred)
        outputs.append({'filename':r['filename'],'predictions':pred})
        if (ri+1)%10==0 or ri==0:print(f'BIO {ri+1}/172 crops={count} elapsed={time.perf_counter()-start:.1f}s',flush=True)
    save(ROOT/'predictions/gen3.json',{'images':outputs,'seconds':time.perf_counter()-start})
def ground_truth(rows):
    gt={'info':{'description':'Frozen 172-image eight-species validation'},'images':[],'annotations':[],'categories':[{'id':i+1,'name':n} for i,n in enumerate(NAMES)]}
    for iid,r in enumerate(rows,1):
        gt['images'].append({'id':iid,'file_name':r['filename'],'width':r['width'],'height':r['height']})
        for g in r['truth']:
            x,y,x2,y2=g['bbox_xyxy'];b=[x,y,x2-x,y2-y]
            gt['annotations'].append({'id':len(gt['annotations'])+1,'image_id':iid,'category_id':g['class_id']+1,'bbox':b,'area':b[2]*b[3],'iscrowd':0})
    return gt
def score(p):return p['detector_score'] if p['classifier_top_score'] is None else p['detector_score']*p['classifier_top_score']
def match(truth,pred,class_aware):
    remaining=set(range(len(truth)));pairs=[]
    for pi,p in sorted(enumerate(pred),key=lambda z:(-z[1]['detector_score'],z[1]['candidate_index'])):
        opts=sorted(((cfg.iou(p['bbox_xyxy'],truth[gi]['bbox_xyxy']),gi) for gi in remaining if not class_aware or p['class_id']==truth[gi]['class_id']),reverse=True)
        if opts and opts[0][0]>=.5:
            ov,gi=opts[0];remaining.remove(gi);pairs.append((gi,pi,ov))
    return pairs
def evaluate():
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    inputs=read(ROOT/'inputs.json');rows=inputs['rows'];gt=ground_truth(rows);save(ROOT/'coco_truth.json',gt)
    frozen_gt=read(RF_MAP/'固定验证集八类COCO真值.json')
    assert [(g['image_id'],g['category_id'],g['bbox']) for g in gt['annotations']]==[(g['image_id'],g['category_id'],g['bbox']) for g in frozen_gt['annotations']]
    results={};all_images=[];perclass=[]
    for key,label in MODELS.items():
        pd={r['filename']:r['predictions'] for r in read(ROOT/f'predictions/{key}.json')['images']};assert set(pd)=={r['filename'] for r in rows}
        records=[];metrics={'TP':0,'FP':0,'FN':0,'exact_correct_images':0,'no_box_images':0,'geometry_matched_boxes':0,'geometry_matched_class_correct':0,'class_disagreement_images':0};detail=[]
        for iid,r in enumerate(rows,1):
            pred=pd[r['filename']]
            for p in pred:
                assert 0<=p['class_id']<8 and NAMES[p['class_id']]==p['class_name']
                if p['classifier_scores']:assert abs(sum(p['classifier_scores'].values())-1)<1e-4
                x,y,x2,y2=p['bbox_xyxy'];assert 0<=x<x2<=r['width'] and 0<=y<y2<=r['height']
                records.append({'image_id':iid,'category_id':p['class_id']+1,'bbox':[x,y,x2-x,y2-y],'score':float(score(p))})
            shown=[p for p in pred if p['detector_score']>.3]
            correct=match(r['truth'],shown,True);spatial=match(r['truth'],shown,False)
            class_correct=sum(r['truth'][gi]['class_id']==shown[pi]['class_id'] for gi,pi,_ in spatial)
            tp=len(correct);fp=len(shown)-tp;fn=len(r['truth'])-tp
            d={'model':key,'filename':r['filename'],'GT':len(r['truth']),'displayed_boxes':len(shown),'TP':tp,'FP':fp,'FN':fn,'exact_correct':fp==0 and fn==0,'geometry_matched':len(spatial),'geometry_class_correct':class_correct,'no_boxes':not shown,'spatial_matches':[{'truth_index':gi,'candidate_index':shown[pi]['candidate_index'],'iou':ov,'truth_class':r['truth'][gi]['class_name'],'prediction_class':shown[pi]['class_name']} for gi,pi,ov in spatial]}
            for k in ['TP','FP','FN']:metrics[k]+=d[k]
            metrics['exact_correct_images']+=d['exact_correct'];metrics['no_box_images']+=d['no_boxes'];metrics['geometry_matched_boxes']+=len(spatial);metrics['geometry_matched_class_correct']+=class_correct;metrics['class_disagreement_images']+=class_correct<len(spatial)
            detail.append(d)
        save(ROOT/f'coco_predictions/{key}.json',records)
        with contextlib.redirect_stdout(io.StringIO()) as buffer:
            c=COCO();c.dataset=gt;c.createIndex();dt=c.loadRes(records);ev=COCOeval(c,dt,'bbox');ev.evaluate();ev.accumulate();ev.summarize()
        (ROOT/f'{key}_coco_log.txt').write_text(buffer.getvalue(),encoding='utf-8')
        precision=ev.eval['precision'];ap=[]
        for cid,name in enumerate(NAMES):
            a=precision[:,:,cid,0,-1];a50=precision[0,:,cid,0,-1]
            ap.append({'model':key,'class_name':name,'GT':sum(g['category_id']==cid+1 for g in gt['annotations']),'AP50':float(a50[a50>=0].mean()),'AP50_95':float(a[a>=0].mean())})
        assert abs(np.mean([a['AP50'] for a in ap])-ev.stats[1])<1e-10
        tp,fp,fn=metrics['TP'],metrics['FP'],metrics['FN'];assert tp+fn==204
        metrics.update({'model':label,'mAP50':float(ev.stats[1]),'mAP50_95':float(ev.stats[0]),'AP75':float(ev.stats[2]),'low_score_candidates':len(records),'Precision':tp/(tp+fp) if tp+fp else 0,'Recall':tp/(tp+fn),'F1':2*tp/(2*tp+fp+fn),'image_exact_accuracy':metrics['exact_correct_images']/172,'matched_box_classification_accuracy':metrics['geometry_matched_class_correct']/metrics['geometry_matched_boxes'],'images':172,'GT':204,'per_class':ap})
        save(ROOT/f'{key}_image_metrics.json',detail);all_images.extend(detail);perclass.extend(ap);results[key]=metrics
        print(key,json.dumps({k:v for k,v in metrics.items() if k!='per_class'},ensure_ascii=False),flush=True)
    old=read(RF_MAP/'最终八类端到端mAP.json')['primary']
    assert abs(results['gen4']['mAP50']-old['mAP50'])<1e-10 and abs(results['gen4']['mAP50_95']-old['mAP50_95'])<1e-10
    assert (results['gen4']['TP'],results['gen4']['FP'],results['gen4']['FN'])==(191,5,13)
    save(ROOT/'统一评估结果.json',{'rules':read(ROOT/'rules.json'),'provenance':inputs['provenance'],'results':results,'fourth_generation_historical_AP_reproduced':True})
    for name,data in [('总体指标.csv',[{k:v for k,v in r.items() if k!='per_class'} for r in results.values()]),('各类别AP.csv',perclass),('逐图指标.csv',[{k:v for k,v in r.items() if k!='spatial_matches'} for r in all_images])]:
        with (ROOT/name).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    lines=['# 四代模型统一评估','', '同一固定验证集：172张图，204个标注鸟实例，八类。未训练、未调参、未修改XML和网站。','', '|方案|八类mAP50|mAP50:95|整图完全正确率|已匹配框分类准确率|端到端F1|无框图|','|---|---:|---:|---:|---:|---:|---:|']
    for r in results.values():lines.append(f"|{r['model']}|{r['mAP50']:.2%}|{r['mAP50_95']:.2%}|{r['exact_correct_images']}/172 = {r['image_exact_accuracy']:.2%}|{r['geometry_matched_class_correct']}/{r['geometry_matched_boxes']} = {r['matched_box_classification_accuracy']:.2%}|{r['F1']:.2%}|{r['no_box_images']}|")
    lines+=['','## 指标解释','', 'mAP与第四代历史主口径一致：原生低分候选下限0.001，原分数优先IoU≥0.5额外去重，每框只保留一个物种。第一代直接使用其联合检测类别分数；后面三代用检测分×分类top-1分排序。COCOeval以IoU0.5、101点插值和maxDets=100计算八类平均。没有先以0.30或0.50删掉低分框。','', '整图完全正确率：检测分数严格大于0.30后，按类别与IoU≥0.5一对一匹配，全部标注目标都匹配且无多余预测框的图片数/172。漏鸟、多框、分错类别或定位不足均会导致该图不计为完全正确。','', '已匹配框分类准确率：先忽略物种按检测分数与IoU≥0.5进行一对一几何匹配，再计算物种正确数/几何匹配数。没有检出的鸟不进入此分母，因此必须结合无框图、召回和F1理解。','', '这些都是与现有XML的一致性指标。固定验证集已经反复用于开发，历史训练清单与预算不完全一致，不能据此宣称独立测试准确率或公平的同预算算法优劣。','', '第二代两个分类器分别报告，没有将其误写为投票融合。原PPT的历史AP与本轮统一后的结果可能不同，因为本轮采用冻结最佳分类权重、额外去重和统一评估口径。第四代历史mAP已经逐位复现。']
    (ROOT/'统一指标说明.md').write_text('\n'.join(lines),encoding='utf-8')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('role',choices=['prepare','cnn','bio','evaluate']);a=ap.parse_args()
    {'prepare':prepare,'cnn':classify_cnn,'bio':classify_bio,'evaluate':evaluate}[a.role]()
