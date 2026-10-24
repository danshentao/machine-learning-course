from pathlib import Path
import os, json, hashlib, csv, random

ROOT=Path(__file__).resolve().parents[1]
for k,v in {'HF_HOME':ROOT/'.cache/huggingface','HF_HUB_CACHE':ROOT/'.cache/huggingface/hub','TORCH_HOME':ROOT/'.cache/torch','MPLCONFIGDIR':ROOT/'.cache/matplotlib','TORCHINDUCTOR_CACHE_DIR':ROOT/'.cache/inductor','PIP_CACHE_DIR':ROOT/'.cache/pip','TEMP':ROOT/'.cache/tmp','TMP':ROOT/'.cache/tmp'}.items():
    Path(v).mkdir(parents=True,exist_ok=True);os.environ[k]=str(v)
os.environ['HF_HUB_DISABLE_XET']='1'
ARCHIVE=Path(json.loads((ROOT/'reports/第二代归档位置.json').read_text(encoding='utf-8'))['archive'])
GEN2=ARCHIVE/'工程'
BASELINE=ARCHIVE/'第二代实验报告/最终与最佳分类权重对照/最终与最佳权重比较结果.json'
MANIFEST=GEN2/'reports/classification_crop_manifest.csv'
NAMES=[c['name'] for c in json.loads((GEN2/'configs/class_mapping.json').read_text(encoding='utf-8'))['classes']]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
def manifest_rows():
    with MANIFEST.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    train=[r for r in rows if r['split']=='train' and r['use_for_classifier_train']=='True']
    valid=[r for r in rows if r['split']=='val' and r['review_pending']!='True']
    assert len(train)==1594 and len(valid)==146,(len(train),len(valid))
    return train,valid
def baseline():return json.loads(BASELINE.read_text(encoding='utf-8'))
def square(im):
    from PIL import Image
    w,h=im.size;s=max(w,h);out=Image.new('RGB',(s,s),(114,114,114));out.paste(im,((s-w)//2,(s-h)//2));return out
def crop(im,b,margin=.15):
    import math
    x1,y1,x2,y2=b;dx=(x2-x1)*margin;dy=(y2-y1)*margin
    return im.crop((max(0,math.floor(x1-dx)),max(0,math.floor(y1-dy)),min(im.width,math.ceil(x2+dx)),min(im.height,math.ceil(y2+dy))))
def jobs():
    # Classifiers receive exactly the same 177 grouped detector boxes, never folder hints.
    result=[]
    for ri,r in enumerate(baseline()['rows']):
        for pi,p in enumerate(r['models']['bilinear']['best']['displayed']):
            result.append({'row':ri,'prediction':pi,'image':r['image_path'],'bbox':p['bbox_xyxy']})
    assert len(result)==177,len(result)
    return result
def seed(value=123):
    import numpy as np, torch
    random.seed(value);np.random.seed(value);torch.manual_seed(value);torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.benchmark=False
def publish(model,scores,metadata):
    import copy,sys,numpy as np
    sys.path.insert(0,str(GEN2/'scripts'))
    from compare_display_thresholds import inspect
    rows=copy.deepcopy(baseline()['rows']);j=jobs();preds=[[] for _ in rows]
    assert len(scores)==len(j)
    for job,s in zip(j,scores):
        ri,pi=job['row'],job['prediction'];old=rows[ri]['models']['bilinear']['best']['displayed'][pi]
        p={k:v for k,v in old.items() if k not in ['same_class_correct_match','different_from_folder','spatial_status','matched_truth_class','matched_truth_iou']}
        top=int(np.argmax(s));p.update(class_id=top,class_name=NAMES[top],classifier_top_score=float(s[top]),classifier_scores=dict(zip(NAMES,map(float,s))),final_score_product=float(old['detector_score']*s[top]))
        preds[ri].append(p)
    for r,p in zip(rows,preds):r['models']={model:inspect(r['truth'],p,r['expected'])}
    a=[r['models'][model] for r in rows];tp=sum(x['TP'] for x in a);fp=sum(x['FP'] for x in a);fn=sum(x['FN'] for x in a)
    precision=tp/(tp+fp);recall=tp/(tp+fn)
    summary={'images':len(rows),'GT':tp+fn,'displayed_boxes':tp+fp,'TP':tp,'FP':fp,'FN':fn,'precision':precision,'recall':recall,'F1':2*precision*recall/(precision+recall),'class_disagreement_images':sum(x['class_disagreement'] for x in a),'no_display_images':sum(x['no_display'] for x in a)}
    metadata.update({'detector_display_threshold_strictly_greater':.30,'crop_margin':.15,'baseline_sha256':sha(BASELINE),'crop_manifest_sha256':sha(MANIFEST),'scope':'172 fixed validation images, 177 frozen displayed detections; no full-candidate AP computed; reused validation, not independent test'})
    write(ROOT/'reports'/model/'固定验证集结果.json',{'metadata':metadata,'summary':summary,'rows':rows});print(json.dumps({'model':model,'summary':summary},ensure_ascii=False),flush=True)
