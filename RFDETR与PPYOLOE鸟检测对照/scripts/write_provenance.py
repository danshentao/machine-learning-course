from __future__ import annotations
import csv, hashlib, json
from datetime import datetime
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
from PIL import Image, ImageDraw
R=Path(__file__).resolve().parents[1]
CFG=json.loads((R/'configs/experiment.json').read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def pkg(n):
    try:return version(n)
    except PackageNotFoundError:return None
out=R/'reports';out.mkdir(exist_ok=True)
rows=list(csv.DictReader((R/'runs/formal/metrics.csv').open(encoding='utf-8-sig',newline='')))
val=[]
for x in rows:
    v=x.get('val/ema_mAP_50_95') or x.get('val/mAP_50_95')
    if v:
        try:val.append({'epoch':int(float(x['epoch']))+1,'ap50_95':float(v),'ap50':float(x.get('val/ema_mAP_50') or x.get('val/mAP_50') or 0)})
        except (ValueError,TypeError):pass
# simplify logged validation rows if duplicated
by={x['epoch']:x for x in val}; val=list(by.values())
weights={}
for name,p in [('official_pretrained',R/'weights/cache/rf-detr-small.pth'),('best',R/'weights/rfdetr_small_bird_best.pth'),('final_ema',R/'weights/rfdetr_small_bird_final_ema.pth'),('best_ema_checkpoint',R/'runs/formal/checkpoint_best_ema.pth'),('last_resume',R/'runs/formal/last.ckpt')]:
    if p.is_file():weights[name]={'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
launch=json.loads((R/'runs/formal/launcher_config.json').read_text(encoding='utf-8'))
train_cfg=json.loads((R/'runs/formal/training_config.json').read_text(encoding='utf-8')) if (R/'runs/formal/training_config.json').is_file() else {}
runtime_versions=json.loads((R/'reports/runtime_versions.json').read_text(encoding='utf-8'))
ppy_val_doc=json.loads((R/'runs/baseline_ppyoloe/fixed_val_raw.json').read_text(encoding='utf-8'))
ppy_ext_doc=json.loads((R/'runs/baseline_ppyoloe/external80_raw.json').read_text(encoding='utf-8'))
for doc in (ppy_val_doc,ppy_ext_doc):
    doc['metadata'].update({'paddle_version':runtime_versions['paddle'],'numpy_version':runtime_versions['paddle_numpy']})
    doc['metadata']['cudnn_version_query_skipped_for_cpu']=True
(R/'runs/baseline_ppyoloe/fixed_val_raw.json').write_text(json.dumps(ppy_val_doc,ensure_ascii=False),encoding='utf-8')
(R/'runs/baseline_ppyoloe/external80_raw.json').write_text(json.dumps(ppy_ext_doc,ensure_ascii=False),encoding='utf-8')
best=max(val,key=lambda x:x['ap50_95']) if val else None
nan_log=any('nan' in line.lower() for line in (R/'logs/formal_train.log').read_text(encoding='utf-8',errors='replace').splitlines())
report={
 'generated_at':datetime.now().astimezone().isoformat(),
 'official_project':'https://github.com/roboflow/rf-detr',
 'official_docs':'https://rfdetr.roboflow.com/learn/train/',
 'package':{'rfdetr':pkg('rfdetr'),'torch':pkg('torch'),'torchvision':pkg('torchvision'),'paddlepaddle':runtime_versions['paddle'],'paddle_numpy':runtime_versions['paddle_numpy']},
 'model':{'name':'RF-DETR Small','encoder':launch.get('encoder'),'input_size':launch.get('resolution'),'num_classes':1,'class':'bird','pretrained_initialization':'official COCO detector checkpoint; not DINOv2 classifier head'},
 'license_note':'Official repository distinguishes Apache-2.0 models from Plus PML-1.0; selected RF-DETR Small is distributed as Apache-2.0 per repository model table.',
 'pretrained_source':{'url':'https://storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth','official_md5_expected':'fb37061c1af7bace359c91b723a8d5c1','local_sha256':weights.get('official_pretrained',{}).get('sha256')},
 'weights':weights,
 'baseline':{'weights':CFG['deployed_detector_weights'],'weights_sha256':sha(Path(CFG['deployed_detector_weights'])),'config':CFG['deployed_detector_config'],'config_sha256':sha(Path(CFG['deployed_detector_config']))},
 'baseline_inference':{'device':'CPU due to unavailable matching cuDNN 8 in existing Paddle environment','fixed_val_seconds':ppy_val_doc['metadata']['elapsed_seconds'],'fixed_val_seconds_per_image':ppy_val_doc['metadata']['elapsed_seconds']/172,'external80_seconds':ppy_ext_doc['metadata']['elapsed_seconds'],'external80_seconds_per_image':ppy_ext_doc['metadata']['elapsed_seconds']/80,'fixed_input_shapes':ppy_val_doc['metadata']['input_shapes'],'resize':ppy_val_doc['metadata']['resize'],'cudnn_cpu_version_probe_skipped':True},
 'rfdetr_inference':{'fixed_val':json.loads((R/'runs/formal/rfdetr_fixed_val_raw.json').read_text(encoding='utf-8'))['metadata'],'external80':json.loads((R/'runs/formal/rfdetr_external80_raw.json').read_text(encoding='utf-8'))['metadata']},
 'actual_training_config':train_cfg.get('train_config',{}),
 'fixed_data':{'validation_json':CFG['fixed_validation_json'],'validation_json_sha256':sha(Path(CFG['fixed_validation_json'])),'train_json':str(R/'data/dataset/train/_annotations.coco.json'),'train_json_sha256':sha(R/'data/dataset/train/_annotations.coco.json'),'data_audit':str(out/'data_audit.json')},
 'train_args':launch.get('train_args'),'train_elapsed_seconds':launch.get('elapsed_seconds'),'train_peak_cuda_memory_gib':launch.get('peak_cuda_memory_gib'),'probe_peak_cuda_memory_gib':launch.get('probe_peak_cuda_memory_gib'),'no_nan_text_in_log':not nan_log,
 'validation_history':val,'selected_best_epoch_1_based':best['epoch'] if best else None,'selected_best_ema_map50_95':best['ap50_95'] if best else None,
 'limitations':['固定验证集已经被重复探索，不是独立测试。','部署PP-YOLOE+历史训练数据含有待补标训练样本；本轮RF训练副本排除了该样本，存在数据边界差异。','不同官方预处理与分数校准不可比，统一0.30只提供一个共同观察点；同时报告AP/PR。','外来80图无逐实例真值，不计算检测准确率或mAP。']}
(out/'model_provenance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
# 无额外绘图库依赖地绘制真实训练曲线。
W,H=1100,650; im=Image.new('RGB',(W,H),'white');d=ImageDraw.Draw(im)
d.text((30,18),'RF-DETR-S training: recorded validation mAP and training loss',fill='#183a2e')
plot=(80,70,1040,570);x0,y0,x1,y1=plot
for i in range(6):
 y=y1-(y1-y0)*i/5;d.line((x0,y,x1,y),fill='#e0e6e2');d.text((25,y-7),f'{i/5:.1f}',fill='#56635b')
d.rectangle(plot,outline='#68766e')
def line(vals,color,label,scale=(0,1)):
 if len(vals)<2:return
 a,b=scale;pts=[]
 for j,v in enumerate(vals):
  xx=x0+(x1-x0)*j/max(len(vals)-1,1); yy=y1-(y1-y0)*(v-a)/(b-a);pts.append((xx,yy))
 d.line(pts,fill=color,width=3);d.text((820,25+20*(0 if color=='#1878ba' else 1)),label,fill=color)
epoch_rows={}
for x in rows:
 ep=int(float(x['epoch']))+1
 q=epoch_rows.setdefault(ep,{})
 for key,name in [('train/loss','loss'),('val/ema_mAP_50_95','map')]:
  v=x.get(key)
  if v:
   try:q[name]=float(v)
   except ValueError:pass
hist=[epoch_rows[k] for k in sorted(epoch_rows)]
line([x['map'] for x in hist if 'map' in x],'#1878ba','EMA validation mAP50:95')
loss=[x['loss'] for x in hist if 'loss' in x]
if loss:
 mx=max(loss); mn=min(loss); n=[(v-mn)/(mx-mn or 1) for v in loss]
 line(n,'#d85b35','train loss (normalized for display)')
for k in range(0,max(len(hist),2),max(1,len(hist)//10)):
 xx=x0+(x1-x0)*k/max(len(hist)-1,1);d.text((xx-8,y1+10),str(k+1),fill='#56635b')
d.text((x0,610),'Training loss is normalized to its observed range so both series fit the same 0–1 plot.',fill='#56635b')
im.save(out/'training_curve.png')
print(json.dumps({'best_epoch':report['selected_best_epoch_1_based'],'best_ema_map50_95':report['selected_best_ema_map50_95'],'weights':weights,'curve':str(out/'training_curve.png')},ensure_ascii=False))
