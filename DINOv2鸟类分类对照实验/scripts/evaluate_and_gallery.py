"""在固定检测框上重放BioCLIP 2并运行DINOv2，生成评估表、图册及外来80图对照。"""
from __future__ import annotations
import csv, hashlib, html, json, math, os, shutil, sys, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'configs'/'experiment.json').read_text(encoding='utf-8'))
GEN3=Path(CFG['gen3_project']); ARCHIVE=Path(CFG['gen2_archive']); EXT=Path(CFG['external_80_report'])
WEIGHTS=ROOT/CFG['weights_relative_path']; DINO_SRC=ROOT/'third_party'/'dinov2'/'dinov2-7764ea0f912e53c92e82eb78a2a1631e92725fc8'
for k,rel in [('TORCH_HOME','cache/torch'),('HF_HOME','cache/huggingface'),('MPLCONFIGDIR','cache/matplotlib'),('TEMP','cache/tmp'),('TMP','cache/tmp')]:
 p=ROOT/rel;p.mkdir(parents=True,exist_ok=True);os.environ[k]=str(p)

def sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def loadjson(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def box4(v):
 if isinstance(v,str):
  try: return [float(x) for x in v.replace(',',' ').split()[:4]]
  except Exception: pass
 return [float(x) for x in v[:4]]

def class_scores(logits):
 import torch
 return torch.softmax(logits.detach().float(),dim=1).cpu().numpy()

def build_dino():
 import torch
 sys.path.insert(0,str(DINO_SRC))
 from dinov2.hub.backbones import dinov2_vitb14
 model=dinov2_vitb14(pretrained=False)
 model.load_state_dict(torch.load(WEIGHTS,map_location='cpu',weights_only=True),strict=True)
 model.eval().cuda()
 for p in model.parameters():p.requires_grad_(False)
 head=torch.nn.Linear(768,8).cuda()
 head.load_state_dict(torch.load(ROOT/'runs'/'dinov2'/'best_head.pt',map_location='cuda',weights_only=True))
 head.eval()
 return model,head

def dino_batch(images,model,head,torch):
 import torch.nn.functional as F
 from PIL import Image
 mean=np.array([.485,.456,.406],np.float32).reshape(1,1,3);std=np.array([.229,.224,.225],np.float32).reshape(1,1,3);arr=[]
 for im in images:
  im=im.convert('RGB');side=max(im.size);sq=Image.new('RGB',(side,side),(114,114,114));sq.paste(im,((side-im.width)//2,(side-im.height)//2));sq=sq.resize((224,224),Image.Resampling.BICUBIC)
  arr.append(((np.asarray(sq,dtype=np.float32)/255-mean)/std).transpose(2,0,1))
 x=torch.from_numpy(np.stack(arr)).cuda(non_blocking=True)
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16): z=model.forward_features(x)['x_norm_clstoken']
 z=F.normalize(z.float(),dim=1);return z,head(z)

def bio_model():
 import torch,open_clip
 p=GEN3/'weights'/'bioclip2'/'open_clip_model.safetensors';cfg=loadjson(p.parent/'open_clip_config.json')
 model,_,transform=open_clip.create_model_and_transforms('ViT-L-14',pretrained=str(p),image_mean=cfg['preprocess_cfg']['mean'],image_std=cfg['preprocess_cfg']['std'],image_interpolation='bicubic')
 visual=model.visual.cuda().eval();head=torch.nn.Linear(768,8).cuda();head.load_state_dict(torch.load(GEN3/'runs'/'bioclip2'/'best_head.pt',map_location='cuda',weights_only=True));head.eval()
 return visual,transform,head

def crop_image(path,bbox,margin=.15):
 from PIL import Image
 x1,y1,x2,y2=box4(bbox);im=Image.open(path).convert('RGB');dx=(x2-x1)*margin;dy=(y2-y1)*margin
 return im.crop((max(0,math.floor(x1-dx)),max(0,math.floor(y1-dy)),min(im.width,math.ceil(x2+dx)),min(im.height,math.ceil(y2+dy))))

def make_candidate_tasks():
 fixed=loadjson(GEN3/'reports'/'bioclip2'/'固定验证集结果.json');tasks=[];fixed_links=[]
 for ri,row in enumerate(fixed['rows']):
  for pi,p in enumerate(row['models']['bioclip2']['displayed']):
   task={'kind':'fixed_val','row':ri,'pred':pi,'image_path':row['image_path'],'bbox':box4(p['bbox_xyxy'])}
   fixed_links.append(task);tasks.append(task)
 ext_manifest=loadjson(EXT/'manifest.json');ext_result=loadjson(EXT/'完整测试结果.json');ext_byid={r['id']:r for r in ext_result['images']};ext_links=[];candidate_mismatches=[]
 for m in ext_manifest:
  result=ext_byid[m['id']];prop=loadjson(Path(m['directory'])/'gen2'/'proposals.json');job=loadjson(Path(m['directory'])/'gen2'/'job.json')
  assert prop['weights']==str(ARCHIVE/'runs'/'detector'/'best_model_ap50.pdparams'),(m['id'],prop.get('weights'))
  assert float(job['threshold'])==.3 and job['model']=='gen2'
  bio=result['results']['bioclip2'];shown=prop['displayed']
  if len(bio)!=len(shown):candidate_mismatches.append({'id':m['id'],'reason':'box_count','stored_bioclip':len(bio),'formal_detector':len(shown)})
  for i,(a,b) in enumerate(zip(bio,shown)):
   d=max(abs(x-y) for x,y in zip(box4(a['bbox_xyxy']),box4(b['bbox_xyxy'])))
   if d>1e-4 or abs(float(a['detector_score'])-float(b['detector_score']))>1e-6:candidate_mismatches.append({'id':m['id'],'candidate':i,'coordinate_max_abs_diff':d})
  for pi,p in enumerate(shown):
   task={'kind':'external80','id':m['id'],'row':m,'result':result,'pred':pi,'image_path':m['image'],'bbox':box4(p['bbox_xyxy'])}
   ext_links.append(task);tasks.append(task)
 return fixed,tasks,fixed_links,ext_manifest,ext_byid,ext_links,candidate_mismatches

def infer_dino(tasks,batch_size=8):
 import torch
 from PIL import Image
 model,head=build_dino();out=[];start=time.perf_counter();peak=0
 for i in range(0,len(tasks),batch_size):
  group=tasks[i:i+batch_size];ims=[crop_image(t['image_path'],t['bbox']) for t in group]
  z,logits=dino_batch(ims,model,head,torch);scores=class_scores(logits)
  for task,s in zip(group,scores):out.append(s.astype(float).tolist())
  peak=max(peak,torch.cuda.max_memory_allocated())
  if i%64==0: print(json.dumps({'phase':'candidate_inference','done':min(i+len(group),len(tasks)),'total':len(tasks),'seconds':time.perf_counter()-start,'peak_cuda_gib':round(peak/1024**3,3)}),flush=True)
 del model,head;torch.cuda.empty_cache()
 return out,{'count':len(out),'seconds':time.perf_counter()-start,'peak_cuda_gib':peak/1024**3}

def infer_bio(tasks,batch_size=8):
 import torch,torch.nn.functional as F
 visual,transform,head=bio_model();scores=[];start=time.perf_counter();peak=0
 for i in range(0,len(tasks),batch_size):
  ims=[crop_image(t['image_path'],t['bbox']) for t in tasks[i:i+batch_size]]
  x=torch.stack([transform(im) for im in ims]).cuda(non_blocking=True)
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):z=visual(x)
  z=F.normalize(z.float(),dim=1);probs=class_scores(head(z));scores.extend(probs.astype(float).tolist());peak=max(peak,torch.cuda.max_memory_allocated())
  if i%64==0: print(json.dumps({'phase':'bioclip_replay','done':min(i+len(ims),len(tasks)),'total':len(tasks),'seconds':time.perf_counter()-start,'peak_cuda_gib':round(peak/1024**3,3)}),flush=True)
 del visual,head;torch.cuda.empty_cache()
 return scores,{'count':len(scores),'seconds':time.perf_counter()-start,'peak_cuda_gib':peak/1024**3}

def crop_classification_metrics(y,p,names):
 k=len(names);cm=[[0]*k for _ in range(k)]
 for a,b in zip(y,p):cm[int(a)][int(b)]+=1
 per=[]
 for i,n in enumerate(names):
  tp=cm[i][i];fp=sum(cm[r][i] for r in range(k) if r!=i);fn=sum(cm[i][c] for c in range(k) if c!=i)
  pr=tp/(tp+fp) if tp+fp else 0.;re=tp/(tp+fn) if tp+fn else 0.
  per.append({'class_id':i,'class_name':n,'support':sum(cm[i]),'precision':pr,'recall':re,'f1':2*pr*re/(pr+re) if pr+re else 0.})
 return {'instances':len(y),'accuracy':sum(int(a==b) for a,b in zip(y,p))/len(y),'macro_f1':float(np.mean([x['f1'] for x in per])),'per_class':per,'confusion_matrix':cm}

def iou(a,b):
 x=max(0,min(a[2],b[2])-max(a[0],b[0]));y=max(0,min(a[3],b[3])-max(a[1],b[1]));inter=x*y
 aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]);bb=max(0,b[2]-b[0])*max(0,b[3]-b[1]);return inter/(aa+bb-inter) if aa+bb>inter else 0.

def endpoint_metrics(rows,preds,names):
 k=len(names);loc_matches=[];unmatched_pred=[];unmatched_gt=[];counts=[{'tp':0,'fp':0,'fn':0} for _ in names];nbox=0;nogt=0
 cm=[[0]*(k+1) for _ in range(k+1)]
 image_rows=[];box_rows=[];cls_disagree=[]
 for row,ps in zip(rows,preds):
  truths=row['truth'];ps=sorted(ps,key=lambda p:-p['detector_score']);nbox+=len(ps);nogt+=len(truths)
  used=set();m=[];up=[]
  for p in ps:
   choices=[(iou(p['bbox_xyxy'],g['bbox_xyxy']),gi) for gi,g in enumerate(truths) if gi not in used]
   ov,gi=max(choices,default=(0.,-1))
   if ov>=.5:
    used.add(gi);g=truths[gi];m.append((p,gi,ov))
    if p['class_id']==g['class_id']:
     counts[g['class_id']]['tp']+=1;cm[g['class_id']][p['class_id']]+=1
    else:
     counts[p['class_id']]['fp']+=1;counts[g['class_id']]['fn']+=1;cm[g['class_id']][p['class_id']]+=1
     cls_disagree.append({'filename':row['filename'],'truth_class':g['class_name'],'predicted_class':p['class_name'],'iou':ov,'bbox_xyxy':p['bbox_xyxy']})
   else:
    up.append(p);unmatched_pred.append({'filename':row['filename'],**p});counts[p['class_id']]['fp']+=1;cm[k][p['class_id']]+=1
  for gi,g in enumerate(truths):
   if gi not in used:
    counts[g['class_id']]['fn']+=1;cm[g['class_id']][k]+=1;unmatched_gt.append({'filename':row['filename'],**g})
  image_rows.append({'filename':row['filename'],'expected':row.get('expected'),'ground_truths':truths,'predictions':ps,'matches':[{'candidate_index':p.get('candidate_index'),'bbox_xyxy':p['bbox_xyxy'],'truth_index':gi,'iou':ov,'class_match':p['class_id']==truths[gi]['class_id']} for p,gi,ov in m],'unmatched_prediction_records':up,'unmatched_truth_records':[g for gi,g in enumerate(truths) if gi not in used],'correct_matches':sum(p['class_id']==truths[gi]['class_id'] for p,gi,_ in m),'spatial_matches':len(m),'wrong_class_spatial_matches':sum(p['class_id']!=truths[gi]['class_id'] for p,gi,_ in m),'unmatched_display_boxes':len(up),'unmatched_truths':len(truths)-len(used),'no_display':len(ps)==0,'class_disagreement':any(p['class_id']!=truths[gi]['class_id'] for p,gi,_ in m)})
  for p in ps:box_rows.append({'filename':row['filename'],**p})
 tp=sum(c['tp'] for c in counts);fp=sum(c['fp'] for c in counts);fn=sum(c['fn'] for c in counts)
 # 错分类位置匹配按预测类计FP、真值类计FN；位置未匹配框计预测类FP。
 prec=tp/(tp+fp) if tp+fp else 0.;rec=tp/(tp+fn) if tp+fn else 0.
 per=[]
 for i,n in enumerate(names):
  c=counts[i];pr=c['tp']/(c['tp']+c['fp']) if c['tp']+c['fp'] else 0.;re=c['tp']/(c['tp']+c['fn']) if c['tp']+c['fn'] else 0.
  per.append({'class_id':i,'class_name':n,'TP':c['tp'],'FP':c['fp'],'FN':c['fn'],'precision':pr,'recall':re,'F1':2*pr*re/(pr+re) if pr+re else 0.})
 summ={'images':len(rows),'GT_instances':nogt,'displayed_boxes':nbox,'TP':tp,'FP':fp,'FN':fn,'precision':prec,'recall':rec,'F1':2*prec*rec/(prec+rec) if prec+rec else 0.,'unmatched_displayed_boxes':len(unmatched_pred),'unmatched_truths':len(unmatched_gt),'no_display_images':sum(len(p)==0 for p in preds),'class_disagreement_images':len({x['filename'] for x in cls_disagree}),'spatially_matched_wrong_class_instances':len(cls_disagree),'per_class':per}
 return {'summary':summ,'per_image':image_rows,'per_box':box_rows,'unmatched_predictions':unmatched_pred,'unmatched_truths':unmatched_gt,'class_disagreements':cls_disagree,'confusion_matrix_with_missed_column':{'row_labels':names+['未匹配显示框'],'column_labels':names+['漏检'],'matrix':cm}}

def external_summary(records,names):
 def summary(sub):
  nonempty=[r for r in sub if r['bio_predictions'] or r['dino_predictions']]
  return {'images':len(sub),'bio_nonempty':sum(bool(r['bio_predictions']) for r in sub),'dino_nonempty':sum(bool(r['dino_predictions']) for r in sub),
   'bio_expected_category_agree_images':sum(bool(r['bio_predictions']) and all(p['class_name']==r['expected'] for p in r['bio_predictions']) for r in sub),
   'dino_expected_category_agree_images':sum(bool(r['dino_predictions']) and all(p['class_name']==r['expected'] for p in r['dino_predictions']) for r in sub),
   'bio_expected_category_differ_images':sum(bool(r['bio_predictions']) and any(p['class_name']!=r['expected'] for p in r['bio_predictions']) for r in sub),
   'dino_expected_category_differ_images':sum(bool(r['dino_predictions']) and any(p['class_name']!=r['expected'] for p in r['dino_predictions']) for r in sub),
   'both_have_no_box':sum(not r['bio_predictions'] and not r['dino_predictions'] for r in sub),
   'classifier_disagreement_images':sum(bool(r['bio_predictions']) and bool(r['dino_predictions']) and [p['class_id'] for p in r['bio_predictions']] != [p['class_id'] for p in r['dino_predictions']] for r in sub),
   'expected_category_agreement_among_nonempty_bio':sum(bool(r['bio_predictions']) and all(p['class_name']==r['expected'] for p in r['bio_predictions']) for r in sub)/max(1,sum(bool(r['bio_predictions']) for r in sub)),
   'expected_category_agreement_among_nonempty_dino':sum(bool(r['dino_predictions']) and all(p['class_name']==r['expected'] for p in r['dino_predictions']) for r in sub)/max(1,sum(bool(r['dino_predictions']) for r in sub))}
 return {'all_80':summary(records),'excluding_9_same_source':summary([r for r in records if not r['same_photo']]),'excluding_9_same_source_and_related_scene':summary([r for r in records if not r['same_photo'] and not r['related_scene']]),'metric_caveat':'目录名仅作为事后预期类别；无逐实例人工真值；一致数不是识别准确率，也不计算检测P/R/AP。无框图片不计类别一致。'}

def csv_write(path,rows,fields):
 with path.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def render_html(title,desc,records,path,kind):
 # HTML依赖同目录下相对图像资产；不依赖在线JS、CSS或目录标签做推理。
 data=json.dumps(records,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
 if kind=='fixed': filters=['all','类别分歧','无显示框','未匹配XML框','定位/分类退步']
 else: filters=['all','模型分歧','无框','同源回放','相关场景','麻雀三类']
 body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>body{{font:14px "Microsoft YaHei",sans-serif;margin:0;background:#f3f6f9;color:#1c2b39}}header{{position:sticky;top:0;z-index:3;background:#17324d;color:white;padding:14px 22px;box-shadow:0 2px 6px #aaa}}h1{{margin:0 0 5px;font-size:21px}}.note{{font-size:12px;line-height:1.6;color:#dbe6f1}}.toolbar{{display:flex;gap:10px;flex-wrap:wrap;margin-top:10px}}input,select,button{{padding:8px;border:1px solid #b7c6d4;border-radius:5px}}main{{max-width:1800px;margin:16px auto;padding:0 14px}}article{{background:white;border:1px solid #d6e0e9;border-radius:9px;padding:12px;margin-bottom:14px}}article h2{{margin:0 0 9px;font-size:17px}}.badges{{display:flex;gap:6px;flex-wrap:wrap;margin:6px 0}}.badge{{padding:3px 7px;border-radius:10px;background:#e9eff4;font-size:11px}}.grid{{display:grid;grid-template-columns:repeat(3,minmax(220px,1fr));gap:9px}}.panel{{border:1px solid #d6e0e9;padding:7px;border-radius:6px;min-width:0}}.panel b{{display:block;margin-bottom:5px}}canvas{{display:block;width:100%;height:auto;background:#eceff2}}.small{{font-size:12px;line-height:1.5;color:#394d60}}details{{margin-top:8px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:11px/1.45 Consolas,monospace;background:#f6f8fa;padding:8px;max-height:320px;overflow:auto}}.warning{{color:#8a4b00;background:#fff7e5;padding:8px;border-radius:5px;margin:8px 0}}[hidden]{{display:none!important}}@media(max-width:850px){{.grid{{grid-template-columns:1fr}}header{{position:static}}}}</style>
<header><h1>{html.escape(title)}</h1><div class="note">{html.escape(desc)}</div><div class="toolbar"><input id="q" placeholder="搜索文件名或类别"><select id="filter">{''.join(f'<option>{html.escape(v)}</option>' for v in filters)}</select><label><input type="checkbox" id="xml" checked>显示XML框</label><label><input type="checkbox" id="bio" checked>显示BioCLIP 2框</label><label><input type="checkbox" id="dino" checked>显示DINOv2框</label></div></header>
<main id="items"></main><script id="data" type="application/json">{data}</script><script>
const DATA=JSON.parse(document.querySelector('#data').textContent),labels=DATA.length?DATA[0].class_names:[],esc=s=>String(s??'').replace(/[&<>\"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}}[c]));
function draw(cv,item,preds,color,showTruth){{const im=new Image();im.onload=()=>{{cv.width=im.naturalWidth;cv.height=im.naturalHeight;const c=cv.getContext('2d');c.drawImage(im,0,0);const sx=cv.width/item.width,sy=cv.height/item.height;if(showTruth&&document.querySelector('#xml').checked)for(const g of item.truth||[])rect(c,g.bbox_xyxy,'#00a35a','XML '+g.class_name,sx,sy);if(preds)for(const p of preds)rect(c,p.bbox_xyxy,color,p.class_name+' D'+Number(p.detector_score).toFixed(2)+' C'+Number(p.classifier_top_score).toFixed(2),sx,sy)}};im.src=item.asset}}
function rect(c,b,color,text,sx,sy){{c.strokeStyle=color;c.lineWidth=Math.max(2,Math.min(sx,sy)*3);c.strokeRect(b[0]*sx,b[1]*sy,(b[2]-b[0])*sx,(b[3]-b[1])*sy);c.font='bold 13px sans-serif';c.fillStyle=color;c.fillText(text,Math.max(2,b[0]*sx),Math.max(14,b[1]*sy-3))}}
function details(row){{const d=[];for(const [title,list] of [['BioCLIP 2',row.bio_predictions||[]],['DINOv2',row.dino_predictions||[]]])for(const p of list)d.push(title+' | '+p.class_name+' | detector='+Number(p.detector_score).toFixed(5)+' | top='+Number(p.classifier_top_score).toFixed(5)+' | box='+JSON.stringify(p.bbox_xyxy)+' | scores='+JSON.stringify(p.classifier_scores));return d.join('\\n')}}
function include(r,f,q){{const txt=(r.filename+' '+(r.expected||'')+' '+JSON.stringify(r)).toLowerCase();if(q&&!txt.includes(q.toLowerCase()))return false;if(f==='all')return true;if(f==='类别分歧')return !!r.class_disagreement;if(f==='无显示框')return !!r.no_display;if(f==='未匹配XML框')return (r.unmatched_display_boxes||0)>0;if(f==='定位/分类退步')return !!r.regression;if(f==='模型分歧')return !!r.model_disagreement;if(f==='无框')return !!r.both_empty;if(f==='同源回放')return !!r.same_photo;if(f==='相关场景')return !!r.related_scene;if(f==='麻雀三类')return ['家麻雀','树麻雀','山麻雀'].includes(r.expected);return true}}
function render(){{const root=document.querySelector('#items'),q=document.querySelector('#q').value.trim(),f=document.querySelector('#filter').value;root.innerHTML='';for(const r of DATA){{if(!include(r,f,q))continue;const a=document.createElement('article');const badges=(r.badges||[]).map(x=>'<span class="badge">'+esc(x)+'</span>').join('');const warning=r.caveat?'<div class="warning">'+esc(r.caveat)+'</div>':'';let panels=`<div class="grid"><div class="panel"><b>原图（无预测框）</b><canvas data-view="raw"></canvas></div><div class="panel"><b>BioCLIP 2 · ${{(r.bio_predictions||[]).length}}框</b><canvas data-view="bio"></canvas></div><div class="panel"><b>DINOv2 · ${{(r.dino_predictions||[]).length}}框</b><canvas data-view="dino"></canvas></div></div>`;a.innerHTML='<h2>'+esc(r.filename)+' · 预期类别（事后）：'+esc(r.expected||'固定验证集')+'</h2><div class="badges">'+badges+'</div>'+warning+panels+'<details><summary>查看逐框八类分数、框坐标与匹配信息</summary><pre>'+esc(details(r))+'</pre><pre>'+esc(JSON.stringify({{truth:r.truth||[],bio:r.bio_predictions||[],dino:r.dino_predictions||[],matches:r.matches||[],unmatched:r.unmatched||[]}},null,2))+'</pre></details>';root.appendChild(a);draw(a.querySelector('[data-view=raw]'),r,null,'',false);draw(a.querySelector('[data-view=bio]'),r,document.querySelector('#bio').checked?r.bio_predictions:[],'#1670d4',true);draw(a.querySelector('[data-view=dino]'),r,document.querySelector('#dino').checked?r.dino_predictions:[],'#e33a35',true)}}}}
['q','filter','xml','bio','dino'].forEach(id=>document.querySelector('#'+id).addEventListener(id==='q'?'input':'change',render));render();
</script></html>'''
 path.write_text(body,encoding='utf-8')

def main():
 import torch,torch.nn.functional as F
 assert torch.cuda.is_available()
 names=[c['name'] for c in loadjson(ROOT/'data'/'class_mapping.json')['classes']]
 fixed,tasks,fixed_links,ext_manifest,ext_byid,ext_links,candidate_mismatches=make_candidate_tasks()
 assert len(fixed['rows'])==172 and len(fixed_links)==177
 # BioCLIP2正式结果可复用的前提：固定报告记录的训练基线SHA与官方冻结编码器缓存元数据一致。
 bio_report_meta=fixed['metadata']
 baseline_path=ARCHIVE.parent/'第二代实验报告'/'最终与最佳分类权重对照'/'最终与最佳权重比较结果.json'
 assert sha(baseline_path)==bio_report_meta['baseline_sha256'], 'BioCLIP候选源基线哈希不一致'
 # 固定集直接复用正式BioCLIP结果（缓存SHA及逐框候选来源验证通过）；外来图复用相同正式权重下的已保存结果。
 # 外来候选数、坐标或检测分数若无法与PP-YOLOE+ proposals对应，停止而不拼接不公平结果。
 if candidate_mismatches: raise RuntimeError('外来80图的已保存BioCLIP结果与正式detector proposals不一致；需在同一候选重新推理：'+json.dumps(candidate_mismatches[:8],ensure_ascii=False))
 oldbio=[]
 for link in fixed_links: oldbio.append(fixed['rows'][link['row']]['models']['bioclip2']['displayed'][link['pred']])
 dino_scores,dino_time=infer_dino(tasks)
 dino_fixed=[];bio_fixed=[];offset=0
 for ix,link in enumerate(fixed_links):
  row=fixed['rows'][link['row']];old=row['models']['bioclip2']['displayed'][link['pred']];bb=link['bbox']
  dprob=dino_scores[ix];ci=int(np.argmax(dprob))
  dino_fixed.append({'candidate_index':int(old.get('candidate_index',link['pred'])),'bbox_xyxy':bb,'detector_score':float(old['detector_score']),'class_id':ci,'class_name':names[ci],'classifier_top_score':float(dprob[ci]),'classifier_scores':dict(zip(names,map(float,dprob))),'final_score_product':float(old['detector_score']*dprob[ci])})
  bio_fixed.append(dict(old))
 bio_byrow=[[] for _ in fixed['rows']];dino_byrow=[[] for _ in fixed['rows']]
 for link,b,d in zip(fixed_links,bio_fixed,dino_fixed):bio_byrow[link['row']].append(b);dino_byrow[link['row']].append(d)

 # 纯裁剪分类验证：BioCLIP沿用正式缓存/分类头，DINOv2使用同一146张无待复核标注裁剪。
 with (ROOT/'data'/'val_crops.csv').open(encoding='utf-8-sig',newline='') as f: valc=list(csv.DictReader(f))
 train_count=1594;bio_cache=np.load(GEN3/'runs'/'bioclip2'/'features.npz');bio_feat=torch.from_numpy(bio_cache['features'][train_count:train_count+len(valc)]).cuda()
 biohead=torch.nn.Linear(bio_feat.shape[1],8).cuda();biohead.load_state_dict(torch.load(GEN3/'runs'/'bioclip2'/'best_head.pt',map_location='cuda',weights_only=True));biohead.eval()
 with torch.inference_mode():bio_p=class_scores(biohead(bio_feat))
 dino_train=np.load(ROOT/'runs'/'dinov2'/'features_train_val.npz')['features'];dino_val=torch.from_numpy(dino_train[len(load_rows('train_crops.csv')):]).cuda()
 dinohead=torch.nn.Linear(768,8).cuda();dinohead.load_state_dict(torch.load(ROOT/'runs'/'dinov2'/'best_head.pt',map_location='cuda',weights_only=True));dinohead.eval()
 with torch.inference_mode():dino_p=class_scores(dinohead(dino_val))
 y=[int(r['class_id']) for r in valc];pure={'validation_crops':len(valc),'source_images':len({r['source_image'] for r in valc}),'bioCLIP2':crop_classification_metrics(y,np.argmax(bio_p,1),names),'DINOv2':crop_classification_metrics(y,np.argmax(dino_p,1),names),'rule':'same fixed 146 validated manually annotated 15%-expanded crops; excludes 58 review-pending validation crops; BioCLIP uses its frozen encoder/cache and existing best head; DINOv2 uses current frozen encoder and newly trained linear head'}
 (ROOT/'reports'/'pure_crop_classification.json').write_text(json.dumps(pure,ensure_ascii=False,indent=2),encoding='utf-8')
 pure_rows=[]
 for i,row in enumerate(valc):pure_rows.append({'crop_path':row['crop_path'],'source_image':row['source_image'],'class_name':row['class_name'],'class_id':int(row['class_id']),'bio_pred':names[int(np.argmax(bio_p[i]))],'bio_scores':json.dumps(dict(zip(names,map(float,bio_p[i]))),ensure_ascii=False),'dino_pred':names[int(np.argmax(dino_p[i]))],'dino_scores':json.dumps(dict(zip(names,map(float,dino_p[i]))),ensure_ascii=False)})
 csv_write(ROOT/'reports'/'pure_crop_classification.csv',pure_rows,['crop_path','source_image','class_name','class_id','bio_pred','bio_scores','dino_pred','dino_scores'])

 # 固定172张端到端结果：检测器候选组不变，只替换分类分数。XML只在预测结束后用于评估。
 bio_eval=endpoint_metrics(fixed['rows'],bio_byrow,names);dino_eval=endpoint_metrics(fixed['rows'],dino_byrow,names)
 # 复核Bio重放实现与正式历史结果的计数是否一致。
 for key in ['TP','FP','FN','no_display_images']:
  historical=fixed['summary'].get(key) if key in fixed.get('summary',{}) else None
  if historical is not None and bio_eval['summary'][key]!=historical: raise RuntimeError(f'BioCLIP重放指标不一致 {key}: {bio_eval["summary"][key]} vs {historical}')
 fixed_out=[];asset_dir=ROOT/'reports'/'assets'/'fixed_val';asset_dir.mkdir(parents=True,exist_ok=True)
 for ri,(row,bb,dd,irb,ird) in enumerate(zip(fixed['rows'],bio_byrow,dino_byrow,bio_eval['per_image'],dino_eval['per_image'])):
  src=Path(row['image_path']);asset=asset_dir/f'{ri:03d}_{src.name}';
  if not asset.exists():shutil.copy2(src,asset)
  truth=row['truth'];bad=ird['class_disagreement'];loc_bad=ird['unmatched_display_boxes']>0 or ird['unmatched_truths']>0
  fixed_out.append({'class_names':names,'filename':row['filename'],'expected':row.get('expected'),'width':row['width'],'height':row['height'],'asset':'assets/fixed_val/'+asset.name,'truth':truth,'bio_predictions':bb,'dino_predictions':dd,'class_disagreement':bad,'no_display':ird['no_display'],'unmatched_display_boxes':ird['unmatched_display_boxes'],'regression':loc_bad and (ird['unmatched_display_boxes']>irb['unmatched_display_boxes'] or ird['unmatched_truths']>irb['unmatched_truths']) or (bad and not irb['class_disagreement']),'matches':ird,'unmatched':{'predictions':[p for p in dd if not any(iou(p['bbox_xyxy'],g['bbox_xyxy'])>=.5 for g in truth)],'truth':[g for g in truth if not any(iou(g['bbox_xyxy'],p['bbox_xyxy'])>=.5 for p in dd)]}})
 fixed_report={'metadata':{'images':172,'ground_truth_instances':204,'display_threshold_strictly_greater':.30,'iou_matching':.5,'detector_dedup':'descending detector score, greedy non-transitive IoU>=0.5 winner grouping','crop_expansion':.15,'detector_checkpoint':str(ARCHIVE/'runs'/'detector'/'best_model_ap50.pdparams'),'detector_checkpoint_sha256':sha(ARCHIVE/'runs'/'detector'/'best_model_ap50.pdparams'),'classifier_backbone_frozen':True,'teacher_test_used':False,'bioCLIP2_source':'existing official fixed-validation prediction replay; source baseline SHA matches fixed-result metadata; existing frozen feature-cache logits reproduce saved scores exactly','validation_scope':'Repeatedly used fixed validation set; not an independent test. Detection AP is unchanged by replacing only the classifier and is not recomputed.'},'BioCLIP2':bio_eval,'DINOv2':dino_eval,'candidate_replay':{'same_177_boxes':True,'bioCLIP2_existing_prediction_reused':True,'bioCLIP2_baseline_sha256':bio_report_meta['baseline_sha256'],'external_candidate_mismatch_count':len(candidate_mismatches),'external_mismatches':candidate_mismatches},'per_image':fixed_out}
 (ROOT/'reports'/'fixed_validation_end_to_end.json').write_text(json.dumps(fixed_report,ensure_ascii=False,indent=2),encoding='utf-8')
 # 输出逐图、逐框和逐类CSV，分数保留八类，框坐标为原图像素。
 csv_write(ROOT/'reports'/'fixed_val_per_class.csv',[{'model':m,**x} for m,e in [('BioCLIP2',bio_eval),('DINOv2',dino_eval)] for x in e['summary']['per_class']],['model','class_id','class_name','TP','FP','FN','precision','recall','F1'])
 imagecsv=[];boxcsv=[]
 for m,e in [('BioCLIP2',bio_eval),('DINOv2',dino_eval)]:
  for r in e['per_image']:imagecsv.append({'model':m,'filename':r['filename'],'correct_matches':r['correct_matches'],'spatial_matches':r['spatial_matches'],'wrong_class_spatial_matches':r['wrong_class_spatial_matches'],'unmatched_display_boxes':r['unmatched_display_boxes'],'unmatched_truths':r['unmatched_truths'],'no_display':r['no_display'],'class_disagreement':r['class_disagreement']})
  for p in e['per_box']:boxcsv.append({'model':m,**p,'bbox_xyxy':json.dumps(p['bbox_xyxy']),'classifier_scores':json.dumps(p['classifier_scores'],ensure_ascii=False)})
 csv_write(ROOT/'reports'/'fixed_val_per_image.csv',imagecsv,['model','filename','correct_matches','spatial_matches','wrong_class_spatial_matches','unmatched_display_boxes','unmatched_truths','no_display','class_disagreement'])
 csv_write(ROOT/'reports'/'fixed_val_per_box.csv',boxcsv,['model','filename','candidate_index','bbox_xyxy','detector_score','class_id','class_name','classifier_top_score','classifier_scores','final_score_product'])

 # 外来80图：使用现有正式PP-YOLOE+ proposals，核验后对共同框分别推理；目录类仅事后统计。
 same_doc=loadjson(EXT/'人工同源核查.json');same=set(same_doc.get('same_photo',[]));related=set(same_doc.get('related_scene',[]));ext_out=[];pos=0
 for m in ext_manifest:
  er=ext_byid[m['id']];prop=loadjson(Path(m['directory'])/'gen2'/'proposals.json');preds=[]
  old=er['results']['bioclip2'];same_count=len(old)==len(prop['displayed']);maxbox=0.0
  for pi,p in enumerate(prop['displayed']):
   dp=dino_scores[len(fixed_links)+pos];pos+=1;dpidx=int(np.argmax(dp));
   if pi<len(old):maxbox=max(maxbox,max(abs(x-y) for x,y in zip(box4(old[pi]['bbox_xyxy']),box4(p['bbox_xyxy']))))
   preds.append({'proposal':p,'bio_old':old[pi] if pi<len(old) else None,'dino':{'class_id':dpidx,'class_name':names[dpidx],'classifier_top_score':float(dp[dpidx]),'classifier_scores':dict(zip(names,map(float,dp)))}})
  src=Path(m['image']);asset=ROOT/'reports'/'assets'/'external80'/f"{m['id']}.png";asset.parent.mkdir(parents=True,exist_ok=True)
  if not asset.exists():shutil.copy2(Path(m['image']),asset)
  bio_simple=[{'bbox_xyxy':box4(p['bbox_xyxy']),'detector_score':float(p['detector_score']),'class_id':int(p['class_id']),'class_name':p['class_name'],'classifier_top_score':float(p['classifier_top_score']),'classifier_scores':p['classifier_scores']} for p in old]
  dino_simple=[{'bbox_xyxy':box4(p['proposal']['bbox_xyxy']),'detector_score':float(p['proposal']['detector_score']),'class_id':p['dino']['class_id'],'class_name':p['dino']['class_name'],'classifier_top_score':p['dino']['classifier_top_score'],'classifier_scores':p['dino']['classifier_scores']} for p in preds]
  mismatch=not same_count or maxbox>1e-4
  ext_out.append({'class_names':names,'id':m['id'],'filename':m['filename'],'expected':m['expected'],'width':ImageSize(m['image'])[0],'height':ImageSize(m['image'])[1],'asset':'assets/external80/'+asset.name,'same_photo':m['filename'] in same,'related_scene':m['filename'] in related,'bio_predictions':bio_simple,'dino_predictions':dino_simple,'model_disagreement':bool(bio_simple and dino_simple and [p['class_id'] for p in bio_simple]!=[p['class_id'] for p in dino_simple]),'both_empty':not bio_simple and not dino_simple,'candidate_match_to_previous_report':not mismatch,'candidate_match_max_box_diff':maxbox,'badges':(['与原数据同源回放'] if m['filename'] in same else [])+(['相关场景候选'] if m['filename'] in related else []),'caveat':'目录类别只用于事后对照；无逐实例人工真值。无框不计类别一致。' if not bio_simple or not dino_simple else ''})
 assert pos==len(tasks)-len(fixed_links)
 external={'metadata':{'images':len(ext_out),'shared_detection_candidates':True,'detector_checkpoint':str(ARCHIVE/'runs'/'detector'/'best_model_ap50.pdparams'),'strict_threshold':.30,'iou_dedup':.5,'crop_margin':.15,'teacher_test_used':False,'note':'所有80张仅用于事后外部回放与对照，不用于训练、权重选择、阈值选择或调参。BioCLIP 2重用已保存的同候选检测预测；DINOv2由冻结骨干和线性头预测。'},'summary':external_summary(ext_out,names),'images':ext_out,'per_class_posthoc':{n:{'images':sum(r['expected']==n for r in ext_out),'bio_expected_agreement_nonempty':sum(r['expected']==n and bool(r['bio_predictions']) and all(p['class_name']==n for p in r['bio_predictions']) for r in ext_out),'dino_expected_agreement_nonempty':sum(r['expected']==n and bool(r['dino_predictions']) and all(p['class_name']==n for p in r['dino_predictions']) for r in ext_out)} for n in names},'sparrow_family_disagreements':[r for r in ext_out if r['expected'] in {'家麻雀','树麻雀','山麻雀'} and r['model_disagreement']]}
 (ROOT/'reports'/'external80_comparison.json').write_text(json.dumps(external,ensure_ascii=False,indent=2),encoding='utf-8')
 csv_rows=[]
 for r in ext_out:
  for model,preds in [('BioCLIP2',r['bio_predictions']),('DINOv2',r['dino_predictions'])]:
   if not preds:csv_rows.append({'id':r['id'],'filename':r['filename'],'expected_posthoc':r['expected'],'model':model,'status':'无框','same_photo':r['same_photo'],'related_scene':r['related_scene']})
   for p in preds:csv_rows.append({'id':r['id'],'filename':r['filename'],'expected_posthoc':r['expected'],'model':model,'status':'目录类别相同' if p['class_name']==r['expected'] else '目录类别不同','same_photo':r['same_photo'],'related_scene':r['related_scene'],'detector_score':p['detector_score'],'class_name':p['class_name'],'classifier_top_score':p['classifier_top_score'],'bbox_xyxy':json.dumps(p['bbox_xyxy']),'classifier_scores':json.dumps(p['classifier_scores'],ensure_ascii=False)})
 csv_write(ROOT/'reports'/'external80_per_box.csv',csv_rows,['id','filename','expected_posthoc','model','status','same_photo','related_scene','detector_score','class_name','classifier_top_score','bbox_xyxy','classifier_scores'])

 # 训练曲线、总指标摘要和图册资产全部保存在本实验目录。
 import matplotlib.pyplot as plt
 logs=[json.loads(x) for x in (ROOT/'runs'/'dinov2'/'training.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
 fig,ax=plt.subplots(1,2,figsize=(10,3.6));ax[0].plot([r['epoch'] for r in logs],[r['loss'] for r in logs]);ax[0].set_title('Training loss');ax[1].plot([r['epoch'] for r in logs],[r['validation_crop_accuracy'] for r in logs]);ax[1].set_title('Validation crop accuracy');
 for a in ax:a.set_xlabel('epoch');a.grid(alpha=.25)
 fig.tight_layout();fig.savefig(ROOT/'reports'/'training_curve.png',dpi=150);plt.close(fig)
 summary={'data_audit':loadjson(ROOT/'reports'/'data_audit.json'),'training':loadjson(ROOT/'runs'/'dinov2'/'training_metadata.json'),'pure_crop_classification':pure,'fixed_validation':{'BioCLIP2':bio_eval['summary'],'DINOv2':dino_eval['summary'],'detector_AP_changed':False,'independent_test':False},'external80':external['summary'],'runtime':{'DINOv2':dino_time,'BioCLIP2':'reused frozen predictions/cache; no encoder inference rerun'},'limitations':['固定验证集曾反复用于训练选择，非独立测试。','外来80图没有逐实例人工真值，目录类别一致仅为事后预期对照。','检测器未变化，分类器替换不会提高检测AP，也无法恢复无检测框的图片。','本次DINOv2训练遵守jiamaque_087待补标隔离，使用训练裁剪比历史BioCLIP头少8条。']}
 (ROOT/'reports'/'metrics_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
 render_html('DINOv2 与 BioCLIP 2：固定172图端到端对照','同一PP-YOLOE+最佳基准检测器、同一177个IoU去重框，分数>0.30，框外扩15%。类别分歧、未匹配框仅表示与现有XML的一致性差异；验证集反复用于探索，不是独立测试。检测AP不会因替换分类器而改变。',fixed_out,ROOT/'reports'/'fixed_validation_comparison.html','fixed')
 render_html('DINOv2 与 BioCLIP 2：80张外来鸟图回放','文件夹类别仅是事后预期标签；无实例真值，不报告准确率/mAP。9张同源回放和1张相关场景单独筛选。无检测框的图不参与类别一致统计。',ext_out,ROOT/'reports'/'external80_comparison.html','external')
 index='''<!doctype html><meta charset="utf-8"><title>DINOv2鸟类分类对照</title><style>body{font:16px Microsoft YaHei;background:#f2f6fa;margin:50px;color:#19324a}main{max-width:900px;margin:auto;background:white;padding:28px;border-radius:12px}a{display:block;padding:14px;margin:12px 0;background:#eaf2fb;border-radius:8px;color:#11508c}p{line-height:1.7}</style><main><h1>PP-YOLOE+ + 冻结DINOv2 ViT-B/14 对照</h1><p>分类器对照实验；BioCLIP 2保留为网站主方案。固定验证集是反复使用的探索集，外来80图无逐实例真值。</p><a href="fixed_validation_comparison.html">固定172张验证图端到端图册</a><a href="external80_comparison.html">外来80图回放对照</a><a href="pure_crop_classification.json">人工真值框纯分类指标 JSON</a><a href="metrics_summary.json">指标与限制汇总 JSON</a><a href="training_curve.png">训练曲线</a></main>'''
 (ROOT/'reports'/'index.html').write_text(index,encoding='utf-8')
 print(json.dumps({'fixed':fixed_report['DINOv2']['summary'],'pure':{'DINOv2_accuracy':pure['DINOv2']['accuracy'],'BioCLIP2_accuracy':pure['bioCLIP2']['accuracy']},'external':external['summary'],'candidate_mismatch_count':len(candidate_mismatches),'htmls':['reports/index.html','reports/fixed_validation_comparison.html','reports/external80_comparison.html']},ensure_ascii=False,indent=2),flush=True)

def ImageSize(path):
 from PIL import Image
 with Image.open(path) as im:return im.size

def load_rows(name):
 with (ROOT/'data'/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

if __name__=='__main__':main()
