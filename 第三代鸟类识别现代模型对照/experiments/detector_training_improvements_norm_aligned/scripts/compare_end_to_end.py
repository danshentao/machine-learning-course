"""复用冻结BioCLIP 2和固定框规则，对A/B/C做同框端到端验证比较。"""
from __future__ import annotations
import base64,copy,csv,io,json,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch,torch.nn.functional as F,open_clip

ROOT=Path(__file__).resolve().parents[1];GEN3=Path('E:/机器学习课设/第三代鸟类识别现代模型对照')
sys.path.insert(0,str(GEN3/'scripts'));from common import NAMES,seed,square,crop,sha,write
sys.path.insert(0,str(Path('E:/机器学习课设/第二代鸟类检测与细粒度分类')/'scripts'))
from build_deduplicated_gallery import group_candidates
from compare_display_thresholds import inspect
from evaluate_two_stage import match

OUT=ROOT/'reports'/'fixed_val_comparison'
def summarize(rows,key):
 xs=[r['models'][key] for r in rows];tp=sum(v['TP'] for v in xs);fp=sum(v['FP'] for v in xs);fn=sum(v['FN'] for v in xs)
 p=tp/(tp+fp) if tp+fp else 0;r=tp/(tp+fn) if tp+fn else 0
 spatial=0;spfp=spfn=0;cm=np.zeros((9,9),dtype=int)
 for row in rows:
  pred=row['models'][key]['displayed'];gt=row['truth'];ug,up,pairs=match(gt,pred,False);spatial+=len(pairs);spfp+=len(pred)-len(up);spfn+=len(gt)-len(ug)
  for gi,pi,_ in pairs:
   if pred[pi]['class_id']==gt[gi]['class_id']:cm[gt[gi]['class_id'],gt[gi]['class_id']]+=1
   else:cm[gt[gi]['class_id'],pred[pi]['class_id']]+=1
  for gi,g in enumerate(gt):
   if gi not in ug:cm[g['class_id'],8]+=1
  for pi,p0 in enumerate(pred):
   if pi not in up:cm[8,p0['class_id']]+=1
 classwise=[]
 for c,name in enumerate(NAMES):
  tp_c=fp_c=fn_c=0
  for row in rows:
   gt_c=[g for g in row['truth'] if g['class_id']==c]
   pr_c=[p0 for p0 in row['models'][key]['displayed'] if p0['class_id']==c]
   ug_c,up_c,pair_c=match(gt_c,pr_c,True);tp_c+=len(pair_c);fp_c+=len(pr_c)-len(up_c);fn_c+=len(gt_c)-len(ug_c)
  pc=tp_c/(tp_c+fp_c) if tp_c+fp_c else 0;rc=tp_c/(tp_c+fn_c) if tp_c+fn_c else 0
  classwise.append({'class_id':c,'class_name':name,'TP':tp_c,'FP':fp_c,'FN':fn_c,'precision':pc,'recall':rc,'F1':2*pc*rc/(pc+rc) if pc+rc else 0})
 macro={k:float(np.mean([x[k] for x in classwise])) for k in ['precision','recall','F1']}
 return {'images':len(xs),'GT':tp+fn,'displayed':tp+fp,'TP':tp,'FP':fp,'FN':fn,'precision':p,'recall':r,'F1':2*p*r/(p+r) if p+r else 0,
  'spatial_detection_TP':spatial,'spatial_detection_FP':spfp,'spatial_detection_FN':spfn,'spatial_detection_P':spatial/(spatial+spfp) if spatial+spfp else 0,'spatial_detection_R':spatial/(spatial+spfn) if spatial+spfn else 0,
  'class_disagreement_images':sum(x['class_disagreement'] for x in xs),'no_display_images':sum(x['no_display'] for x in xs),
  'spatially_matched_wrong_class':sum(x['spatially_matched_wrong_class'] for x in xs),'unmatched_displayed_boxes':sum(x['unmatched_annotation_boxes'] for x in xs),
  'macro_classwise':macro,'per_class':classwise,'confusion_matrix_true_rows_predicted_columns_including_background':cm.tolist()}
def main():
 t0=time.perf_counter();seed(123);torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
 if device!='cuda':raise RuntimeError('固定分类头评估应使用现有CUDA环境，当前CUDA不可用')
 base=json.loads((GEN3/'reports'/'bioclip2'/'固定验证集结果.json').read_text(encoding='utf-8'));rows=copy.deepcopy(base['rows'])
 oldkey='bioclip2';keys={'original':'原PP-YOLOE+基准','A':'A继续训练控制组','B':'B分阶段增强组','C':'C轻量遮挡组'}
 baseline_key='gen2_bilinear_best'
 for row in rows:
  row['models'][baseline_key]=row['models'][oldkey];row['original_no_display']=row['models'][baseline_key]['no_display']
 wt=GEN3/'weights'/'bioclip2'/'open_clip_model.safetensors';head=GEN3/'runs'/'bioclip2'/'best_head.pt';m=json.loads((GEN3/'runs'/'bioclip2'/'metadata.json').read_text(encoding='utf-8'))
 assert sha(head)==m['best_head_sha256']
 cfg=json.loads((wt.parent/'open_clip_config.json').read_text(encoding='utf-8'))
 model,_,transform=open_clip.create_model_and_transforms('ViT-L-14',pretrained=str(wt),image_mean=cfg['preprocess_cfg']['mean'],image_std=cfg['preprocess_cfg']['std'],image_interpolation='bicubic')
 model=model.visual.to(device).eval();headmod=torch.nn.Linear(768,8).to(device);headmod.load_state_dict(torch.load(head,map_location=device,weights_only=True));headmod.eval()
 detail={};elapsed={}
 for group in 'ABC':
  t=time.perf_counter();doc=json.loads((ROOT/'runs'/f'{group}_inference'/'proposals.json').read_text(encoding='utf-8'))
  assert doc['metadata']['source_images']==172
  lookup={x['filename']:x for x in doc['images']};jobs=[];group_rows=[]
  for ri,row in enumerate(rows):
   raw=lookup[row['filename']]['candidates'];groups=group_candidates(raw);wins=[g['winner'] for g in groups if g['winner']['detector_score']>.30]
   preds=[]
   for g in groups:
    w=g['winner']
    if w['detector_score']>.30:jobs.append({'row':ri,'image':row['image_path'],'group':g})
   group_rows.append({'raw_candidate_count':len(raw),'groups':groups,'winners':wins,'predictions':preds})
  scores=[]
  with torch.inference_mode():
   for st in range(0,len(jobs),8):
    ims=[]
    for job in jobs[st:st+8]:
     d=job['group']['winner']
     with Image.open(job['image']) as im:ims.append(transform(square(crop(im.convert('RGB'),d['bbox_xyxy'],.15))))
    with torch.autocast(device_type='cuda',enabled=True,dtype=torch.float16):z=model(torch.stack(ims).to(device))
    scores.extend(headmod(F.normalize(z.float(),dim=1)).softmax(1).float().cpu().tolist())
  for job,s in zip(jobs,scores):
   d=job['group']['winner'];top=int(np.argmax(s));p={**d,'class_id':top,'class_name':NAMES[top],
    'classifier_top_score':float(s[top]),'classifier_scores':dict(zip(NAMES,map(float,s))),
    'final_score_product':float(d['detector_score']*s[top])}
   group_rows[job['row']]['predictions'].append(p)
  for ri,row in enumerate(rows):row['models'][group]=inspect(row['truth'],group_rows[ri]['predictions'],row['expected'])
  detail[group]={'metadata':doc['metadata'],'images':group_rows};elapsed[group]=time.perf_counter()-t
 summaries={baseline_key:summarize(rows,baseline_key),**{g:summarize(rows,g) for g in 'ABC'}}
 original_five=[r['filename'] for r in rows if r['original_no_display']]
 deltas={'original_no_display_images':original_five,'found_from_original_no_display':{},'new_no_display':{},'end_to_end_TP_change':{},'spatially_matched_wrong_class':{},'unmatched_displayed_boxes':{},'new_unmatched_displayed_boxes_vs_baseline':{},'timing_seconds':elapsed}
 base_tp=summaries[baseline_key]['TP']
 for g in 'ABC':
  deltas['found_from_original_no_display'][g]=[r['filename'] for r in rows if r['original_no_display'] and not r['models'][g]['no_display']]
  deltas['new_no_display'][g]=[r['filename'] for r in rows if not r['original_no_display'] and r['models'][g]['no_display']]
  deltas['end_to_end_TP_change'][g]=summaries[g]['TP']-base_tp
  deltas['spatially_matched_wrong_class'][g]=[{'filename':r['filename'],'count':r['models'][g]['spatially_matched_wrong_class']} for r in rows if r['models'][g]['spatially_matched_wrong_class']]
  deltas['unmatched_displayed_boxes'][g]=[{'filename':r['filename'],'count':r['models'][g]['unmatched_annotation_boxes']} for r in rows if r['models'][g]['unmatched_annotation_boxes']]
  deltas['new_unmatched_displayed_boxes_vs_baseline'][g]=[{'filename':r['filename'],'baseline_count':r['models'][baseline_key]['unmatched_annotation_boxes'],'group_count':r['models'][g]['unmatched_annotation_boxes']} for r in rows if r['models'][g]['unmatched_annotation_boxes']>r['models'][baseline_key]['unmatched_annotation_boxes']]
 deltas['original_five_outcomes']={}
 for g in 'ABC':
  deltas['original_five_outcomes'][g]=[]
  for r in rows:
   if not r['original_no_display']:continue
   v=r['models'][g]
   deltas['original_five_outcomes'][g].append({'filename':r['filename'],'has_display':not v['no_display'],'displayed_count':len(v['displayed']),
    'location_matched':bool(v['spatially_matched_wrong_class']+v['TP']>0),'end_to_end_correct_TP':v['TP'],'spatially_matched_wrong_class':v['spatially_matched_wrong_class'],
    'predictions':[{'class_name':p['class_name'],'detector_score':p['detector_score'],'classifier_top_score':p['classifier_top_score'],'spatial_status':p['spatial_status']} for p in v['displayed']]})
 metadata={'fixed_validation_images':172,'manual_instances':204,'strict_detector_threshold':.30,'overall_metrics':'micro P/R/F1 from aggregate TP/FP/FN across all images and eight categories; separate per-class metrics and macro means are included','matching':'predictions and GT matched one-to-one at IoU>=0.5; classwise metrics require class equality. Bird-location diagnostics use class-agnostic one-to-one matching.','cross_class_dedup':'PPDetection stored candidates, confidence descending, IoU>=0.5 grouped to retained winner; non-transitive','classification':'Frozen BioCLIP 2 encoder and fixed best linear head, same 15% expanded square padded crops and preprocessing','final_score':'detector_score * classifier_top_score; ranking score, not calibrated correctness probability','train_val_scope':'reused validation for checkpoint exploration, not independent test','classifier_head_sha256':sha(head),'classifier_encoder_sha256':sha(wt),'seconds':time.perf_counter()-t0,'class_score_interpretation':'all eight softmax scores saved per retained final display box','folder_disagreement_caveat':'class_disagreement_images is a directory-category diagnostic only; multi-bird images may contain valid other classes. Object-level conclusions come from the eight-class XML match and confusion matrix.'}
 report={'metadata':metadata,'models':keys,'summary':summaries,'changes':deltas,'rows':rows}
 OUT.mkdir(parents=True,exist_ok=True);write(OUT/'fixed_val_results.json',report)
 for g in 'ABC':write(OUT/f'{g}_all_candidates_and_groups.json',{'metadata':detail[g]['metadata'],'images':[{'filename':r['filename'],'groups':x['groups'],'low_score_winners':[z for z in x['winners'] if z['detector_score']<=.30]} for r,x in zip(rows,detail[g]['images'])]})
 with (OUT/'fixed_val_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.writer(f);w.writerow(['方案','验证图','TP','FP','FN','Precision_micro','Recall_micro','F1_micro','鸟框定位TP','定位FP','定位FN','无显示框','类别分歧图','位置对但类别错实例','未匹配显示框'])
  for k,n in keys.items():s=summaries[baseline_key if k=='original' else k];w.writerow([n,s['images'],s['TP'],s['FP'],s['FN'],s['precision'],s['recall'],s['F1'],s['spatial_detection_TP'],s['spatial_detection_FP'],s['spatial_detection_FN'],s['no_display_images'],s['class_disagreement_images'],s['spatially_matched_wrong_class'],s['unmatched_displayed_boxes']])
 with (OUT/'fixed_val_per_class.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.writer(f);w.writerow(['方案','类别编号','标准类别','TP','FP','FN','Precision','Recall','F1'])
  for k,n in keys.items():
   s=summaries[baseline_key if k=='original' else k]
   for c in s['per_class']:w.writerow([n,c['class_id'],c['class_name'],c['TP'],c['FP'],c['FN'],c['precision'],c['recall'],c['F1']])
 for k,n in keys.items():
  s=summaries[baseline_key if k=='original' else k];cm=s['confusion_matrix_true_rows_predicted_columns_including_background']
  with (OUT/f'confusion_matrix_{k}.csv').open('w',encoding='utf-8-sig',newline='') as f:
   w=csv.writer(f);w.writerow(['真实\\预测',*NAMES,'背景']);
   for ci,row0 in enumerate(cm):w.writerow([NAMES[ci] if ci<8 else '背景',*row0])
 # 给浏览器静态HTML内嵌中等尺寸缩略图，页面不依赖Python服务。
 for row in rows:
  with Image.open(row['image_path']) as im:
   im=im.convert('RGB');im.thumbnail((960,960));buf=io.BytesIO();im.save(buf,format='JPEG',quality=78);row['image']='data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode('ascii')
 template='''<!doctype html><meta charset="utf-8"><title>第三代检测器训练改进固定验证集对照</title><style>body{font:15px system-ui;background:#f3f5f7;margin:20px;color:#1f2937}header,article{background:white;padding:16px;border-radius:12px;margin-bottom:14px}input,select{font:inherit;padding:8px;margin:5px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:12px}canvas{width:100%;height:auto}.bad{color:#bc2636}details{background:#f5f6f8;padding:8px;margin:4px}table{border-collapse:collapse}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:left}.score{font-size:13px}</style><header><h1>第三代检测器训练改进实验</h1><p>固定172张验证图 / 204个人工实例。四种检测器统一640输入、BioCLIP 2冻结分类器、鸟框外扩15%、检测分数严格&gt;0.30及跨类别IoU≥0.5检测分数优先去重。验证集用于反复探索与选权重，不是独立测试。仅显示通过统一阈值的去重框；所有原始候选、低分框和被抑制框另存JSON。目录类别分歧仅作提示，多鸟图中的不同类别须按XML实例逐只匹配。</p><div id="summary"></div><input id="search" placeholder="搜索图片名"><select id="filter"><option value="original_empty">原模型原5张无框图</option><option value="all">全部172张</option><option value="found">原无框图找回</option><option value="regress">无框新增/显示指标退步</option><option value="empty">所有无显示框</option><option value="unmatched">有未匹配显示框</option></select><label><input type="checkbox" id="gt">显示人工框</label> <span id="count"></span></header><main id="cards"></main><button id="more">更多</button><script>const D=__DATA__;const names=D.models,keys=['original','A','B','C'],rows=D.rows;const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));document.querySelector('#summary').innerHTML='<table><tr><th>方案</th><th>TP/FP/FN</th><th>P/R/F1</th><th>无框图</th><th>目录分歧提示</th><th>错类实例</th><th>未匹配显示框</th></tr>'+keys.map(k=>{let q=k==='original'?'gen2_bilinear_best':k,s=D.summary[q];return `<tr><td>${names[k]}</td><td>${s.TP}/${s.FP}/${s.FN}</td><td>${s.precision.toFixed(4)} / ${s.recall.toFixed(4)} / ${s.F1.toFixed(4)}</td><td>${s.no_display_images}</td><td>${s.class_disagreement_images}</td><td>${s.spatially_matched_wrong_class}</td><td>${s.unmatched_displayed_boxes}</td></tr>`}).join('')+'</table>';let filtered=[],shown=0;function draw(c,r,k){let im=new Image();im.onload=()=>{c.width=im.width;c.height=im.height;let x=c.getContext('2d');x.drawImage(im,0,0);let sx=im.width/r.width,sy=im.height/r.height;x.font='bold 14px system-ui';x.lineWidth=2;function box(p,col,label){let b=p.bbox_xyxy;x.strokeStyle=col;x.strokeRect(b[0]*sx,b[1]*sy,(b[2]-b[0])*sx,(b[3]-b[1])*sy);let xx=Math.max(0,b[0]*sx),yy=Math.max(18,b[1]*sy),tw=x.measureText(label).width;x.fillStyle=col;x.fillRect(xx,yy-18,tw+8,20);x.fillStyle='white';x.fillText(label,xx+4,yy-3)}if(document.querySelector('#gt').checked)r.truth.forEach((p,i)=>box(p,'#15803d','GT '+(i+1)+' '+p.class_name));r.models[k==='original'?'gen2_bilinear_best':k].displayed.forEach((p,i)=>box(p,'#1d4ed8',`${p.class_name} D${(p.detector_score*100).toFixed(0)} C${(p.classifier_top_score*100).toFixed(0)}`))};im.src=r.image}function render(){let q=document.querySelector('#search').value.toLowerCase(),f=document.querySelector('#filter').value;filtered=rows.filter(r=>{let b=r.models.gen2_bilinear_best,any=keys.slice(1).some(k=>r.models[k].TP<b.TP||r.models[k].no_display&&!b.no_display),found=keys.slice(1).some(k=>r.original_no_display&&!r.models[k].no_display),unmatched=keys.some(k=>{let m=k==='original'?b:r.models[k];return m.unmatched_annotation_boxes>0});return r.filename.toLowerCase().includes(q)&&(f==='all'||f==='original_empty'&&r.original_no_display||f==='found'&&found||f==='regress'&&any||f==='empty'&&keys.some(k=>{let m=k==='original'?b:r.models[k];return m.no_display})||f==='unmatched'&&unmatched)});shown=0;document.querySelector('#cards').innerHTML='';more()}function more(){let sub=filtered.slice(shown,shown+8);shown+=sub.length;for(let r of sub){let a=document.createElement('article');a.innerHTML=`<h2>${esc(r.filename)} <small>${r.truth.length}个人工框</small></h2><div class="grid">`+keys.map(k=>{let v=r.models[k==='original'?'gen2_bilinear_best':k];return `<section><h3>${names[k]} ${v.no_display?'<span class="bad">无显示框</span>':''}</h3><canvas data-key="${k}"></canvas><p>显示${v.displayed.length}框；TP ${v.TP}/FP ${v.FP}/FN ${v.FN}</p>`+v.displayed.map((p,i)=>`<details><summary>${i+1}. ${esc(p.class_name)} | 检测 ${(p.detector_score*100).toFixed(1)}% | 分类 ${(p.classifier_top_score*100).toFixed(1)}%</summary><div class="score">`+Object.entries(p.classifier_scores).sort((x,y)=>y[1]-x[1]).map(([n,s])=>`${esc(n)} ${(s*100).toFixed(2)}%`).join('<br>')+'</div></details>').join('')+'</section>'}).join('')+'</div>';document.querySelector('#cards').appendChild(a);a.querySelectorAll('canvas').forEach(c=>draw(c,r,c.dataset.key))}document.querySelector('#more').hidden=shown>=filtered.length;document.querySelector('#count').textContent=`筛选 ${filtered.length} 张，已显示 ${shown}`}document.querySelector('#search').oninput=render;document.querySelector('#filter').onchange=render;document.querySelector('#gt').onchange=render;document.querySelector('#more').onclick=more;render()</script>'''
 payload=json.dumps({'models':keys,'summary':summaries,'rows':rows},ensure_ascii=False).replace('<','\\u003c')
 (OUT/'fixed_val_comparison.html').write_text(template.replace('__DATA__',payload),encoding='utf-8')
 print(json.dumps({'summary':summaries,'previous_no_display':len(original_five),'changes':deltas,'output':str(OUT)},ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
