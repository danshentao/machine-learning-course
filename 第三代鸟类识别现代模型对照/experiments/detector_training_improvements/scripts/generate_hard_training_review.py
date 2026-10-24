"""用原PP-YOLOE+检查训练原图并产出D组人工复核页面；只生成候选，绝不训练或改标签。"""
from __future__ import annotations
import argparse,base64,csv,io,json,sys,time
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];P=json.loads((ROOT/'paths.json').read_text(encoding='utf-8'))
PD=Path(P['paddledetection']);sys.path.insert(0,str(PD))
from ppdet.core.workspace import load_config
from ppdet.engine import Trainer
from ppdet.utils.logger import setup_logger
setup_logger('hard_training_review')

def iou(a,b):
 x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3]);inter=max(0,x2-x1)*max(0,y2-y1)
 aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]);bb=max(0,b[2]-b[0])*max(0,b[3]-b[1]);return inter/(aa+bb-inter) if aa+bb-inter else 0
def match(gt,preds):
 pairs=sorted([(iou(g['bbox_xyxy'],p['bbox_xyxy']),-pi,gi,pi) for gi,g in enumerate(gt) for pi,p in enumerate(preds) if iou(g['bbox_xyxy'],p['bbox_xyxy'])>=.5],reverse=True)
 ug=set();up=set();out=[]
 for ov,_,gi,pi in pairs:
  if gi not in ug and pi not in up:ug.add(gi);up.add(pi);out.append((gi,pi,ov))
 return ug,up,out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--weight',type=Path,default=Path(P['generation2_archive_engineering'])/'runs'/'detector'/'best_model_ap50.pdparams');a=ap.parse_args()
 arch=Path(P['generation2_archive_engineering']);coco=json.loads((arch/'data'/'annotations'/'instances_train_bird.json').read_text(encoding='utf-8'))
 excluded={'dashanque_119.jpg','shanmaque_090.jpg','meishanque_012.jpg','meishanque_101.jpg'}
 images=[im for im in coco['images'] if not Path(im['file_name']).stem.endswith('_brightness') and im['file_name'] not in excluded]
 assert len(images)==784,len(images)
 anns={im['id']:[] for im in images}
 for a0 in coco['annotations']:
  if a0['image_id'] in anns:
   b=a0['bbox'];anns[a0['image_id']].append({'bbox_xyxy':[b[0],b[1],b[0]+b[2],b[1]+b[3]],'class_name':None,'class_id':None})
 # 复核页附上同一标注版本的八类实例名，仅供人工核对；训练标注仍使用单类bird。
 from csv import DictReader
 manifest=arch/'reports'/'classification_crop_manifest.csv'
 labelmap={}
 with manifest.open(encoding='utf-8-sig',newline='') as f:
  for r in DictReader(f):
   if r['split']!='train' or r['source_variant']!='original' or r['review_pending']=='True':continue
   try:bbox=list(map(float,r['gt_xyxy'].split(',')));labelmap.setdefault(r['source_image'],[]).append((int(r['instance']),bbox,r['class_name'],int(r['class_id'])))
   except Exception:continue
 for im in images:
  name=im['file_name'];labels=labelmap.get(name,[])
  for idx,gt in enumerate(anns[im['id']],1):
   box=gt['bbox_xyxy'];hit=max(labels,key=lambda x:iou(box,x[1]),default=None)
   if hit and iou(box,hit[1])>=.98:gt.update({'class_name':hit[2],'class_id':hit[3],'instance':hit[0]})
 cfg=load_config(str(ROOT/'configs'/'A_phase2.yml'));cfg.weights=str(a.weight);cfg.worker_num=0;cfg.TestReader.update({'batch_size':1,'worker_num':0})
 cfg.PPYOLOEHead['eval_size']=[640,640];cfg.TestReader['inputs_def']['image_shape']=[3,640,640]
 for t in cfg.TestReader['sample_transforms']:
  if 'Resize' in t:t['Resize']['target_size']=[640,640]
 import paddle
 paddle.set_device('gpu');paddle.seed(123);trainer=Trainer(cfg,mode='test');trainer.load_weights(str(a.weight))
 paths=[str(Path(P['prepared_images'])/'images'/'train'/im['file_name']) for im in images]
 outdir=ROOT/'runs'/'hard_train_review';outdir.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
 batches=trainer.predict(paths,output_dir=str(outdir/'predict_output'),save_results=False,visualize=False)
 mapping=trainer.dataset.get_imid2path();byname={};
 for batch in batches:
  boxes=np.asarray(batch['bbox']);ids=np.asarray(batch['im_id']).reshape(-1);counts=np.asarray(batch['bbox_num']).reshape(-1);off=0
  for imid,cnt in zip(ids,counts):
   p=Path(mapping[int(imid)]);part=boxes[off:off+int(cnt)];off+=int(cnt)
   with Image.open(p) as im:w,h=im.size
   candidates=[]
   for z in sorted(part,key=lambda q:-float(q[1]))[:100]:
    b=[min(max(float(x),0.),float(l)) for x,l in zip(z[2:6],[w,h,w,h])]
    if b[2]>b[0] and b[3]>b[1]:candidates.append({'bbox_xyxy':b,'detector_score':float(z[1])})
   byname[p.name]=candidates
 assert len(byname)==784
 rows=[]
 for im in images:
  name=im['file_name'];gt=anns[im['id']];groups=[]
  for ci,d in enumerate(sorted(byname[name],key=lambda p:-p['detector_score'])):
   possible=[(iou(d['bbox_xyxy'],g['winner']['bbox_xyxy']),j) for j,g in enumerate(groups)]
   ov,j=max(possible,default=(0,None))
   if ov>=.5:groups[j]['hidden'].append({'candidate_index':ci,**d,'iou_with_winner':ov})
   else:groups.append({'winner':{'candidate_index':ci,**d},'hidden':[]})
  winners=[g['winner'] for g in groups];display=[p for p in winners if p['detector_score']>.30]
  ug,up,pairs=match(gt,display);issues=[]
  for gi,g in enumerate(gt):
   if gi in ug:continue
   best=max(((iou(g['bbox_xyxy'],d['bbox_xyxy']),d['detector_score']) for d in winners),default=(0.,0.))
   if best[0]>=.5 and best[1]<=.30:issues.append({'type':'score_below_display_threshold','truth_index':gi,'max_iou':best[0],'best_score':best[1]})
   elif best[0]>=.15:issues.append({'type':'localization_or_box_extent','truth_index':gi,'max_iou':best[0],'best_score':best[1]})
   else:issues.append({'type':'ground_truth_missed','truth_index':gi,'max_iou':best[0],'best_score':best[1]})
  for pi,d in enumerate(display):
   if pi in up:continue
   ov=max((iou(d['bbox_xyxy'],g['bbox_xyxy']) for g in gt),default=0.)
   typ='duplicate_or_unmatched_box' if ov>=.5 else 'suspected_background_or_omitted_bird' if ov<.1 else 'localization_or_box_extent'
   issues.append({'type':typ,'prediction_index':pi,'max_iou':ov,'score':d['detector_score']})
  if issues:
   imgpath=str(Path(P['prepared_images'])/'images'/'train'/name)
   rows.append({'filename':name,'image_path':imgpath,'width':im['width'],'height':im['height'],'truth':gt,'displayed':display,'groups':groups,'issues':issues,'human_review':'pending','eligible_for_resampling':False})
 # Encode candidate-only images as medium-sized thumbnails for a portable single-file review page.
 for r in rows:
  with Image.open(r['image_path']) as im:
   im=im.convert('RGB');im.thumbnail((1000,1000));b=io.BytesIO();im.save(b,format='JPEG',quality=78);r['image']='data:image/jpeg;base64,'+base64.b64encode(b.getvalue()).decode('ascii')
 out=ROOT/'reports'/'hard_training_review';out.mkdir(parents=True,exist_ok=True)
 label_total=sum(len(anns[im['id']]) for im in images);label_mapped=sum(gt.get('class_name') is not None for im in images for gt in anns[im['id']])
 metadata={'weight':str(a.weight),'weight_sha256':__import__('hashlib').sha256(a.weight.read_bytes()).hexdigest(),'train_original_images':784,'train_original_instances':label_total,'species_label_mapped_instances':label_mapped,'species_label_source':'archived classification_crop_manifest.csv, original train rows only, exact box IoU >=0.98; supplemental display metadata, not detector training labels','exclusions':sorted(excluded),'val_images_used':0,'brightness_derivatives_in_inference':0,'inference_seconds':time.perf_counter()-start,'threshold_strictly_greater':.30,'dedup_iou':.5,'matching':'class-agnostic, one-to-one IoU>=0.5, detector-score priority','meaning':'All issue types are model-vs-existing-annotation review candidates only. Unmatched predictions are not confirmed background; no labels or training assignments were changed.'}
 doc={'metadata':metadata,'rows':rows};(out/'hard_training_review_cases.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2),encoding='utf-8')
 with (out/'hard_training_review_cases.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.writer(f);w.writerow(['图片','训练原图路径','GT数','检测数','问题类型','GT实例编号','最高IoU','检测分数','人工结论','允许重采样'])
  for r in rows:
   for x in r['issues']:w.writerow([r['filename'],r['image_path'],len(r['truth']),len(r['displayed']),x['type'],x.get('truth_index',''),x.get('max_iou',''),x.get('best_score',x.get('score','')),'待人工复核','否'])
 payload=json.dumps(doc,ensure_ascii=False).replace('<','\\u003c')
 template=(ROOT/'templates'/'hard_review_template.html').read_text(encoding='utf-8')
 (out/'index.html').write_text(template.replace('__DATA__',payload),encoding='utf-8')
 print(json.dumps({'output':str(out),'candidate_images':len(rows),'types':{t:sum(any(x['type']==t for x in r['issues']) for r in rows) for t in sorted({x['type'] for r in rows for x in r['issues']})},'inference_seconds':metadata['inference_seconds']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
