"""使用固定验证集XML，对预测框下整图结果做八类端到端评估。"""
from __future__ import annotations
import argparse,csv,json,zipfile
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

ROOT=Path(__file__).resolve().parents[1];CFG=json.loads((ROOT/'configs/paths.json').read_text(encoding='utf-8'));MAP=json.loads((ROOT/'configs/class_mapping.json').read_text(encoding='utf-8'))
NAMES=[x['name'] for x in MAP['classes']];ID={n:i for i,n in enumerate(NAMES)}
def iou(a,b):
 x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3]);inter=max(0,x2-x1)*max(0,y2-y1)
 aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]);bb=max(0,b[2]-b[0])*max(0,b[3]-b[1]);return inter/(aa+bb-inter) if aa+bb-inter else 0
def load_gt():
 out={}
 with zipfile.ZipFile(CFG['original_annotations_zip']) as z:
  for n in z.namelist():
   if not n.lower().endswith('.xml'):continue
   name=Path(n).stem+'.jpg';root=ET.fromstring(z.read(n));arr=[]
   for obj in root.findall('object'):
    raw=obj.findtext('name','').strip();standard=MAP['xml_label_to_standard'].get(raw)
    if standard is None:continue
    b=obj.find('bndbox');arr.append({'class_name':standard,'class_id':ID[standard],'bbox_xyxy':[float(b.findtext(k)) for k in ('xmin','ymin','xmax','ymax')]})
   out[name]=arr
 return out
def match(gt,preds,require_class=True):
 pairs=[]
 for gi,g in enumerate(gt):
  for pi,p in enumerate(preds):
   ov=iou(g['bbox_xyxy'],p['bbox_xyxy'])
   if ov>=.5 and (not require_class or g['class_id']==p['class_id']):pairs.append((ov,float(p.get('final_score_product',0)),gi,pi))
 ug=set();up=set();res=[]
 for ov,sc,gi,pi in sorted(pairs,reverse=True):
  if gi not in ug and pi not in up:ug.add(gi);up.add(pi);res.append((gi,pi,ov))
 return ug,up,res
def ap101(gt_count, scored):
 if gt_count==0:return None
 scored.sort(key=lambda x:x[0],reverse=True);tp=np.cumsum([x[1] for x in scored]);fp=np.cumsum([not x[1] for x in scored]);rec=tp/gt_count;prec=tp/np.maximum(tp+fp,1)
 return float(np.mean([prec[rec>=r].max() if np.any(rec>=r) else 0 for r in np.linspace(0,1,101)]))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--predictions',type=Path,required=True);ap.add_argument('--split',default='val');ap.add_argument('--detector-threshold',type=float,default=.5);ap.add_argument('--output-dir',type=Path,required=True);a=ap.parse_args()
 pred=json.loads(a.predictions.read_text(encoding='utf-8'));gtall=load_gt();split={}
 with Path(CFG['first_generation_split_manifest']).open(encoding='utf-8-sig',newline='') as f:
  for r in csv.DictReader(f):split[r['source_image']]=r['split']
 records=[x for x in pred['images'] if split.get(x['filename'])==a.split];prs={x['filename']:[{**d,'class_id':d['class_id']} for d in x['candidates']] for x in records}
 gt={x['filename']:gtall[x['filename']] for x in records};class_rows=[];cm=np.zeros((9,9),dtype=np.int64);all_tp=all_fp=all_fn=0
 for c,name in enumerate(NAMES):
  truths=sum(g['class_id']==c for arr in gt.values() for g in arr);tp=fp=fn=0
  threshold_preds={fnm:[p for p in prs[fnm] if p['class_id']==c and p['detector_score']>a.detector_threshold] for fnm in gt}
  for fnm in gt:
   gc=[g for g in gt[fnm] if g['class_id']==c];ps=threshold_preds[fnm]
   ug,up,ms=match(gc,ps,True);tp+=len(ms);fp+=len(ps)-len(up);fn+=len(gc)-len(ug)
  # AP逐类别按最终排序分数全局降序；每个预测只匹配同图中IoU最高且尚未匹配的GT。
  ranked=sorted([(float(p['final_score_product']),fnm,p) for fnm in gt for p in prs[fnm] if p['class_id']==c],reverse=True,key=lambda x:x[0])
  used={fnm:set() for fnm in gt};scored=[]
  for score,fnm,p in ranked:
   options=[(iou(p['bbox_xyxy'],g['bbox_xyxy']),gi) for gi,g in enumerate(gt[fnm]) if g['class_id']==c and gi not in used[fnm]]
   best=max(options,default=(0,None))
   ok=best[0]>=.5
   if ok:used[fnm].add(best[1])
   scored.append((score,ok))
  pr=tp/(tp+fp) if tp+fp else 0;re=tp/(tp+fn) if tp+fn else 0
  all_tp+=tp;all_fp+=fp;all_fn+=fn
  class_rows.append({'id':c,'class_name':name,'AP50_101point':ap101(truths,scored),'precision_at_detector_threshold':pr,'recall_at_detector_threshold':re,'F1':2*pr*re/(pr+re) if pr+re else 0,'TP':tp,'FP':fp,'FN':fn,'GT':truths})
 # confusion matrix用任意类别空间IoU>=0.5一对一配对；错误类别落入真值类->预测类；未配对落背景。
 for fnm in gt:
  gc=gt[fnm];ps=[p for p in prs[fnm] if p['detector_score']>a.detector_threshold];ug,up,ms=match(gc,ps,False)
  for gi,pi,_ in ms:cm[gc[gi]['class_id'],ps[pi]['class_id']]+=1
  for gi,g in enumerate(gc):
   if gi not in ug:cm[g['class_id'],8]+=1
  for pi,p in enumerate(ps):
   if pi not in up:cm[8,p['class_id']]+=1
 precision=all_tp/(all_tp+all_fp) if all_tp+all_fp else 0;recall=all_tp/(all_tp+all_fn) if all_tp+all_fn else 0
 metrics={'split':a.split,'images':len(records),'mAP50_macro_101point':float(np.mean([x['AP50_101point'] for x in class_rows if x['AP50_101point'] is not None])),
  'micro_precision_at_detector_threshold':precision,'micro_recall_at_detector_threshold':recall,'micro_F1':2*precision*recall/(precision+recall) if precision+recall else 0,
  'detector_threshold_strictly_greater_than':a.detector_threshold,'NMS_IoU':.7,'match_IoU':.5,
  'AP_score':'detector_score * classifier_top_score; from PP-YOLOE score_threshold=0.001 outputs, top 100 detector-score candidates per image are retained (COCO maxDets=100) and ranked for 101-point AP',
  'max_detections_per_image':100,
  'confusion_matrix_matching':'不看类别先按IoU>=0.5按重叠降序一对一配对；错类记真值行/预测列；未匹配记入背景行或列。',
  'per_class':class_rows,'matching':'P/R/F1要求类别正确且IoU>=0.5的一对一贪心匹配；重复预测计FP。正式方案以预测框端到端评估。'}
 a.output_dir.mkdir(parents=True,exist_ok=True);(a.output_dir/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
 with (a.output_dir/'metrics_by_class.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(class_rows[0]));w.writeheader();w.writerows(class_rows)
 with (a.output_dir/'confusion_matrix.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.writer(f);w.writerow(['真实\\预测',*NAMES,'背景']);
  for i,row in enumerate(cm):w.writerow([NAMES[i] if i<8 else '背景',*row.tolist()])
 print(json.dumps({'images':len(records),'mAP50':metrics['mAP50_macro_101point'],'micro_P':precision,'micro_R':recall,'micro_F1':metrics['micro_F1']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
