import json,sys,torch,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];GEN3=Path(json.loads((ROOT/'configs'/'experiment.json').read_text(encoding='utf-8'))['gen3_project'])
def main():
 cfg=json.loads((ROOT/'configs'/'experiment.json').read_text(encoding='utf-8'));archive=Path(cfg['gen2_archive'])
 f=json.loads((GEN3/'reports'/'bioclip2'/'固定验证集结果.json').read_text(encoding='utf-8'));items=[p for r in f['rows'] for p in r['models']['bioclip2']['displayed']]
 baseline_path=archive.parent/'第二代实验报告'/'最终与最佳分类权重对照'/'最终与最佳权重比较结果.json';base=json.loads(baseline_path.read_text(encoding='utf-8'))
 x=np.load(GEN3/'runs'/'bioclip2'/'features.npz')['features'];n=1594+146
 z=torch.from_numpy(x[n:n+len(items)]).cuda();head=torch.nn.Linear(z.shape[1],8).cuda();head.load_state_dict(torch.load(GEN3/'runs'/'bioclip2'/'best_head.pt',map_location='cuda',weights_only=True));head.eval()
 with torch.inference_mode(): probs=head(z).softmax(1).cpu().numpy()
 names=[c['name'] for c in json.loads((GEN3/'reports'/'第二代归档位置.json').read_text(encoding='utf-8'))['class_names']] if False else json.loads((ROOT/'data'/'class_mapping.json').read_text(encoding='utf-8-sig'))
 names=[c['name'] for c in names['classes']]
 dif=[]
 for i,(p,s) in enumerate(zip(items,probs)):
  old=np.array([p['classifier_scores'][n] for n in names]);d=float(np.max(np.abs(old-s)))
  if d>1e-4 or int(np.argmax(s))!=int(p['class_id']):dif.append({'i':i,'diff':d,'old':p['class_name'],'cache':names[int(np.argmax(s))]})
 bm={r['filename']:r for r in base['rows']};bmis=[];pathmis=[]
 for r in f['rows']:
  old=bm.get(r['filename']); src=old.get('models',{}).get('bilinear',{}).get('best',{}).get('displayed',[]) if old else []
  if old and old.get('image_path')!=r.get('image_path'):
   pa=Path(old['image_path']);pb=Path(r['image_path']);pathmis.append({'filename':r['filename'],'baseline_path':str(pa),'fixed_path':str(pb),'same_bytes':pa.is_file() and pb.is_file() and pa.read_bytes()==pb.read_bytes()})
  now=r['models']['bioclip2']['displayed']
  if len(src)!=len(now):bmis.append({'filename':r['filename'],'count_baseline':len(src),'count_fixed':len(now)});continue
  for j,(a,b) in enumerate(zip(src,now)):
   aa=a['bbox_xyxy'];bb=b['bbox_xyxy'];md=max(abs(float(x)-float(y)) for x,y in zip(aa,bb))
   if md>1e-5 or abs(float(a['detector_score'])-float(b['detector_score']))>1e-6:bmis.append({'filename':r['filename'],'candidate':j,'coord_diff':md})
 print(json.dumps({'cache_shape':list(x.shape),'candidate_count':len(items),'top_changes':len(dif),'max_score_difference':max([d['diff'] for d in dif],default=0),'baseline_vs_fixed_mismatch_count':len(bmis),'baseline_vs_fixed_examples':bmis[:10],'different_image_path_count':len(pathmis),'different_image_paths':pathmis[:4],'examples':dif[:8]},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
