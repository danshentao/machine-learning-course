"""用正式PP-YOLOE+权重按固定640 reader在172张验证图上保存低阈值原始候选。"""
from __future__ import annotations
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];P=json.loads((ROOT/'paths.json').read_text(encoding='utf-8'))
PD=Path(P['paddledetection']);sys.path.insert(0,str(PD))
from ppdet.core.workspace import load_config
from ppdet.engine import Trainer

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def clip(b,w,h):
 z=[min(max(float(x),0.),float(l)) for x,l in zip(b,[w,h,w,h])]
 return z if z[2]>z[0] and z[3]>z[1] else None
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--group',choices=list('ABC'),required=True);ap.add_argument('--weight',type=Path);a=ap.parse_args()
 group=a.group;weight=a.weight or (ROOT/'runs'/group/'best_model_ap50.pdparams')
 if not weight.exists():raise FileNotFoundError(weight)
 base=json.loads((Path(P['generation3'])/'reports'/'bioclip2'/'固定验证集结果.json').read_text(encoding='utf-8'))
 rows=base['rows'];cfg=load_config(str(ROOT/'configs'/f'{group}_phase2.yml'));cfg.weights=str(weight);cfg.worker_num=0
 cfg.PPYOLOEHead['eval_size']=[640,640];cfg.TestReader['inputs_def']['image_shape']=[3,640,640]
 for t in cfg.TestReader['sample_transforms']:
  if 'Resize' in t:t['Resize']['target_size']=[640,640]
 cfg.TestReader.update({'batch_size':1,'worker_num':0})
 import paddle
 paddle.set_device('gpu');paddle.seed(123)
 trainer=Trainer(cfg,mode='test');trainer.load_weights(str(weight));seen=[];forward=trainer.model.forward
 def record(inputs,*args,**kwargs):
  s=list(inputs['image'].shape)
  if s not in seen:seen.append(s)
  return forward(inputs,*args,**kwargs)
 trainer.model.forward=record
 outdir=ROOT/'runs'/f'{group}_inference';outdir.mkdir(parents=True,exist_ok=True)
 start=time.perf_counter();pred=trainer.predict([r['image_path'] for r in rows],output_dir=str(outdir/'predict_output'),save_results=False,visualize=False)
 mapping=trainer.dataset.get_imid2path();by={}
 for batch in pred:
  boxes=np.asarray(batch['bbox']);ids=np.asarray(batch['im_id']).reshape(-1);counts=np.asarray(batch['bbox_num']).reshape(-1);off=0
  for imid,cnt in zip(ids,counts):
   path=Path(mapping[int(imid)]);part=boxes[off:off+int(cnt)];off+=int(cnt)
   with Image.open(path) as im:w,h=im.size
   det=[]
   for z in sorted(part,key=lambda x:-float(x[1]))[:100]:
    raw=list(map(float,z[2:6]));b=clip(raw,w,h)
    if b is not None:det.append({'bbox_xyxy':b,'raw_bbox_xyxy':raw,'detector_score':float(z[1]),'box_clipped':b!=raw})
   by[path.name]={'filename':path.name,'image_path':str(path),'width':w,'height':h,'candidates':det}
 assert len(by)==172 and seen==[[1,3,640,640]],(len(by),seen)
 meta={'group':group,'weight':str(weight),'weight_sha256':sha(weight),'source_images':172,'input_shapes':seen,'seconds':time.perf_counter()-start,
  'preprocess':'official PP-YOLOE+ TestReader, square stretch Resize to640, raw pixel NormalizeImage norm_type=none/is_scale=false','max_candidates_per_image':100,
  'low_score_retention':'inherits PP-YOLOE+ NMS score_threshold=0.001, not the final display threshold','scope':'fixed validation; no teacher test images'}
 out=ROOT/'runs'/f'{group}_inference'/'proposals.json';out.write_text(json.dumps({'metadata':meta,'images':[by[r['filename']] for r in rows]},ensure_ascii=False),encoding='utf-8')
 print(json.dumps({'group':group,'images':len(by),'seconds':meta['seconds'],'input_shapes':seen,'output':str(out)},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
