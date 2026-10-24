"""Run the unchanged bird detector at a specified size on the same 172 images."""
from common import ROOT, ARCHIVE, baseline, sha, write
import sys, argparse, time
from pathlib import Path
import numpy as np
from PIL import Image
import paddle
import yaml

PD=Path('E:/机器学习课设/自然场景鸟类检测PPYOLOE/third_party/PaddleDetection')
sys.path.insert(0,str(PD))
from ppdet.core.workspace import load_config
from ppdet.engine import Trainer

def clip(b,w,h):
    v=[min(max(float(x),0.),float(lim)) for x,lim in zip(b,[w,h,w,h])]
    return v if v[2]>v[0] and v[3]>v[1] else None

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--size',type=int,choices=[640,960],required=True);a=ap.parse_args()
    source=Path('E:/机器学习课设/第二代鸟类检测与细粒度分类/configs/ppyoloe_plus_bird_640.yml')
    weight=ARCHIVE/'工程/runs/detector/best_model_ap50.pdparams'
    out=ROOT/'runs'/f'detector_resolution_{a.size}';out.mkdir(parents=True,exist_ok=True)
    try:text=source.read_text(encoding='utf-8');source_encoding='utf-8'
    except UnicodeDecodeError:text=source.read_text(encoding='gb18030');source_encoding='gb18030'
    config_doc=yaml.safe_load(text)
    config_doc['_BASE_']=[str((source.parent/x).resolve()) for x in config_doc['_BASE_']]
    local_config=out/'resolved_source_config.yml'
    local_config.write_text(yaml.safe_dump(config_doc,allow_unicode=False),encoding='utf-8')
    cfg=load_config(str(local_config));cfg.weights=str(weight);cfg.worker_num=0
    # Cached anchor grids must correspond to the actual feature map dimensions.
    cfg.PPYOLOEHead['eval_size']=[a.size,a.size]
    cfg.TestReader['inputs_def']['image_shape']=[3,a.size,a.size]
    changed=0
    for t in cfg.TestReader['sample_transforms']:
        if 'Resize' in t:t['Resize']['target_size']=[a.size,a.size];changed+=1
    assert changed==1,changed
    cfg.TestReader['batch_size']=1;cfg.TestReader['worker_num']=0
    rows=baseline()['rows'];paddle.set_device('gpu');paddle.seed(123)
    trainer=Trainer(cfg,mode='test');trainer.load_weights(str(weight));seen=[]
    old_forward=trainer.model.forward
    def recorded_forward(inputs,*args,**kwargs):
        shape=list(inputs['image'].shape)
        if shape not in seen:seen.append(shape)
        return old_forward(inputs,*args,**kwargs)
    trainer.model.forward=recorded_forward
    start=time.perf_counter();outputs=trainer.predict([r['image_path'] for r in rows],output_dir=str(out/'predict_output'),save_results=False,visualize=False)
    id2path=trainer.dataset.get_imid2path();byname={}
    for batch in outputs:
        b=np.asarray(batch['bbox']);ids=np.asarray(batch['im_id']).reshape(-1);counts=np.asarray(batch['bbox_num']).reshape(-1);offset=0
        for imid,count in zip(ids,counts):
            path=Path(id2path[int(imid)]);part=b[offset:offset+int(count)];offset+=int(count)
            with Image.open(path) as im:w,h=im.size
            proposals=[]
            for z in sorted(part,key=lambda z:-float(z[1]))[:100]:
                raw=list(map(float,z[2:6]));clipped=clip(raw,w,h)
                if clipped is not None:proposals.append({'bbox_xyxy':clipped,'raw_bbox_xyxy':raw,'detector_score':float(z[1]),'box_clipped':clipped!=raw})
            assert path.name not in byname
            byname[path.name]={'filename':path.name,'image':str(path),'width':w,'height':h,'detections':proposals}
    assert len(byname)==len(rows)==172
    assert seen and all(s[-2:]==[a.size,a.size] for s in seen),seen
    metadata={'size':a.size,'actual_input_shapes':seen,'detector_weights':str(weight),'detector_weights_sha256':sha(weight),'original_config':str(source),'original_config_sha256':sha(source),'original_config_encoding':source_encoding,'test_resize':cfg.TestReader['sample_transforms'],'image_count':len(rows),'max_candidates':100,'nms_iou':.7,'seconds':time.perf_counter()-start,'coordinates':'original image pixels, clipped to image bounds','display_threshold':.30,'group_iou':.5,'no_training':True}
    write(out/'proposals.json',{'metadata':metadata,'images':[byname[r['filename']] for r in rows]});print(metadata,flush=True)
if __name__=='__main__':main()
