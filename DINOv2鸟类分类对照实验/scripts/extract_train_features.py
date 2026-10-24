"""用官方冻结DINOv2 ViT-B/14提取固定训练/验证裁剪特征。"""
from __future__ import annotations
import csv, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for key, rel in [('TORCH_HOME','cache/torch'),('HF_HOME','cache/huggingface'),('MPLCONFIGDIR','cache/matplotlib'),('TEMP','cache/tmp'),('TMP','cache/tmp')]:
    p=ROOT/rel; p.mkdir(parents=True,exist_ok=True); os.environ[key]=str(p)
CFG=json.loads((ROOT/'configs'/'experiment.json').read_text(encoding='utf-8'))
WEIGHTS=ROOT/CFG['weights_relative_path']
SRC=ROOT/'third_party'/'dinov2'/'dinov2-7764ea0f912e53c92e82eb78a2a1631e92725fc8'
sys.path.insert(0,str(SRC))

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def load_rows(name):
    with (ROOT/'data'/name).open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))

def make_tensor(images, torch):
    from PIL import Image
    arr=[]
    mean=np.array([.485,.456,.406],dtype=np.float32).reshape(1,1,3)
    std=np.array([.229,.224,.225],dtype=np.float32).reshape(1,1,3)
    for im in images:
        im=im.convert('RGB')
        side=max(im.size)
        canvas=Image.new('RGB',(side,side),(114,114,114))
        canvas.paste(im,((side-im.width)//2,(side-im.height)//2))
        im=canvas.resize((224,224),Image.Resampling.BICUBIC)
        x=np.asarray(im,dtype=np.float32)/255.0
        arr.append(((x-mean)/std).transpose(2,0,1))
    return torch.from_numpy(np.stack(arr)).to('cuda',non_blocking=True)

def main():
    import torch
    import torch.nn.functional as F
    from PIL import Image
    torch.set_num_threads(4)
    assert torch.cuda.is_available(), 'DINOv2特征提取要求CUDA；未运行CPU慢速替代'
    from dinov2.hub.backbones import dinov2_vitb14
    model=dinov2_vitb14(pretrained=False)
    state=torch.load(WEIGHTS,map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True)
    model=model.cuda().eval()
    for p in model.parameters(): p.requires_grad_(False)
    train=load_rows('train_crops.csv'); valid=load_rows('val_crops.csv')
    signature=hashlib.sha256((sha(WEIGHTS)+sha(ROOT/'data'/'train_crops.csv')+sha(ROOT/'data'/'val_crops.csv')+CFG['crop_input']+CFG['feature']).encode()).hexdigest()
    tasks=train+valid; feats=[]; start=time.perf_counter(); bs=int(CFG['feature_batch_size'])
    with torch.inference_mode():
        for i in range(0,len(tasks),bs):
            group=tasks[i:i+bs]
            ims=[]
            for row in group:
                with Image.open(row['crop_path']) as im: ims.append(im.copy())
            x=make_tensor(ims,torch)
            with torch.autocast('cuda',dtype=torch.float16): z=model.forward_features(x)['x_norm_clstoken']
            z=F.normalize(z.float(),dim=1)
            if z.shape[1]!=768 or not torch.isfinite(z).all(): raise RuntimeError(f'bad DINO features: {tuple(z.shape)}')
            feats.append(z.cpu().numpy())
            if i%128==0:
                print(json.dumps({'phase':'feature_extraction','done':min(i+len(group),len(tasks)),'total':len(tasks),'seconds':time.perf_counter()-start,'allocated_gib':round(torch.cuda.max_memory_allocated()/1024**3,3)},ensure_ascii=False),flush=True)
    features=np.concatenate(feats).astype(np.float32)
    path=ROOT/'runs'/'dinov2'/'features_train_val.npz';path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,features=features,signature=np.array(signature))
    meta={'encoder':'DINOv2 ViT-B/14 official frozen','weight_path':str(WEIGHTS),'weight_sha256':sha(WEIGHTS),'source_revision':CFG['official_dinov2_revision'],'feature':CFG['feature'],'preprocess':CFG['crop_input'],'train_rows':len(train),'validation_rows':len(valid),'feature_shape':list(features.shape),'signature':signature,'device':torch.cuda.get_device_name(0),'max_cuda_memory_gib':torch.cuda.max_memory_allocated()/1024**3,'elapsed_seconds':time.perf_counter()-start,'cache_path':str(path)}
    (ROOT/'runs'/'dinov2'/'feature_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(meta,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__': main()
