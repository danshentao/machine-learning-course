from pathlib import Path
import json,sys,math
import numpy as np, torch, open_clip
from PIL import Image
import torch.nn.functional as F
ROOT=Path(r'E:\机器学习课设')
G3=ROOT/'第三代鸟类识别现代模型对照'
R=G3/'reports'/'bioclip2'/'固定验证集结果.json'
doc=json.loads(R.read_text(encoding='utf-8')); row=doc['rows'][0]; p=row['models']['bioclip2']['displayed'][0]
img=Image.open(row['image_path']).convert('RGB'); x1,y1,x2,y2=p['bbox_xyxy'];dx=(x2-x1)*.15;dy=(y2-y1)*.15
im=img.crop((max(0,math.floor(x1-dx)),max(0,math.floor(y1-dy)),min(img.width,math.ceil(x2+dx)),min(img.height,math.ceil(y2+dy))))
w,h=im.size;s=max(w,h);sq=Image.new('RGB',(s,s),(114,114,114));sq.paste(im,((s-w)//2,(s-h)//2))
local=G3/'weights'/'bioclip2'/'open_clip_model.safetensors';cfg=json.loads((local.parent/'open_clip_config.json').read_text())
model,_,transform=open_clip.create_model_and_transforms('ViT-L-14',pretrained=str(local),image_mean=cfg['preprocess_cfg']['mean'],image_std=cfg['preprocess_cfg']['std'],image_interpolation='bicubic')
device='cuda' if torch.cuda.is_available() else 'cpu'; model=model.visual.to(device).eval()
for q in model.parameters():q.requires_grad_(False)
with torch.inference_mode(), torch.autocast(device_type=device,enabled=device=="cuda",dtype=torch.float16): z=F.normalize(model(transform(sq).unsqueeze(0).to(device)).float(),dim=1)
head=torch.nn.Linear(z.shape[1],8).to(device);head.load_state_dict(torch.load(G3/'runs'/'bioclip2'/'best_head.pt',map_location=device,weights_only=True));head.eval()
with torch.inference_mode(): pr=head(z).softmax(1)[0].cpu().numpy()
cache=np.load(G3/'runs'/'bioclip2'/'features.npz')['features'];n=1594+146
with torch.inference_mode():cached=head(torch.tensor(cache[n:n+1],device=device)).softmax(1)[0].cpu().numpy()
print(json.dumps({'device':device,'open_clip':open_clip.__version__,'features_shape':list(cache.shape),'image':row['filename'],'predicted_current':int(pr.argmax()),'predicted_cached':int(cached.argmax()),'max_probability_delta':float(np.max(np.abs(pr-cached))),'current_probs':pr.tolist(),'cached_probs':cached.tolist()},ensure_ascii=False))

