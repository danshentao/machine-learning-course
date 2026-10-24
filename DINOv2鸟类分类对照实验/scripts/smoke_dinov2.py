"""正式数据训练前检查官方权重加载、特征形状及八类线性头反向传播。"""
import hashlib,json,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'configs'/'experiment.json').read_text(encoding='utf-8'))
WEIGHTS=ROOT/CFG['weights_relative_path']
SRC=ROOT/'third_party'/'dinov2'/'dinov2-7764ea0f912e53c92e82eb78a2a1631e92725fc8'
sys.path.insert(0,str(SRC))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 from dinov2.hub.backbones import dinov2_vitb14
 assert torch.cuda.is_available()
 torch.manual_seed(123);start=time.perf_counter()
 model=dinov2_vitb14(pretrained=False)
 model.load_state_dict(torch.load(WEIGHTS,map_location='cpu',weights_only=True),strict=True)
 for p in model.parameters(): p.requires_grad_(False)
 model.eval().cuda()
 x=torch.randn(4,3,224,224,device='cuda')
 with torch.inference_mode(): z=model.forward_features(x)['x_norm_clstoken']
 head=torch.nn.Linear(z.shape[1],8).cuda();y=torch.tensor([0,1,2,3],device='cuda')
 loss=torch.nn.functional.cross_entropy(head(torch.nn.functional.normalize(z,dim=1)),y)
 loss.backward()
 assert z.shape==(4,768) and torch.isfinite(z).all() and torch.isfinite(loss)
 result={'status':'passed','official_model':'dinov2_vitb14','official_source_revision':CFG['official_dinov2_revision'],'weights_path':str(WEIGHTS),'weights_sha256':sha(WEIGHTS),'test_batch':4,'feature_shape':list(z.shape),'class_count':8,'one_step_linear_loss':float(loss.detach()),'backbone_trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'gpu':torch.cuda.get_device_name(0),'peak_allocated_gib':round(torch.cuda.max_memory_allocated()/1024**3,3),'elapsed_seconds':time.perf_counter()-start}
 (ROOT/'reports'/'smoke_test.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
