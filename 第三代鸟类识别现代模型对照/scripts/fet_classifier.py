"""Author FET-FGVC Swin-B + local GCN + PFI, adapted to frozen bird crops."""
from common import *
import sys,argparse,time,math
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms as T
sys.path.insert(0,str(ROOT/'third_party/FET-FGVC'))
from network.GlobalBranch import FET_FGVC
from modules.losses import LabelSmoothingCrossEntropy,ConvNextDistillDiffPruningLoss

def build(load_pretrained=True):
    cfg=dict(num_classes=8,image_size=224,size=224,load_pretrained=load_pretrained,drop_rate=0.,attn_drop_rate=0.,drop_path_rate=0.,window_size=7,dynamic=True,base_keep_rate=.5,pruning_loc=6)
    local=dict(depth=3,num_parts=8,batch_size=4,part_channels=16,gaussian_ksize=15)
    return FET_FGVC('swin-base',cfg,local,False,False,str(ROOT/'weights'),-1)
class Dataset(torch.utils.data.Dataset):
    def __init__(self,rows,train=False):
        self.rows=rows
        self.transform=T.Compose(([T.RandomHorizontalFlip()] if train else [])+[T.Resize((224,224)),T.ToTensor(),T.Normalize((.485,.456,.406),(.229,.224,.225))])
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        with Image.open(r['crop_path']) as im:x=self.transform(square(im.convert('RGB')))
        return x,int(r['class_id'])
class BalancedSampler(torch.utils.data.Sampler):
    def __init__(self,rows):
        self.buckets={c:[i for i,r in enumerate(rows) if int(r['class_id'])==c] for c in range(8)};self.steps=math.ceil(len(rows)/4)
    def __len__(self):return self.steps
    def __iter__(self):
        for _ in range(self.steps):
            classes=random.sample(range(8),2)
            yield [i for c in classes for i in random.sample(self.buckets[c],2)]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=60);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    seed();torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu';model=build().to(device)
    train,valid=manifest_rows();ds=Dataset(train,True);vd=Dataset(valid)
    loader=torch.utils.data.DataLoader(ds,batch_sampler=BalancedSampler(train),num_workers=0);vl=torch.utils.data.DataLoader(vd,batch_size=4,num_workers=0)
    run=ROOT/'runs'/('fet_smoke' if a.smoke else 'fet_fgvc');run.mkdir(parents=True,exist_ok=True)
    _,pg=model.get_param_groups();opt=torch.optim.AdamW([{'params':pg[0],'lr':1e-5},{'params':pg[1],'lr':5e-5}],weight_decay=.01)
    ce=LabelSmoothingCrossEntropy();dynamic=ConvNextDistillDiffPruningLoss(None,ratio_weight=10,distill_weight=.5,keep_ratio=model.keep_rate,swin_token=True)
    scaler=torch.amp.GradScaler('cuda',enabled=device=='cuda');start=time.perf_counter();best=-1;bestepoch=0;epochs=1 if a.smoke else a.epochs
    with (run/'training.jsonl').open('w',encoding='utf-8') as log:
        for epoch in range(1,epochs+1):
            # Same 10-epoch warmup/cosine form as reference, sized for this run.
            scale=min(epoch/10,1.) if epoch<=10 else .5*(1+math.cos(math.pi*(epoch-10)/max(epochs-10,1)))
            for group,base in zip(opt.param_groups,[1e-5,5e-5]):group['lr']=base*scale
            model.train();losses=[]
            for step,(x,y) in enumerate(loader,1):
                x,y=x.to(device),y.to(device);opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device,enabled=device=='cuda',dtype=torch.float16):
                    out=model(x,y);logits,pfi_logits,pfi_targets,self_scores,other_scores,global_feature,masks,_=out
                    loss=ce(logits,y)+2*ce(pfi_logits,pfi_targets)+F.margin_ranking_loss(self_scores,other_scores,torch.ones_like(self_scores),margin=.05)+dynamic(x,[global_feature,masks])
                if not torch.isfinite(loss):raise RuntimeError('Non-finite FET loss')
                scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),5);scaler.step(opt);scaler.update();losses.append(loss.item())
                if step%25==0 or step==1:print(json.dumps({'epoch':epoch,'step':step,'steps':len(loader),'loss':loss.item(),'seconds':time.perf_counter()-start}),flush=True)
                if a.smoke:break
            model.eval();correct=total=0
            with torch.inference_mode():
                for x,y in vl:
                    with torch.autocast(device_type=device,enabled=device=='cuda',dtype=torch.float16):z=model(x.to(device))
                    correct+=int((z.argmax(1).cpu()==y).sum());total+=len(y)
                    if a.smoke:break
            info={'epoch':epoch,'loss':float(np.mean(losses)),'validation_crop_accuracy':correct/total,'validation_crops':total,'seconds':time.perf_counter()-start};log.write(json.dumps(info)+'\n');log.flush();print(json.dumps(info),flush=True)
            if correct/total>best:best=correct/total;bestepoch=epoch;torch.save(model.state_dict(),run/'best_model.pt')
    torch.save(model.state_dict(),run/'final_model.pt')
    metadata={'model':'FET-FGVC','author_revision':(ROOT/'third_party/FET_source_revision.txt').read_text(),'backbone':'Swin-B 224 22kto1k','pretrain_sha256':sha(ROOT/'weights/swin/swin_base_patch4_window7_224_22kto1k.pth'),'global_local':True,'pairwise_feature_interaction':True,'dynamic_pruning':True,'teacher_distillation':False,'seed':123,'batch_size':4,'epochs':epochs,'balanced_sampling':'2 classes x 2 instances, 399 steps/epoch (sampling with replacement across batches)','train_crops':len(train),'validation_selection_crops':len(valid),'best_validation_crop_accuracy':best,'best_epoch':bestepoch,'best_model_sha256':sha(run/'best_model.pt'),'elapsed_seconds':time.perf_counter()-start,'status':'smoke only' if a.smoke else 'trained','adaptations':'8 classes, 224 square padding, AMP, batch 4, own data adapter, no teacher distillation, corrected graph offsets separating images; not full paper benchmark reproduction'}
    write(run/'metadata.json',metadata)
    if a.smoke:return
    model.load_state_dict(torch.load(run/'best_model.pt',map_location=device,weights_only=True));model.eval();transform=vd.transform;scores=[];jj=jobs()
    with torch.inference_mode():
        for st in range(0,len(jj),4):
            xs=[]
            for j in jj[st:st+4]:
                with Image.open(j['image']) as im:xs.append(transform(square(crop(im.convert('RGB'),j['bbox']))))
            with torch.autocast(device_type=device,enabled=device=='cuda',dtype=torch.float16):z=model(torch.stack(xs).to(device))
            scores.extend(z.float().softmax(1).cpu().tolist())
    publish('fet_fgvc',scores,metadata)
if __name__=='__main__':main()
