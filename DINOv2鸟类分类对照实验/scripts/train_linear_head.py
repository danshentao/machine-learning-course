"""固定DINOv2特征上训练八类线性头，逐轮保存可恢复状态。"""
from __future__ import annotations
import csv, json, os, random, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(ROOT/'cache'/'matplotlib');Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True,exist_ok=True)
CFG=json.loads((ROOT/'configs'/'experiment.json').read_text(encoding='utf-8'))

def main(resume=True):
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(4); random.seed(CFG['seed']);np.random.seed(CFG['seed']);torch.manual_seed(CFG['seed']);torch.cuda.manual_seed_all(CFG['seed'])
    assert torch.cuda.is_available(), '线性头训练要求CUDA'
    device='cuda'; run=ROOT/'runs'/'dinov2'; run.mkdir(parents=True,exist_ok=True)
    data=np.load(run/'features_train_val.npz'); feat=torch.from_numpy(data['features']).to(device)
    with (ROOT/'data'/'train_crops.csv').open(encoding='utf-8-sig',newline='') as f: tr=list(csv.DictReader(f))
    with (ROOT/'data'/'val_crops.csv').open(encoding='utf-8-sig',newline='') as f: va=list(csv.DictReader(f))
    mapping=json.loads((ROOT/'data'/'class_mapping.json').read_text(encoding='utf-8-sig')); names=[x['name'] for x in mapping['classes']]
    y=torch.tensor([int(r['class_id']) for r in tr],device=device); vy=torch.tensor([int(r['class_id']) for r in va],device=device)
    x=feat[:len(tr)];vx=feat[len(tr):len(tr)+len(va)]
    assert x.shape[0]==len(y) and vx.shape[0]==len(vy) and x.shape[1]==768
    head=torch.nn.Linear(x.shape[1],len(names)).to(device)
    opt=torch.optim.AdamW(head.parameters(),lr=float(CFG['head_learning_rate']),weight_decay=float(CFG['weight_decay']))
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=int(CFG['epochs']))
    best=-1.0;best_epoch=None;start=time.perf_counter();logs=run/'training.jsonl';first=1
    checkpoint=run/'last_checkpoint.pt'
    if resume and checkpoint.exists():
        st=torch.load(checkpoint,map_location=device,weights_only=False)
        head.load_state_dict(st['head']);opt.load_state_dict(st['optimizer']);scheduler.load_state_dict(st['scheduler'])
        best=st['best_accuracy'];best_epoch=st['best_epoch'];first=st['epoch']+1
        if st.get('torch_rng'): torch.set_rng_state(st['torch_rng'].cpu())
        if st.get('cuda_rng'): torch.cuda.set_rng_state_all(st['cuda_rng'])
    mode='a' if first>1 else 'w'
    with logs.open(mode,encoding='utf-8') as log:
        for epoch in range(first,int(CFG['epochs'])+1):
            ep=time.perf_counter();head.train();order=torch.randperm(len(x),device=device);losses=[]
            for ids in order.split(int(CFG['head_batch_size'])):
                loss=F.cross_entropy(head(x[ids]),y[ids]);opt.zero_grad(set_to_none=True);loss.backward();opt.step();losses.append(float(loss.detach()))
            head.eval()
            with torch.inference_mode(): val_acc=float((head(vx).argmax(1)==vy).float().mean())
            rec={'epoch':epoch,'loss':float(np.mean(losses)),'validation_crop_accuracy':val_acc,'learning_rate':opt.param_groups[0]['lr'],'seconds':time.perf_counter()-ep,'total_seconds':time.perf_counter()-start,'nan':bool(not np.isfinite(losses).all())}
            if rec['nan']: raise FloatingPointError(f'NaN loss: {rec}')
            log.write(json.dumps(rec)+'\n');log.flush();print(json.dumps(rec),flush=True)
            if val_acc>best:
                best=val_acc;best_epoch=epoch;torch.save(head.state_dict(),run/'best_head.pt')
            scheduler.step()
            torch.save({'epoch':epoch,'head':head.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'best_accuracy':best,'best_epoch':best_epoch,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'seed':CFG['seed']},checkpoint)
    torch.save(head.state_dict(),run/'final_head.pt')
    meta={'status':'completed','epochs_budget':CFG['epochs'],'last_epoch':min(int(CFG['epochs']),first-1 if first>int(CFG['epochs']) else int(CFG['epochs'])),'best_epoch':best_epoch,'best_validation_crop_accuracy':best,'train_crops':len(tr),'validation_crops':len(va),'excluded_pending_crop_source':'jiamaque_087.jpg','seed':CFG['seed'],'optimizer':'AdamW','lr':CFG['head_learning_rate'],'weight_decay':CFG['weight_decay'],'batch_size':CFG['head_batch_size'],'scheduler':'CosineAnnealingLR','backbone_frozen':True,'encoder_feature_dimension':768,'best_head_path':str(run/'best_head.pt'),'final_head_path':str(run/'final_head.pt'),'last_checkpoint_path':str(checkpoint),'elapsed_seconds':time.perf_counter()-start}
    (run/'training_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(meta,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__': main()
