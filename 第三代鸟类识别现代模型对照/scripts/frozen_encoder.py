"""Frozen official BioCLIP2/DINOv3 encoders, train only our eight-class head."""
from common import *
import argparse,time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['bioclip2','dinov3'],required=True);ap.add_argument('--model-path');ap.add_argument('--epochs',type=int,default=60);ap.add_argument('--batch-size',type=int,default=8);a=ap.parse_args()
    seed();torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu';start=time.perf_counter()
    run=ROOT/'runs'/a.model;run.mkdir(parents=True,exist_ok=True)
    source=a.model_path or ('hf-hub:imageomics/bioclip-2' if a.model=='bioclip2' else 'facebook/dinov3-vits16-pretrain-lvd1689m')
    if a.model=='bioclip2':
        import open_clip
        local=ROOT/'weights/bioclip2/open_clip_model.safetensors'
        if local.exists():
            assert sha(local)=='b7b2bf6fbc95799e42630e394cf95803892ab447c1a8ab629dbc82fbeaf7dfef'
            cfg=json.loads((local.parent/'open_clip_config.json').read_text())
            model,_,transform=open_clip.create_model_and_transforms('ViT-L-14',pretrained=str(local),image_mean=cfg['preprocess_cfg']['mean'],image_std=cfg['preprocess_cfg']['std'],image_interpolation='bicubic')
            source=str(local)
        else:
            model,_,transform=open_clip.create_model_and_transforms(source)
        model=model.visual.to(device).eval()
        def batch(images):return torch.stack([transform(square(im)) for im in images]).to(device)
        def encode(x):return model(x)
    else:
        if Path(source).is_file() and Path(source).suffix=='.pth':
            import sys
            from torchvision import transforms as T
            sys.path.insert(0,str(ROOT/'third_party/dinov3'))
            from dinov3.hub.backbones import dinov3_vits16
            model=dinov3_vits16(weights=source).to(device).eval()
            transform=T.Compose([T.Resize((256,256),antialias=True),T.ToTensor(),T.Normalize((.485,.456,.406),(.229,.224,.225))])
            def batch(images):return torch.stack([transform(square(im)) for im in images]).to(device)
            def encode(x):
                out=model.forward_features(x);return torch.cat([out['x_norm_clstoken'],out['x_norm_patchtokens'].mean(1)],dim=1)
        else:
            from transformers import AutoImageProcessor,AutoModel
            processor=AutoImageProcessor.from_pretrained(source);model=AutoModel.from_pretrained(source).to(device).eval()
            def batch(images):return processor(images=[square(im) for im in images],return_tensors='pt')['pixel_values'].to(device)
            def encode(x):
                out=model(pixel_values=x);return torch.cat([out.last_hidden_state[:,0],out.last_hidden_state[:,5:].mean(1)],dim=1)
    for p in model.parameters():p.requires_grad_(False)
    train,valid=manifest_rows();jj=jobs();tasks=[{'crop_path':r['crop_path']} for r in train+valid]+jj
    cached=run/'features.npz';signature=sha(MANIFEST)+sha(BASELINE)+str(source)+(sha(source) if a.model=='dinov3' and Path(source).is_file() else '')
    if cached.exists():
        loaded=np.load(cached);assert str(loaded['signature'])==signature;features=loaded['features']
    else:
        results=[]
        with torch.inference_mode():
            for st in range(0,len(tasks),a.batch_size):
                images=[]
                for task in tasks[st:st+a.batch_size]:
                    with Image.open(task.get('crop_path',task.get('image'))) as im:
                        im=im.convert('RGB');images.append(im if 'crop_path' in task else crop(im,task['bbox']))
                with torch.autocast(device_type=device,enabled=device=='cuda',dtype=torch.float16):z=encode(batch(images))
                results.extend(F.normalize(z.float(),dim=1).cpu().numpy())
                if st%80==0:print(json.dumps({'phase':'features','done':st+len(images),'total':len(tasks),'seconds':time.perf_counter()-start}),flush=True)
        features=np.stack(results);np.savez_compressed(cached,features=features,signature=signature)
    del model
    if device=='cuda':torch.cuda.empty_cache()
    x=torch.tensor(features[:len(train)],device=device);y=torch.tensor([int(r['class_id']) for r in train],device=device)
    vx=torch.tensor(features[len(train):len(train)+len(valid)],device=device);vy=torch.tensor([int(r['class_id']) for r in valid],device=device)
    head=torch.nn.Linear(x.shape[1],len(NAMES)).to(device);opt=torch.optim.AdamW(head.parameters(),lr=.02,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs);best=-1;bestepoch=None
    with (run/'training.jsonl').open('w',encoding='utf-8') as log:
        for epoch in range(1,a.epochs+1):
            head.train();order=torch.randperm(len(x),device=device);losses=[]
            for ids in order.split(128):
                loss=F.cross_entropy(head(x[ids]),y[ids]);opt.zero_grad();loss.backward();opt.step();losses.append(loss.item())
            head.eval()
            with torch.inference_mode():acc=(head(vx).argmax(1)==vy).float().mean().item()
            info={'epoch':epoch,'loss':float(np.mean(losses)),'validation_crop_accuracy':acc,'seconds':time.perf_counter()-start};log.write(json.dumps(info)+'\n');log.flush();print(json.dumps(info),flush=True)
            if acc>best:best=acc;bestepoch=epoch;torch.save(head.state_dict(),run/'best_head.pt')
            scheduler.step()
    torch.save(head.state_dict(),run/'final_head.pt');head.load_state_dict(torch.load(run/'best_head.pt',map_location=device,weights_only=True));head.eval()
    with torch.inference_mode():scores=head(torch.tensor(features[-len(jj):],device=device)).softmax(1).cpu().tolist()
    metadata={'model':a.model,'encoder_source':source,'encoder_frozen':True,'head':'linear 8-class','seed':123,'epochs':a.epochs,'best_epoch':bestepoch,'best_validation_crop_accuracy':best,'train_crops':len(train),'validation_selection_crops':len(valid),'best_head_sha256':sha(run/'best_head.pt'),'feature_sha256':sha(cached),'input_policy':'15% expansion, aspect-preserving square padding then official model preprocessing','device':device,'elapsed_seconds':time.perf_counter()-start}
    write(run/'metadata.json',metadata);publish(a.model,scores,metadata)
if __name__=='__main__':main()
