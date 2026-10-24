"""训练普通ResNet18或共享骨干Bilinear CNN；人工框分类只作诊断。"""
from __future__ import annotations
import argparse,csv,json,random,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageEnhance
import paddle
from paddle import nn
from paddle.vision.models import resnet18

ROOT=Path(__file__).resolve().parents[1]; REPORTS=ROOT/'reports'; WEIGHTS=ROOT/'weights'
CLASSES=json.loads((ROOT/'configs/class_mapping.json').read_text(encoding='utf-8'))['classes']
NCLASS=len(CLASSES); MEAN=np.array([.485,.456,.406],np.float32);STD=np.array([.229,.224,.225],np.float32)

def read_rows(path,part):
    with path.open(encoding='utf-8-sig',newline='') as f: rows=[r for r in csv.DictReader(f) if r['split']==part]
    if part=='train':rows=[r for r in rows if r.get('use_for_classifier_train')=='True']
    else:rows=[r for r in rows if r.get('review_pending')!='True']
    return rows
def pad_resize(im,size=224):
    w,h=im.size;scale=size/max(w,h);nw=max(1,round(w*scale));nh=max(1,round(h*scale));im=im.resize((nw,nh),Image.Resampling.BILINEAR)
    canvas=Image.new('RGB',(size,size),(114,114,114));canvas.paste(im,((size-nw)//2,(size-nh)//2));return canvas
def augment_crop(im):
    # 小幅缩放/平移模拟预测框边缘误差；固定15%背景外扩使鸟体仍留在裁剪内。
    w,h=im.size;scale=random.uniform(.95,1.05);nw=max(1,round(w*scale));nh=max(1,round(h*scale));im=im.resize((nw,nh),Image.Resampling.BILINEAR)
    canvas=Image.new('RGB',(max(w,nw),max(h,nh)),(114,114,114));canvas.paste(im,((canvas.width-nw)//2+random.randint(-max(1,w//30),max(1,w//30)),(canvas.height-nh)//2+random.randint(-max(1,h//30),max(1,h//30))))
    cx=(canvas.width-w)//2;cy=(canvas.height-h)//2;im=canvas.crop((cx,cy,cx+w,cy+h))
    if random.random()<.5:im=im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if random.random()<.25:im=ImageEnhance.Brightness(im).enhance(random.uniform(.92,1.08))
    return im
class CropSet(paddle.io.Dataset):
    def __init__(self,rows,train,size):self.rows=rows;self.train=train;self.size=size
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        with Image.open(r['crop_path']) as fp: im=fp.convert('RGB')
        if self.train:im=augment_crop(im)
        im=pad_resize(im,self.size);x=np.asarray(im,dtype=np.float32)/255.;x=(x-MEAN)/STD;x=np.transpose(x,(2,0,1)).copy()
        return x,np.int64(r['class_id'])

class BirdClassifier(nn.Layer):
    def __init__(self,bilinear=False,weight_path=None):
        super().__init__();self.bilinear=bilinear
        self.backbone=resnet18(pretrained=False,num_classes=0,with_pool=False)
        if weight_path and Path(weight_path).exists():
            state=paddle.load(str(weight_path));state={k:v for k,v in state.items() if not k.startswith('fc.')}
            self.backbone.set_dict(state)
        if bilinear:
            self.head=nn.Linear(512*512,NCLASS)
        else:
            self.pool=nn.AdaptiveAvgPool2D((1,1));self.head=nn.Linear(512,NCLASS)
    def forward(self,x):
        b=self.backbone;x=self.backbone.conv1(x);x=self.backbone.bn1(x);x=self.backbone.relu(x);x=self.backbone.maxpool(x)
        x=self.backbone.layer1(x);x=self.backbone.layer2(x);x=self.backbone.layer3(x);x=self.backbone.layer4(x)
        if self.bilinear:
            # 空间位置逐点通道外积后求均值；signed sqrt + L2归一化压缩动态范围。
            b,c,h,w=x.shape;f=x.reshape([b,c,h*w]);g=paddle.bmm(f,f.transpose([0,2,1]))/float(h*w)
            g=paddle.sign(g)*paddle.sqrt(paddle.abs(g)+1e-8);g=g.reshape([b,c*c]);g=nn.functional.normalize(g,p=2,axis=1)
            return self.head(g)
        return self.head(self.pool(x).reshape([x.shape[0],512]))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['resnet18','bilinear'],required=True)
    ap.add_argument('--epochs',type=int,default=60);ap.add_argument('--batch-size',type=int,default=16);ap.add_argument('--seed',type=int,default=123)
    ap.add_argument('--max-steps',type=int,default=0);ap.add_argument('--size',type=int,default=224);ap.add_argument('--lr',type=float,default=.0003)
    ap.add_argument('--run-name',default='classifier_smoke');ap.add_argument('--device',choices=['auto','cpu','gpu'],default='auto');args=ap.parse_args()
    random.seed(args.seed);np.random.seed(args.seed);paddle.seed(args.seed)
    chosen=args.device
    if chosen=='auto':chosen='gpu' if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count()>0 else 'cpu'
    paddle.set_device(chosen)
    manifest=REPORTS/'classification_crop_manifest.csv';train=CropSet(read_rows(manifest,'train'),True,args.size)
    valid=CropSet(read_rows(manifest,'val'),False,args.size)
    if not len(train):raise RuntimeError('训练裁剪为空；请先运行prepare_data.py并完成所需标签确认')
    loader=paddle.io.DataLoader(train,batch_size=args.batch_size,shuffle=True,drop_last=False,num_workers=0)
    valid_loader=paddle.io.DataLoader(valid,batch_size=args.batch_size,shuffle=False,drop_last=False,num_workers=0) if len(valid) else None
    model=BirdClassifier(args.model=='bilinear',WEIGHTS/'resnet18.pdparams');model.train()
    optimizer=paddle.optimizer.AdamW(learning_rate=args.lr,parameters=model.parameters(),weight_decay=.0001)
    run=ROOT/'runs'/args.run_name;run.mkdir(parents=True,exist_ok=True);log=run/'training.jsonl'
    start=time.perf_counter();step=0;best_accuracy=-1.0;best_epoch=None
    with log.open('w',encoding='utf-8') as f:
        for epoch in range(args.epochs):
            losses=[]
            for x,y in loader:
                logits=model(x);loss=nn.functional.cross_entropy(logits,y);loss.backward();optimizer.step();optimizer.clear_grad();step+=1;loss_value=float(loss.numpy().item());losses.append(loss_value)
                if step==1 or step%10==0:f.write(json.dumps({'epoch':epoch+1,'step':step,'train_loss':loss_value,'elapsed_seconds':time.perf_counter()-start},ensure_ascii=False)+'\n');f.flush()
                if args.max_steps and step>=args.max_steps:break
            if valid_loader is not None and not args.max_steps:
                model.eval();right=total=0
                with paddle.no_grad():
                    for vx,vy in valid_loader:
                        vp=model(vx).argmax(axis=1);right+=int((vp==vy).astype('int64').sum().numpy().item());total+=int(vy.shape[0])
                accuracy=right/total if total else 0.0
                f.write(json.dumps({'epoch':epoch+1,'validation_crop_accuracy_diagnostic':accuracy,'validation_crop_count':total,'elapsed_seconds':time.perf_counter()-start},ensure_ascii=False)+'\n');f.flush()
                if accuracy>best_accuracy:
                    best_accuracy=accuracy;best_epoch=epoch+1;paddle.save(model.state_dict(),str(run/'best_model.pdparams'))
                model.train()
            if args.max_steps and step>=args.max_steps:break
        paddle.save(model.state_dict(),str(run/'model_final.pdparams'))
    meta={'architecture':'Bilinear CNN: shared ResNet18 feature map, spatial mean of per-location channel outer products, signed-sqrt, L2 norm, linear 512^2-to-8 head' if args.model=='bilinear' else 'PaddleVision ResNet18 ImageNet-pretrained backbone + linear 8-class head',
        'model':args.model,'initial_weights':str(WEIGHTS/'resnet18.pdparams'),'seed':args.seed,'epochs_requested':args.epochs,
        'steps_run':step,'batch_size':args.batch_size,'input_size':args.size,'learning_rate':args.lr,'train_crops':len(train),'validation_crops_diagnostic_only':len(valid),
        'training_seconds':time.perf_counter()-start,'device':paddle.get_device(),'best_validation_crop_accuracy':None if best_accuracy<0 else best_accuracy,
        'best_epoch_by_validation_crop_accuracy':best_epoch,'validation_limit':'人工框裁剪分类诊断；端到端评价必须使用检测预测框。',
        'status':'smoke only; not a formal model' if args.max_steps else 'training run'}
    (run/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(meta,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
