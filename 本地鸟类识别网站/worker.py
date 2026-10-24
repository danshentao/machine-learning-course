"""Separate Paddle and Torch processes; all inputs are app-generated jobs."""
import argparse, json, os, sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from settings import *

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def save(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

def paddle_setup():
    handles = []
    # 共享嵌入式运行时的 RF-DETR local_packages 含 NumPy 2 构建的 pyarrow，
    # 不能与旧 Paddle 环境的 NumPy 1.26混用；Paddle推理先隔离该扩展包目录。
    rf_local = str(RF_PACKAGE.resolve()).casefold()
    sys.path[:] = [p for p in sys.path if str(Path(p or '.').resolve()).casefold() != rf_local]
    sys.path.insert(0, str(PADDLE_SITE_PACKAGES))
    site = PADDLE_SITE_PACKAGES / 'nvidia'
    for directory in site.glob('*/bin'):
        existing = os.environ.get('PATH', '').split(os.pathsep)
        if str(directory).casefold() not in {p.casefold() for p in existing}:
            os.environ['PATH'] = str(directory)+os.pathsep+os.environ.get('PATH','')
        if hasattr(os, 'add_dll_directory'):
            handles.append(os.add_dll_directory(str(directory)))
    sys.path.insert(0, str(FIRST/'third_party/PaddleDetection'))
    import paddle, cv2
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    # Paddle 与 Torch 分进程加载各自的 CUDA/cuDNN，避免混用动态库。
    requested = os.environ.get('BIRD_PADDLE_DEVICE', 'auto').lower()
    if requested not in {'auto', 'cpu', 'gpu'}:
        raise ValueError('BIRD_PADDLE_DEVICE 必须为 auto、cpu 或 gpu。')
    available = paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
    if requested == 'gpu' and not available:
        raise RuntimeError('已指定 Paddle GPU 推理，但当前没有可用 CUDA 设备。')
    device = 'gpu:0' if available and requested != 'cpu' else 'cpu'
    paddle.set_device(device)
    print('Paddle inference device: ' + paddle.get_device(), flush=True)
    return paddle, handles

def detect(job):
    paddle, handles = paddle_setup()
    import yaml, numpy as np
    from PIL import Image
    from ppdet.core.workspace import load_config
    from ppdet.engine import Trainer
    source = FIRST/'configs/ppyoloe_s_bird_640.yml' if job['model']=='gen1' else SECOND/'configs/ppyoloe_plus_bird_640.yml'
    try: text=source.read_text(encoding='utf-8')
    except UnicodeDecodeError: text=source.read_text(encoding='gb18030')
    doc=yaml.safe_load(text)
    doc['_BASE_']=[str((source.parent/p).resolve()) for p in doc['_BASE_']]
    local=Path(job['directory'])/'detector_config.yml'
    local.write_text(yaml.safe_dump(doc, allow_unicode=False), encoding='utf-8')
    cfg=load_config(str(local))
    for key in ['TrainDataset','EvalDataset','TestDataset']:
        ds=cfg.get(key)
        if ds and ds.get('dataset_dir') and not Path(ds['dataset_dir']).is_absolute():
            ds['dataset_dir']=str((source.parent.parent/ds['dataset_dir']).resolve())
    cfg.worker_num=0
    # pycocotools on Windows opens JSON with the system encoding; preserve the
    # category mapping in an ASCII JSON copy without changing the old dataset.
    ds=cfg.get('TestDataset')
    if ds and ds.get('anno_path'):
        annotation=Path(ds['anno_path'])
        if not annotation.is_absolute(): annotation=Path(ds.get('dataset_dir',''))/annotation
        if annotation.is_file():
            local_annotation=Path(job['directory'])/'category_reference.json'
            local_annotation.write_text(json.dumps(json.loads(annotation.read_text(encoding='utf-8-sig')),ensure_ascii=True),encoding='ascii')
            ds['anno_path']=str(local_annotation)
    cfg.TestReader['worker_num']=0
    cfg.TestReader['batch_size']=1
    weights=FIRST_WEIGHTS if job['model']=='gen1' else DETECTOR_WEIGHTS
    cfg.weights=str(weights)
    trainer=Trainer(cfg, mode='test')
    trainer.load_weights(str(weights))
    output=trainer.predict([job['image']], output_dir=job['directory'], save_results=False, visualize=False)
    with Image.open(job['image']) as im: w,h=im.size
    candidates=[]
    for batch in output:
        for row in np.asarray(batch['bbox']):
            raw=[float(v) for v in row[2:6]]
            b=[min(max(v,0),lim) for v,lim in zip(raw,[w,h,w,h])]
            if b[2]<=b[0] or b[3]<=b[1] or not np.isfinite(row).all(): continue
            cls=int(row[0])
            if cls<0: continue
            p={'bbox_xyxy':b,'raw_bbox_xyxy':raw,'detector_score':float(row[1])}
            if job['model']=='gen1':
                if cls>=len(NAMES): continue
                p.update(class_id=cls,class_name=NAMES[cls],classifier_top_score=None,classifier_scores={})
            candidates.append(p)
    candidates=sorted(candidates,key=lambda x:-x['detector_score'])[:100]
    for i,p in enumerate(candidates): p['candidate_index']=i
    groups=grouped(candidates)
    winners=[g['winner'] for g in groups if g['winner']['detector_score']>job['threshold']]
    save(Path(job['directory'])/'proposals.json',{'width':w,'height':h,'groups':groups,'displayed':winners,
         'weights':str(weights),'config':str(source),'device':paddle.get_device()})

def detect_rfdetr(job):
    """用RF-DETR官方API推理，预测框直接是上传图的原图像素坐标。"""
    import hashlib, time
    import numpy as np
    import torch
    from PIL import Image
    sys.path.insert(0, str(RF_PACKAGE))
    from rfdetr import RFDETRSmall
    root=RF_PROJECT
    os.environ.update(RF_HOME=str(root/'weights/cache'),ROBOFLOW_HOME=str(root/'weights/cache'),
                      HF_HOME=str(root/'cache/huggingface'),HUGGINGFACE_HUB_CACHE=str(root/'cache/huggingface/hub'))
    if not torch.cuda.is_available():
        raise RuntimeError('第四代 RF-DETR 需要 CUDA；当前运行环境未检测到可用 CUDA。')
    model=RFDETRSmall.from_checkpoint(str(RF_CHECKPOINT),device='cuda')
    with Image.open(job['image']) as im: width,height=im.size
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    pred=model.predict(str(job['image']),threshold=0.001)
    elapsed=time.perf_counter()-start
    det=pred[0] if isinstance(pred,list) else pred
    boxes=np.asarray(det.xyxy,dtype=np.float32).reshape(-1,4)
    scores=np.asarray(det.confidence,dtype=np.float32).reshape(-1)
    classes=np.asarray(det.class_id,dtype=np.int64).reshape(-1)
    if not (len(boxes)==len(scores)==len(classes)):
        raise RuntimeError('RF-DETR返回的框、分数和类别数量不一致。')
    candidates=[]
    for i,(box,score,class_id) in enumerate(zip(boxes,scores,classes)):
        raw=[float(x) for x in box]
        b=[min(max(raw[j],0.0),float([width,height,width,height][j])) for j in range(4)]
        if b[2]<=b[0] or b[3]<=b[1] or not np.isfinite(score): continue
        candidates.append({'candidate_index':i,'bbox_xyxy':b,'raw_bbox_xyxy':raw,
            'detector_score':float(score),'class_id':int(class_id),'class_name':'bird','box_clipped':b!=raw})
    groups=grouped(candidates)
    winners=[g['winner'] for g in groups if g['winner']['detector_score']>job['threshold']]
    digest=hashlib.sha256(RF_CHECKPOINT.read_bytes()).hexdigest()
    save(Path(job['directory'])/'proposals.json',{'width':width,'height':height,'groups':groups,'displayed':winners,
        'weights':str(RF_CHECKPOINT),'weights_sha256':digest,'config':str(root/'configs/experiment.json'),
        'device':'cuda','model':'RF-DETR Small','backbone':'dinov2_windowed_small','input_size':512,
        'candidate_threshold':0.001,'inference_seconds':elapsed,
        'peak_cuda_memory_gib':torch.cuda.max_memory_allocated()/1024**3,
        'coordinates':'官方API返回原图像素坐标并裁到图像边界，没有进行二次缩放换算。'})

def classify_cnn(job, proposals):
    paddle, handles = paddle_setup()
    import numpy as np
    from PIL import Image
    sys.path.insert(0, str(SECOND/'scripts'))
    from train_classifier import BirdClassifier, pad_resize, MEAN, STD
    items=proposals['displayed']; arrays=[]
    with Image.open(job['image']) as fp:
        im=fp.convert('RGB')
        for p in items:
            x,y,x2,y2=p['bbox_xyxy'];dx=(x2-x)*.15;dy=(y2-y)*.15
            patch=im.crop((max(0,int(x-dx)),max(0,int(y-dy)),min(im.width,int(x2+dx)),min(im.height,int(y2+dy))))
            a=np.asarray(pad_resize(patch,224),dtype=np.float32)/255
            arrays.append(((a-MEAN)/STD).transpose(2,0,1).copy())
    keys=['resnet18','bilinear'] if job['classifier']=='both' else [job['classifier']]
    outputs={}
    for key in keys:
        model=BirdClassifier(key=='bilinear')
        model.set_state_dict(paddle.load(str(CNN_WEIGHTS[key]))); model.eval()
        scores=[]
        with paddle.no_grad():
            for st in range(0,len(arrays),16):
                scores.extend(paddle.nn.functional.softmax(model(paddle.to_tensor(np.stack(arrays[st:st+16]))),axis=1).numpy().tolist())
        outputs[key]=decorate(items,scores)
        del model
    return outputs

def decorate(items, scores):
    result=[]
    for p,s in zip(items,scores):
        cls=max(range(len(s)),key=lambda i:s[i])
        result.append({**p,'class_id':cls,'class_name':NAMES[cls], 'classifier_top_score':float(s[cls]),
                       'classifier_scores':dict(zip(NAMES,map(float,s)))})
    return result

def classify_bio(job, proposals):
    import numpy as np, torch, open_clip
    import torch.nn.functional as F
    from PIL import Image
    torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    cfg=read(BIO_WEIGHTS.parent/'open_clip_config.json')
    model,_,transform=open_clip.create_model_and_transforms('ViT-L-14',pretrained=str(BIO_WEIGHTS),
             image_mean=cfg['preprocess_cfg']['mean'],image_std=cfg['preprocess_cfg']['std'],image_interpolation='bicubic')
    visual=model.visual.to(device).eval();del model
    head=torch.nn.Linear(768,8).to(device)
    head.load_state_dict(torch.load(BIO_HEAD,map_location=device,weights_only=True));head.eval()
    patches=[]
    with Image.open(job['image']) as fp:
        im=fp.convert('RGB')
        for p in proposals['displayed']:
            x,y,x2,y2=p['bbox_xyxy'];dx=(x2-x)*.15;dy=(y2-y)*.15
            patch=im.crop((max(0,math.floor(x-dx)),max(0,math.floor(y-dy)),min(im.width,math.ceil(x2+dx)),min(im.height,math.ceil(y2+dy))))
            w,h=patch.size;s=max(w,h);square=Image.new('RGB',(s,s),(114,114,114));square.paste(patch,((s-w)//2,(s-h)//2))
            patches.append(transform(square))
    scores=[]
    with torch.inference_mode():
        for st in range(0,len(patches),8):
            batch=torch.stack(patches[st:st+8]).to(device)
            with torch.autocast(device_type=device,enabled=device=='cuda',dtype=torch.float16): features=visual(batch)
            z=F.normalize(features.float(),dim=1)
            scores.extend(head(z).softmax(1).cpu().tolist())
    key='rfdetr_bioclip2' if job['model']=='gen4' else 'bioclip2'
    return {key:decorate(proposals['displayed'],scores)}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('role',choices=['detect','detect-rf','cnn','bio']);ap.add_argument('job');args=ap.parse_args()
    job=read(args.job)
    if args.role=='detect':detect(job)
    elif args.role=='detect-rf':detect_rfdetr(job)
    else:
        prop=read(Path(job['directory'])/'proposals.json')
        outputs=classify_cnn(job,prop) if args.role=='cnn' else classify_bio(job,prop)
        save(Path(job['directory'])/'classifications.json',outputs)
