"""任意图片目录两阶段推理；检测只输出bird，分类器给每个预测框八类分数。"""
from __future__ import annotations
import argparse,csv,json,sys,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import paddle

ROOT=Path(__file__).resolve().parents[1]; CFG=json.loads((ROOT/'configs/paths.json').read_text(encoding='utf-8'))
PD=Path(CFG['first_generation'])/'third_party/PaddleDetection'; sys.path.insert(0,str(PD))
sys.path.insert(0,str(ROOT/'scripts'))
from train_classifier import BirdClassifier,pad_resize,MEAN,STD
from ppdet.core.workspace import load_config
from ppdet.engine import Trainer

EXTS={'.jpg','.jpeg','.png','.bmp','.webp'}
def images_in(path):return sorted(p for p in path.rglob('*') if p.suffix.lower() in EXTS)
def crop_box(path,box,margin=.15):
    with Image.open(path) as im: im=im.convert('RGB');w,h=im.size
    x1,y1,x2,y2=box;dx=(x2-x1)*margin;dy=(y2-y1)*margin
    rect=(max(0,int(x1-dx)),max(0,int(y1-dy)),min(w,int(x2+dx)),min(h,int(y2+dy)))
    with Image.open(path) as im:return im.convert('RGB').crop(rect)
def clip_xyxy(box,width,height):
    """候选框以原图像素坐标保存；超界部分裁回图内，退化框不送分类器。"""
    x1,y1,x2,y2=map(float,box)
    clipped=[min(max(x1,0.0),float(width)),min(max(y1,0.0),float(height)),
             min(max(x2,0.0),float(width)),min(max(y2,0.0),float(height))]
    if clipped[2]<=clipped[0] or clipped[3]<=clipped[1]:return None
    return clipped
def classifier_tensor(im,size=224):
    im=pad_resize(im,size);x=np.asarray(im,dtype=np.float32)/255.;x=(x-MEAN)/STD
    return paddle.to_tensor(x.transpose(2,0,1)[None].copy())
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--image-dir',type=Path,required=True);ap.add_argument('--detector-config',type=Path,default=ROOT/'configs/ppyoloe_plus_bird_640.yml')
    ap.add_argument('--detector-weights',type=Path,required=True);ap.add_argument('--classifier-weights',type=Path,required=True)
    ap.add_argument('--classifier',choices=['resnet18','bilinear'],default='resnet18');ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--proposals-json',type=Path,default=None,help='可复用同一检测器已保存的预测框，保证CNN与Bilinear CNN比较时框完全相同')
    ap.add_argument('--display-threshold',type=float,default=.5);ap.add_argument('--device',default='gpu')
    ap.add_argument('--max-detections',type=int,default=100,help='每图按检测分数保留上限；与COCO评估maxDets=100对齐')
    args=ap.parse_args()
    if not args.image_dir.exists():raise FileNotFoundError(args.image_dir)
    ims=images_in(args.image_dir)
    if not ims:raise RuntimeError(f'目录中没有可推理图片: {args.image_dir}')
    args.output_dir.mkdir(parents=True,exist_ok=True);(args.output_dir/'visualizations').mkdir(exist_ok=True)
    paddle.set_device(args.device if args.device=='cpu' or paddle.is_compiled_with_cuda() else 'cpu')
    started=time.perf_counter();proposal_by_path={}
    if args.proposals_json:
        proposal_doc=json.loads(args.proposals_json.read_text(encoding='utf-8'))
        for rec in proposal_doc['images']:
            proposal_by_path[str(Path(rec['image']).resolve())]=rec['detections']
        missing=[str(p) for p in ims if str(p.resolve()) not in proposal_by_path]
        if missing:raise ValueError(f'proposal file misses input images: {missing[:3]}')
    else:
        cfg=load_config(str(args.detector_config));cfg.weights=str(args.detector_weights)
        trainer=Trainer(cfg,mode='test');trainer.load_weights(str(args.detector_weights));outs=trainer.predict([str(p) for p in ims],save_results=False,visualize=False)
        id2path=trainer.dataset.get_imid2path()
        for batch in outs:
            arr=np.asarray(batch['bbox']);imids=np.asarray(batch['im_id']).reshape(-1);counts=np.asarray(batch['bbox_num']).reshape(-1)
            offset=0
            for imid,count in zip(imids,counts):
                path=Path(id2path[int(imid)]).resolve();rows=arr[offset:offset+int(count)];offset+=int(count)
                proposal_by_path[str(path)]=[{'bbox_xyxy':[float(v) for v in row[2:6]],'detector_score':float(row[1])} for row in rows]
    model=BirdClassifier(args.classifier=='bilinear',ROOT/'weights/resnet18.pdparams');model.set_state_dict(paddle.load(str(args.classifier_weights)));model.eval()
    class_list=json.loads((ROOT/'configs/class_mapping.json').read_text(encoding='utf-8'))['classes'];names=[x['name'] for x in class_list]
    results=[];flat=[];proposal_records=[]
    with paddle.no_grad():
        for path0 in ims:
            path=path0.resolve();raw=proposal_by_path.get(str(path),[])
            # 官方COCO AP按每图最多100个检测评估；分类及导出也统一采用该上限。
            raw=sorted(raw,key=lambda x:float(x['detector_score']),reverse=True)[:args.max_detections]
            with Image.open(path) as im:width,height=im.size
            stored=[];pending=[]
            for prop in raw:
                det_score=float(prop['detector_score'])
                # 复用proposal时保留第一阶段记录的原始坐标；分类和评估统一使用裁边框。
                raw_xyxy=[float(v) for v in prop.get('raw_bbox_xyxy',prop['bbox_xyxy'])]
                xyxy=clip_xyxy(prop['bbox_xyxy'],width,height)
                if xyxy is None:
                    stored.append({'bbox_xyxy':raw_xyxy,'raw_bbox_xyxy':raw_xyxy,'detector_score':det_score,'valid_after_clipping':False})
                    continue
                stored.append({'bbox_xyxy':xyxy,'raw_bbox_xyxy':raw_xyxy,'detector_score':det_score,
                               'valid_after_clipping':True,'box_clipped':xyxy!=raw_xyxy})
                pending.append({'tensor':classifier_tensor(crop_box(path,xyxy)),'xyxy':xyxy,'detector_score':det_score,
                                'raw_xyxy':raw_xyxy,'box_clipped':xyxy!=raw_xyxy})
            # 每张图的候选鸟框分批推理，减少逐框调用造成的GPU启动开销。
            for start in range(0,len(pending),64):
                chunk=pending[start:start+64]
                batch=paddle.concat([x['tensor'] for x in chunk],axis=0)
                scores_batch=paddle.nn.functional.softmax(model(batch),axis=1).numpy()
                for item,scores in zip(chunk,scores_batch):
                    pred_id=int(scores.argmax());class_score=float(scores[pred_id]);det_score=item['detector_score']
                    flat.append({'source_image':str(path),'filename':path.name,'bbox_xyxy':item['xyxy'],'detector_score':det_score,
                      'classifier_scores':{names[i]:float(scores[i]) for i in range(len(names))},'classifier_top_score':class_score,
                      'class_id':pred_id,'class_name':names[pred_id],'final_score_product':det_score*class_score,
                      'raw_bbox_xyxy':item['raw_xyxy'],'box_clipped':item['box_clipped'],
                      'displayed':det_score>args.display_threshold,'display_rule':'检测分数严格大于阈值；分类分数独立列示'})
            proposal_records.append({'image':str(path),'filename':path.name,'width':width,'height':height,'detections':stored})
    bypath={str(p.resolve()):[] for p in ims}
    for d in flat:bypath[d['source_image']].append(d)
    for p in ims:
        dets=bypath[str(p.resolve())];results.append({'image':str(p.resolve()),'filename':p.name,'width':Image.open(p).width,'height':Image.open(p).height,
          'candidates':dets,'displayed':[d for d in dets if d['displayed']],'hidden':[d for d in dets if not d['displayed']]})
        with Image.open(p) as src: canvas=src.convert('RGB')
        draw=ImageDraw.Draw(canvas)
        for d in results[-1]['displayed']:
            x1,y1,x2,y2=d['bbox_xyxy'];draw.rectangle((x1,y1,x2,y2),outline=(255,60,20),width=max(2,canvas.width//500))
            draw.text((x1,max(0,y1-18)),f"{d['class_name']} det={d['detector_score']:.3f} cls={d['classifier_top_score']:.3f}",fill=(255,60,20))
        canvas.save(args.output_dir/'visualizations'/f'{p.stem}.jpg',quality=92)
    elapsed=time.perf_counter()-started
    payload={'metadata':{'detector_config':str(args.detector_config),'detector_weights':str(args.detector_weights),'classifier':args.classifier,
      'classifier_weights':str(args.classifier_weights),'input_size_detector':[640,640],'classifier_input_size':[224,224],
      'detector_score_threshold_for_display':args.display_threshold,'comparison':'strict >','max_detections_per_image':args.max_detections,
      'final_score_rule':'detector_score * classifier_top_score; ranking aid, not calibrated probability',
      'detector_nms':'one-class PP-YOLOE+ official MultiClassNMS, configured IoU 0.7','elapsed_seconds':elapsed,'image_count':len(ims),
      'inference_device':paddle.get_device(),'truth_used_for_prediction':False,'proposals_reused':bool(args.proposals_json),
      'proposals_source':str(args.proposals_json) if args.proposals_json else 'current PP-YOLOE+ run'},'images':results}
    proposal_payload={'metadata':{'detector_config':str(args.detector_config),'detector_weights':str(args.detector_weights),'nms_iou':.7,'one_class':'bird','proposal_count':sum(len(r['detections']) for r in proposal_records)},'images':proposal_records}
    (args.output_dir/'proposals.json').write_text(json.dumps(proposal_payload,ensure_ascii=False,indent=2),encoding='utf-8')
    (args.output_dir/'predictions.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    with (args.output_dir/'predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['source_image','filename','x1','y1','x2','y2','detector_score','classifier_top_score','class_name','final_score_product','displayed','classifier_scores']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for d in flat:w.writerow({'source_image':d['source_image'],'filename':d['filename'],'x1':d['bbox_xyxy'][0],'y1':d['bbox_xyxy'][1],'x2':d['bbox_xyxy'][2],'y2':d['bbox_xyxy'][3],
          'detector_score':d['detector_score'],'classifier_top_score':d['classifier_top_score'],'class_name':d['class_name'],'final_score_product':d['final_score_product'],'displayed':d['displayed'],'classifier_scores':json.dumps(d['classifier_scores'],ensure_ascii=False)})
    print(json.dumps({'image_count':len(ims),'detection_count':len(flat),'display_count':sum(d['displayed'] for d in flat),'elapsed_seconds':elapsed,'output':str(args.output_dir)},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
