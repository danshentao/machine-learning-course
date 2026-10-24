"""冻结30%检测框和15%裁剪，比较最终与人工框验证最佳分类权重。"""
from pathlib import Path
import base64, hashlib, io, json, time
import numpy as np
import paddle
from PIL import Image
from train_classifier import BirdClassifier
from diagnose_classification_crops import crop, tensor
from compare_display_thresholds import inspect

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path('E:/机器学习课设/模型检查报告/第二代实验/30_40_50阈值比较/阈值比较完整结果.json')
OUT=Path('E:/机器学习课设/模型检查报告/第二代实验/最终与最佳分类权重对照')

def summarize(rows,model,version):
    arr=[r['models'][model][version] for r in rows]
    tp=sum(a['TP'] for a in arr);fp=sum(a['FP'] for a in arr);fn=sum(a['FN'] for a in arr)
    p=tp/(tp+fp) if tp+fp else 0;rec=tp/(tp+fn) if tp+fn else 0
    return {'images':len(rows),'GT':tp+fn,'displayed_boxes':tp+fp,'TP':tp,'FP':fp,'FN':fn,
        'precision':p,'recall':rec,'F1':2*p*rec/(p+rec) if p+rec else 0,
        'class_disagreement_images':sum(a['class_disagreement'] for a in arr),
        'no_display_images':sum(a['no_display'] for a in arr),
        'unmatched_annotation_boxes':sum(a['unmatched_annotation_boxes'] for a in arr),
        'spatially_matched_wrong_class':sum(a['spatially_matched_wrong_class'] for a in arr)}

def main():
    start=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    source=json.loads(SOURCE.read_text(encoding='utf-8'));mapping=json.loads((ROOT/'configs/class_mapping.json').read_text(encoding='utf-8'))
    names=[c['name'] for c in mapping['classes']];rows=[];jobs=[];arrays=[]
    for row in source['rows']:
        out={k:v for k,v in row.items() if k!='models'};out['models']={}
        rows.append(out)
        for m in ['resnet','bilinear']:out['models'][m]={'final':row['models'][m]['30']}
        a=row['models']['resnet']['30']['displayed'];b=row['models']['bilinear']['30']['displayed']
        assert [(d['bbox_xyxy'],d['detector_score']) for d in a]==[(d['bbox_xyxy'],d['detector_score']) for d in b]
        with Image.open(row['image_path']) as fp:image=fp.convert('RGB')
        for pi,d in enumerate(a):
            jobs.append((len(rows)-1,pi));arrays.append(tensor(crop(image,d['bbox_xyxy'],.15)))
    device='gpu' if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count()>0 else 'cpu';paddle.set_device(device)
    weights={};replay={};epochs={}
    for m,run in [('resnet','classifier_resnet_v2'),('bilinear','classifier_bilinear_v2')]:
        meta=json.loads((ROOT/'runs'/run/'metadata.json').read_text(encoding='utf-8'));epochs[m]=meta['best_epoch_by_validation_crop_accuracy']
        weights[m]={}
        for version,filename in [('final','model_final.pdparams'),('best','best_model.pdparams')]:
            path=ROOT/'runs'/run/filename
            weights[m][version]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'epoch':60 if version=='final' else epochs[m]}
            model=BirdClassifier(m=='bilinear');model.set_state_dict(paddle.load(str(path)));model.eval();results=[]
            with paddle.no_grad():
                for st in range(0,len(arrays),16):
                    results.extend(paddle.nn.functional.softmax(model(paddle.to_tensor(np.stack(arrays[st:st+16]))),axis=1).numpy().tolist())
            preds={i:[] for i in range(len(rows))};delta=0;class_changes=0
            for (ri,pi),scores in zip(jobs,results):
                old=rows[ri]['models'][m]['final']['displayed'][pi];top=int(np.argmax(scores))
                if version=='final':
                    delta=max(delta,max(abs(scores[i]-old['classifier_scores'][n]) for i,n in enumerate(names)))
                    class_changes+=names[top]!=old['class_name']
                else:
                    d={k:v for k,v in old.items() if k not in ['same_class_correct_match','different_from_folder','spatial_status','matched_truth_class','matched_truth_iou']}
                    d.update({'class_id':top,'class_name':names[top],'classifier_top_score':scores[top],
                        'classifier_scores':dict(zip(names,scores)),'final_score_product':old['detector_score']*scores[top]})
                    preds[ri].append(d)
            if version=='final':
                replay[m]={'max_score_difference':delta,'changed_top_class':class_changes}
                if class_changes or delta>.002:raise RuntimeError('Final replay mismatch '+str(replay[m]))
            else:
                for ri,r in enumerate(rows):r['models'][m]['best']=inspect(r['truth'],preds[ri],r['expected'])
            del model
    summary={m:{v:summarize(rows,m,v) for v in ['final','best']} for m in ['resnet','bilinear']}
    changes={}
    for m in ['resnet','bilinear']:
        changes[m]={'changed_class_images':[],'folder_disagreement_removed':[],'folder_disagreement_added':[],
            'correct_matches_increased':[],'correct_matches_decreased':[]}
        for r in rows:
            a=r['models'][m]['final'];b=r['models'][m]['best']
            assert [(d['candidate_index'],d['bbox_xyxy'],d['detector_score']) for d in a['displayed']]==[(d['candidate_index'],d['bbox_xyxy'],d['detector_score']) for d in b['displayed']]
            r.setdefault('class_changed',{})[m]=[d['class_name'] for d in a['displayed']]!=[d['class_name'] for d in b['displayed']]
            if r['class_changed'][m]:changes[m]['changed_class_images'].append(r['filename'])
            if a['class_disagreement'] and not b['class_disagreement']:changes[m]['folder_disagreement_removed'].append(r['filename'])
            if not a['class_disagreement'] and b['class_disagreement']:changes[m]['folder_disagreement_added'].append(r['filename'])
            if b['TP']>a['TP']:changes[m]['correct_matches_increased'].append(r['filename'])
            if b['TP']<a['TP']:changes[m]['correct_matches_decreased'].append(r['filename'])
    metadata={'source':str(SOURCE),'detector_display_threshold_strictly_greater':.3,'crop_margin':.15,
        'dedup':'same frozen IoU>=0.5 highest detector-score groups','classifier_input_size':224,
        'weights':weights,'final_prediction_replay':replay,'device':device,'elapsed_seconds':time.perf_counter()-start,
        'best_selection':'existing best_model selected by accuracy on 146 eligible manual validation crops, not end-to-end selection',
        'metric_scope':'P/R/F1 and displayed-class disagreement on 172 fixed validation images; no full candidate AP calculated',
        'limits':'探索使用已参与权重选择的验证集，不是独立测试成绩；未修改标签、原始预测或正式模型配置。'}
    (OUT/'最终与最佳权重比较结果.json').write_text(json.dumps({'metadata':metadata,'summary':summary,'changes':changes,'rows':rows},ensure_ascii=False,indent=2),encoding='utf-8')
    for r in rows:
        with Image.open(r['image_path']) as fp:
            image=fp.convert('RGB');image.thumbnail((1400,1400));buf=io.BytesIO();image.save(buf,format='JPEG',quality=85)
        r['image']='data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode('ascii')
    template=(ROOT/'scripts/classifier_checkpoint_template.html').read_text(encoding='utf-8')
    payload=json.dumps({'metadata':metadata,'summary':summary,'changes':changes,'rows':rows},ensure_ascii=False).replace('<','\\u003c')
    (OUT/'最终权重与最佳权重对照.html').write_text(template.replace('__PAYLOAD__',payload),encoding='utf-8')
    print(json.dumps({'output':str(OUT),'summary':summary,'changes':{m:{k:len(v) for k,v in ds.items()} for m,ds in changes.items()},'replay':replay,'elapsed_seconds':metadata['elapsed_seconds']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
