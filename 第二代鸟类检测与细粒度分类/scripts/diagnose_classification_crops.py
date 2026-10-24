"""对30%显示类别分歧做裁剪诊断，冻结权重，原标注和正式输出只读。"""
from pathlib import Path
import base64, hashlib, io, json, time
import numpy as np
import paddle
from PIL import Image
from train_classifier import BirdClassifier, pad_resize, MEAN, STD
from evaluate_two_stage import iou

ROOT=Path(__file__).resolve().parents[1]
OUT=Path('E:/机器学习课设/模型检查报告/第二代实验/30阈值错分类裁剪诊断')
REPORT=Path('E:/机器学习课设/模型检查报告/第二代实验/30_40_50阈值比较/阈值比较完整结果.json')

def crop(image,box,margin):
    x,y,x2,y2=box;dx=(x2-x)*margin;dy=(y2-y)*margin
    rect=(max(0,int(x-dx)),max(0,int(y-dy)),min(image.width,int(x2+dx)),min(image.height,int(y2+dy)))
    return image.crop(rect)

def tensor(image):
    x=np.asarray(pad_resize(image,224),dtype=np.float32)/255.;x=(x-MEAN)/STD
    return x.transpose(2,0,1).copy()

def encode(image):
    image=image.copy();image.thumbnail((1400,1400));buf=io.BytesIO();image.save(buf,format='JPEG',quality=92)
    return 'data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode('ascii')

def main():
    started=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    names=[c['name'] for c in json.loads((ROOT/'configs/class_mapping.json').read_text(encoding='utf-8'))['classes']]
    report=json.loads(REPORT.read_text(encoding='utf-8'));cases=[];jobs=[];arrays=[]
    for row in report['rows']:
        # 取两分类器类别分歧的并集；检测框候选编号一致，避免把同框列两遍。
        ids={d['candidate_index'] for model in ['resnet','bilinear'] for d in row['models'][model]['30']['displayed'] if d['different_from_folder']}
        if not ids:continue
        lookup={m:{d['candidate_index']:d for d in row['models'][m]['30']['displayed']} for m in ['resnet','bilinear']}
        with Image.open(row['image_path']) as fp:image=fp.convert('RGB')
        for candidate_id in sorted(ids):
            d=lookup['resnet'][candidate_id]
            best=max([(iou(d['bbox_xyxy'],g['bbox_xyxy']),i,g) for i,g in enumerate(row['truth'])],key=lambda x:x[0],default=(0,None,None))
            # 必须被原空间一对一匹配接纳，避免把额外背景/重复框当真鸟分类问题。
            linked=best[0]>=.5 and d['spatial_status']!='未匹配人工框（待确认误检或漏标等）'
            case={'filename':row['filename'],'image_path':row['image_path'],'folder_class':row['expected'],
                'candidate_index':candidate_id,'detector_score':d['detector_score'],'detector_box':d['bbox_xyxy'],
                'linked_annotation':best[2] if linked else None,'annotation_iou':best[0],
                'saved_predictions':{m:lookup[m][candidate_id] for m in ['resnet','bilinear']},
                'results':{},'crops':{},'image':encode(image)}
            case_index=len(cases);cases.append(case)
            for margin in [0,.05,.15,.25]:
                key=f'det_{int(margin*100)}';im=crop(image,d['bbox_xyxy'],margin)
                arrays.append(tensor(im));jobs.append((case_index,key));case['crops'][key]=encode(im)
            if linked:
                im=crop(image,best[2]['bbox_xyxy'],.15);arrays.append(tensor(im));jobs.append((case_index,'gt_15'));case['crops']['gt_15']=encode(im)
    device='gpu' if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count()>0 else 'cpu';paddle.set_device(device)
    weights={};max_delta={};summary={}
    for model_name,run in [('resnet','classifier_resnet_v2'),('bilinear','classifier_bilinear_v2')]:
        path=ROOT/'runs'/run/'model_final.pdparams';weights[model_name]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        model=BirdClassifier(model_name=='bilinear');model.set_state_dict(paddle.load(str(path)));model.eval()
        scores_all=[]
        with paddle.no_grad():
            for start in range(0,len(arrays),16):
                scores_all.extend(paddle.nn.functional.softmax(model(paddle.to_tensor(np.stack(arrays[start:start+16]))),axis=1).numpy().tolist())
        for (ci,key),scores in zip(jobs,scores_all):
            top=int(np.argmax(scores));cases[ci]['results'].setdefault(model_name,{})[key]={'class_name':names[top],'top_score':scores[top],'scores':dict(zip(names,scores))}
        max_delta[model_name]=max(max(abs(c['results'][model_name]['det_15']['scores'][n]-c['saved_predictions'][model_name]['classifier_scores'][n]) for n in names) for c in cases)
        if max_delta[model_name]>.001:raise RuntimeError(f'Existing prediction replay mismatch: {model_name} {max_delta[model_name]}')
        counts={}
        for c in cases:
            if not c['saved_predictions'][model_name]['different_from_folder']:continue
            result=c['results'][model_name];gt=c['linked_annotation']
            if not gt:category='未匹配人工鸟框，先复核是否误检或漏标'
            elif c['saved_predictions'][model_name]['class_name']==gt['class_name']:category='与实例标注一致，但与所属目录不同'
            elif result['gt_15']['class_name']==gt['class_name']:category='人工框能分对，检测框分错：优先检查裁剪'
            else:category='人工框与检测框都分错：优先检查分类器及标签'
            c.setdefault('diagnosis',{})[model_name]=category
            counts[category]=counts.get(category,0)+1
        linked_wrong=[c for c in cases if c['linked_annotation'] and c['saved_predictions'][model_name]['class_name']!=c['linked_annotation']['class_name']]
        summary[model_name]={'folder_disagreement_images':len({c['filename'] for c in cases if c['saved_predictions'][model_name]['different_from_folder']}),
            'diagnosis_counts':counts,'wrong_class_cases_with_linked_annotation':len(linked_wrong),
            'fixed_by_margin_on_selected_cases':{k:sum(c['results'][model_name][k]['class_name']==c['linked_annotation']['class_name'] for c in linked_wrong) for k in ['det_0','det_5','det_15','det_25','gt_15']}}
        del model
    doc={'metadata':{'source':str(REPORT),'device':device,'weights':weights,'elapsed_seconds':time.perf_counter()-started,
        'prediction_replay_max_score_difference':max_delta,'no_training_or_label_changes':True,
        'limits':'仅30%显示类别分歧图片的裁剪诊断；人工框仅作为原因分析；在错误子集上试裁剪不能证明全体验证提升。未匹配框不得自动视为背景。'},'summary':summary,'cases':cases}
    # JSON只存结果和路径；HTML嵌入图片以便离线查看。
    compact={**doc,'cases':[{k:v for k,v in c.items() if k not in ['image','crops']} for c in cases]}
    (OUT/'错分类裁剪诊断结果.json').write_text(json.dumps(compact,ensure_ascii=False,indent=2),encoding='utf-8')
    template=(ROOT/'scripts/classification_crop_diagnostic_template.html').read_text(encoding='utf-8')
    payload=json.dumps(doc,ensure_ascii=False).replace('<','\\u003c')
    (OUT/'错分类裁剪对照.html').write_text(template.replace('__PAYLOAD__',payload),encoding='utf-8')
    print(json.dumps({'output':str(OUT),'summary':summary,'cases':len(cases),'max_replay_difference':max_delta,'elapsed_seconds':doc['metadata']['elapsed_seconds']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
