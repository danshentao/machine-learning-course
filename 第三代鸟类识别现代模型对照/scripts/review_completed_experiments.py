"""Verify saved weights, re-run the 177 fixed crops, and reconcile metrics."""
from common import *
import time,copy,sys
import numpy as np
import torch
from PIL import Image

def main():
    start=time.perf_counter();device='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(4)
    jj=jobs();result={};seed()
    for name in ['bioclip2','fet_fgvc']:
        p=ROOT/'reports'/name/'固定验证集结果.json';d=json.loads(p.read_text(encoding='utf-8'));meta=json.loads((ROOT/'runs'/name/'metadata.json').read_text(encoding='utf-8'))
        logs=[json.loads(s) for s in (ROOT/'runs'/name/'training.jsonl').read_text().splitlines()]
        assert len(logs)==60 and [r['epoch'] for r in logs]==list(range(1,61))
        stored=[q for r in d['rows'] for q in r['models'][name]['displayed']]
        assert len(stored)==len(jj)==177
        if name=='bioclip2':
            checkpoint=ROOT/'runs/bioclip2/best_head.pt';assert sha(checkpoint)==meta['best_head_sha256']
            featurepath=ROOT/'runs/bioclip2/features.npz';assert sha(featurepath)==meta['feature_sha256']
            features=np.load(featurepath)['features'];assert features.shape[0]==1594+146+177
            model=torch.nn.Linear(features.shape[1],8).to(device);model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True));model.eval()
            with torch.inference_mode():scores=model(torch.tensor(features[-177:],device=device)).softmax(1).cpu().numpy()
        else:
            from fet_classifier import build,Dataset
            checkpoint=ROOT/'runs/fet_fgvc/best_model.pt';assert sha(checkpoint)==meta['best_model_sha256']
            model=build(load_pretrained=False).to(device);model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True));model.eval();transform=Dataset([]).transform;arrays=[]
            for j in jj:
                with Image.open(j['image']) as im:arrays.append(transform(square(crop(im.convert('RGB'),j['bbox']))))
            scores=[]
            with torch.inference_mode():
                for st in range(0,len(arrays),4):
                    with torch.autocast(device_type=device,enabled=device=='cuda',dtype=torch.float16):z=model(torch.stack(arrays[st:st+4]).to(device))
                    scores.extend(z.float().softmax(1).cpu().numpy())
                # Ensure one image's result is not contaminated by its batch companions.
                with torch.autocast(device_type=device,enabled=device=='cuda',dtype=torch.float16):single=model(arrays[0].unsqueeze(0).to(device))
                batch_delta=float(np.max(np.abs(single.float().softmax(1).cpu().numpy()[0]-scores[0])))
                assert batch_delta<.005,batch_delta
            scores=np.array(scores)
        old=np.array([[q['classifier_scores'][n] for n in NAMES] for q in stored]);delta=float(np.max(np.abs(scores-old)))
        top_changes=int(np.sum(scores.argmax(1)!=old.argmax(1)));assert top_changes==0 and delta<.002,(name,delta,top_changes)
        replayrows=copy.deepcopy(d['rows']);byrow=[[] for _ in replayrows]
        for j,q,s in zip(jj,stored,scores):
            q=copy.deepcopy(q);top=int(np.argmax(s));q.update(class_id=top,class_name=NAMES[top],classifier_top_score=float(s[top]),classifier_scores=dict(zip(NAMES,map(float,s))),final_score_product=float(q['detector_score']*s[top]));byrow[j['row']].append(q)
        sys.path.insert(0,str(GEN2/'scripts'));from compare_display_thresholds import inspect
        checks=[inspect(r['truth'],qs,r['expected']) for r,qs in zip(replayrows,byrow)]
        totals={k:sum(x[k] for x in checks) for k in ['TP','FP','FN','class_disagreement','no_display']}
        for k in ['TP','FP','FN']:assert totals[k]==d['summary'][k]
        assert totals['class_disagreement']==d['summary']['class_disagreement_images'] and totals['no_display']==d['summary']['no_display_images']
        def cases(test):return [r['filename'] for r in d['rows'] if test(r)]
        result[name]={'epochs_verified':60,'weight_sha256':sha(checkpoint),'best_epoch':meta['best_epoch'],'best_validation_crop_accuracy':meta['best_validation_crop_accuracy'],'prediction_replay_max_score_difference':delta,'prediction_replay_top_class_changes':top_changes,'recomputed_totals':totals,'summary':d['summary'],'disagreement_images':cases(lambda r:r['models'][name]['class_disagreement']),'no_display_images':cases(lambda r:r['models'][name]['no_display']),'spatially_matched_wrong_class':sum(x['spatially_matched_wrong_class'] for x in checks),'unmatched_displayed_boxes':sum(sum('未匹配' in q['spatial_status'] for q in x['displayed']) for x in checks)}
        if name=='fet_fgvc':result[name]['single_vs_batch_first_crop_max_difference']=batch_delta
        del model
        if device=='cuda':torch.cuda.empty_cache()
    a=json.loads((ROOT/'reports/bioclip2/固定验证集结果.json').read_text(encoding='utf-8'));b=json.loads((ROOT/'reports/fet_fgvc/固定验证集结果.json').read_text(encoding='utf-8'))
    lookup={r['filename']:r for r in b['rows']};complement=[]
    for r in a['rows']:
        other=lookup[r['filename']];x=r['models']['bioclip2'];y=other['models']['fet_fgvc']
        assert [(p['bbox_xyxy'],p['detector_score']) for p in x['displayed']]==[(p['bbox_xyxy'],p['detector_score']) for p in y['displayed']]
        if x['class_disagreement'] or y['class_disagreement']:complement.append({'filename':r['filename'],'expected':r['expected'],'bioclip2':[p['class_name'] for p in x['displayed']],'fet_fgvc':[p['class_name'] for p in y['displayed']]})
    report={'review_date':'2026-10-04','device':device,'models':result,'disagreement_comparison':complement,'elapsed_seconds':time.perf_counter()-start,'decision':'Generation3 uses BioCLIP2 and FET-FGVC only; DINOv3 reserved for conditional generation4','limits':'172 reused validation images, no independent teacher test; folder labels and existing XML unchanged; no full-candidate AP recalculated'}
    write(ROOT/'reports/第三代两方案完成情况核验.json',report)
    md='# 第三代两方案检查（2026-10-04）\n\n两种方案均完成60轮训练和固定验证集检查，权重SHA-256核验通过，177个相同检测框的重新推理未产生类别变化。第二代保持封存。DINOv3保留为第三代效果不足时的第四代候选，本次不运行。\n\n|模型|所属类别分歧图片|无框图片|Precision|Recall|F1|\n|---|---:|---:|---:|---:|---:|\n'
    for label,s in [('第二代最佳Bilinear CNN',baseline()['summary']['bilinear']['best'])]+[(n,result[n]['summary']) for n in result]:md+=f"|{label}|{s['class_disagreement_images']}|{s['no_display_images']}|{s['precision']:.4f}|{s['recall']:.4f}|{s['F1']:.4f}|\n"
    md+='\nBioCLIP 2当前最好。两种方案均有8个显示框未与人工框一对一匹配。BioCLIP 2还有2个位置匹配但类别不一致的实例，FET-FGVC为6个；不能把类别分歧张数当作所有模型错误数。5张无框图和定位漏检需另看检测器，换分类器不会消除这些问题。\n\nBioCLIP 2最佳分类头在第1轮（146个人工框验证裁剪全对）；FET-FGVC最佳在第7轮（145/146）。这些是人工框裁剪诊断，不能当作172张图的端到端准确率，也不能当作未见过的独立测试成绩。FET-FGVC训练用时约1小时28分。\n\n剩余类别分歧：\n'
    for n in result:md+='\n- '+n+'：'+', '.join(result[n]['disagreement_images'])+'。\n'
    md+='\n逐图图册：第三代对照/第三代分类模型逐图对照.html；八类得分：第三代对照/逐图分类对照.csv。\n';(ROOT/'reports/第三代两方案检查说明.md').write_text(md,encoding='utf-8')
    print(json.dumps({'models':result,'disagreement_comparison':complement},ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
