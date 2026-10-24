"""复核全量推理中固定验证集的指标，并记录八图坐标修正。"""
import json
from pathlib import Path
from collections import Counter
from evaluate_predictions import ROOT, read_truth, class_correct_counts, ap_101

def main():
    run = ROOT/'runs/confidence_audit/all_originals'
    doc = json.loads((run/'class_confidences.json').read_text(encoding='utf-8'))
    truth = read_truth(ROOT/'data/prepared/annotations/instances_val.json')
    records = {r['filename']: r for r in doc['images']}
    assert len(records) == len(doc['images']) == 966
    assert not {'meishanque_012.jpg', 'meishanque_101.jpg'} & records.keys()
    assert set(truth) <= records.keys()
    gt = {name: r['gt'] for name,r in truth.items()}
    pred = {name: records[name]['detections'] for name in truth}
    tp,fp,fn,_,_ = class_correct_counts(gt,pred,.25,.5)
    p=float(tp.sum()/(tp.sum()+fp.sum())); r=float(tp.sum()/(tp.sum()+fn.sum()))
    metrics={'mAP@0.5': sum(ap_101(gt,pred,c) for c in range(8))/8,
             'micro_precision':p, 'micro_recall':r, 'micro_F1':2*p*r/(p+r),
             'TP':int(tp.sum()),'FP':int(fp.sum()),'FN':int(fn.sum()),'validation_images':len(truth)}
    original=json.loads((ROOT/'runs/baseline_640/evaluation_ap50_epoch_13/metrics.json').read_text(encoding='utf-8'))
    corrected=json.loads((ROOT/'runs/baseline_640/input_change_check/corrected_summary/input_change_summary.json').read_text(encoding='utf-8'))
    audit={'fresh_validation_metrics':metrics,'previous_metrics_document':original,
           'image_split_counts':dict(Counter(r['split'] for r in doc['images'])),
           'excluded_invalid_images':doc['metadata']['excluded_images'],
           'corrected_eight_image_check':corrected,
           'note':'原八图汇总用小图坐标匹配原图真值，错误得到0个TP。本次从comparison.json读取已还原坐标，原图和降采样图均9个TP。'}
    (run/'verification.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    # 为已有评估器保存同一172张验证图的新预测，独立复算不混入训练图。
    (run/'validation_predictions.json').write_text(json.dumps({'metadata':doc['metadata'],
        'images':[r for r in doc['images'] if r['filename'] in truth]},ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'fresh_validation_metrics':metrics,'split_counts':audit['image_split_counts']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
