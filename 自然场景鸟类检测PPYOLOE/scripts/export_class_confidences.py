"""逐图导出八类候选得分，并提取每个保留检测框同一候选位置的八类得分。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
import types
from pathlib import Path

import numpy as np
from batch_infer import ROOT, CLASS_NAMES, collect_images, detector


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--weights', type=Path, default=ROOT/'runs/baseline_640/best_model_ap50.pdparams')
    ap.add_argument('--config', type=Path, default=ROOT/'configs/ppyoloe_s_bird_640.yml')
    ap.add_argument('--device', choices=['gpu', 'cpu'], default='gpu')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    excluded = {'meishanque_012.jpg', 'meishanque_101.jpg'}
    images = [p.resolve() for p in collect_images(args.input) if p.name not in excluded]
    if args.limit: images = images[:args.limit]
    assert images, '没有可推理图片'
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    manifest = list(csv.DictReader((ROOT/'reports/split_manifest.csv').open(encoding='utf-8-sig')))
    splits = {r['source_image']: r['split'] for r in manifest if r['source_type'] == 'original'}
    model = detector(args.config.resolve(), args.weights.resolve(), args.device)
    head = model.model.yolo_head
    # 开启索引只增加输出，不改变模型得分、NMS阈值或模型权重。
    head.nms.return_index = True
    original_post_process = head.post_process
    captured = []

    def capture(this, head_outs, scale_factor):
        result = original_post_process(head_outs, scale_factor)
        scores = head_outs[0].numpy()
        boxes, nums, indices = result
        assert scores.shape[0] == 1 and scores.shape[1] == 8, scores.shape
        rows = boxes.numpy(); idx = indices.numpy().reshape(-1).astype(int)
        assert len(rows) == len(idx)
        vectors = []
        for row, k in zip(rows, idx):
            if int(row[0]) < 0:
                vectors.append(None); continue
            assert 0 <= k < scores.shape[2], k
            vector = scores[0, :, k]
            # 验证NMS索引确实指向该框的分类候选，避免错位导出八类得分。
            assert abs(float(vector[int(row[0])]) - float(row[1])) < 1e-5
            vectors.append(vector.tolist())
        captured.append({'class_max_scores': scores[0].max(axis=1).tolist(),
                         'candidate_indices': idx.tolist(), 'vectors': vectors})
        return result

    head.post_process = types.MethodType(capture, head)
    started = time.perf_counter()
    raw = model.predict([str(p) for p in images], output_dir=str(out/'infer_work'),
                        visualize=False, save_results=False)
    paths = model.dataset.get_imid2path()
    assert len(raw) == len(captured) == len(images)
    records = []
    for batch, data in zip(raw, captured):
        path = Path(paths[int(batch['im_id'].reshape(-1)[0])]).resolve()
        detections = []
        for row, vector, k in zip(batch['bbox'], data['vectors'], data['candidate_indices']):
            cid = int(row[0])
            if cid < 0: continue
            detections.append({'class_id': cid, 'class_name': CLASS_NAMES[cid],
                'score': float(row[1]), 'bbox_xyxy': [float(v) for v in row[2:6]],
                'candidate_index': k, 'class_scores': vector})
        kept_max = [max((d['score'] for d in detections if d['class_id']==c), default=None) for c in range(8)]
        records.append({'image': str(path), 'filename': path.name,
            'split': splits.get(path.name, '未参与训练或验证'),
            'class_max_scores_before_nms': data['class_max_scores'],
            'class_max_scores_after_nms': kept_max, 'detections': detections})
    metadata = {'weights': str(args.weights.resolve()),
        'weights_sha256': hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        'config': str(args.config.resolve()), 'image_count': len(records),
        'excluded_images': sorted(excluded), 'class_names': CLASS_NAMES,
        'input_size': 640, 'nms_iou': float(head.nms.nms_threshold),
        'model_score_threshold': float(head.nms.score_threshold),
        'detail_display_threshold': 0.25, 'device': args.device,
        'inference_seconds': time.perf_counter()-started,
        'score_definition': '逐图八类得分是每类在全图所有候选位置的NMS前最大检测得分，可来自不同位置；检测框明细的八类得分来自该框同一候选位置。独立sigmoid得分，不归一化为总和1，不是整图物种概率。'}
    payload = {'metadata': metadata, 'images': records}
    (out/'class_confidences.json').write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
