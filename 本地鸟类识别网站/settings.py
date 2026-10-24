from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
FIRST = WORK / '自然场景鸟类检测PPYOLOE'
SECOND = WORK / '第二代鸟类检测与细粒度分类'
THIRD = WORK / '第三代鸟类识别现代模型对照'
RF_PROJECT = WORK / 'RFDETR与PPYOLOE鸟检测对照'
RF_PYTHON = RF_PROJECT / 'runtime/python311-embed/python.exe'
RF_CHECKPOINT = RF_PROJECT / 'weights/rfdetr_small_bird_best.pth'
RF_PACKAGE = RF_PROJECT / 'runtime/local_packages'
ARCHIVE = WORK / '第二代结果归档/20261003_211135/工程'
PADDLE_PYTHON = RF_PYTHON
TORCH_PYTHON = RF_PYTHON
PADDLE_SITE_PACKAGES = FIRST / '.venv/Lib/site-packages'
FIRST_WEIGHTS = WORK / '第一代结果归档/模型与训练/best_model_ap50.pdparams'
DETECTOR_WEIGHTS = ARCHIVE / 'runs/detector/best_model_ap50.pdparams'
CNN_WEIGHTS = {'resnet18': ARCHIVE / 'runs/classifier_resnet_v2/best_model.pdparams',
               'bilinear': ARCHIVE / 'runs/classifier_bilinear_v2/best_model.pdparams'}
BIO_WEIGHTS = THIRD / 'weights/bioclip2/open_clip_model.safetensors'
BIO_HEAD = THIRD / 'runs/bioclip2/best_head.pt'
NAMES = ['树麻雀', '家麻雀', '山麻雀', '白头鹎', '红耳鹎', '黑喉红臀鹎', '大山雀', '煤山雀']
MODELS = {'gen1': '第一代 · PP-YOLOE-s', 'gen2': '第二代 · PP-YOLOE+ + CNN',
          'gen3': '第三代 · PP-YOLOE+ + BioCLIP 2',
          'gen4': '第四代 · RF-DETR + BioCLIP 2'}

def iou(a, b):
    area = lambda x: max(0, x[2]-x[0]) * max(0, x[3]-x[1])
    inter = max(0, min(a[2],b[2])-max(a[0],b[0])) * max(0, min(a[3],b[3])-max(a[1],b[1]))
    union = area(a)+area(b)-inter
    return inter/union if union else 0.

def grouped(candidates):
    # Only retained winners are compared: no containment or transitive merging.
    groups = []
    for item in sorted(candidates, key=lambda x: (-x['detector_score'], x['candidate_index'])):
        choices = [(iou(item['bbox_xyxy'], g['winner']['bbox_xyxy']), j) for j,g in enumerate(groups)]
        overlap, index = max(choices, default=(0., None))
        if overlap >= .5:
            groups[index]['hidden'].append({**item, 'iou_with_winner': overlap})
        else:
            groups.append({'winner': item, 'hidden': []})
    return groups
