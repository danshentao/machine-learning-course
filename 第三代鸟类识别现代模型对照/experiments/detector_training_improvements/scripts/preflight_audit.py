"""正式微调前核查归档、划分、预处理、标签尺度和运行环境。"""
from __future__ import annotations
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
P=json.loads((ROOT/'paths.json').read_text(encoding='utf-8'))
ARCH=Path(P['generation2_archive_engineering']);GEN2=Path(P['generation2_live_engineering'])
PD=Path(P['paddledetection']);PREP=Path(P['prepared_images'])
EXP=ROOT/'reports';EXP.mkdir(exist_ok=True)
EXCLUDED={'dashanque_119.jpg','shanmaque_090.jpg','meishanque_012.jpg','meishanque_101.jpg'}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def root_source(n):
    p=Path(n);stem=p.stem
    if stem.endswith('_brightness'):stem=stem[:-11]
    return stem+p.suffix

def main():
    train_json=ARCH/'data/annotations/instances_train_bird.json';val_json=ARCH/'data/annotations/instances_val_bird.json'
    tr=json.loads(train_json.read_text(encoding='utf-8'));va=json.loads(val_json.read_text(encoding='utf-8'))
    train_names=[x['file_name'] for x in tr['images']];val_names=[x['file_name'] for x in va['images']]
    train_sources={root_source(n) for n in train_names};val_sources={root_source(n) for n in val_names}
    train_gt={x['id']:[] for x in tr['images']}
    for a in tr['annotations']:train_gt[a['image_id']].append(a)
    sizes=[480,512,544,576,608,640,672,704,736,768]
    per_size={}
    for size in sizes:
        short=[];areas=[]
        for im in tr['images']:
            for a in train_gt[im['id']]:
                w=a['bbox'][2]*size/im['width'];h=a['bbox'][3]*size/im['height']
                short.append(min(w,h));areas.append(w*h)
        v=np.asarray(short);ar=np.asarray(areas)
        per_size[str(size)]={'instances':len(v),'min_short_side_px':float(v.min()),
          'p05_short_side_px':float(np.percentile(v,5)),'p50_short_side_px':float(np.percentile(v,50)),
          'fraction_short_side_below_8px':float(np.mean(v<8)),
          'fraction_box_area_below_64px':float(np.mean(ar<64))}

    sys.path.insert(0,str(PD))
    from ppdet.core.workspace import load_config
    configs={}
    for p in sorted((ROOT/'configs').glob('[ABC]_phase*.yml')):
        c=load_config(str(p))
        configs[p.name]={'architecture':c.architecture,'num_classes':c.num_classes,'epochs_in_phase':c.epoch,
          'use_ema':bool(c.use_ema),'base_lr':float(c.LearningRate['base_lr']),
          'optimizer':dict(c.OptimizerBuilder.get('optimizer',{})),
          'head_keys':sorted(c.PPYOLOEHead.keys()),
          'assigner':str(c.PPYOLOEHead.get('assigner','inherited/default')),
          'train_transforms':c.TrainReader['sample_transforms'],
          'batch_transforms':c.TrainReader['batch_transforms'],
          'eval_transforms':c.EvalReader['sample_transforms']}
    excluded_presence={n:{'train_json':n in train_names,'val_json':n in val_names,
      'train_source_group':root_source(n) in train_sources,'val_source_group':root_source(n) in val_sources} for n in sorted(EXCLUDED)}
    report={'generated_from_local_files':True,'device_environment':{'python':sys.version,
      'paddle':__import__('paddle').__version__,'paddle_cuda':__import__('paddle').is_compiled_with_cuda()},
      'versioning':{'generation2_archive':str(Path(P['generation2_archive'])),'archive_best_weight':str(ARCH/'runs/detector/best_model_ap50.pdparams'),
        'initial_weight_sha256':sha(ARCH/'runs/detector/best_model_ap50.pdparams'),
        'train_annotation_sha256':sha(train_json),'val_annotation_sha256':sha(val_json),
        'class_mapping_sha256':sha(ARCH/'configs/class_mapping.json'),
        'source_split_manifest_sha256':sha(Path(P['generation2_live_engineering'])/'configs/paths.json') if False else None},
      'dataset':{'train_images_including_qualified_brightness_variants':len(tr['images']),
        'train_instances':len(tr['annotations']),'val_images':len(va['images']),'val_instances':len(va['annotations']),
        'train_source_groups':len(train_sources),'val_source_groups':len(val_sources),
        'train_val_source_overlap':sorted(train_sources&val_sources),
        'brightness_variants':sum(Path(n).stem.endswith('_brightness') for n in train_names),
        'excluded_images_presence':excluded_presence,'all_categories':tr['categories'],
        'nonbird_category_ids':sorted({a['category_id'] for a in tr['annotations']}),
        'split_manifest':'Generation2 COCO train/val files copied into immutable generation2 archive; source assignment reused; no reshuffle.'},
      'target_pixel_distribution_after_stretch_resize':per_size,
      'effective_official_configs':configs,
      'coordinate_and_preprocess_review':{'COCO_bbox':'[x,y,width,height] converts from unchanged XML endpoints; no +/-1 adjustment; archived conversion pre-existing and reused.',
        'official_resize':'PP-YOLOE+ Reader uses independent x/y resize (keep_ratio=false) to 640; batch random resize also uses keep_ratio=false and Resize updates boxes and scale_factor.',
        'normalization':'New A/B/C train readers use raw 0..255 pixels (mean=0,std=1,is_scale=false,norm_type=none) then HWC-to-CHW Permute, matching inherited PP-YOLOE+ Eval/Test readers. Historical Gen2 finetune TrainReader used ImageNet mean/std and /255 while Eval/Test used raw pixels; the new groups jointly correct this mismatch, so comparisons against the untouched original baseline include this common correction.',
        'assigner_loss':'PP-YOLOE+ head/TaskAlignedAssigner and Varifocal classification loss are inherited from pinned official config; no custom label assignment/loss patch.',
        'prediction_coordinates':'official PaddleDetection predictor applies scale_factor when mapping detector outputs back to source image pixels; current generation-3 640 inference replay verified max box delta 0.'},
      'notes':['Existing correct split and exclusions are retained.','Low-confidence unmatched training predictions are review candidates only; this experiment does not mine them as negatives.','Teacher test images are not opened. DINOv3 is not loaded or launched.']}
    split=Path(json.loads((GEN2/'configs/paths.json').read_text(encoding='utf-8'))['first_generation_split_manifest'])
    report['versioning']['source_split_manifest_sha256']=sha(split)
    out=EXP/'preflight_audit.json';out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(out),'train_images':len(tr['images']),'train_instances':len(tr['annotations']),
      'val_images':len(va['images']),'val_instances':len(va['annotations']),
      'train_val_source_overlap':len(train_sources&val_sources),'configs_loaded':len(configs),
      'excluded_present':{k:v['train_json'] or v['val_json'] for k,v in excluded_presence.items()},
      '640_p05_box_short_side':per_size['640']['p05_short_side_px'],'480_p05_box_short_side':per_size['480']['p05_short_side_px']},ensure_ascii=False,indent=2))
    if train_sources&val_sources:raise SystemExit('检测到来源组跨训练/验证')
    if len(configs)!=6:raise SystemExit('有训练配置未能加载')
    if any(v['train_json'] or v['val_json'] for v in excluded_presence.values()):raise SystemExit('排除图片混入派生COCO数据')

if __name__=='__main__':main()
