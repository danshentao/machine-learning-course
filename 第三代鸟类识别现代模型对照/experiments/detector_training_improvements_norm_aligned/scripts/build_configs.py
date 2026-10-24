"""从归档中的单类COCO数据与官方PP-YOLOE+ s配置生成隔离实验配置。"""
from __future__ import annotations
import copy,json
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]
P=json.loads((ROOT/'paths.json').read_text(encoding='utf-8'))
ARCH=Path(P['generation2_archive_engineering']);PD=Path(P['paddledetection'])
SOURCE=ARCH/'configs/ppyoloe_plus_bird_640.yml'
WEIGHT=ARCH/'runs/detector/best_model_ap50.pdparams'
OUT=ROOT/'configs';RUNS=ROOT/'runs'
SIZES=[480,512,544,576,608,640,672,704,736,768]
# 初始权重来自归档微调，实际TrainReader用ImageNet mean/std及/255；保持该训练输入分布。
# 官方继承Eval/Test reader仍为原像素口径，和现有基准推理一致。
MEAN=[0.485,0.456,0.406];STD=[0.229,0.224,0.225]

def main():
    try:src=yaml.safe_load(SOURCE.read_text(encoding='utf-8'))
    except UnicodeDecodeError:src=yaml.safe_load(SOURCE.read_text(encoding='gb18030'))
    base=copy.deepcopy(src)
    base['_BASE_']=[str(PD/'configs/ppyoloe/ppyoloe_plus_crn_s_80e_coco.yml')]
    base.update({'metric':'COCO','num_classes':1,'epoch':40,'snapshot_epoch':5,
      'log_iter':50,'worker_num':0,'weights':None,'pretrain_weights':str(WEIGHT),
      'save_dir':'','use_gpu':True})
    base['LearningRate']={'base_lr':float(P['base_learning_rate']),'schedulers':[
      {'name':'CosineDecay','max_epochs':40},
      {'name':'LinearWarmup','start_factor':0.0,'epochs':2}]}
    base['TrainDataset'].update({'name':'COCODataSet','image_dir':'images/train',
      'anno_path':str(ARCH/'data/annotations/instances_train_bird.json'),
      'dataset_dir':P['prepared_images'],'data_fields':['image','gt_bbox','gt_class','is_crowd']})
    base['EvalDataset'].update({'name':'COCODataSet','image_dir':'images/val',
      'anno_path':str(ARCH/'data/annotations/instances_val_bird.json'),
      'dataset_dir':P['prepared_images'],'allow_empty':True})
    base['TestDataset'].update({'name':'ImageFolder',
      'anno_path':str(ARCH/'data/annotations/instances_val_bird.json'),
      'dataset_dir':P['prepared_images']})
    base['EvalReader']={'batch_size':1,'worker_num':0}
    base['TestReader']={'inputs_def':{'image_shape':[3,640,640]},'batch_size':1,'worker_num':0}
    base['PPYOLOEHead']['eval_size']=[640,640]
    base['TrainReader']={'batch_size':int(P['batch_size']),'collate_batch':True,
      'worker_num':0,'use_shared_memory':False,'shuffle':True,'drop_last':False}

    fixed=[{'Decode':{}},{'RandomFlip':{'prob':0.5}},
      {'Resize':{'target_size':[640,640],'keep_ratio':False,'interp':2}},
      {'NormalizeImage':{'mean':MEAN,'std':STD,'is_scale':True,'norm_type':'mean_std'}},{'Permute':{}}]
    fixed_batch=[{'PadGT':{}}]
    multi=[{'Decode':{}},{'RandomDistort':{'hue':[-6,6,0.5],
      'saturation':[0.85,1.15,0.5],'contrast':[0.85,1.15,0.5],
      'brightness':[0.85,1.15,0.5],'random_apply':True,'count':4,
      'random_channel':False,'prob':0.6}},
      {'RandomCrop':{'aspect_ratio':[0.75,1.33],'thresholds':[0.3,0.5,0.7],
        'scaling':[0.75,1.0],'num_attempts':20,'allow_no_crop':True,
        'cover_all_box':False,'ioumode':'iou','prob':0.35,
        'use_box_candidates':True,'wh_thr':4,'ar_thr':40,'area_thr':0.70}},
      {'RandomFlip':{'prob':0.5}}]
    multibatch=[{'BatchRandomResize':{'target_size':SIZES,'random_size':True,
       'random_interp':False,'keep_ratio':False}},
       {'NormalizeImage':{'mean':MEAN,'std':STD,'is_scale':True,'norm_type':'mean_std'}},
       {'Permute':{}},{'PadGT':{}}]
    erase={'RandomErasingImage':{'prob':0.15,'lower':0.05,'higher':0.15,'aspect_ratio':0.3}}
    for group in ['A','B','C']:
      for phase in [1,2]:
        doc=copy.deepcopy(base);doc['save_dir']=str(RUNS/group)
        doc['epoch']=30 if phase==1 else 40
        if phase==1 and group in ['B','C']:
          doc['TrainReader']['sample_transforms']=copy.deepcopy(multi)
          if group=='C':doc['TrainReader']['sample_transforms'].append(copy.deepcopy(erase))
          doc['TrainReader']['batch_transforms']=copy.deepcopy(multibatch)
        else:
          doc['TrainReader']['sample_transforms']=copy.deepcopy(fixed)
          doc['TrainReader']['batch_transforms']=copy.deepcopy(fixed_batch)
        (OUT/f'{group}_phase{phase}.yml').write_text(yaml.safe_dump(doc,allow_unicode=True,sort_keys=False),encoding='gb18030')
    (OUT/'training_design.json').write_text(json.dumps({'initialized_from':str(WEIGHT),
      'initial_weight_sha256':__import__('hashlib').sha256(WEIGHT.read_bytes()).hexdigest(),
      'fresh_optimizer':'no --resume at epoch 0; new Momentum optimizer and cosine scheduler, avoiding old near-zero LR state',
      'budget':{'epochs':40,'phase1':30,'phase2':10,'checkpoint_interval':5,'batch_size':2,'input_final':640,'seed':123,
        'base_learning_rate':P['base_learning_rate'],'ema':'inherited from official PP-YOLOE+ s config'},
      'groups':{'A':'fixed 640 + horizontal flip; 30+10 fixed 640 control',
        'B':'phase1 mild RandomDistort + box-aware RandomCrop + multi-scale 480..768 step32; phase2 fixed640, no color/crop/scale aug',
        'C':'same as B plus per-box RandomErasingImage p=.15, target-box area .05-.15; off in final 10 epochs'},
      'same_train_annotation':str(ARCH/'data/annotations/instances_train_bird.json'),
      'same_validation_annotation':str(ARCH/'data/annotations/instances_val_bird.json'),
      'augmentation_source':'PaddleDetection 2.9 operators; A/B/C share the same official model and original Gen2 best checkpoint',
      'training_normalization':'ImageNet mean/std, scale 1/255, matching archived Gen2 fine-tuning TrainReader; official inherited Eval/Test remains raw pixels, matching the baseline inference reader. A one-epoch diagnostic preserved official val AP50 0.879 from initial 0.876; an input-mismatched raw-pixel fine-tune collapsed and is archived separately.',
      'note':'The multi-scale interval is a short-run candidate set. The actual subset used by the final config is recorded after preflight memory/target-size review.'},ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Wrote six phase configs to {OUT}')

if __name__=='__main__':main()
