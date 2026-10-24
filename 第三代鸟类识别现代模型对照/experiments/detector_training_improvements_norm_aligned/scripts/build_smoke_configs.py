"""为A/B/C建立仅供管线检查的16图烟测配置，不改正式数据。"""
from __future__ import annotations
import copy,json
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]
P=json.loads((ROOT/'paths.json').read_text(encoding='utf-8'))
ARCH=Path(P['generation2_archive_engineering'])

def main():
    src=json.loads((ARCH/'data/annotations/instances_train_bird.json').read_text(encoding='utf-8'))
    # 固定前16张原图，刻意排除brightness衍生图；所有烟测组使用完全相同清单。
    imgs=[x for x in src['images'] if not Path(x['file_name']).stem.endswith('_brightness')][:16]
    ids={x['id'] for x in imgs}
    ann=[a for a in src['annotations'] if a['image_id'] in ids]
    sub={'images':imgs,'annotations':ann,'categories':src['categories']}
    data=ROOT/'data';data.mkdir(exist_ok=True)
    (data/'smoke_train_16.json').write_text(json.dumps(sub,ensure_ascii=False),encoding='utf-8')
    for group in 'ABC':
        base=yaml.safe_load((ROOT/'configs'/f'{group}_phase1.yml').read_text(encoding='gb18030'))
        base['epoch']=1;base['snapshot_epoch']=1;base['log_iter']=1
        base['save_dir']=str(ROOT/'runs'/'smoke'/group)
        base['TrainDataset']['anno_path']=str(data/'smoke_train_16.json')
        base['LearningRate']['schedulers'][0]['max_epochs']=1
        base['LearningRate']['schedulers'][1]['epochs']=0
        (ROOT/'configs'/f'smoke_{group}.yml').write_text(yaml.safe_dump(base,allow_unicode=True,sort_keys=False),encoding='gb18030')
    (ROOT/'reports'/'smoke_dataset.json').write_text(json.dumps({'images':len(imgs),'annotations':len(ann),'file_names':[x['file_name'] for x in imgs]},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'images':len(imgs),'instances':len(ann),'configs':3},ensure_ascii=False))
if __name__=='__main__':main()
