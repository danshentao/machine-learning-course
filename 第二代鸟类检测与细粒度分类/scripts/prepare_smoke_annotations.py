"""从暂用固定清单中抽取极少量图片，单独构建短跑JSON，不覆盖正式候选标注。"""
import json
import shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; ann=ROOT/'data/annotations'
for part,n in [('train',12),('val',8)]:
    src=json.loads((ann/f'instances_{part}_bird.json').read_text(encoding='utf-8'))
    images=src['images'][:n]; ids={x['id'] for x in images}
    anns=[a for a in src['annotations'] if a['image_id'] in ids]
    (ann/f'instances_{part}_bird_smoke.json').write_text(json.dumps({'images':images,'annotations':anns,'categories':src['categories']},ensure_ascii=False,indent=2),encoding='utf-8')
    print(part,'images',len(images),'instances',len(anns))
smoke_images=json.loads((ann/'instances_val_bird_smoke.json').read_text(encoding='utf-8'))['images']
infer_dir=ROOT/'data/smoke_infer_inputs';infer_dir.mkdir(parents=True,exist_ok=True)
gen1=ROOT.parent/'自然场景鸟类检测PPYOLOE'
for rec in smoke_images[:2]:shutil.copy2(gen1/'data/prepared/images/val'/rec['file_name'],infer_dir/rec['file_name'])
