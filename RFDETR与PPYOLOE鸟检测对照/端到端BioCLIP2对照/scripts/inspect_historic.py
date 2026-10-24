import json
from pathlib import Path
p=Path(r'E:\机器学习课设\DINOv2鸟类分类对照实验\reports\fixed_validation_end_to_end.json');d=json.loads(p.read_text(encoding='utf-8'))
for key in ['BioCLIP2']:
 x=d[key];print('summary',x['summary']);print('per_image first',json.dumps(x['per_image'][0],ensure_ascii=False)[:1000]);print('per_box first',json.dumps(x['per_box'][0],ensure_ascii=False)[:1200])
p=Path(r'E:\机器学习课设\DINOv2鸟类分类对照实验\reports\external80_comparison.json');d=json.loads(p.read_text(encoding='utf-8'));r=d['images'][0];print('external first',json.dumps(r,ensure_ascii=False)[:2500])
