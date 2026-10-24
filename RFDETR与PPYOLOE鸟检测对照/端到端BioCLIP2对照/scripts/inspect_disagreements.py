import json
from pathlib import Path
p=Path(r'E:\机器学习课设\DINOv2鸟类分类对照实验\reports\fixed_validation_end_to_end.json');d=json.loads(p.read_text(encoding='utf-8'))
for m in ['BioCLIP2']:
 print(json.dumps(d[m]['class_disagreements'],ensure_ascii=False))
