import json
from pathlib import Path
p=Path(r'E:\机器学习课设\RFDETR与PPYOLOE鸟检测对照\reports\fixed_validation_comparison.json')
b=Path(r'E:\机器学习课设\第三代鸟类识别现代模型对照\reports\bioclip2\固定验证集结果.json')
a=json.loads(p.read_text(encoding='utf-8'))['images'][0];z=json.loads(b.read_text(encoding='utf-8'))['rows'][0]
print('ours',a['models']['ppyoloe']['display_candidates'][0])
print('old',z['models']['bioclip2']['displayed'][0])
print('old detector source',json.loads(b.read_text(encoding='utf-8'))['metadata']['detector_checkpoint_sha256'])
