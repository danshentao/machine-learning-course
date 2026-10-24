import json
from pathlib import Path
r1=json.loads(Path(r'E:\机器学习课设\RFDETR与PPYOLOE鸟检测对照\reports\fixed_validation_comparison.json').read_text(encoding='utf-8'))['images'][0]
r2=json.loads(Path(r'E:\机器学习课设\第三代鸟类识别现代模型对照\reports\bioclip2\固定验证集结果.json').read_text(encoding='utf-8'))['rows'][0]
print(r1['filename'],r1['truth']);print(r2['filename'],r2['truth'])
