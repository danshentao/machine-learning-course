import json,hashlib
from pathlib import Path
from PIL import Image,ImageChops
root=Path(r'E:\机器学习课设\RFDETR与PPYOLOE鸟检测对照');gen=Path(r'E:\机器学习课设\DINOv2鸟类分类对照实验')
fix=json.loads((root/'reports/fixed_validation_comparison.json').read_text(encoding='utf-8'))['images'];bio=json.loads((Path(r'E:\机器学习课设\第三代鸟类识别现代模型对照/reports/bioclip2/固定验证集结果.json')).read_text(encoding='utf-8'))['rows']; bm={x['filename']:x for x in bio}; ok=0;bad=[]
for r in fix:
 a=Path(r['image_path']);b=Path(bm[r['filename']]['image_path'])
 with Image.open(a) as x,Image.open(b) as y:
  if x.convert('RGB').size==y.convert('RGB').size and ImageChops.difference(x.convert('RGB'),y.convert('RGB')).getbbox() is None:ok+=1
  else:bad.append((r['filename'],x.size,y.size))
print('fixed pixel same',ok,'bad',bad[:5])
ext=json.loads((root/'reports/external80_comparison.json').read_text(encoding='utf-8'))['images'];old=json.loads((gen/'reports/external80_comparison.json').read_text(encoding='utf-8'))['images'];om={x['filename']:x for x in old};ok=0;bad=[]
for r in ext:
 a=Path(r['image_path']);b=(gen/'reports'/om[r['filename']]['asset']).resolve()
 with Image.open(a) as x,Image.open(b) as y:
  if x.convert('RGB').size==y.convert('RGB').size and ImageChops.difference(x.convert('RGB'),y.convert('RGB')).getbbox() is None:ok+=1
  else:bad.append((r['filename'],x.size,y.size,str(b)))
print('ext pixel same',ok,'bad count',len(bad),'ex',bad[:5])
