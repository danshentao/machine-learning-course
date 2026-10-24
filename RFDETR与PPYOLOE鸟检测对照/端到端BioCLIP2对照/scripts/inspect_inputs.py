import json
from pathlib import Path
root=Path(r'E:\机器学习课设\RFDETR与PPYOLOE鸟检测对照')
for f in ['reports/fixed_validation_comparison.json','reports/external80_comparison.json']:
 d=json.loads((root/f).read_text(encoding='utf-8')); print(f, d.keys());r=(d['rows'] if 'rows' in d else d['images'])[0];print('rowkeys',r.keys(),'name',r['filename'],'dims',r['width'],r['height'],'truth',r.get('truth')[:1]);print('models',[(m, v.keys(),len(v.get('display_candidates',[]))) for m,v in r['models'].items()]);print('candidate',r['models']['rfdetr']['display_candidates'][:1])

