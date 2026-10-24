from pathlib import Path
s=Path(r'E:\机器学习课设\RFDETR与PPYOLOE鸟检测对照\端到端BioCLIP2对照\reports\external80_end_to_end.html').read_text(encoding='utf-8')
for x in ['"asset"','../assets/external80','大山雀_01.jpg']:
 i=s.find(x);print(x,i,s[i-60:i+150] if i>=0 else '')
