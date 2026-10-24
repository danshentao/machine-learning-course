from pathlib import Path
import re
p=Path(r'E:\机器学习课设\RFDETR与PPYOLOE鸟检测对照\端到端BioCLIP2对照\reports\browser_qa\fixed_dom.html');s=p.read_text(encoding='utf-8',errors='replace')
for k in ['data-js-error','data-qa','class="card"','loadedImages']:
 m=re.search(k,s);print(k,m.start() if m else None, s[m.start()-100:m.start()+300] if m else '')
