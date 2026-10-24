from pathlib import Path
s=Path(r'E:\机器学习课设\RFDETR与PPYOLOE鸟检测对照\端到端BioCLIP2对照\reports\browser_qa\fixed_dom.html').read_text(encoding='utf-8',errors='replace')
print(repr(s[-2500:]))
