from pathlib import Path
p=Path(r'E:\机器学习课设\第三代鸟类识别现代模型对照\experiments\detector_training_improvements_norm_aligned\reports\同鸟重复框后处理对照\compare_postprocess_rules.py')
s=p.read_text(encoding='utf-8-sig')
old="payload=json.dumps({'meta':metadata,'names':keys,'policy_names':policy_names,'summaries':summaries,'records':records,'images':img_payload},ensure_ascii=False,separators=(',',':')).replace('</','<\\\\/')"
new="payload=json.dumps({'meta':metadata,'names':keys,'policy_names':policy_names,'summaries':summaries,'per_image_deltas':per_image_deltas,'records':records,'images':img_payload},ensure_ascii=False,separators=(',',':')).replace('</','<\\\\/')"
if old not in s:
    print('old not found; locating payload source')
    for line in s.splitlines():
        if 'payload=json.dumps' in line: print(repr(line))
    raise SystemExit(2)
p.write_text(s.replace(old,new),encoding='utf-8')
