"""Compare website inference with saved reference outputs, never use them to predict."""
import json, sqlite3
from settings import *

db=sqlite3.connect(ROOT/'data/history.sqlite');db.row_factory=sqlite3.Row
tasks=json.loads((ROOT/'验收任务_有鸟.json').read_text(encoding='utf-8'))
refs=json.loads((WORK/'第二代结果归档/20261003_211135/第二代实验报告/最终与最佳分类权重对照/最终与最佳权重比较结果.json').read_text(encoding='utf-8'))
ref=next(r for r in refs['rows'] if r['filename']=='baitoubei_009.jpg')
bio=json.loads((THIRD/'reports/bioclip2/固定验证集结果.json').read_text(encoding='utf-8'))
bio_ref=next(r for r in bio['rows'] if r['filename']=='baitoubei_009.jpg')['models']['bioclip2']['displayed']
report={'sample':'baitoubei_009.jpg uploaded under unrelated filename','scope':'管线验收及历史结果重放，不是外来图片准确率评估','models':[]}
for task in tasks:
    row=db.execute('SELECT * FROM history WHERE id=?',[task['id']]).fetchone()
    assert row['status']=='done',(task,row['error'])
    result=json.loads(row['result']);checks=[]
    for key,items in result['variants'].items():
        assert items,'Expected a displayed bird in this sample'
        assert all(p['detector_score']>result['threshold'] for p in items)
        assert all(0<=p['bbox_xyxy'][0]<p['bbox_xyxy'][2]<=result['width'] and 0<=p['bbox_xyxy'][1]<p['bbox_xyxy'][3]<=result['height'] for p in items)
        for i,p in enumerate(items):
            for q in items[i+1:]: assert iou(p['bbox_xyxy'],q['bbox_xyxy'])<.5
        if key=='ppyoloe':checks.append({'variant':key,'birds':len(items),'class':items[0]['class_name']});continue
        expected=bio_ref if key=='bioclip2' else ref['models']['resnet' if key=='resnet18' else 'bilinear']['best']['displayed']
        assert len(expected)==len(items)
        delta=0.;coordinates=0.
        for p,q in zip(items,expected):
            assert p['class_name']==q['class_name']
            assert set(p['classifier_scores'])==set(NAMES)
            assert abs(sum(p['classifier_scores'].values())-1)<1e-4
            delta=max(delta,max(abs(p['classifier_scores'][n]-q['classifier_scores'][n]) for n in NAMES))
            coordinates=max(coordinates,max(abs(a-b) for a,b in zip(p['bbox_xyxy'],q['bbox_xyxy'])))
        assert coordinates<.01 and delta<.002,(key,coordinates,delta)
        checks.append({'variant':key,'birds':len(items),'class':items[0]['class_name'],'max_coordinate_delta':coordinates,'max_class_score_delta':delta})
    report['models'].append({'model':task['model'],'seconds_including_load':result['elapsed_seconds'],'checks':checks})
report['no_box_handling']=all(json.loads(r['result'])['variants'][next(iter(json.loads(r['result'])['variants']))]==[] for r in db.execute("SELECT result FROM history WHERE filename='external_test_renamed.jpg' AND status='done'"))
(ROOT/'真实模型验收.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
