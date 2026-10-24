"""读取已有预测，在同一验证图集上比较最终显示类别与原始目录类别。"""
from pathlib import Path
import base64, io, json
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path('E:/机器学习课设/模型检查报告/第二代实验/最高置信度框对照')
OUT=Path('E:/机器学习课设/模型检查报告/第二代实验/同一172张类别对照')

def main():
    cfg=json.loads((ROOT/'configs/paths.json').read_text(encoding='utf-8'))
    mapping=json.loads((ROOT/'configs/class_mapping.json').read_text(encoding='utf-8'))
    names=[c['name'] for c in mapping['classes']];sources={}
    for c in mapping['classes']:
        for p in (Path(cfg['source_images'])/c['folder']).rglob('*.jpg'):
            if p.name in sources:raise ValueError('Duplicate filename '+p.name)
            sources[p.name]={'path':str(p),'expected':c['name'],'folder':c['folder']}
    second={model:json.loads((SOURCE/f'{model}_postprocessed_predictions.json').read_text(encoding='utf-8')) for model in ['resnet','bilinear']}
    lookup={model:{r['filename']:r for r in doc['images']} for model,doc in second.items()}
    filenames=set(lookup['resnet']);assert len(filenames)==172 and filenames==set(lookup['bilinear'])
    firstfile=Path(cfg['first_generation_review_archive'])/'严格阈值全量逐图候选/逐图原始候选_隐藏候选_最终分组.json'
    firstdoc=json.loads(firstfile.read_text(encoding='utf-8'));first={r['filename']:r for r in firstdoc['images']}
    assert filenames<=set(first)
    # 两代使用各自已保存的最高分重叠组，避免新处理改变第一代归档口径。
    rows=[]
    for name in sorted(filenames):
        origin=sources[name];r=lookup['resnet'][name]
        assert first[name]['split']=='val',name
        g1=[]
        for g in first[name]['candidate_groups']:
            d=g['winner']
            if d['score']>.5:
                g1.append({'class_name':d['class_name'],'bbox_xyxy':d['bbox_xyxy'],'score':d['score'],
                           'classifier_top_score':None,'scores':dict(zip(names,d.get('class_scores',[])))})
        row={'filename':name,'expected':origin['expected'],'folder':origin['folder'],'source_image':origin['path'],
             'width':r['width'],'height':r['height'],'models':{'gen1':g1}}
        for model in ['resnet','bilinear']:
            row['models'][model]=[{'class_name':d['class_name'],'bbox_xyxy':d['bbox_xyxy'],'score':d['detector_score'],
              'classifier_top_score':d['classifier_top_score'],'scores':d['classifier_scores']} for d in lookup[model][name]['after']]
        row['different']={model:any(d['class_name']!=row['expected'] for d in ds) for model,ds in row['models'].items()}
        row['no_display']={model:not ds for model,ds in row['models'].items()}
        rows.append(row)
    summary={}
    for model in ['gen1','resnet','bilinear']:
        summary[model]={'images':172,'class_disagreement_images':sum(r['different'][model] for r in rows),
                        'no_display_images':sum(r['no_display'][model] for r in rows),
                        'displayed_boxes':sum(len(r['models'][model]) for r in rows)}
    transitions={}
    for model in ['resnet','bilinear']:
        transitions[model]={
            'first_disagreement_second_no_disagreement_with_display':[r['filename'] for r in rows if r['different']['gen1'] and not r['different'][model] and not r['no_display'][model]],
            'first_disagreement_second_no_display':[r['filename'] for r in rows if r['different']['gen1'] and r['no_display'][model]],
            'new_disagreement':[r['filename'] for r in rows if not r['different']['gen1'] and r['different'][model]],
            'both_disagreement':[r['filename'] for r in rows if r['different']['gen1'] and r['different'][model]]}
    OUT.mkdir(parents=True,exist_ok=True)
    report={'rule':'同一172张验证图；读取各代现有去重最高分框；第一代score>0.5，第二代detector_score>0.5；任一最终显示类别不同于原图目录类别即计分歧，不检查框数量/位置。无框另列，不作为类别正确。',
            'score_caveat':'两代分数含义不同：第一代联合检测类别分数，第二代检测分数；本比较按用户要求采用各代现有50%显示规则，分数不视为等价概率。',
            'source_gen1':str(firstfile),'summary':summary,'transitions':transitions,'rows':rows}
    (OUT/'同范围同筛选规则结果.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    selected=[r for r in rows if any(r['different'].values()) or any(r['no_display'].values())]
    for r in selected:
        with Image.open(r['source_image']) as im:
            im=im.convert('RGB');im.thumbnail((1400,1400));buf=io.BytesIO();im.save(buf,format='JPEG',quality=86)
        r['image']='data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode('ascii')
    payload=json.dumps({'rows':selected,'summary':summary,'transitions':transitions},ensure_ascii=False).replace('<','\\u003c')
    template=(ROOT/'scripts/same_rule_comparison_template.html').read_text(encoding='utf-8')
    (OUT/'第一代与第二代类别分歧对照.html').write_text(template.replace('__PAYLOAD__',payload),encoding='utf-8')
    print(json.dumps({'output':str(OUT),'summary':summary,'transitions':{m:{k:len(v) for k,v in t.items()} for m,t in transitions.items()},'gallery_images':len(selected)},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
