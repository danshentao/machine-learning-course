from common import *
import sys,time,copy,csv,io,base64
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import open_clip
sys.path.insert(0,str(GEN2/'scripts'))
from build_deduplicated_gallery import group_candidates
from compare_display_thresholds import inspect

OUT=ROOT/'reports/640与960检测输入对照'
def summarize(rows,key):
    xs=[r['models'][key] for r in rows];tp=sum(x['TP'] for x in xs);fp=sum(x['FP'] for x in xs);fn=sum(x['FN'] for x in xs)
    p=tp/(tp+fp);rec=tp/(tp+fn)
    return {'images':len(rows),'GT':tp+fn,'displayed_boxes':tp+fp,'TP':tp,'FP':fp,'FN':fn,'precision':p,'recall':rec,'F1':2*p*rec/(p+rec),'class_disagreement_images':sum(x['class_disagreement'] for x in xs),'no_display_images':sum(x['no_display'] for x in xs),'spatially_matched_wrong_class':sum(x['spatially_matched_wrong_class'] for x in xs),'unmatched_displayed_boxes':sum(sum('未匹配' in p['spatial_status'] for p in x['displayed']) for x in xs)}

def main():
    start=time.perf_counter();seed();torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu';OUT.mkdir(parents=True,exist_ok=True)
    rows=copy.deepcopy(baseline()['rows']);jobs=[];docs={};groups_by_size={}
    for size in [640,960]:
        doc=json.loads((ROOT/f'runs/detector_resolution_{size}/proposals.json').read_text(encoding='utf-8'));docs[size]=doc
        assert doc['metadata']['actual_input_shapes']==[[1,3,size,size]]
        lookup={r['filename']:r for r in doc['images']};gs=[]
        for ri,row in enumerate(rows):
            groups=group_candidates(lookup[row['filename']]['detections']);gs.append(groups)
            for g in groups:
                if g['winner']['detector_score']>.30:jobs.append({'size':size,'row':ri,'image':row['image_path'],'proposal':g['winner']})
        groups_by_size[size]=gs
    weights=ROOT/'weights/bioclip2/open_clip_model.safetensors';headpath=ROOT/'runs/bioclip2/best_head.pt';meta=json.loads((ROOT/'runs/bioclip2/metadata.json').read_text(encoding='utf-8'))
    assert sha(headpath)==meta['best_head_sha256'];assert sha(weights)=='b7b2bf6fbc95799e42630e394cf95803892ab447c1a8ab629dbc82fbeaf7dfef'
    cfg=json.loads((weights.parent/'open_clip_config.json').read_text());model,_,transform=open_clip.create_model_and_transforms('ViT-L-14',pretrained=str(weights),image_mean=cfg['preprocess_cfg']['mean'],image_std=cfg['preprocess_cfg']['std'],image_interpolation='bicubic');model=model.visual.to(device).eval()
    head=torch.nn.Linear(768,8).to(device);head.load_state_dict(torch.load(headpath,map_location=device,weights_only=True));head.eval();scores=[]
    with torch.inference_mode():
        for st in range(0,len(jobs),8):
            arrays=[]
            for j in jobs[st:st+8]:
                with Image.open(j['image']) as im:arrays.append(transform(square(crop(im.convert('RGB'),j['proposal']['bbox_xyxy']))))
            with torch.autocast(device_type=device,enabled=device=='cuda',dtype=torch.float16):z=model(torch.stack(arrays).to(device))
            scores.extend(head(F.normalize(z.float(),dim=1)).softmax(1).cpu().tolist())
    preds={size:[[] for _ in rows] for size in [640,960]}
    for j,s in zip(jobs,scores):
        p=copy.deepcopy(j['proposal']);top=int(np.argmax(s));p.update(class_id=top,class_name=NAMES[top],classifier_scores=dict(zip(NAMES,s)),classifier_top_score=s[top],final_score_product=p['detector_score']*s[top]);preds[j['size']][j['row']].append(p)
    keys={640:'gen2_bilinear_best',960:'size960'} # Compatible gallery baseline key; label explicitly says BioCLIP2.
    for ri,row in enumerate(rows):
        row['original_no_display']=row['models']['bilinear']['best']['no_display'];row['models']={keys[s]:inspect(row['truth'],preds[s][ri],row['expected']) for s in [640,960]}
    summary={keys[s]:summarize(rows,keys[s]) for s in [640,960]}
    old=json.loads((ROOT/'reports/bioclip2/固定验证集结果.json').read_text(encoding='utf-8'));delta=0;classchanges=0;maxboxdelta=0
    for row,prev in zip(rows,old['rows']):
        now=row['models'][keys[640]]['displayed'];before=prev['models']['bioclip2']['displayed'];assert len(now)==len(before)
        for p,q in zip(now,before):
            maxboxdelta=max(maxboxdelta,max(abs(x-y) for x,y in zip(p['bbox_xyxy'],q['bbox_xyxy'])));delta=max(delta,max(abs(p['classifier_scores'][n]-q['classifier_scores'][n]) for n in NAMES));classchanges+=p['class_name']!=q['class_name']
    assert classchanges==0 and delta<.002 and maxboxdelta<.02,(classchanges,delta,maxboxdelta)
    changes={'previous_five':[],'no_display_removed':[],'no_display_added':[],'spatial_TP_increased':[],'spatial_TP_decreased':[],'end_to_end_TP_increased':[],'end_to_end_TP_decreased':[],'unmatched_displayed_boxes_increased':[]}
    for row in rows:
        a=row['models'][keys[640]];b=row['models'][keys[960]]
        def unmatched(x):return sum('未匹配' in p['spatial_status'] for p in x['displayed'])
        def spatialtp(x):return sum('未匹配' not in p['spatial_status'] for p in x['displayed'])
        if row['original_no_display']:changes['previous_five'].append({'filename':row['filename'],'640_displayed':len(a['displayed']),'960_displayed':len(b['displayed']),'960_TP':b['TP'],'960_FP':b['FP'],'960_FN':b['FN'],'960_predictions':[{'class':p['class_name'],'detector_score':p['detector_score'],'spatial_status':p['spatial_status']} for p in b['displayed']]})
        if a['no_display'] and not b['no_display']:changes['no_display_removed'].append(row['filename'])
        if not a['no_display'] and b['no_display']:changes['no_display_added'].append(row['filename'])
        for k,c in [('spatial_TP_increased',spatialtp(b)>spatialtp(a)),('spatial_TP_decreased',spatialtp(b)<spatialtp(a)),('end_to_end_TP_increased',b['TP']>a['TP']),('end_to_end_TP_decreased',b['TP']<a['TP']),('unmatched_displayed_boxes_increased',unmatched(b)>unmatched(a))]:
            if c:changes[k].append(row['filename'])
    metadata={'date':'2026-10-04','sizes':[640,960],'detector_weight_sha256':docs[640]['metadata']['detector_weights_sha256'],'classifier_weight_sha256':sha(headpath),'same_weights':True,'strict_display_threshold':.30,'group_iou':.5,'crop_margin':.15,'classifier_input':224,'replay_640':{'class_changes':classchanges,'max_classifier_score_difference':delta,'max_coordinate_difference':maxboxdelta},'seconds':time.perf_counter()-start,'limits':'Frozen detector trained at640; size960 changes inference only. Reused validation, not independent test. No full-candidate AP computed. Unmatched predicted boxes can reflect false positives, annotation omissions or duplicate detections; no automatic label changes.'}
    write(OUT/'完整结果.json',{'metadata':metadata,'summary':summary,'changes':changes,'rows':rows});write(OUT/'汇总.json',{'metadata':metadata,'summary':summary,'changes':changes})
    # Save every candidate and its suppression group; only winners >30% are drawn.
    for size in [640,960]:write(OUT/f'{size}_原始候选与去重分组.json',{'metadata':docs[size]['metadata'],'images':[{'filename':r['filename'],'groups':g} for r,g in zip(rows,groups_by_size[size])]})
    with (OUT/'逐图比较.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['图片','所属类别','输入尺寸','显示框数','TP','FP','FN','类别分歧','无显示框'])
        for r in rows:
            for size in [640,960]:x=r['models'][keys[size]];w.writerow([r['filename'],r['expected'],size,len(x['displayed']),x['TP'],x['FP'],x['FN'],x['class_disagreement'],x['no_display']])
    for r in rows:
        with Image.open(r['image_path']) as im:
            im=im.convert('RGB');im.thumbnail((1000,1000));buf=io.BytesIO();im.save(buf,format='JPEG',quality=85);r['image']='data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode()
    template=(ROOT/'scripts/gallery.html').read_text(encoding='utf-8');template=template.replace('第三代鸟类分类对照','640与960检测输入对照').replace('第二代与第三代：同框分类对照','640与960：相同权重，不同检测输入').replace('同一172张验证图、同一177个检测分数&gt;30%的去重框；外扩15%。','同一172张验证图、相同检测权重及BioCLIP 2分类权重；只改变检测输入尺寸。检测分数&gt;30%、IoU≥0.5去重、外扩15%。').replace('相对第二代','相对640').replace('<option value="diff">','<option value="previous_empty">原来5张无框图</option><option value="diff">').replace("(f==='all'||", "(f==='previous_empty'&&r.original_no_display||f==='all'||")
    payload=json.dumps({'rows':rows,'models':{keys[640]:'640输入 + BioCLIP 2',keys[960]:'960输入 + BioCLIP 2'},'summary':summary},ensure_ascii=False).replace('<','\\u003c');(OUT/'640与960逐图对照.html').write_text(template.replace('__DATA__',payload),encoding='utf-8')
    print(json.dumps({'summary':summary,'changes':changes,'replay':metadata['replay_640']},ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
