from __future__ import annotations
import base64,csv,hashlib,io,json,math,os,sys,time,copy
from pathlib import Path
from PIL import Image

EXP=Path(r'E:\机器学习课设\第三代鸟类识别现代模型对照\experiments\detector_training_improvements_norm_aligned')
GEN3=Path(r'E:\机器学习课设\第三代鸟类识别现代模型对照')
GEN2=Path(r'E:\机器学习课设\第二代鸟类检测与细粒度分类')
OUT=EXP/'reports'/'同鸟重复框后处理对照'
for k,v in {'HF_HOME':OUT/'.cache/huggingface','HF_HUB_CACHE':OUT/'.cache/huggingface/hub','TORCH_HOME':OUT/'.cache/torch','MPLCONFIGDIR':OUT/'.cache/matplotlib','TORCHINDUCTOR_CACHE_DIR':OUT/'.cache/inductor','TEMP':OUT/'.cache/tmp','TMP':OUT/'.cache/tmp'}.items():
    Path(v).mkdir(parents=True,exist_ok=True);os.environ[k]=str(v)
os.environ['HF_HUB_DISABLE_XET']='1'

sys.path.insert(0,str(GEN3/'scripts'))
from common import NAMES,seed,square,crop,sha
sys.path.insert(0,str(GEN2/'scripts'))
from build_deduplicated_gallery import group_candidates
from evaluate_two_stage import match,iou
import numpy as np
import torch,torch.nn.functional as F,open_clip

BASE_PROPOSALS=GEN3/'runs'/'detector_resolution_640'/'proposals.json'
C_PROPOSALS=EXP/'runs'/'C_inference'/'proposals.json'
FIXED=EXP/'reports'/'fixed_val_comparison'/'fixed_val_results.json'
BIO_WEIGHT=GEN3/'weights'/'bioclip2'/'open_clip_model.safetensors'
HEAD=GEN3/'runs'/'bioclip2'/'best_head.pt'
HEAD_META=GEN3/'runs'/'bioclip2'/'metadata.json'
IMAGE_LIMIT=100
DISPLAY_THRESHOLD=.30
MATCH_IOU=.50

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def area(b):return max(0,b[2]-b[0])*max(0,b[3]-b[1])
def intersection(b1,b2):
    return max(0,min(b1[2],b2[2])-max(b1[0],b2[0]))*max(0,min(b1[3],b2[3])-max(b1[1],b2[1]))
def containment(b1,b2):
    small=min(area(b1),area(b2));return intersection(b1,b2)/small if small>0 else 0.0
def center_distance_ratio(b1,b2):
    c1=((b1[0]+b1[2])/2,(b1[1]+b1[3])/2);c2=((b2[0]+b2[2])/2,(b2[1]+b2[3])/2)
    diag=min(math.hypot(b1[2]-b1[0],b1[3]-b1[1]),math.hypot(b2[2]-b2[0],b2[3]-b2[1]))
    return math.hypot(c1[0]-c2[0],c1[1]-c2[1])/diag if diag>0 else float('inf')
def normalize_candidate(c,idx):
    d=dict(c);d['proposal_index']=int(idx);d['bbox_xyxy']=[float(x) for x in c['bbox_xyxy']]
    d['detector_score']=float(c['detector_score']);return d
def clone_members(g):return [g['winner'],*g.get('hidden',[])]

def containment_groups(cands,threshold):
    groups=[]
    for d in sorted(cands,key=lambda x:(-x['detector_score'],x['proposal_index'])):
        options=[]
        for j,g in enumerate(groups):
            w=g['winner'];ov=iou(d['bbox_xyxy'],w['bbox_xyxy']);cov=containment(d['bbox_xyxy'],w['bbox_xyxy'])
            if ov>=.5 or cov>=threshold:options.append((max(ov/.5,cov/threshold),ov,cov,j))
        if options:
            ratio,ov,cov,j=max(options,key=lambda x:(x[0],x[1],x[2],-x[3]))
            groups[j]['hidden'].append({**d,'suppression_reason':{'iou':ov,'intersection_over_smaller':cov,'iou_threshold':.5,'containment_threshold':threshold}})
        else:groups.append({'winner':d,'hidden':[]})
    return groups

MERGE_RULE={'iou_min':.20,'intersection_over_smaller_min':.50,'center_distance_over_smaller_diagonal_max':.45,'max_members':3,
            'clustering':'score-descending greedy complete-link; every new member must satisfy all pair criteria with every group member; join only when exactly one eligible group exists; otherwise start singleton. No transitive bridge.'}
def merge_groups(cands):
    groups=[]
    for d in sorted(cands,key=lambda x:(-x['detector_score'],x['proposal_index'])):
        eligible=[]
        for gi,g in enumerate(groups):
            if len(g['members'])>=MERGE_RULE['max_members']:continue
            pair_metrics=[{'iou':iou(d['bbox_xyxy'],m['bbox_xyxy']),'intersection_over_smaller':containment(d['bbox_xyxy'],m['bbox_xyxy']),'center_distance_over_smaller_diagonal':center_distance_ratio(d['bbox_xyxy'],m['bbox_xyxy'])} for m in g['members']]
            ok=all(q['iou']>=MERGE_RULE['iou_min'] and q['intersection_over_smaller']>=MERGE_RULE['intersection_over_smaller_min'] and q['center_distance_over_smaller_diagonal']<=MERGE_RULE['center_distance_over_smaller_diagonal_max'] for q in pair_metrics)
            if ok:eligible.append((gi,pair_metrics))
        if len(eligible)==1:
            gi,pm=eligible[0];groups[gi]['members'].append(d);groups[gi]['pair_checks'].append({'proposal_index':d['proposal_index'],'against':[m['proposal_index'] for m in groups[gi]['members'][:-1]],'metrics':pm})
        else:
            groups.append({'members':[d],'pair_checks':[],'ambiguous_eligible_group_count':len(eligible)})
    return groups

def make_groups(policy,cands):
    if policy=='iou':return group_candidates(cands)
    if policy=='contain_08':return containment_groups(cands,.80)
    if policy=='contain_09':return containment_groups(cands,.90)
    if policy=='merge':return merge_groups([c for c in cands if c['detector_score']>DISPLAY_THRESHOLD])
    raise ValueError(policy)

def result_specs(policy,groups,width,height,gt):
    specs=[];group_records=[]
    for gi,g in enumerate(groups):
        if policy=='merge':
            members=g['members'];score_owner=max(members,key=lambda x:(x['detector_score'],-x['proposal_index']))
            box=[min(x['bbox_xyxy'][0] for x in members),min(x['bbox_xyxy'][1] for x in members),max(x['bbox_xyxy'][2] for x in members),max(x['bbox_xyxy'][3] for x in members)]
            box=[max(0,box[0]),max(0,box[1]),min(width,box[2]),min(height,box[3])]
            score=score_owner['detector_score'];indices=[x['proposal_index'] for x in members]
            spec={'bbox_xyxy':box,'detector_score':score,'candidate_indices':indices,'group_id':gi,'is_merged':len(members)>1,
                  'score_source_candidate_index':score_owner['proposal_index'],'score_rule':'沿用组内最大检测分数，不求和，不是检测器重新预测',
                  'group_members':[{'candidate_index':x['proposal_index'],'bbox_xyxy':x['bbox_xyxy'],'detector_score':x['detector_score']} for x in members],
                  'grouping_reason':MERGE_RULE,'pair_checks':g.get('pair_checks',[]),'ambiguous_eligible_group_count':g.get('ambiguous_eligible_group_count',0)}
            suppressed=[]
        else:
            w=g['winner'];members=clone_members(g);box=w['bbox_xyxy'];score=w['detector_score'];indices=[x['proposal_index'] for x in members]
            spec={'bbox_xyxy':box,'detector_score':score,'candidate_indices':[w['proposal_index']],'group_id':gi,'is_merged':False,
                  'score_source_candidate_index':w['proposal_index'],'score_rule':'原检测候选分数沿用',
                  'group_members':[{'candidate_index':x['proposal_index'],'bbox_xyxy':x['bbox_xyxy'],'detector_score':x['detector_score']} for x in members],
                  'grouping_reason': 'IoU≥0.5按检测分数降序非传递贪心' if policy=='iou' else f'IoU≥0.5或交集/较小框面积≥{.8 if policy=="contain_08" else .9:.2f}；降序非传递贪心',
                  'pair_checks':[],'ambiguous_eligible_group_count':0}
            suppressed=[]
            for h in g.get('hidden',[]):
                detail={k:h.get(k) for k in ('proposal_index','bbox_xyxy','detector_score','iou_with_winner','suppression_reason') if k in h}
                suppressed.append(detail)
        if len(members)>1 and policy=='merge':
            member_truth=[]
            for m in members:
                best=max([(iou(m['bbox_xyxy'],t['bbox_xyxy']),ti) for ti,t in enumerate(gt)],default=(0,None))
                if best[0]>=MATCH_IOU:member_truth.append({'candidate_index':m['proposal_index'],'gt_instance':best[1]+1,'iou':best[0]})
            union_gt=[{'gt_instance':ti+1,'iou':iou(box,t['bbox_xyxy'])} for ti,t in enumerate(gt) if iou(box,t['bbox_xyxy'])>=MATCH_IOU]
            matched_instances={x['gt_instance'] for x in member_truth}
            spec['posthoc_gt_diagnostic']={'member_candidate_gt_matches':member_truth,'union_box_gt_matches':union_gt,
                'possible_merge_of_multiple_birds':len(matched_instances)>1 or len(union_gt)>1}
        else:spec['posthoc_gt_diagnostic']={'member_candidate_gt_matches':[],'union_box_gt_matches':[],'possible_merge_of_multiple_birds':False}
        if score>DISPLAY_THRESHOLD:specs.append(spec)
        group_records.append({'group_id':gi,'winner_or_members':indices,'output_candidate_indices':indices if score>DISPLAY_THRESHOLD else [],'winner_bbox_xyxy':box,'winner_score':score,
          'suppressed_candidates':suppressed,'below_display_threshold':score<=DISPLAY_THRESHOLD,'is_merged':spec['is_merged'],'possible_merge':spec['posthoc_gt_diagnostic']['possible_merge_of_multiple_birds']})
    return specs,group_records

def pmetrics(truth,pred):
    # 与现有 evaluate_two_stage.match 完全一致：类别正确、IoU≥0.5，按IoU/最终分数排序一对一匹配。
    ug,up,pairs=match(truth,pred,True)
    spatial_gt,spatial_pred,spatial_pairs=match(truth,pred,False)
    wrong=sum(pred[pi]['class_id']!=truth[gi]['class_id'] for gi,pi,_ in spatial_pairs)
    return {'TP':len(pairs),'FP':len(pred)-len(up),'FN':len(truth)-len(ug),'unmatched_display_boxes':len(pred)-len(spatial_pred),
            'spatially_matched_wrong_class':wrong,'no_boxes':len(pred)==0,
            'pairs':[{'gt_instance':gi+1,'prediction_index':pi,'iou':ov} for gi,pi,ov in pairs],
            'spatial_pairs':[{'gt_instance':gi+1,'prediction_index':pi,'iou':ov,'class_correct':pred[pi]['class_id']==truth[gi]['class_id']} for gi,pi,ov in spatial_pairs]}
def score_metrics(records):
    tp=sum(r['metric']['TP'] for r in records);fp=sum(r['metric']['FP'] for r in records);fn=sum(r['metric']['FN'] for r in records)
    p=tp/(tp+fp) if tp+fp else 0;r=tp/(tp+fn) if tp+fn else 0
    return {'images':len(records),'TP':tp,'FP':fp,'FN':fn,'displayed_boxes':tp+fp,'unmatched_display_boxes':sum(x['metric']['unmatched_display_boxes'] for x in records),
            'no_box_images':sum(x['metric']['no_boxes'] for x in records),'spatially_matched_wrong_class':sum(x['metric']['spatially_matched_wrong_class'] for x in records),'Precision_micro':p,'Recall_micro':r,'F1_micro':2*p*r/(p+r) if p+r else 0,
            'possible_merge_groups':sum(sum(g['possible_merge'] for g in x['group_records']) for x in records)}
def image_f1(metric):
    p=metric['TP']/(metric['TP']+metric['FP']) if metric['TP']+metric['FP'] else 0
    r=metric['TP']/(metric['TP']+metric['FN']) if metric['TP']+metric['FN'] else 0
    return 2*p*r/(p+r) if p+r else 0

def main():
    t0=time.perf_counter();seed(123);torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('BioCLIP 2 frozen crop classification requires the existing CUDA environment')
    fixed=read(FIXED);fixed_rows=fixed['rows'];truth_by={r['filename']:r['truth'] for r in fixed_rows};row_by={r['filename']:r for r in fixed_rows}
    base_doc=read(BASE_PROPOSALS);c_doc=read(C_PROPOSALS)
    assert len(base_doc['images'])==len(c_doc['images'])==len(fixed_rows)==172
    assert base_doc['metadata']['detector_weights_sha256']=='dc394290449ddf40fef31c144320acb8b13d0fb57b6d35f39419c3aeadfca539'
    docs={'baseline':base_doc,'C':c_doc};keys={'baseline':'原PP-YOLOE+基准','C':'C组轻量遮挡最佳权重'}
    source_key={'baseline':'detections','C':'candidates'}
    raw_by={};image_meta={}
    for model,doc in docs.items():
        lookup={x['filename']:x for x in doc['images']};raw_by[model]={}
        for fn in row_by:
            r=lookup[fn];items=[normalize_candidate(c,i) for i,c in enumerate(r[source_key[model]]) if c.get('valid_after_clipping',True)]
            raw_by[model][fn]=items;image_meta[fn]={'path':r.get('image',r.get('image_path')),'width':r['width'],'height':r['height']}
    policies=['iou','contain_08','contain_09','merge']
    spec_rows={m:{} for m in docs};groups_rows={m:{} for m in docs};raw_counts={m:{} for m in docs};jobs=[];job_lookup={}
    for m in docs:
        for fn,cands in raw_by[m].items():
            raw_counts[m][fn]={'all_candidates':len(cands),'above_threshold':sum(c['detector_score']>DISPLAY_THRESHOLD for c in cands),'below_or_equal':sum(c['detector_score']<=DISPLAY_THRESHOLD for c in cands)}
            for policy in policies:
                groups=make_groups(policy,cands)
                specs,grecs=result_specs(policy,groups,image_meta[fn]['width'],image_meta[fn]['height'],truth_by[fn])
                spec_rows[m].setdefault(policy,{})[fn]=specs;groups_rows[m].setdefault(policy,{})[fn]=grecs
                for si,s in enumerate(specs):
                    box=tuple(round(float(x),4) for x in s['bbox_xyxy']);key=(m,fn,box)
                    if key not in job_lookup:
                        job_lookup[key]=len(jobs);jobs.append({'model':m,'filename':fn,'image':image_meta[fn]['path'],'bbox':s['bbox_xyxy']})
                    s['classification_job']=job_lookup[key]
    print(f'classification_crops={len(jobs)}',flush=True)
    cfg=read(BIO_WEIGHT.parent/'open_clip_config.json');head_meta=read(HEAD_META)
    assert sha(HEAD)==head_meta['best_head_sha256']
    model,_,transform=open_clip.create_model_and_transforms('ViT-L-14',pretrained=str(BIO_WEIGHT),image_mean=cfg['preprocess_cfg']['mean'],image_std=cfg['preprocess_cfg']['std'],image_interpolation='bicubic')
    model=model.visual.cuda().eval();head=torch.nn.Linear(768,len(NAMES)).cuda();head.load_state_dict(torch.load(HEAD,map_location='cuda',weights_only=True));head.eval()
    scores=[]
    with torch.inference_mode():
        for st in range(0,len(jobs),8):
            batch=jobs[st:st+8];ims=[]
            for j in batch:
                with Image.open(j['image']) as im:ims.append(transform(square(crop(im.convert('RGB'),j['bbox'],.15))))
            x=torch.stack(ims).cuda()
            with torch.autocast(device_type='cuda',enabled=True,dtype=torch.float16):z=model(x)
            logits=head(F.normalize(z.float(),dim=1));scores.extend(logits.softmax(1).float().cpu().tolist())
            if st%160==0:print(f'classified {min(st+8,len(jobs))}/{len(jobs)} crops',flush=True)
    for m in docs:
        for policy in policies:
            for fn,preds in spec_rows[m][policy].items():
                for p in preds:
                    s=scores[p['classification_job']];top=int(np.argmax(s))
                    p.update(class_id=top,class_name=NAMES[top],classifier_top_score=float(s[top]),classifier_scores=dict(zip(NAMES,map(float,s))),final_score_product=float(p['detector_score']*s[top]))
                    p['final_score_interpretation']='detector_score × BioCLIP 2 top softmax，仅排序分数，非校准概率'
                    p['crop_rule']='最终框外扩15%，与固定BioCLIP 2预处理相同'
    records={'baseline':{},'C':{}}
    for m in docs:
        for policy in policies:
            out=[]
            for fn in row_by:
                pred=spec_rows[m][policy][fn];metric=pmetrics(truth_by[fn],pred);grecs=groups_rows[m][policy][fn]
                out.append({'filename':fn,'image_path':image_meta[fn]['path'],'width':image_meta[fn]['width'],'height':image_meta[fn]['height'],
                    'truth':truth_by[fn],'multi_bird_image':len(truth_by[fn])>=2,'raw_candidate_count':raw_counts[m][fn]['all_candidates'],
                    'raw_candidates_over_threshold':raw_counts[m][fn]['above_threshold'],'predictions':pred,'group_records':grecs,'metric':metric,
                    'possible_mismerge':any(g['possible_merge'] for g in grecs),'image_F1':image_f1(metric)})
                
            records[m][policy]=out
    summaries={};per_image_deltas={}
    policy_names={'iou':'原IoU去重','contain_08':'IoU+包含阈值0.8','contain_09':'IoU+包含阈值0.9','merge':'受约束分组合并'}
    for m in docs:
        base=records[m]['iou'];summaries[m]={};lookup={p:{r['filename']:r for r in records[m][p]} for p in policies}
        for policy in policies:
            sm=score_metrics(records[m][policy]);sm['policy_name']=policy_names[policy];summaries[m][policy]=sm
        per_image_deltas[m]={}
        for policy in policies:
            per=[]
            for fn in row_by:
                b=lookup['iou'][fn];q=lookup[policy][fn];bf=b['image_F1'];qf=q['image_F1']
                status='改善' if qf>bf+1e-12 else ('退步' if qf<bf-1e-12 else '无变化')
                if q['metric']['no_boxes']:status='无框'
                per.append({'filename':fn,'baseline_F1':bf,'policy_F1':qf,'status_vs_iou':status,'TP':q['metric']['TP'],'FP':q['metric']['FP'],'FN':q['metric']['FN'],
                    'display_boxes':len(q['predictions']),'possible_mismerge':q['possible_mismerge'],'multi_bird_image':q['multi_bird_image']})
            per_image_deltas[m][policy]=per
    # embed validation images once; boxes remain in original pixel coordinates and scale factors are explicitly stored.
    img_payload={}
    for fn,r in row_by.items():
        path=Path(image_meta[fn]['path'])
        with Image.open(path) as im:
            im=im.convert('RGB');ow,oh=im.size;im.thumbnail((1150,1150),Image.Resampling.LANCZOS);buf=io.BytesIO();im.save(buf,'JPEG',quality=79,optimize=True)
            img_payload[fn]={'src':'data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode(),'width':im.width,'height':im.height,'scale_x':im.width/ow,'scale_y':im.height/oh,'path':str(path)}
    metadata={'created_local':'2026-10-06 Asia/Shanghai','validation_images':172,'xml_instances':sum(len(x) for x in truth_by.values()),'scope':'固定验证集探索/后处理对照，不是独立测试；教师图片未读取',
      'candidate_stage':'使用已保存的 detector outputs；这些候选已经过PP-YOLOE官方NMS（原基准NMS IoU 0.7，候选上限100/图；C组保留 score_threshold=0.001 输出上限100/图），但尚未过本项目的额外IoU/包含/合并后处理。官方NMS前原始网络输出未保存，无法恢复。',
      'baseline_proposal_file':str(BASE_PROPOSALS),'baseline_proposal_sha256':sha(BASE_PROPOSALS),'baseline_detector_weight_sha256':base_doc['metadata']['detector_weights_sha256'],
      'C_proposal_file':str(C_PROPOSALS),'C_proposal_sha256':sha(C_PROPOSALS),'C_detector_weight_sha256':c_doc['metadata']['weight_sha256'],
      'classifier':'固定BioCLIP 2 ViT-L-14编码器+冻结八类头；与本轮一致的15%外扩裁剪、square灰边、预处理；未使用目录类别或XML决定类别',
      'classifier_weight_sha256':sha(BIO_WEIGHT),'classifier_head_sha256':sha(HEAD),'display_detector_threshold_strictly_greater_than':DISPLAY_THRESHOLD,
      'original_iou_rule':'复用第二代原group_candidates实现：全候选按检测分数降序，逐个与已保留winner比较；最高IoU≥0.5则隐藏；不传递聚类；仅最后保留框分数>0.30显示。',
      'containment_rule':'候选检测分数>0.30后按分数降序贪心；与任何保留winner的IoU≥0.5或交集面积/较小框面积达到0.8/0.9时，隐藏较低分框；非传递；不使用分类类别。',
      'merge_rule':{**MERGE_RULE,'score_threshold_strictly_greater_than':DISPLAY_THRESHOLD,'merged_box':'组内坐标并集并裁到原图范围','score':'沿用组内最大检测分数，不求和且不重新预测','class':'合并框重新裁剪后由BioCLIP 2分类'},
      'matching':'与现有评估脚本一致：P/R/F1按类别正确、IoU≥0.5匹配；候选pair按IoU、final_score_product优先级降序一对一贪心；错类体现FP/FN；未匹配框只表示与当前XML未匹配。',
      'image_status':'改善/退步按每张图的XML匹配F1相对原IoU规则变化标记；不等同真实识别结论。possible_mismerge只在分组后用XML作事后诊断：合并组内成员分别匹配到多个XML实例，或合并框与多个XML框IoU≥0.5。分组过程从未读XML。',
      'candidate_threshold_pool':'从已保存的100个/图官方NMS候选中取detector_score>0.30的候选再执行后处理。'}
    result={'metadata':metadata,'policy_names':policy_names,'summaries':summaries,'per_image_deltas':per_image_deltas,'records':records,'images':img_payload}
    outjson=OUT/'postprocess_comparison.json';outjson.write_text(json.dumps(result,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    with (OUT/'逐图逐规则指标.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['model','model_name','policy','policy_name','filename','truth_instances','raw_candidates_over_0_30','display_boxes','TP','FP','FN','unmatched_display_boxes','image_F1','status_vs_iou','possible_mismerge','multi_bird_image']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for m in docs:
            for p in policies:
                stat={x['filename']:x for x in per_image_deltas[m][p]}
                for r in records[m][p]:
                    d=stat[r['filename']];w.writerow({'model':m,'model_name':keys[m],'policy':p,'policy_name':policy_names[p],'filename':r['filename'],'truth_instances':len(r['truth']),'raw_candidates_over_0_30':r['raw_candidates_over_threshold'],'display_boxes':len(r['predictions']),**{k:r['metric'][k] for k in ('TP','FP','FN','unmatched_display_boxes')},'image_F1':r['image_F1'],'status_vs_iou':d['status_vs_iou'],'possible_mismerge':r['possible_mismerge'],'multi_bird_image':r['multi_bird_image']})
    with (OUT/'总体指标.csv').open('w',encoding='utf-8-sig',newline='') as f:
        cols=['model','model_name','policy','policy_name','images','TP','FP','FN','displayed_boxes','unmatched_display_boxes','spatially_matched_wrong_class','no_box_images','Precision_micro','Recall_micro','F1_micro','possible_merge_groups']
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        for m in docs:
            for p in policies:w.writerow({'model':m,'model_name':keys[m],'policy':p,**summaries[m][p]})
    # Offline HTML draws source image, XML truth and each postprocess output side by side.
    payload=json.dumps({'meta':metadata,'names':keys,'policy_names':policy_names,'summaries':summaries,'per_image_deltas':per_image_deltas,'records':records,'images':img_payload},ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
    template=r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>同鸟重复框后处理对照</title><style>
body{font:14px Segoe UI,"Microsoft YaHei",sans-serif;background:#f3f6f8;color:#1d2832;margin:0}.head{position:sticky;top:0;z-index:4;background:#20364b;color:white;padding:14px 18px}.head h1{margin:0 0 5px;font-size:21px}.head p{margin:5px 0;color:#dbe6ef}.tools{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:8px}.tools button,.tools select,.tools input{font:inherit;padding:7px;border-radius:5px;border:1px solid #b9c6d2}.notice{margin:12px;padding:12px;background:#fff2d6;border-left:5px solid #dd9c25;line-height:1.5}.summary{background:white;margin:12px;padding:12px;border-radius:8px;overflow:auto}.summary table{border-collapse:collapse;width:100%;min-width:960px}.summary th,.summary td{border-bottom:1px solid #dae0e4;padding:7px;text-align:right}.summary th:first-child,.summary td:first-child{text-align:left}.cards{padding:0 12px 30px}.card{background:white;margin:12px 0;padding:12px;border-radius:8px;box-shadow:0 1px 5px #0001}.card h2{margin:0 0 7px;font-size:18px}.meta{font-size:12px;color:#566370}.grid{display:grid;grid-template-columns:repeat(5,minmax(180px,1fr));gap:8px;overflow:auto}.panel{min-width:180px;background:#f8fafb;border:1px solid #dce3e8;border-radius:5px;padding:6px}.panel h3{margin:2px 0 6px;font-size:14px}.panel canvas{width:100%;height:auto;background:#121619}.details{margin-top:8px}.details details{background:#f6f8fa;margin:4px 0;padding:7px;border-radius:4px}.table{font-family:Consolas,monospace;font-size:12px}.badge{display:inline-block;background:#e7eef3;border-radius:12px;padding:3px 8px;margin:2px}.improve{color:#087443}.regress{color:#bd2e34}.warning{color:#9a6111}.case-detail{max-height:260px;overflow:auto;white-space:pre-wrap}.more{padding:8px 16px;margin:0 12px 20px}.hidden{display:none}@media(max-width:1050px){.grid{grid-template-columns:repeat(5,220px)}.head{position:static}}
</style><header class="head"><h1>原基准与 C 组：重复框后处理对照</h1><p>固定172张验证图；检测分数严格&gt;0.30；检测候选、BioCLIP 2冻结分类、15%外扩裁剪均固定；只改变额外后处理，不训练、不改权重。</p><div class="tools"><label>检测器 <select id="model"><option value="baseline">原PP-YOLOE+基准</option><option value="C">C组轻量遮挡权重</option></select></label><label>筛选 <select id="filter"><option value="all">全部验证图</option><option value="improve">XML指标改善</option><option value="regress">XML指标退步</option><option value="no_boxes">无框</option><option value="possible_merge">可能误合并</option><option value="multi_bird">多鸟图</option></select></label><input id="search" placeholder="按文件名筛选"><label><input type="checkbox" id="showgt" checked>XML框</label><label><input type="checkbox" id="showpred" checked>预测框</label><button id="export">导出当前筛选JSON</button><span id="shown"></span></div></header>
<div class="notice"><b>读数提醒：</b>未匹配只表示与现有XML在类别正确且IoU≥0.5的匹配规则下未匹配，不自动等于背景误检。这里的“改善/退步”只表示与XML的一致性F1变化。包含阈值分别是交集面积/较小框面积≥0.8或≥0.9；同鸟合并采用完整链接、非传递分组，不能仅凭类别或单纯相交合并。合并分数沿用组内最大检测分数，类别来自合并框重新裁剪后的BioCLIP 2，不是检测器重预测概率。</div><section class="summary"><h2>总体指标（micro，一对一，类别正确且 IoU≥0.5）</h2><div id="summary"></div><div id="rules" class="meta"></div></section><main id="cards" class="cards"></main><button id="more" class="more">加载更多</button><script id="payload" type="application/json">__PAYLOAD__</script><script>
const D=JSON.parse(document.querySelector('#payload').textContent),policies=['iou','contain_08','contain_09','merge'],color={xml:'#21d68b',pred:'#ed5252'},esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function summarize(){let m=document.querySelector('#model').value,S=D.summaries[m],html='<table><tr><th>规则</th><th>TP/FP/FN</th><th>显示框</th><th>未匹配显示框</th><th>位置匹配但类别错</th><th>无框图片</th><th>P</th><th>R</th><th>F1</th><th>疑似误合并组</th></tr>';for(let p of policies){let x=S[p];html+=`<tr><td>${D.policy_names[p]}</td><td>${x.TP}/${x.FP}/${x.FN}</td><td>${x.displayed_boxes}</td><td>${x.unmatched_display_boxes}</td><td>${x.spatially_matched_wrong_class}</td><td>${x.no_box_images}</td><td>${x.Precision_micro.toFixed(4)}</td><td>${x.Recall_micro.toFixed(4)}</td><td>${x.F1_micro.toFixed(4)}</td><td>${x.possible_merge_groups}</td></tr>`}document.querySelector('#summary').innerHTML=html+'</table>';document.querySelector('#rules').textContent=`合并规则：IoU≥${D.meta.merge_rule.iou_min} 且交集/较小框≥${D.meta.merge_rule.intersection_over_smaller_min} 且中心距/较小框对角线≤${D.meta.merge_rule.center_distance_over_smaller_diagonal_max}；完整链接、最多${D.meta.merge_rule.max_members}框、只允许唯一可加入组。基准阈值及权重hash见导出JSON。`}
function draw(c,img,truth,preds,focus=null){let ctx=c.getContext('2d'),im=new Image();im.onload=()=>{let s=Math.min(700/im.width,470/im.height,1);c.width=Math.round(im.width*s);c.height=Math.round(im.height*s);ctx.drawImage(im,0,0,c.width,c.height);const drawbox=(xctx,b,col,label,scale)=>{let [x1,y1,x2,y2]=b.map((v,i)=>v*(i%2===0?img.scale_x:img.scale_y)*scale);xctx.strokeStyle=col;xctx.lineWidth=2.5;xctx.strokeRect(x1,y1,x2-x1,y2-y1);xctx.font='bold 12px Segoe UI';let w=xctx.measureText(label).width;xctx.fillStyle=col;xctx.fillRect(x1,Math.max(0,y1-17),w+7,17);xctx.fillStyle='#111';xctx.fillText(label,x1+3,Math.max(13,y1-4))};let boxes=[];if(document.querySelector('#showgt').checked)truth.forEach((g,i)=>{let b={box:g.bbox_xyxy,type:'xml',label:`XML${i+1} ${g.class_name}`};boxes.push(b);drawbox(ctx,b.box,color.xml,b.label,s)});if(document.querySelector('#showpred').checked)preds.forEach((p,i)=>{let b={box:p.bbox_xyxy,type:'pred',label:`${i+1} ${p.class_name} D${p.detector_score.toFixed(2)} C${p.classifier_top_score.toFixed(2)}`};boxes.push(b);drawbox(ctx,b.box,color.pred,b.label,s)});let z=c.parentElement.querySelector('canvas.zoom'),zctx=z.getContext('2d'),f=focus||preds[0]?.bbox_xyxy||truth[0]?.bbox_xyxy;if(!f){return}let fx=[f[0]*img.scale_x,f[1]*img.scale_y,f[2]*img.scale_x,f[3]*img.scale_y],pad=Math.max((fx[2]-fx[0])*.6,(fx[3]-fx[1])*.6,18),x0=Math.max(0,fx[0]-pad),y0=Math.max(0,fx[1]-pad),x1=Math.min(im.width,fx[2]+pad),y1=Math.min(im.height,fx[3]+pad),sw=x1-x0,sh=y1-y0,zs=Math.min(360/sw,220/sh);z.width=Math.max(1,Math.round(sw*zs));z.height=Math.max(1,Math.round(sh*zs));zctx.drawImage(im,x0,y0,sw,sh,0,0,z.width,z.height);for(let b of boxes){let q=b.box.map((v,i)=>v*(i%2===0?img.scale_x:img.scale_y));zctx.strokeStyle=b.type==='xml'?color.xml:color.pred;zctx.lineWidth=2.5;zctx.strokeRect((q[0]-x0)*zs,(q[1]-y0)*zs,(q[2]-q[0])*zs,(q[3]-q[1])*zs)}};im.src=img.src}
let selected=[],limit=0;function render(){let m=document.querySelector('#model').value,f=document.querySelector('#filter').value,q=document.querySelector('#search').value.toLowerCase();selected=[];for(let i=0;i<D.records[m].iou.length;i++){let ref=D.records[m].iou[i],image=D.images[ref.filename],deltas={};for(let p of policies)deltas[p]=D.records[m][p][i];let st=D.per_image_deltas[m];let status={};for(let p of policies)status[p]=st[p][i].status_vs_iou;let possible=deltas.merge.possible_mismerge,hasbox=p=>deltas[p].predictions.length>0;let match=(f==='all'||f==='multi_bird'&&ref.multi_bird_image||f==='possible_merge'&&possible||f==='no_boxes'&&policies.some(p=>!hasbox(p))||['improve','regress'].includes(f)&&status[policies.slice(1).find(p=>status[p]===(f==='improve'?'改善':'退步'))]);if(!ref.filename.toLowerCase().includes(q)||!match)continue;selected.push({ref,image,deltas,status,possible})}limit=0;document.querySelector('#cards').innerHTML='';summarize();more()}
function more(){let chunk=selected.slice(limit,limit+8);limit+=chunk.length;for(let x of chunk){let card=document.createElement('article');card.className='card';let candidates=x.ref.raw_candidates_over_threshold;card.innerHTML=`<h2>${esc(x.ref.filename)} <span class="badge">XML实例 ${x.ref.truth.length}</span>${x.ref.multi_bird_image?'<span class="badge">多鸟图</span>':''}${x.possible?'<span class="badge warning">可能误合并</span>':''}</h2><div class="meta">高于阈值的原候选：${candidates}；XML仅用于下方事后匹配及可能误合并诊断。评估状态按图片F1相对原IoU规则：${policies.map(p=>`${D.policy_names[p]} ${x.status[p]}`).join('；')}</div><div class="grid">`+`<section class="panel"><h3>XML人工框（现有标注）</h3><canvas data-kind="gt"></canvas><canvas class="zoom"></canvas><div class="meta">每框标注实例编号和标准类别</div></section>`+policies.map(p=>`<section class="panel"><h3>${D.policy_names[p]} <span class="badge ${x.status[p]==='改善'?'improve':x.status[p]==='退步'?'regress':''}">${x.status[p]}</span></h3><canvas data-policy="${p}"></canvas><canvas class="zoom"></canvas><div class="meta">框 ${x.deltas[p].predictions.length}；TP ${x.deltas[p].metric.TP}/FP ${x.deltas[p].metric.FP}/FN ${x.deltas[p].metric.FN}；未匹配显示框 ${x.deltas[p].metric.unmatched_display_boxes}</div><div class="details">${x.deltas[p].predictions.map((d,j)=>`<details><summary>${j+1}. ${esc(d.class_name)} D=${d.detector_score.toFixed(4)} C=${d.classifier_top_score.toFixed(4)} 组${d.group_id}</summary><div class="table">框=${JSON.stringify(d.bbox_xyxy)}<br>分类八类分数=${esc(JSON.stringify(d.classifier_scores))}<br>来源候选=${esc(JSON.stringify(d.candidate_indices))}<br>分组依据=${esc(JSON.stringify(d.grouping_reason))}<br>score来源候选=${d.score_source_candidate_index}；${esc(d.score_rule)}<br>${d.posthoc_gt_diagnostic?.possible_merge_of_multiple_birds?'<b class="warning">事后XML提示可能覆盖多只鸟，请人工看图。</b>':''}</div></details>`).join('')}</div></section>`).join('')+`</div>`;document.querySelector('#cards').appendChild(card);card.querySelectorAll('.panel').forEach(panel=>{let c=panel.querySelector('canvas:not(.zoom)');if(c.dataset.kind==='gt')draw(c,x.image,x.ref.truth,[]);else draw(c,x.image,x.ref.truth,x.deltas[c.dataset.policy].predictions)})}document.querySelector('#shown').textContent=`筛选${selected.length}张，已显示${limit}张`;document.querySelector('#more').hidden=limit>=selected.length}
document.querySelector('#model').onchange=render;document.querySelector('#filter').onchange=render;document.querySelector('#search').oninput=render;document.querySelector('#showgt').onchange=render;document.querySelector('#showpred').onchange=render;document.querySelector('#more').onclick=more;document.querySelector('#export').onclick=()=>{let m=document.querySelector('#model').value,f=document.querySelector('#filter').value,q=document.querySelector('#search').value.toLowerCase();let keep=new Set(selected.map(x=>x.ref.filename));let subset={metadata:D.meta,model:m,filter:f,images:D.records[m].iou.filter(x=>keep.has(x.filename)).map(base=>({filename:base.filename,truth:base.truth,policies:Object.fromEntries(policies.map(p=>[p,{predictions:D.records[m][p][D.records[m].iou.indexOf(base)].predictions,metric:D.records[m][p][D.records[m].iou.indexOf(base)].metric,groups:D.records[m][p][D.records[m].iou.indexOf(base)].group_records}]))}))};let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(subset,null,2)],{type:'application/json'}));a.download=`${m}_当前筛选后处理复核.json`;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};render()
</script></html>'''
    (OUT/'原基准与C组重复框后处理对照.html').write_text(template.replace('__PAYLOAD__',payload),encoding='utf-8')
    result_summary={'metadata':metadata,'models':keys,'summaries':summaries,'total_runtime_seconds':time.perf_counter()-t0,
       'special_cases':{fn:{m:{p:next(r for r in records[m][p] if r['filename']==fn) for p in policies} for m in docs} for fn in ('dashanque_022.jpg','baitoubei_084.jpg','jiamaque_045.jpg')},
       'not_training':True,'not_main_model_change':True}
    (OUT/'postprocess_summary.json').write_text(json.dumps(result_summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'runtime_s':round(time.perf_counter()-t0,2),'crops':len(jobs),'summary':summaries,'special':{fn:{m:{p:{'display':len(next(r for r in records[m][p] if r['filename']==fn)['predictions']),'TP':next(r for r in records[m][p] if r['filename']==fn)['metric']['TP'],'FP':next(r for r in records[m][p] if r['filename']==fn)['metric']['FP'],'FN':next(r for r in records[m][p] if r['filename']==fn)['metric']['FN']} for p in policies} for m in docs} for fn in ('dashanque_022.jpg','baitoubei_084.jpg','jiamaque_045.jpg')},'html':str(OUT/'原基准与C组重复框后处理对照.html')},ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
