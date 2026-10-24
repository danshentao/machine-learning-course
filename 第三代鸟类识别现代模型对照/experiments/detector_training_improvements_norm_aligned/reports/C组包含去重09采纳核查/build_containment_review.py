from __future__ import annotations
import base64,csv,hashlib,json,shutil
from pathlib import Path

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'同鸟重复框后处理对照'
SRC=OLD/'postprocess_comparison.json'
OUT=HERE/'C组包含阈值09逐框人工复核.html'
JSON_OUT=HERE/'新增抑制框与匹配变化.json'
CSV_OUT=HERE/'四方案统一指标.csv'
NOTE=HERE/'核查摘要.md'

def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def iou(a,b):
    iw=max(0,min(a[2],b[2])-max(a[0],b[0])); ih=max(0,min(a[3],b[3])-max(a[1],b[1])); inter=iw*ih
    aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]); ab=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter/(aa+ab-inter) if aa+ab>inter else 0
def predmap(r): return {p['candidate_indices'][0]:p for p in r['predictions'] if p.get('candidate_indices')}
def bestgt(box,truth):
    vals=[(iou(box,t['bbox_xyxy']),j+1) for j,t in enumerate(truth)]
    return max(vals,default=(0,None))
def disag(records): return sum(any(not p.get('class_correct',True) for p in r['metric'].get('spatial_pairs',[])) for r in records)

def build():
    src=read(SRC); rec=src['records']; summaries={};events=[];deltas={};rows=[];exposed={'baseline':[],'C':[]}
    labels={'baseline':'原基准','C':'C组'}
    for model in ('baseline','C'):
        old={r['filename']:r for r in rec[model]['iou']}; new={r['filename']:r for r in rec[model]['contain_09']}
        lost=[];gained=[];byfile={}
        for fn,a in old.items():
            b=new[fn];op=predmap(a);npred=predmap(b);supmap={}
            for cid,pred in npred.items():
                if cid not in op:
                    exposed[model].append({'model':model,'filename':fn,'candidate_index':cid,'bbox_xyxy':pred['bbox_xyxy'],
                      'detector_score':pred['detector_score'],'class_name':pred['class_name'],
                      'classifier_top_score':pred['classifier_top_score'],'classifier_scores':pred.get('classifier_scores',{}),
                      'reason':'包含规则的贪心分组使该候选成为新保留框；原IoU输出中未显示','truth_match':b['metric'].get('pairs',[])})
            for group in b.get('group_records',[]):
                ids=group.get('winner_or_members',[]);winner_id=ids[0] if ids else None
                for s in group.get('suppressed_candidates',[]): supmap[s['proposal_index']]=(winner_id,s)
            one=[]
            for cid,p in op.items():
                if cid in npred: continue
                if cid not in supmap: raise RuntimeError(f'缺失抑制链 {model} {fn} #{cid}')
                wid,detail=supmap[cid];w=npred.get(wid)
                if not w: raise RuntimeError(f'抑制触发框不是显示框 {model} {fn} #{cid}->{wid}')
                reason=detail['suppression_reason']; pi,pg=bestgt(p['bbox_xyxy'],a['truth']);wi,wg=bestgt(w['bbox_xyxy'],a['truth'])
                e={'model':model,'filename':fn,'directory_class':fn.split('_')[0],
                   'suppressed_candidate_index':cid,'suppressor_candidate_index':wid,
                   'suppressed_box_xyxy':p['bbox_xyxy'],'suppressor_box_xyxy':w['bbox_xyxy'],
                   'suppressed_detector_score':p['detector_score'],'suppressor_detector_score':w['detector_score'],
                   'suppressed_class':p['class_name'],'suppressed_classifier_top_score':p['classifier_top_score'],
                   'suppressed_classifier_scores':p.get('classifier_scores',{}),
                   'suppressor_class':w['class_name'],'suppressor_classifier_top_score':w['classifier_top_score'],
                   'suppressor_classifier_scores':w.get('classifier_scores',{}),
                   'iou':reason['iou'],'intersection_over_smaller':reason['intersection_over_smaller'],
                   'rule_trigger':'IoU>=0.5' if reason['iou']>=.5 else '交集/较小框面积>=0.9',
                   'old_spatial_gt_instance':pg if pi>=.5 else None,'old_spatial_gt_iou':pi,
                   'winner_spatial_gt_instance':wg if wi>=.5 else None,'winner_spatial_gt_iou':wi,
                   'different_gt_instances_overlap':bool(pi>=.5 and wi>=.5 and pg!=wg),
                   'multi_bird_image':len(a['truth'])>1,'review_choice':'','review_note':''}
                one.append(e);events.append(e)
            byfile[fn]=one
            ag={x['gt_instance'] for x in a['metric']['pairs']};bg={x['gt_instance'] for x in b['metric']['pairs']}
            for gt in sorted(ag-bg): lost.append({'filename':fn,'gt_instance':gt,'old_match':next(x for x in a['metric']['pairs'] if x['gt_instance']==gt),'new_match':None})
            for gt in sorted(bg-ag): gained.append({'filename':fn,'gt_instance':gt,'old_match':None,'new_match':next(x for x in b['metric']['pairs'] if x['gt_instance']==gt)})
        deltas[model]={'lost_correct_matches':lost,'gained_correct_matches':gained,'lost_count':len(lost),'gained_count':len(gained),'net_tp_change':len(gained)-len(lost)}
        summaries[model]={}
        for policy in ('iou','contain_09'):
            s=src['summaries'][model][policy]
            d={'model':model,'model_name':labels[model],'policy':policy,'policy_name':'原IoU' if policy=='iou' else '包含0.9',
               **{k:s[k] for k in ('images','TP','FP','FN','displayed_boxes','unmatched_display_boxes','spatially_matched_wrong_class','no_box_images','Precision_micro','Recall_micro','F1_micro')},
               'class_disagreement_images':disag(rec[model][policy])}
            summaries[model][policy]=d;rows.append(d)
        src['review_events_'+model]=byfile
    fixed_no_boxes=['baitoubei_084.jpg','dashanque_022.jpg','heihouhongtunbei_121.jpg','jiamaque_045.jpg','meishanque_080.jpg']
    special=['baitoubei_084.jpg','dashanque_022.jpg','jiamaque_045.jpg']
    no_box_status={}
    for fn in fixed_no_boxes:
        no_box_status[fn]={}
        for model in ('baseline','C'):
            no_box_status[fn][model]={}
            for policy in ('iou','contain_09'):
                r=next(x for x in rec[model][policy] if x['filename']==fn)
                no_box_status[fn][model][policy]={'displayed_boxes':len(r['predictions']),'TP':r['metric']['TP'],'FP':r['metric']['FP'],'FN':r['metric']['FN'],'predictions':r['predictions']}
    fns=set(fixed_no_boxes+special)
    for m in ('baseline','C'):
        fns.update(fn for fn,es in src['review_events_'+m].items() if es)
        fns.update(r['filename'] for r in rec[m]['iou'] if len(r['truth'])>1)
    # 页面只内嵌确有抑制、所有多鸟图和固定特殊图；完整172张指标仍在汇总中。
    visual={'baseline':{},'C':{}}
    for m in ('baseline','C'):
        for p in ('iou','contain_09'):
            visual[m][p]=[r for r in rec[m][p] if r['filename'] in fns]
    result={'metadata':{'source_json':str(SRC),'source_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),
             'validation_images':172,'threshold':'detector score strictly > 0.30',
             'classifier':'冻结BioCLIP 2；同一分类头；检测框外扩15%裁剪',
             'containment':'按检测分数降序贪心；IoU>=0.5或交集/较小框面积>=0.9抑制低分框',
             'xml_usage':'仅用于事后匹配与跨实例重叠诊断，未参与候选分组/抑制',
             'candidate_stage':'已保存候选在官方NMS后、项目额外去重前且每图最多100个；官方NMS前输出未保存'},
             'summaries':summaries,'matched_gt_delta':deltas,'events':events,'newly_exposed_candidates':exposed,'no_box_status':no_box_status,'visual_records':visual,
             'images':{fn:src['images'][fn] for fn in sorted(fns) if fn in src['images']},
             'special_filenames':special,'historical_no_box_filenames':fixed_no_boxes}
    write(JSON_OUT,result)
    with CSV_OUT.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    shutil.copy2(OLD/'训练集困难样本人工复核问题清单.html',HERE/'训练集困难样本人工复核问题清单.html')
    shutil.copy2(OLD/'训练集人工复核问题清单.json',HERE/'训练集人工复核问题清单.json')
    payload=json.dumps(result,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
    page=PAGE.replace('\n+','\n')
    page=page.replace('候选记录处于官方NMS后（官方NMS前输出未保存）。','候选记录处于官方NMS后（官方NMS前输出未保存）。框线颜色：XML绿色；原IoU蓝色；包含0.9红色；新增被抑制候选橙色；触发抑制的保留框紫色。')
    page=page.replace(r'"\\ufeff"+',r'"\ufeff"+').replace(r'"\\r\\n"',r'"\r\n"')
    page=page.replace('ev=events(m,fn),lost=', 'ev=events(m,fn),fresh=D.newly_exposed_candidates[m].filter(x=>x.filename===fn),lost=')
    page=page.replace('XML只用于事后核对。</div><div class=\'grid\'>"+grids', 'XML只用于事后核对。</div>"+(fresh.length?"<div class=\'detail\'>包含0.9新进入显示的候选（原IoU最终未显示）："+fresh.map(x=>"候选#"+x.candidate_index+" "+esc(x.class_name)+" D="+x.detector_score.toFixed(4)+" 分类top="+x.classifier_top_score.toFixed(4)+" 框="+JSON.stringify(x.bbox_xyxy)).join("<br>")+"</div>":"")+"<div class=\'grid\'>"+grids')
    page=page.replace("<div class='detail'>\"+esc(text)+\"</div>", "<div class='grid'><div>被抑制框局部放大<br><canvas class='eventZoom' data-box='\"+JSON.stringify(e.suppressed_box_xyxy)+\"' data-color='#f08c00'></canvas></div><div>触发保留框局部放大<br><canvas class='eventZoom' data-box='\"+JSON.stringify(e.suppressor_box_xyxy)+\"' data-color='#942c9f'></canvas></div></div><div class='detail'>\"+esc(text)+\"</div>")
    page=page.replace('function matchline(r){', "function drawZoom(c,src,b,color){let im=new Image();im.onload=()=>{let pad=Math.max(10,Math.round(Math.max(b[2]-b[0],b[3]-b[1])*.12)),x1=Math.max(0,Math.floor(b[0]-pad)),y1=Math.max(0,Math.floor(b[1]-pad)),x2=Math.min(im.width,Math.ceil(b[2]+pad)),y2=Math.min(im.height,Math.ceil(b[3]+pad)),w=Math.max(1,x2-x1),h=Math.max(1,y2-y1),s=Math.min(600/w,220/h);c.width=Math.max(1,Math.round(w*s));c.height=Math.max(1,Math.round(h*s));let x=c.getContext('2d');x.drawImage(im,x1,y1,w,h,0,0,c.width,c.height);x.strokeStyle=color;x.lineWidth=3;x.strokeRect((b[0]-x1)*s,(b[1]-y1)*s,(b[2]-b[0])*s,(b[3]-b[1])*s)};im.src=src}function matchline(r){")
    page=page.replace('draw(c,img,a.truth,r.predictions,pp==="iou"?events(mm,fn):[],"old")','draw(c,img,a.truth,r.predictions,pp==="iou"?events(mm,fn):[],pp==="iou"?"old":"new")')
    page=page.replace('card.querySelectorAll(".choice").forEach', 'card.querySelectorAll(".eventZoom").forEach(c=>drawZoom(c,img,JSON.parse(c.dataset.box),c.dataset.color));card.querySelectorAll(".choice").forEach')
    OUT.write_text(page.replace('__PAYLOAD__',payload),encoding='utf-8')
    c_events=[e for e in events if e['model']=='C'];c_lost=deltas['C']['lost_correct_matches'];c_gain=deltas['C']['gained_correct_matches']
    c_cross=[e for e in c_events if e['different_gt_instances_overlap']]
    base=summaries['baseline']['iou']; ci=summaries['C']['iou']; cn=summaries['C']['contain_09']
    lines=['# C组包含阈值0.9采纳核查摘要','',
      '本轮复用固定172张验证集预测，没有重训、改权重或切换主方案。它是反复探索用验证集，不是独立测试。未匹配框不自动等于背景误检。','',
      f"- 输入候选记录SHA-256：{result['metadata']['source_sha256']}",
      f"- 候选阶段：{result['metadata']['candidate_stage']}",
      f"- C组新增被抑制的原显示框：{len(c_events)}个；多鸟图事件{sum(e['multi_bird_image'] for e in c_events)}个；两个框事后分别与不同XML实例IoU>=0.5的候选{len(c_cross)}个。",
      f"- C组另有{len(exposed['C'])}个候选在包含规则下新成为显示框："+'；'.join(f"{x['filename']} 候选#{x['candidate_index']} D={x['detector_score']:.4f} {x['class_name']} 分类top={x['classifier_top_score']:.4f}" for x in exposed['C'])+'.',
      f"- C组逐实例正确匹配变化：丢失{len(c_lost)}个，新增{len(c_gain)}个，净TP={len(c_gain)-len(c_lost)}。","",
      '## C组原IoU到包含0.9','',
      f"TP/FP/FN：{ci['TP']}/{ci['FP']}/{ci['FN']} → {cn['TP']}/{cn['FP']}/{cn['FN']}；P/R/F1：{ci['Precision_micro']:.4f}/{ci['Recall_micro']:.4f}/{ci['F1_micro']:.4f} → {cn['Precision_micro']:.4f}/{cn['Recall_micro']:.4f}/{cn['F1_micro']:.4f}；显示框{ci['displayed_boxes']}→{cn['displayed_boxes']}；未匹配框{ci['unmatched_display_boxes']}→{cn['unmatched_display_boxes']}；无框图均{ci['no_box_images']}。类别分歧图{ci['class_disagreement_images']}→{cn['class_disagreement_images']}。","",
      '抑制事件数比显示框净减少多1：0.9规则还让一个旧IoU隐藏的候选成为新显示框。因此不能把抑制事件数量直接当作显示框净变化。该新增候选及全部隐藏关系均可在逐图页面查看。正确匹配丢失/新增分别为：','']
    for x in c_lost:
        ev=next((e for e in c_events if e['filename']==x['filename'] and e['old_spatial_gt_instance']==x['gt_instance']),None)
        lines.append(f"- {x['filename']} XML实例{x['gt_instance']}：原框IoU={x['old_match']['iou']:.4f}；新规则下该实例未再匹配。"+
          (f"抑制链为候选#{ev['suppressed_candidate_index']}（D={ev['suppressed_detector_score']:.4f}、{ev['suppressed_class']}、分类top={ev['suppressed_classifier_top_score']:.4f}，与XML IoU={ev['old_spatial_gt_iou']:.4f}）→候选#{ev['suppressor_candidate_index']}（D={ev['suppressor_detector_score']:.4f}、{ev['suppressor_class']}、分类top={ev['suppressor_classifier_top_score']:.4f}，与XML IoU={ev['winner_spatial_gt_iou']:.4f}）；框IoU={ev['iou']:.4f}、交集/较小框={ev['intersection_over_smaller']:.4f}。" if ev else ""))
    lines += [f"- 新增正确匹配：{x['filename']} XML实例{x['gt_instance']}。" for x in c_gain] or ['- 新增正确匹配：0。']
    lines += ['','## 四方案对照','','|方案|TP/FP/FN|P|R|F1|无框图|类别分歧图|未匹配显示框|显示框|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m in ('baseline','C'):
        for p in ('iou','contain_09'):
            x=summaries[m][p];lines.append(f"|{x['model_name']}＋{x['policy_name']}|{x['TP']}/{x['FP']}/{x['FN']}|{x['Precision_micro']:.4f}|{x['Recall_micro']:.4f}|{x['F1_micro']:.4f}|{x['no_box_images']}|{x['class_disagreement_images']}|{x['unmatched_display_boxes']}|{x['displayed_boxes']}|")
    lines += ['','类别分歧图定义为至少一个空间IoU>=0.5的一对一匹配框类别不同。未匹配显示框仅为XML空间不匹配统计，疑似漏标仍待人工判断。','',
      '## 特殊图片','',
      '- baitoubei_084.jpg：叶枝候选与主鸟框分离，0.9包含去重不会移除；必须单独判断，不可把它算成去重收益。',
      '- dashanque_022.jpg：两框没有达到0.9包含关系；仍保留两个框（1个匹配、1个未匹配）。',
      '- jiamaque_045.jpg：找回的有效家麻雀框在C组两规则下都保留。',
      '- 历史五张无框图可由页面筛选查看。本次没有根据文件名特殊处理。','',
      '### 历史五张无框图逐方案显示框数','','|图片|原基准 IoU / 0.9|C组 IoU / 0.9|','|---|---|---|']
    for fn in fixed_no_boxes:
        q=no_box_status[fn]
        def stat(z): return f"{z['displayed_boxes']}框（TP/FP/FN {z['TP']}/{z['FP']}/{z['FN']}）"
        lines.append(f"|{fn}|{stat(q['baseline']['iou'])} / {stat(q['baseline']['contain_09'])}|{stat(q['C']['iou'])} / {stat(q['C']['contain_09'])}|")
    lines += ['',
      '## 相对原基准与取舍','',
      f"C组＋0.9相对原基准＋原IoU：TP {cn['TP']-base['TP']:+d}、FP {cn['FP']-base['FP']:+d}、FN {cn['FN']-base['FN']:+d}，F1 {cn['F1_micro']-base['F1_micro']:+.4f}，无框图 {cn['no_box_images']-base['no_box_images']:+d}，未匹配显示框 {cn['unmatched_display_boxes']-base['unmatched_display_boxes']:+d}。相对C组原IoU，F1提高{cn['F1_micro']-ci['F1_micro']:+.4f}但Recall变化{cn['Recall_micro']-ci['Recall_micro']:+.4f}，且具体丢失2个原正确匹配、没有新增匹配。现有证据不足以自动采纳；优先复核两个丢失实例以及多鸟图事件。",
      '训练集入口沿用上一轮30张困难样本的原决定和8张优先疑点，单独复制到本目录；jiamaque_087仍待补标或排除决定。验证图与验证复核结果不进入训练。若修订验证标注，必须用同一新版标注重评原基准与A/B/C。','']
    NOTE.write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'html':str(OUT),'json':str(JSON_OUT),'csv':str(CSV_OUT),'note':str(NOTE),
       'events':len(events),'C_events':len(c_events),'C_lost':len(c_lost),'C_gained':len(c_gain),
       'C_cross':len(c_cross),'visual_images':len(fns),'summary':summaries},ensure_ascii=False,indent=2))

PAGE=r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>C组包含去重0.9人工核查</title>
<style>
body{margin:0;background:#f2f5f8;color:#172337;font:14px "Segoe UI","Microsoft YaHei",sans-serif}header{background:#142943;color:white;padding:16px 22px;position:sticky;top:0;z-index:5}h1{font-size:21px;margin:0 0 6px}.muted{color:#5b6877;font-size:12px;line-height:1.55}.headnote{color:#dce8f3}.wrap{max-width:1800px;margin:auto;padding:16px}.box{background:white;border:1px solid #d7e0e8;border-radius:9px;padding:13px;margin-bottom:12px}.controls{display:flex;gap:9px;align-items:center;flex-wrap:wrap}.controls input,.controls select,.controls button{padding:7px;border:1px solid #bdc9d5;border-radius:5px;background:#fff}.controls button{cursor:pointer;background:#e9f1f9}.summary table{border-collapse:collapse;width:100%;font-size:12px}.summary td,.summary th{border:1px solid #d5dee7;padding:6px;text-align:center}.summary th{background:#edf2f7}.card{background:white;border:1px solid #d5dee7;border-radius:9px;padding:13px;margin:12px 0}.card h2{font-size:18px;margin:0 0 5px}.badge{display:inline-block;border-radius:14px;padding:3px 8px;background:#e8eef5;font-size:11px;margin-left:5px}.danger{background:#ffe2df}.notice{background:#fff8df}.good{background:#e1f6e9}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:9px;margin-top:9px}.panel{border:1px solid #dbe2ea;border-radius:7px;padding:8px;min-width:0}.panel h3{font-size:14px;margin:2px 0 7px}.panel canvas{width:100%;height:auto;display:block;background:#eee;border:1px solid #dce2e8}.zoom{margin-top:5px}.event{border:1px solid #e0e5eb;border-radius:7px;padding:9px;margin:7px 0}.eventhead{font-weight:bold;color:#813800}.detail{font:12px/1.5 Consolas,monospace;white-space:pre-wrap;background:#f6f8fa;padding:7px;border-radius:5px;margin:6px 0}.event select,.event textarea{padding:6px;border:1px solid #c5cfd9;border-radius:5px}.event textarea{width:96%;height:38px;display:block;margin-top:4px}a{display:inline-block;background:#e5f0fb;border-radius:5px;padding:7px 9px;text-decoration:none;color:#153c63;margin:5px 0}.smalltag{font-size:12px}.events{max-height:none}@media(max-width:650px){header{position:relative}.wrap{padding:7px}}
+</style></head><body><header><h1>C组＋包含阈值0.9：逐框人工核查</h1><div class="headnote">固定172张验证图｜检测分数严格&gt;0.30｜BioCLIP 2冻结分类头、框外扩15%｜XML只用于事后评估</div></header>
+<div class="wrap"><div class="box notice"><b>边界：</b>验证集反复探索结果，不是独立测试。未匹配只表示与现有XML不匹配，不自动等同于背景误检。去重没有使用XML、目录类别或人工答案。候选记录处于官方NMS后（官方NMS前输出未保存）。训练复核与验证复核隔离，验证图片及其复核结论不得加入训练。<br><a href="训练集困难样本人工复核问题清单.html">打开训练集困难样本复核页：保留30张原决定，优先检查8张疑点</a> jiamaque_087继续待补标/排除决定；若验证标注有修订，须用同一新版标注重评原基准和A/B/C。</div>
+<div class="box summary"><h2>四方案统一比较</h2><div id="summary"></div><div class="muted">实例级micro：类别正确且IoU≥0.5，一对一匹配。类别分歧图指至少一处空间匹配类别错误。FP含错类匹配及空间未匹配候选；未匹配框不代表确认背景误检。</div></div>
+<div class="box controls"><label>查看模型 <select id="model"><option value="C">C组</option><option value="baseline">原基准</option></select></label><label>筛选 <select id="filter"><option value="priority">优先：TP丢失/多鸟/跨实例</option><option value="all_events">所有新增抑制</option><option value="lost_tp">正确匹配丢失</option><option value="multibird">有抑制的多鸟图</option><option value="cross_gt">可能抑制不同鸟</option><option value="no_box">历史5张无框图</option><option value="special">三张重点图</option><option value="all">全部相关图</option></select></label><label>文件名 <input id="search" placeholder="筛选文件名"></label><label><input type="checkbox" id="xml" checked> XML框</label><label><input type="checkbox" id="pred" checked> 预测框</label><label><input type="checkbox" id="supp" checked> 抑制候选/触发框</label><button id="exportj">导出人工复核JSON</button><button id="exportc">导出逐框CSV</button><span id="count" class="muted"></span></div>
+<main id="cards"></main></div><script id="payload" type="application/json">__PAYLOAD__</script><script>
+const D=JSON.parse(document.getElementById("payload").textContent),MT={baseline:"原基准",C:"C组"},PT={iou:"原IoU",contain_09:"包含0.9"};
+const esc=s=>String(s==null?"":s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));
+const rec=(m,p)=>D.visual_records[m][p],by=(m,p,fn)=>rec(m,p).find(x=>x.filename===fn),events=(m,fn)=>D.events.filter(e=>e.model===m&&e.filename===fn);
+const fns=[...new Set(Object.keys(D.images))].sort();
+function summary(){let h="<table><tr><th>方案</th><th>TP/FP/FN</th><th>P</th><th>R</th><th>F1</th><th>无框图</th><th>类别分歧图</th><th>未匹配显示框</th><th>显示框</th></tr>";for(const m of ["baseline","C"])for(const p of ["iou","contain_09"]){let x=D.summaries[m][p];h+="<tr><td>"+MT[m]+"＋"+PT[p]+"</td><td>"+x.TP+"/"+x.FP+"/"+x.FN+"</td><td>"+x.Precision_micro.toFixed(4)+"</td><td>"+x.Recall_micro.toFixed(4)+"</td><td>"+x.F1_micro.toFixed(4)+"</td><td>"+x.no_box_images+"</td><td>"+x.class_disagreement_images+"</td><td>"+x.unmatched_display_boxes+"</td><td>"+x.displayed_boxes+"</td></tr>"}document.getElementById("summary").innerHTML=h+"</table>"}
+function draw(c,src,truth,preds,ev,kind){let im=new Image();im.onload=()=>{let s=Math.min(1,620/im.width),w=Math.round(im.width*s),h=Math.round(im.height*s);c.width=w;c.height=h;let x=c.getContext("2d");x.drawImage(im,0,0,w,h);function box(b,col,label){x.strokeStyle=col;x.lineWidth=2.5;x.strokeRect(b[0]*s,b[1]*s,(b[2]-b[0])*s,(b[3]-b[1])*s);x.fillStyle=col;x.font="bold 12px sans-serif";x.fillText(label,b[0]*s+3,Math.max(13,b[1]*s+13))}if(document.getElementById("xml").checked)truth.forEach((t,i)=>box(t.bbox_xyxy,"#10a968","XML "+(i+1)+" "+t.class_name));if(kind!=="xml"&&document.getElementById("pred").checked)preds.forEach((p,i)=>box(p.bbox_xyxy,kind==="old"?"#1268c4":"#d33232",(i+1)+" "+p.class_name+" D"+p.detector_score.toFixed(2)+" C"+p.classifier_top_score.toFixed(2)));if(kind==="old"&&document.getElementById("supp").checked)ev.forEach(e=>{box(e.suppressed_box_xyxy,"#f08c00","被抑制 #"+e.suppressed_candidate_index);box(e.suppressor_box_xyxy,"#942c9f","触发保留 #"+e.suppressor_candidate_index)})};im.src=src}
+function matchline(r){return "显示框 "+r.predictions.length+"；TP/FP/FN "+r.metric.TP+"/"+r.metric.FP+"/"+r.metric.FN+"；未匹配 "+r.metric.unmatched_display_boxes+"；位置匹配错类 "+r.metric.spatially_matched_wrong_class}
+function localData(){try{return JSON.parse(localStorage.getItem("c09-human-v1")||"{}")}catch(e){return {}}}
+function save(id,k,v){let d=localData();d[id]=d[id]||{};d[id][k]=v;localStorage.setItem("c09-human-v1",JSON.stringify(d))}
+function eventCard(e){let id=e.model+"|"+e.filename+"|"+e.suppressed_candidate_index,s=localData()[id]||{};let opts=["同鸟重复框","不同鸟被误删","背景候选被删除","框偏差","暂不能确定"];let options="<option value=''>待复核</option>"+opts.map(v=>"<option "+(s.choice===v?"selected":"")+">"+v+"</option>").join("");let text="检测器分数："+e.suppressed_detector_score.toFixed(4)+" → "+e.suppressor_detector_score.toFixed(4)+"；分类："+e.suppressed_class+" ("+e.suppressed_classifier_top_score.toFixed(4)+") → "+e.suppressor_class+" ("+e.suppressor_classifier_top_score.toFixed(4)+")\\n触发条件："+e.rule_trigger+"；IoU="+e.iou.toFixed(4)+"；交集/较小框="+e.intersection_over_smaller.toFixed(4)+"\\n被抑制框="+JSON.stringify(e.suppressed_box_xyxy)+"；保留框="+JSON.stringify(e.suppressor_box_xyxy)+"\\nXML事后诊断：候选实例"+(e.old_spatial_gt_instance||"无")+" IoU="+e.old_spatial_gt_iou.toFixed(3)+"；保留实例"+(e.winner_spatial_gt_instance||"无")+" IoU="+e.winner_spatial_gt_iou.toFixed(3)+(e.different_gt_instances_overlap?"；注意：分别与不同XML实例重叠":"");return "<div class='event'><div class='eventhead'>"+MT[e.model]+" "+e.filename+"：候选#"+e.suppressed_candidate_index+" 被保留框#"+e.suppressor_candidate_index+"抑制</div><div class='detail'>"+esc(text)+"</div><label>人工结论 <select class='choice' data-id='"+esc(id)+"'>"+options+"</select></label><textarea class='note' data-id='"+esc(id)+"' placeholder='复核说明'>"+esc(s.note||"")+"</textarea></div>"}
+function render(){summary();let m=document.getElementById("model").value,f=document.getElementById("filter").value,q=document.getElementById("search").value.toLowerCase();let list=fns.filter(fn=>{let a=by(m,"iou",fn);if(!a)return false;let ev=events(m,fn),lost=D.matched_gt_delta[m].lost_correct_matches.some(x=>x.filename===fn),cross=ev.some(x=>x.different_gt_instances_overlap),multi=a.truth.length>1,no=D.historical_no_box_filenames.includes(fn),sp=D.special_filenames.includes(fn);let ok=f==="all"||f==="priority"&&(lost||cross||multi&&ev.length)||f==="all_events"&&ev.length||f==="lost_tp"&&lost||f==="multibird"&&multi&&ev.length||f==="cross_gt"&&cross||f==="no_box"&&no||f==="special"&&sp;return ok&&fn.toLowerCase().includes(q)});let root=document.getElementById("cards");root.innerHTML="";document.getElementById("count").textContent="当前"+MT[m]+"：显示 "+list.length+" 张；本组新增抑制 "+D.events.filter(e=>e.model===m).length+" 框";
+for(const fn of list){let img=D.images[fn].src,a=by(m,"iou",fn),n=by(m,"contain_09",fn),ev=events(m,fn),lost=D.matched_gt_delta[m].lost_correct_matches.filter(x=>x.filename===fn),gain=D.matched_gt_delta[m].gained_correct_matches.filter(x=>x.filename===fn);let card=document.createElement("section");card.className="card";let badges=(a.truth.length>1?"<span class='badge'>多鸟图</span>":"")+(lost.length?"<span class='badge danger'>正确匹配丢失 "+lost.length+"</span>":"")+(ev.some(x=>x.different_gt_instances_overlap)?"<span class='badge danger'>可能跨实例抑制</span>":"")+(D.historical_no_box_filenames.includes(fn)?"<span class='badge good'>历史无框图</span>":"");let grids="<section class='panel'><h3>XML框（现有标注）</h3><canvas class='view' data-kind='xml'></canvas><div class='muted'>"+a.truth.map((t,i)=>"实例"+(i+1)+" "+esc(t.class_name)+" "+JSON.stringify(t.bbox_xyxy)).join("<br>")+"</div></section>";for(let mm of ["baseline","C"])for(let pp of ["iou","contain_09"]){let r=by(mm,pp,fn);if(!r)continue;grids+="<section class='panel'><h3>"+MT[mm]+" · "+PT[pp]+"</h3><canvas class='view' data-model='"+mm+"' data-policy='"+pp+"'></canvas><div class='muted'>"+matchline(r)+"</div></section>"}card.innerHTML="<h2>"+esc(fn)+"<span class='badge'>目录提示 "+esc(fn.split("_")[0])+"</span><span class='badge'>XML实例 "+a.truth.length+"</span>"+badges+"</h2><div class='muted'>"+m+"组实例匹配：丢失 "+(lost.map(x=>"实例"+x.gt_instance).join("、")||"无")+"；新增 "+(gain.map(x=>"实例"+x.gt_instance).join("、")||"无")+"。XML只用于事后核对。</div><div class='grid'>"+grids+"</div><h3>新增抑制关系（原IoU可见、0.9最终不显示）</h3>"+(ev.length?ev.map(eventCard).join(""):"<div class='muted'>本图无新增抑制框。</div>");root.appendChild(card);card.querySelectorAll("canvas.view").forEach(c=>{let kind=c.dataset.kind;if(kind==="xml")draw(c,img,a.truth,[],[],"xml");else{let mm=c.dataset.model,pp=c.dataset.policy,r=by(mm,pp,fn);draw(c,img,a.truth,r.predictions,pp==="iou"?events(mm,fn):[],"old")}});card.querySelectorAll(".choice").forEach(x=>x.onchange=()=>save(x.dataset.id,"choice",x.value));card.querySelectorAll(".note").forEach(x=>x.oninput=()=>save(x.dataset.id,"note",x.value))}}
+function dl(blob,name){let a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1200)}
+function allReviews(){let saved=localData();return D.events.map(e=>{let id=e.model+"|"+e.filename+"|"+e.suppressed_candidate_index;return {...e,review_choice:saved[id]?.choice||"",review_note:saved[id]?.note||""}})}
+function expj(){dl(new Blob([JSON.stringify({metadata:D.metadata,events:allReviews()},null,2)],{type:"application/json"}),"包含去重09人工复核.json")}
+function expc(){let rs=allReviews(),cols=Object.keys(rs[0]||{}),q=x=>'"'+String(typeof x==="object"?JSON.stringify(x):x==null?"":x).replaceAll('"','""')+'"';dl(new Blob(["\\ufeff"+[cols.map(q).join(","),...rs.map(r=>cols.map(c=>q(r[c])).join(","))].join("\\r\\n")],{type:"text/csv;charset=utf-8"}),"新增抑制逐框复核.csv")}
+["model","filter","xml","pred","supp"].forEach(id=>document.getElementById(id).onchange=render);document.getElementById("search").oninput=render;document.getElementById("exportj").onclick=expj;document.getElementById("exportc").onclick=expc;render();
+</script></body></html>'''

if __name__=='__main__': build()

