from __future__ import annotations
import base64,hashlib,json
from pathlib import Path

ROOT=Path(r'E:\机器学习课设\第三代鸟类识别现代模型对照')
EXP=ROOT/'experiments'/'detector_training_improvements_norm_aligned'
OLD=EXP/'reports'/'同鸟重复框后处理对照'
OUT=EXP/'reports'/'包含去重回退记录'
SOURCE=OLD/'postprocess_comparison.json'
EVENTS=EXP/'reports'/'C组包含去重09采纳核查'/'新增抑制框与匹配变化.json'
FNS=['meishanque_046.jpg','shanmaque_078.jpg']

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,d): p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    src=read(SOURCE); eventdoc=read(EVENTS)
    opinions={
      'record_type':'用户人工复核意见',
      'source':'用户人工查看后明确确认',
      'recorded_date_local':'2026-10-08',
      'scope':'只作为框选择与定位意见；不修改XML、类别或模型权重。',
      'decisions':[
        {'filename':'meishanque_046.jpg','opinion':'蓝框优于红框，红框包含过多背景。','interpretation':'蓝框指C组原IoU规则中此前被包含0.9规则抑制的候选框；红框指包含0.9规则保留框。'},
        {'filename':'shanmaque_078.jpg','opinion':'蓝框相对更合适，但尾部覆盖仍不完整，不能将其认定为完美标注。','interpretation':'蓝框指C组原IoU规则中此前被包含0.9规则抑制的候选框；红框指包含0.9规则保留框。'}
      ],
      'actions_taken':{'xml_modified':False,'species_labels_modified':False,'weights_modified':False,'training_restarted':False,'D_group_started':False,'new_dedup_rule_tuned':False}
    }
    write(OUT/'人工复核意见_用户确认.json',opinions)

    files={
      'project_readme':ROOT/'README.md',
      'official_fixed_comparison':EXP/'scripts'/'compare_end_to_end.py',
      'official_iou_grouping':Path(r'E:\机器学习课设\第二代鸟类检测与细粒度分类\scripts\build_deduplicated_gallery.py'),
      'postprocess_experiment':OLD/'compare_postprocess_rules.py',
      'fixed_result_metadata':EXP/'reports'/'fixed_val_comparison'/'fixed_val_results.json',
      'historic_postprocess_json':SOURCE,
    }
    audit={
      'date_local':'2026-10-08',
      'conclusion':'实验规则未采纳；当前主方案原本一直使用IoU去重，本次无需回退代码/配置。',
      'main_scheme':'原PP-YOLOE+基准检测器 + 冻结BioCLIP 2 + detector score严格>0.30 + 检测分数降序、IoU>=0.5非传递去重 + 裁剪外扩15%。',
      'C_group_default_display':'C组固定验证展示继续使用同一原IoU去重；C组作为偏召回实验对照。',
      'verified_evidence':[
        {'file':str(files['project_readme']),'sha256':sha(files['project_readme']),'finding':'项目说明已写明检测框按检测分数优先、IoU>=0.5贪心去重，并采用>0.30阈值及15%外扩。'},
        {'file':str(files['official_fixed_comparison']),'sha256':sha(files['official_fixed_comparison']),'finding':'固定验证的A/B/C对照导入build_deduplicated_gallery.group_candidates，对每组winner使用detector_score>0.30；metadata.cross_class_dedup记为置信度降序、IoU>=0.5、非传递。'},
        {'file':str(files['official_iou_grouping']),'sha256':sha(files['official_iou_grouping']),'finding':'group_candidates的IOU常量为0.5；每个候选只与已保留winner比较。'},
        {'file':str(files['fixed_result_metadata']),'sha256':sha(files['fixed_result_metadata']),'finding':'当前固定验证结果元数据记录IoU>=0.5贪心去重及15%裁剪外扩。'},
        {'file':str(files['postprocess_experiment']),'sha256':sha(files['postprocess_experiment']),'finding':'包含0.8/0.9与合并逻辑只在后处理比较脚本中离线生成独立对照，没有被正式固定验证/展示代码导入。'},
        {'file':str(files['historic_postprocess_json']),'sha256':sha(files['historic_postprocess_json']),'finding':'历史JSON保留baseline/C四种规则结果；本次只读。'}
      ],
      'changes_to_existing_project':[],
      'new_outputs_only':str(OUT),
      'models_or_weights_changed':False,
      'xml_or_labels_changed':False,
      'historical_experiments_deleted_or_overwritten':False,
      'default_rule_switched':False
    }
    write(OUT/'默认规则核查.json',audit)

    previews=[]
    for fn in FNS:
        records={m:{p:next(r for r in src['records'][m][p] if r['filename']==fn) for p in ('iou','contain_09')} for m in ('baseline','C')}
        evs=[e for e in eventdoc['events'] if e['filename']==fn]
        previews.append({'filename':fn,'width':records['C']['iou']['width'],'height':records['C']['iou']['height'],
          'truth':records['C']['iou']['truth'],'baseline_iou':records['baseline']['iou'],
          'C_iou':records['C']['iou'],'C_contain09':records['C']['contain_09'],'suppression_events':evs,
          'user_opinion':next(x['opinion'] for x in opinions['decisions'] if x['filename']==fn),
          'image':src['images'][fn]['src']})
    payload=json.dumps({'items':previews,'threshold':0.30,'iou_threshold':0.5,'crop_margin':0.15,'decision':'实验规则未采纳；正式默认保留原IoU去重'},ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
    # 模板中的行首加号是排版标记，写出HTML前移除。
    html=PAGE.replace('__DATA__',payload).replace('\n+','\n')
    (OUT/'两图IoU回退预览.html').write_text(html,encoding='utf-8')

    print(json.dumps({'output':str(OUT),'default_rule':'IoU >= 0.5','containment_was_default':False,
      'previews':[{ 'filename':x['filename'],'baseline_iou_boxes':len(x['baseline_iou']['predictions']),
       'C_iou_boxes':len(x['C_iou']['predictions']),'C_contain09_boxes':len(x['C_contain09']['predictions']),
       'restored_candidate_ids':[e['suppressed_candidate_index'] for e in x['suppression_events'] if e['model']=='C']} for x in previews],
      'modified_existing_project_files':[]},ensure_ascii=False,indent=2))

PAGE=r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>原IoU规则回退预览：两张人工复核图</title>
<style>
body{margin:0;background:#f3f5f8;color:#1b2838;font:14px "Segoe UI","Microsoft YaHei",sans-serif}header{background:#172d49;color:#fff;padding:18px 24px}h1{font-size:22px;margin:0 0 6px}.muted{font-size:12px;color:#58687a;line-height:1.5}.headnote{color:#dce7f3}.wrap{max-width:1700px;margin:auto;padding:16px}.notice,.card{background:#fff;border:1px solid #d7dfe8;border-radius:9px;padding:14px;margin-bottom:14px}.notice{line-height:1.65;background:#fff9e8}.controls{display:flex;gap:15px;align-items:center;flex-wrap:wrap}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:10px;margin-top:10px}.panel{border:1px solid #dde4eb;border-radius:7px;padding:8px}.panel h3{margin:3px 0 8px;font-size:15px}.panel canvas{display:block;width:100%;height:auto;background:#eee;border:1px solid #d6dee6}.details{background:#f5f7f9;padding:7px;margin-top:7px;border-radius:5px;font:12px/1.5 Consolas,monospace;white-space:pre-wrap}.event{margin:8px 0;padding:9px;border-left:4px solid #f08c00;background:#fff7eb}.badge{background:#e9eff5;border-radius:20px;padding:3px 8px;margin-left:6px;font-size:11px}.blue{color:#1559b5}.red{color:#c5232c}.green{color:#07844d}.orange{color:#c87500}.purple{color:#822997}input,button{padding:7px;border:1px solid #c4d0dc;border-radius:5px}.small{font-size:12px}
+</style></head><body><header><h1>包含去重规则未采纳：恢复原IoU框</h1><div class="headnote">默认方案：原基准 PP-YOLOE+ + 冻结 BioCLIP 2；检测分数严格 &gt;0.30；原IoU≥0.5贪心去重；裁剪外扩15%。C组仅为偏召回实验对照，其默认展示也采用原IoU。</div></header><main class="wrap"><div class="notice"><b>复核记录：</b><span class="green">XML人工框</span> · <span class="blue">原IoU最终显示框</span> · <span class="red">包含0.9最终显示框</span> · <span class="orange">包含0.9抑制的框</span> · <span class="purple">触发抑制的保留框</span><br>包含0.9从未接入项目默认推理或正式展示，本轮只记录“实验规则未采纳”，未改配置、预测、权重或XML。原IoU结果会如实显示同图其余通过阈值的框，不强制只保留一个。框按原图像素坐标绘制，画布缩放系数由原图尺寸/记录图像尺寸计算。</div><div class="notice"><b>用户人工查看意见：</b><br>meishanque_046.jpg：蓝框优于红框，红框包含过多背景。<br>shanmaque_078.jpg：蓝框相对更合适，但尾部覆盖仍不完整，不能将其认定为完美标注。<br>两张意见只针对框选择与定位；没有据此更改物种、XML或权重。</div><section id="cards"></section></main><script id="data" type="application/json">__DATA__</script><script>
+const D=JSON.parse(document.querySelector("#data").textContent),esc=s=>String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));
+function box(ctx,b,color,label,sx,sy){ctx.strokeStyle=color;ctx.lineWidth=Math.max(2,Math.round(Math.min(sx,sy)*2));ctx.strokeRect(b[0]*sx,b[1]*sy,(b[2]-b[0])*sx,(b[3]-b[1])*sy);ctx.font="bold 13px sans-serif";ctx.fillStyle=color;ctx.fillText(label,Math.max(1,b[0]*sx+3),Math.max(14,b[1]*sy+14))}
+function draw(canvas,item,preds,color,kind){let im=new Image();im.onload=()=>{canvas.width=im.naturalWidth;canvas.height=im.naturalHeight;let x=canvas.getContext("2d");x.drawImage(im,0,0);let sx=canvas.width/item.width,sy=canvas.height/item.height;item.truth.forEach((t,i)=>box(x,t.bbox_xyxy,"#079652","XML "+(i+1)+" "+t.class_name,sx,sy));preds.forEach((p,i)=>box(x,p.bbox_xyxy,color,(i+1)+" "+p.class_name+" D"+p.detector_score.toFixed(3)+" C"+p.classifier_top_score.toFixed(3),sx,sy));if(kind==="contain")item.suppression_events.filter(e=>e.model==="C").forEach(e=>{box(x,e.suppressed_box_xyxy,"#ef8500","抑制候选 #"+e.suppressed_candidate_index,sx,sy);box(x,e.suppressor_box_xyxy,"#8c2697","保留触发框 #"+e.suppressor_candidate_index,sx,sy)})};im.src=item.image}
+function crop(canvas,item,b,color){let im=new Image();im.onload=()=>{let sx=im.naturalWidth/item.width,sy=im.naturalHeight/item.height,x0=Math.max(0,Math.floor(b[0]*sx)),y0=Math.max(0,Math.floor(b[1]*sy)),x1=Math.min(im.naturalWidth,Math.ceil(b[2]*sx)),y1=Math.min(im.naturalHeight,Math.ceil(b[3]*sy)),w=Math.max(1,x1-x0),h=Math.max(1,y1-y0),s=Math.min(520/w,250/h);canvas.width=Math.ceil(w*s);canvas.height=Math.ceil(h*s);let c=canvas.getContext("2d");c.drawImage(im,x0,y0,w,h,0,0,canvas.width,canvas.height);c.strokeStyle=color;c.lineWidth=3;c.strokeRect((b[0]*sx-x0)*s,(b[1]*sy-y0)*s,(b[2]-b[0])*sx*s,(b[3]-b[1])*sy*s)};im.src=item.image}
+function render(){let root=document.querySelector("#cards");root.innerHTML="";for(const item of D.items){const c=document.createElement("article");c.className="card";const e=item.suppression_events.find(x=>x.model==="C");let xmlMetric=item.C_iou.metric,newMetric=item.C_contain09.metric;let sections="<section class='panel'><h3>原始图片＋XML实例</h3><canvas data-kind='xml'></canvas><div class='muted'>"+item.truth.map((t,i)=>"实例"+(i+1)+" "+esc(t.class_name)+" "+JSON.stringify(t.bbox_xyxy)).join("<br>")+"</div></section>";
+const panels=[["原基准默认：IoU≥0.5",item.baseline_iou,"#1559b5","normal"],["C组恢复默认：原IoU≥0.5（蓝框恢复显示）",item.C_iou,"#1559b5","normal"],["C组历史实验：包含阈值0.9（不采纳）",item.C_contain09,"#c5232c","contain"]];
+for(const [title,r,col,kind] of panels){sections+="<section class='panel'><h3>"+title+"<span class='badge'>"+r.predictions.length+"个最终框</span></h3><canvas data-kind='"+kind+"' data-panel='"+title+"'></canvas><div class='details'>"+r.predictions.map((p,i)=>"框"+(i+1)+" 候选"+JSON.stringify(p.candidate_indices)+"："+esc(p.class_name)+" 检测="+p.detector_score.toFixed(4)+" 分类top="+p.classifier_top_score.toFixed(4)+" bbox="+JSON.stringify(p.bbox_xyxy)).join("\\n")+"</div><div class='muted'>TP/FP/FN "+r.metric.TP+"/"+r.metric.FP+"/"+r.metric.FN+"；未匹配显示框 "+r.metric.unmatched_display_boxes+"</div></section>"}
+c.innerHTML="<h2>"+esc(item.filename)+"<span class='badge'>"+item.truth.length+"个XML实例</span></h2><div class='notice'>人工意见："+esc(item.user_opinion)+"</div><div class='grid'>"+sections+"</div>"+(e?"<div class='event'><b>包含0.9新增抑制链：</b>候选#"+e.suppressed_candidate_index+"（蓝框在IoU结果中重新显示） → 触发抑制的候选#"+e.suppressor_candidate_index+"（0.9结果保留）。<br>检测分数 "+e.suppressed_detector_score.toFixed(4)+" → "+e.suppressor_detector_score.toFixed(4)+"；两框IoU "+e.iou.toFixed(4)+"；交集/较小框 "+e.intersection_over_smaller.toFixed(4)+"。XML仅用于事后匹配。<div class='grid'><div>原IoU恢复框局部放大：<canvas class='zoom' data-box='"+JSON.stringify(e.suppressed_box_xyxy)+"' data-color='#1559b5'></canvas></div><div>包含0.9保留框局部放大：<canvas class='zoom' data-box='"+JSON.stringify(e.suppressor_box_xyxy)+"' data-color='#c5232c'></canvas></div></div></div>":"");root.appendChild(c);c.querySelectorAll("canvas[data-kind]").forEach(k=>{let kind=k.dataset.kind;if(kind==="xml")draw(k,item,[],"","xml");else {let title=k.dataset.panel;let p=panels.find(x=>x[0]===title);draw(k,item,p[1].predictions,p[2],p[3])}});c.querySelectorAll(".zoom").forEach(z=>crop(z,item,JSON.parse(z.dataset.box),z.dataset.color))}
+}
+render();
+</script></body></html>'''

if __name__=='__main__': main()

