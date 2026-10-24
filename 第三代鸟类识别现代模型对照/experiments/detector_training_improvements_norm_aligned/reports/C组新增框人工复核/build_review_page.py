from __future__ import annotations
import base64, csv, io, json, math
from pathlib import Path
from PIL import Image

EXP = Path(r'E:\机器学习课设\第三代鸟类识别现代模型对照\experiments\detector_training_improvements_norm_aligned')
OUT = EXP / 'reports' / 'C组新增框人工复核'
SOURCE = EXP / 'reports' / 'fixed_val_comparison' / 'fixed_val_results.json'
TRAIN_REVIEW = '../hard_training_review/index.html'

raw = json.loads(SOURCE.read_text(encoding='utf-8-sig'))
rows = raw['rows']
changes = raw['changes']
by_name = {r['filename']: r for r in rows}

def iou(a,b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    area_a=max(0,a[2]-a[0])*max(0,a[3]-a[1]); area_b=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter/(area_a+area_b-inter) if area_a+area_b-inter else 0.0

def label_box(p, model, idx):
    return {'bbox':p.get('bbox_xyxy'), 'class_name':p.get('class_name'), 'score':p.get('detector_score'),
            'classifier_top_score':p.get('classifier_top_score'), 'class_scores':p.get('classifier_scores',{}),
            'final_score_product':p.get('final_score_product'), 'matched_truth_class':p.get('matched_truth_class'),
            'matched_truth_iou':p.get('matched_truth_iou'), 'spatial_status':p.get('spatial_status'),
            'same_class_correct_match':p.get('same_class_correct_match',False), 'candidate_index':p.get('candidate_index',idx),
            'model':model}

images = {}
def image_data(row):
    key=row['filename']
    if key in images: return key
    path=Path(row['image_path'])
    with Image.open(path) as im:
        im=im.convert('RGB')
        original_width, original_height = im.size
        im.thumbnail((1800,1800), Image.Resampling.LANCZOS)
        b=io.BytesIO(); im.save(b,format='JPEG',quality=84,optimize=True)
        images[key]={'src':'data:image/jpeg;base64,'+base64.b64encode(b.getvalue()).decode('ascii'),
                     'width':im.width,'height':im.height,'scale_x':im.width/original_width,'scale_y':im.height/original_height,'path':str(path)}
    return key

found_names=changes.get('found_from_original_no_display',{}).get('C',[])
found=[]
for name in found_names:
    r=by_name[name]; key=image_data(r)
    found.append({'id':'found:'+name,'kind':'found','filename':name,'imageKey':key,'folder':r.get('folder'),
      'truth':r['truth'],'baseline': [label_box(p,'基准',i) for i,p in enumerate(r['models']['bioclip2']['displayed'])],
      'groupC':[label_box(p,'C组',i) for i,p in enumerate(r['models']['C']['displayed'])],
      'note':'该图在原基准没有通过统一显示规则的框；本组C是否正确仍按XML匹配信息人工检查。'})

unmatched=[]
for r in rows:
    c=r['models']['C']
    for i,p in enumerate(c['displayed']):
        # 与XML位置匹配但类别不同的实例归入第3节，不重复计入25个未匹配框。
        if p.get('same_class_correct_match',False) or (p.get('matched_truth_iou') is not None and p.get('matched_truth_class') != p.get('class_name')): continue
        base=r['models']['bioclip2']['displayed']
        overlaps=[iou(p['bbox_xyxy'],q['bbox_xyxy']) for q in base]
        best=max(overlaps,default=0.0)
        novel=best < 0.5
        key=image_data(r)
        unmatched.append({'id':f"unmatched:{r['filename']}:{p.get('candidate_index',i)}:{i}",'kind':'unmatched',
          'filename':r['filename'],'imageKey':key,'folder':r.get('folder'),'truth':r['truth'],
          'candidate':label_box(p,'C组',i),'baseline':[label_box(q,'基准',j) for j,q in enumerate(base)],
          'groupC':[label_box(q,'C组',j) for j,q in enumerate(c['displayed'])],
          'novel':novel,'baseline_overlap_iou':best,
          'novel_rule':'C框与基准所有最终显示框的最大IoU < 0.5' if novel else '与至少一个基准最终显示框IoU ≥ 0.5，视为已有相似框'})

# 按人工XML实例逐一比较：先以预测框和真实框IoU>=0.5建立位置对应，再检查类别。
def status_for(gt,preds):
    spatial=[]
    for p in preds:
        ov=iou(gt['bbox_xyxy'],p.get('bbox') or [])
        if ov>=0.5: spatial.append((ov,p))
    if not spatial: return {'status':'no_spatial_match','predictions':[]}
    spatial.sort(key=lambda x:(x[0],x[1].get('score') or 0),reverse=True)
    correct=[x for x in spatial if x[1].get('class_name')==gt['class_name']]
    if correct: return {'status':'correct','predictions':[{'iou':x[0],**x[1]} for x in correct]}
    return {'status':'class_disagreement','predictions':[{'iou':x[0],**x[1]} for x in spatial]}

diffs=[]
persistent=[]
for r in rows:
    base=[label_box(p,'基准',i) for i,p in enumerate(r['models']['bioclip2']['displayed'])]
    c=[label_box(p,'C组',i) for i,p in enumerate(r['models']['C']['displayed'])]
    for gi,gt in enumerate(r['truth']):
        sb=status_for(gt,base); sc=status_for(gt,c)
        if sb['status']=='correct' and sc['status']!='correct': direction='基准正确，C组出现分歧'
        elif sb['status']!='correct' and sc['status']=='correct': direction='基准有分歧，C组正确匹配'
        elif sb['status']=='class_disagreement' and sc['status']=='class_disagreement': direction='基准与C组均位置匹配但类别有分歧（持续待复核）'
        else: continue
        key=image_data(r)
        target = persistent if direction.startswith('基准与C组均') else diffs
        target.append({'id':f"diff:{r['filename']}:{gi}:{direction}",'kind':'diff','filename':r['filename'],'imageKey':key,
          'folder':r.get('folder'),'truth':gt,'instanceIndex':gi+1,'direction':direction,
          'baselineStatus':sb,'cStatus':sc,'baseline':base,'groupC':c,
          'note':'以现有XML及IoU≥0.5逐实例作程序化对照，仅用于定位分歧，不代表人工确认了真值。'})

# Stable order: show novel candidates first, then largest confidence; per-file all C unmatched remain present.
unmatched.sort(key=lambda x:(not x['novel'],-(x['candidate'].get('score') or 0),x['filename']))

payload={'metadata':{'source_json':str(SOURCE),'fixed_validation_images':len(rows),'xml_instance_count':sum(len(r['truth']) for r in rows),
  'threshold':raw['metadata'].get('strict_detector_threshold',0.3),'dedup':raw['metadata'].get('cross_class_dedup'),
  'classifier':raw['metadata'].get('classification'),'unmatched_definition':'未能与当前XML目标按类别正确且IoU≥0.5一对一匹配的最终显示框；不是自动认定的背景误检。',
  'new_definition':'该C组框与同图基准最终显示框的最大IoU小于0.5；仅表示显示框位置相对基准新增，不表示鸟是新增或模型误检。',
  'validation_warning':'验证数据仅用于模型探索与复核；任何标注修订必须保持图像身份固定，并以同一新版标注重新评估基准、A、B、C。验证图和复核结论不得转入训练。'},
  'images':images,'found':found,'unmatched':unmatched,'diffs':diffs,'persistent_disagreements':persistent}
html_data=json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
page=r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>原基准与C组差异人工复核</title>
<style>
:root{font-family:Segoe UI,"Microsoft YaHei",sans-serif;color:#20252b;background:#f4f6f8}body{margin:0}.top{background:#182a3a;color:#fff;padding:18px 24px;position:sticky;top:0;z-index:4}.top h1{font-size:22px;margin:0 0 8px}.top p{margin:4px 0;color:#dbe5ee}.controls{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}.controls button,.controls input{padding:8px 11px;border:0;border-radius:6px}.controls button{cursor:pointer;background:#d7e6f4}.controls .active{background:#6fc1aa}.notice{background:#fff4d5;border-left:5px solid #dfa929;padding:12px 16px;margin:16px;border-radius:5px}.layout{display:grid;grid-template-columns:300px 1fr;gap:14px;padding:0 16px 20px}.side{background:white;padding:14px;border-radius:8px;height:max-content;position:sticky;top:190px}.side a{display:block;margin:8px 0}.side .stat{font-size:14px;margin:8px 0}.main{min-width:0}.section{display:none}.section.active{display:block}.card{background:#fff;padding:14px;margin-bottom:14px;border-radius:8px;box-shadow:0 1px 5px #0001}.card h3{margin:0 0 7px}.meta{font-size:13px;color:#52606b;margin-bottom:9px}.views{display:grid;grid-template-columns:minmax(300px,1.5fr) minmax(210px,.8fr);gap:10px}.canvasbox{background:#15191d;text-align:center;min-height:180px;display:flex;align-items:center;justify-content:center}.canvasbox canvas{max-width:100%;max-height:600px}.zoom canvas{max-height:360px}.toggles{display:flex;gap:12px;flex-wrap:wrap;margin:9px 0;font-size:13px}.toggles label{white-space:nowrap}.boxes{font-size:13px;background:#f5f7f8;padding:8px;max-height:170px;overflow:auto}.review{display:flex;gap:8px;align-items:center;margin:8px 0}.review select{padding:7px;max-width:300px}.pill{display:inline-block;background:#e8eef3;border-radius:12px;padding:3px 8px;margin:2px;font-size:12px}.score{font-family:Consolas,monospace}.candidate{border:2px solid #d98131;padding:8px;margin-top:10px;border-radius:5px}.foot{color:#5c636a;font-size:12px}.hidden{display:none!important}
@media(max-width:850px){.layout{grid-template-columns:1fr}.side{position:static}.top{position:static}.views{grid-template-columns:1fr}}
</style></head><body><header class="top"><h1>原基准与 C 组预测差异：人工复核</h1><p>固定验证集；模型、阈值、去重和分类设置沿用已完成实验。页面仅整理差异，不改 XML 或预测。</p><div class="controls"><button class="active" data-tab="found">C组找回原无框图</button><button data-tab="unmatched">C组全部未匹配框</button><button data-tab="diff">实例变化</button><input id="search" placeholder="按文件名或类别筛选"><button id="exportJson">导出复核 JSON</button><button id="exportCsv">导出复核 CSV</button></div></header>
<div class="notice"><b>判读约定：</b>“未匹配”只表示当前预测未与现有 XML 达成类别正确且 IoU≥0.5 的一对一匹配。可能涉及漏标、重复框、框偏差或类别分歧；不能自动判为误检或自动修订标签。C组新增指相对基准最终显示框没有 IoU≥0.5 的相似框，不表示新增了真实鸟只。</div>
<div class="layout"><aside class="side"><b>复核队列</b><div id="stats"></div><hr><a href="__TRAIN_REVIEW__" target="_blank">打开 D 组候选：训练集困难样本复核页</a><p class="foot">该入口仅含训练集图像。此页面为固定验证集，验证图片及复核结论不得加入训练。若修订验证标注，需在同一新版标注下重新评估原基准和 A/B/C。</p></aside><main class="main"><section id="found" class="section active"></section><section id="unmatched" class="section"></section><section id="diff" class="section"></section></main></div>
<script id="payload" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('payload').textContent); const opts=['暂不能确定','正确检测但 XML 漏标','背景误检','重复框','框位置偏差','类别错误']; const choices=['xml','base','c'];
const keyPrefix='bird_review_v1:';
function clsColor(n){return ({'树麻雀':'#22a6f2','家麻雀':'#13bb75','山麻雀':'#e3b328','白头鹎':'#ec5b52','红耳鹎':'#bd65dd','黑喉红臀鹎':'#ff8d28','大山雀':'#52d3c1','煤山雀':'#f076a8'})[n]||'#ffae32'}
function boxList(item){let a=[]; (Array.isArray(item.truth)?item.truth:(item.truth?[item.truth]:[])).forEach((x,i)=>a.push({type:'xml',box:x.bbox_xyxy,class_name:x.class_name,tag:`XML实例${i+1}`})); (item.baseline||[]).forEach((x,i)=>a.push({type:'base',box:x.bbox,class_name:x.class_name,tag:`基准${i+1} ${x.class_name} ${Number(x.score||0).toFixed(3)}`})); (item.groupC||[]).forEach((x,i)=>a.push({type:'c',box:x.bbox,class_name:x.class_name,tag:`C${i+1} ${x.class_name} ${Number(x.score||0).toFixed(3)}`})); return a}
function reviewControl(item){const k=keyPrefix+item.id;let val=localStorage.getItem(k)||'暂不能确定';return `<div class="review"><b>人工复核：</b><select data-review="${esc(item.id)}">${opts.map(o=>`<option ${o===val?'selected':''}>${o}</option>`).join('')}</select><input data-note="${esc(item.id)}" placeholder="备注（可选）" value="${esc(localStorage.getItem(k+':note')||'')}"></div>`}
function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function scores(p){let s=p.class_scores||{};return Object.entries(s).sort((a,b)=>b[1]-a[1]).map(([k,v])=>`${k} ${Number(v).toFixed(3)}`).join(' · ')}
function card(item,idx){const img=D.images[item.imageKey];let candidate=item.candidate;let badge=item.kind==='unmatched'?(item.novel?'C组新增候选':'基准已有相似框'):(item.direction||'找回图');let title=item.kind==='unmatched'?`${item.filename} · C候选 ${idx+1}`:item.filename;let boxHtml=boxList(item).map(x=>`<span class="pill" style="border-left:4px solid ${clsColor(x.class_name)}">${esc(x.tag)}</span>`).join('');let details=candidate?`<div class="candidate"><b>${badge}</b>　检测分数 <span class="score">${Number(candidate.score||0).toFixed(4)}</span>　预测类别 <b>${esc(candidate.class_name)}</b>　分类最高分 <span class="score">${Number(candidate.classifier_top_score||0).toFixed(4)}</span><div class="meta">八类分类得分：${esc(scores(candidate))}</div><div class="meta">与基准最终框最大 IoU：${Number(item.baseline_overlap_iou||0).toFixed(3)}。${esc(item.novel_rule||'')}。与XML匹配状态：未匹配；请人工判断。</div></div>`:'';
let direction=item.kind==='diff'?`<div class="candidate"><b>${esc(item.direction)}</b>　XML实例${item.instanceIndex}：${esc(item.truth.class_name)}；基准状态：${esc(item.baselineStatus.status)}；C组状态：${esc(item.cStatus.status)}。<div class="meta">程序按预测框与该XML实例 IoU≥0.5 对照类别。此状态只是现有标注一致性。</div></div>`:'';
return `<article class="card" data-filename="${esc(item.filename.toLowerCase())}" data-folder="${esc(item.folder||'')}"><h3>${esc(title)} <span class="pill">${esc(badge)}</span></h3><div class="meta">目录提示：${esc(item.folder||'未知')}　·　来源：固定验证集　·　真值框按XML逐实例展示</div><div class="toggles"><label><input type="checkbox" data-toggle="xml" checked>XML框</label><label><input type="checkbox" data-toggle="base" checked>基准框</label><label><input type="checkbox" data-toggle="c" checked>C组框</label></div><div class="views"><div class="canvasbox"><canvas class="full"></canvas></div><div><div class="canvasbox zoom"><canvas class="zoom"></canvas></div><div class="meta">局部放大：优先以该未匹配框为中心；其余条目以首个 C 组框为中心。</div></div></div><div class="boxes">${boxHtml}</div>${details}${direction}${reviewControl(item)}<div class="foot">文件：${esc(img.path)}。红/橙等框颜色对应类别；XML框为绿色。</div></article>`}
function draw(canvas,img,boxes,enabled,focus){let ctx=canvas.getContext('2d'),im=new Image();im.onload=()=>{const maxW=900,maxH=560,scale=Math.min(maxW/im.width,maxH/im.height,1);canvas.width=Math.round(im.width*scale);canvas.height=Math.round(im.height*scale);ctx.drawImage(im,0,0,canvas.width,canvas.height);for(const b of boxes){if(!enabled[b.type]||!b.box)continue;let [x1,y1,x2,y2]=b.box.map((v,i)=>v*(i%2===0?img.scale_x:img.scale_y)*scale);ctx.strokeStyle=b.type==='xml'?'#28e08b':clsColor(b.class_name);ctx.lineWidth=3;ctx.strokeRect(x1,y1,x2-x1,y2-y1);ctx.font='bold 14px Segoe UI';let text=b.tag||b.class_name||b.type;ctx.fillStyle=ctx.strokeStyle;ctx.fillRect(x1,Math.max(0,y1-18),ctx.measureText(text).width+8,18);ctx.fillStyle='#111';ctx.fillText(text,x1+4,Math.max(13,y1-5));}};im.src=img.src;
// separate crop canvas
let z=canvas.parentElement.parentElement.querySelector('.zoom canvas');let zc=z.getContext('2d');let zi=new Image();zi.onload=()=>{let f0=focus&&(focus.bbox||focus.box)?(focus.bbox||focus.box):(boxes.find(x=>x.type==='c')||boxes.find(x=>x.type==='xml')||{}).box;if(!f0){return;}let f=[f0[0]*img.scale_x,f0[1]*img.scale_y,f0[2]*img.scale_x,f0[3]*img.scale_y];let pad=Math.max((f[2]-f[0])*.6,(f[3]-f[1])*.6,20),x1=Math.max(0,f[0]-pad),y1=Math.max(0,f[1]-pad),x2=Math.min(zi.width,f[2]+pad),y2=Math.min(zi.height,f[3]+pad);let sw=x2-x1,sh=y2-y1,scale=Math.min(640/sw,360/sh);z.width=Math.round(sw*scale);z.height=Math.round(sh*scale);zc.drawImage(zi,x1,y1,sw,sh,0,0,z.width,z.height);for(const b of boxes){if(!enabled[b.type]||!b.box)continue;let [a0,c0,d0,e0]=b.box;let a=a0*img.scale_x,c=c0*img.scale_y,d=d0*img.scale_x,e=e0*img.scale_y;let xx=(a-x1)*scale,yy=(c-y1)*scale;zc.strokeStyle=b.type==='xml'?'#28e08b':clsColor(b.class_name);zc.lineWidth=3;zc.strokeRect(xx,yy,(d-a)*scale,(e-c)*scale);}};zi.src=img.src;}
function mount(section,items){document.getElementById(section).innerHTML=`<h2>${section==='found'?'C组找回的原无框图片':section==='unmatched'?'C组全部未匹配显示框':'基准与C组逐实例变化'}</h2><p class="meta">${items.length}条待查条目。每张图片可切换 XML、基准与C组框；复核选项保存在本机浏览器，可导出。</p>`+items.map(card).join('');document.querySelectorAll(`#${section} article`).forEach((el,i)=>{let item=items[i],img=D.images[item.imageKey],focus=item.candidate||null;const render=()=>{let en={xml:el.querySelector('[data-toggle=xml]').checked,base:el.querySelector('[data-toggle=base]').checked,c:el.querySelector('[data-toggle=c]').checked};draw(el.querySelector('canvas.full'),img,boxList(item),en,focus)};el.querySelectorAll('[data-toggle]').forEach(x=>x.addEventListener('change',render));render();});}
mount('found',D.found);mount('unmatched',D.unmatched);mount('diff',[...D.diffs,...D.persistent_disagreements]);
const total=D.found.length+D.unmatched.length+D.diffs.length;document.getElementById('stats').innerHTML=`<div class="stat">找回原无框图：<b>${D.found.length}</b></div><div class="stat">C组未匹配框：<b>${D.unmatched.length}</b>（新增候选 ${D.unmatched.filter(x=>x.novel).length}）</div><div class="stat">实例变化：<b>${D.diffs.length}</b>；持续类别分歧：<b>${D.persistent_disagreements.length}</b></div><div class="stat">验证集图像：${D.metadata.fixed_validation_images}；XML实例：${D.metadata.xml_instance_count}</div><div class="stat">复核状态：${total}条，默认“暂不能确定”</div>`;
document.querySelectorAll('[data-review]').forEach(s=>s.addEventListener('change',()=>localStorage.setItem(keyPrefix+s.dataset.review,s.value)));document.querySelectorAll('[data-note]').forEach(s=>s.addEventListener('change',()=>localStorage.setItem(keyPrefix+s.dataset.note+':note',s.value)));
document.querySelectorAll('.controls button[data-tab]').forEach(b=>b.onclick=()=>{document.querySelectorAll('.controls button[data-tab]').forEach(x=>x.classList.remove('active'));b.classList.add('active');document.querySelectorAll('.section').forEach(x=>x.classList.remove('active'));document.getElementById(b.dataset.tab).classList.add('active');});
document.getElementById('search').oninput=e=>{let q=e.target.value.toLowerCase();document.querySelectorAll('.section.active article').forEach(a=>{const text=`${a.dataset.filename} ${a.dataset.folder}`;a.classList.toggle('hidden',!text.includes(q));});};
function reviewRows(){let all=[...D.found,...D.unmatched,...D.diffs,...D.persistent_disagreements];return all.map(x=>({id:x.id,filename:x.filename,section:x.kind,review:localStorage.getItem(keyPrefix+x.id)||'暂不能确定',note:localStorage.getItem(keyPrefix+x.id+':note')||'',candidate_class:x.candidate?.class_name||'',candidate_bbox_json:JSON.stringify(x.candidate?.bbox||''),detector_score:x.candidate?.score??'',classifier_top_score:x.candidate?.classifier_top_score??'',class_scores_json:JSON.stringify(x.candidate?.class_scores||{}),c_new:x.novel??'',baseline_overlap_iou:x.baseline_overlap_iou??'',direction:x.direction||'',xml_instances_json:JSON.stringify((Array.isArray(x.truth)?x.truth:(x.truth?[x.truth]:[])).map((t,i)=>({instance:x.instanceIndex||i+1,class_name:t.class_name,bbox_xyxy:t.bbox_xyxy}))),baseline_status:x.baselineStatus?.status||'',C_status:x.cStatus?.status||'',folder:x.folder||'',source:'fixed_validation'}))}
function down(name,text,type){let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([text],{type}));a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}
document.getElementById('exportJson').onclick=()=>down('C组人工复核结果.json',JSON.stringify({metadata:D.metadata,items:reviewRows()},null,2),'application/json;charset=utf-8');document.getElementById('exportCsv').onclick=()=>{let rs=reviewRows(),cols=Object.keys(rs[0]||{}),q=v=>'"'+String(v??'').replaceAll('"','""')+'"';down('C组人工复核结果.csv','\ufeff'+[cols.map(q).join(','),...rs.map(r=>cols.map(k=>q(r[k])).join(','))].join('\r\n'),'text/csv;charset=utf-8')};
</script></body></html>'''.replace('__TRAIN_REVIEW__',TRAIN_REVIEW).replace('__DATA__',html_data)
(OUT/'原基准与C组差异人工复核.html').write_text(page,encoding='utf-8')
# JSON roster facilitates audit and regeneration without reading HTML internals.
(OUT/'review_manifest.json').write_text(json.dumps({k:v for k,v in payload.items() if k!='images'},ensure_ascii=False,indent=2),encoding='utf-8')
with (OUT/'review_manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['id','section','filename','folder','xml_instances','candidate_class','detector_score','classifier_top_score','c_new','baseline_overlap_iou','direction'])
    w.writeheader()
    for item in found: w.writerow({'id':item['id'],'section':'found','filename':item['filename'],'folder':item.get('folder'),'xml_instances':len(item['truth'])})
    for item in unmatched: w.writerow({'id':item['id'],'section':'unmatched','filename':item['filename'],'folder':item.get('folder'),'xml_instances':len(item['truth']),'candidate_class':item['candidate']['class_name'],'detector_score':item['candidate']['score'],'classifier_top_score':item['candidate']['classifier_top_score'],'c_new':item['novel'],'baseline_overlap_iou':item['baseline_overlap_iou']})
    for item in diffs: w.writerow({'id':item['id'],'section':'diff','filename':item['filename'],'folder':item.get('folder'),'xml_instances':1,'direction':item['direction']})
print(json.dumps({'html':str(OUT/'原基准与C组差异人工复核.html'),'unique_images':len(images),'found_images':len(found),'C_unmatched_boxes':len(unmatched),'C_new_candidates':sum(x['novel'] for x in unmatched),'instance_changes':len(diffs),'review_html_bytes':(OUT/'原基准与C组差异人工复核.html').stat().st_size},ensure_ascii=False,indent=2))




