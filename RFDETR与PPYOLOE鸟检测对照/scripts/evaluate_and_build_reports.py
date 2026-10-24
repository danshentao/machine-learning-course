"""统一计算单类检测结果、PR/AP、逐图JSON/CSV和可筛选人工复核图册。"""
from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs/experiment.json").read_text(encoding="utf-8"))
REPORTS = ROOT / "reports"
ASSETS = REPORTS / "assets"
IOU_MATCH = 0.5
THRESHOLD = 0.30
SUPPRESS_IOU = 0.5
NOBOX_BASELINE = {
    "大山雀_09.jpg", "树麻雀_06.jpg", "树麻雀_08.jpg", "白头鹎_02.jpg", "白头鹎_03.jpg",
    "白头鹎_04.jpg", "白头鹎_09.jpg", "红耳鹎_06.jpg", "黑喉红臀鹎_02.jpg",
}


def iou(a: list[float], b: list[float]) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter) if area_a + area_b > inter else 0.0


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def deduplicate(candidates: list[dict]) -> tuple[list[dict], list[dict]]:
    """与网站原规则一致：分数优先、只同当前组winner比较、不传递合并。"""
    groups: list[dict] = []
    for candidate in sorted(candidates, key=lambda x: (-x["detector_score"], x.get("candidate_index", 0))):
        overlaps = [(iou(candidate["bbox_xyxy"], g["winner"]["bbox_xyxy"]), idx) for idx, g in enumerate(groups)]
        overlap, group_idx = max(overlaps, default=(0.0, None))
        item = dict(candidate)
        if overlap >= SUPPRESS_IOU:
            winner = groups[group_idx]["winner"]
            groups[group_idx]["hidden"].append({**item, "suppressed_by_candidate_index": winner.get("candidate_index"), "iou_to_winner": overlap})
        else:
            groups.append({"winner": item, "hidden": []})
    return groups, [g["winner"] for g in groups]


def match_predictions(preds: list[dict], truths: list[dict]) -> tuple[list[dict], list[dict]]:
    remaining = {t["instance_id"]: t for t in truths}
    annotated_preds = []
    for pred in sorted(preds, key=lambda x: (-x["detector_score"], x.get("candidate_index", 0))):
        overlaps = [(iou(pred["bbox_xyxy"], t["bbox_xyxy"]), key, t) for key, t in remaining.items()]
        overlap, key, truth = max(overlaps, default=(0.0, None, None), key=lambda x: x[0])
        row = dict(pred)
        if truth is not None and overlap >= IOU_MATCH:
            row.update({"xml_match": "matched", "matched_instance_id": key, "matched_iou": overlap,
                        "matched_truth_bbox_xyxy": truth["bbox_xyxy"]})
            del remaining[key]
        else:
            row.update({"xml_match": "unmatched_to_current_xml", "matched_instance_id": None, "matched_iou": overlap,
                        "matched_truth_bbox_xyxy": None})
        annotated_preds.append(row)
    return annotated_preds, list(remaining.values())


def metrics_for_rows(rows: list[dict], model: str) -> dict:
    tp = sum(len(r["models"][model]["matched_predictions"]) for r in rows)
    fp = sum(len(r["models"][model]["unmatched_predictions"]) for r in rows)
    fn = sum(len(r["models"][model]["unmatched_truths"]) for r in rows)
    p = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "images": len(rows), "GT_instances": tp + fn, "displayed_boxes": tp + fp,
        "TP": tp, "FP_against_current_XML": fp, "FN_against_current_XML": fn,
        "Precision": p, "Recall": rec, "F1": 2 * p * rec / (p + rec) if p + rec else 0.0,
        "images_with_no_displayed_boxes": sum(not r["models"][model]["display_candidates"] for r in rows),
        "no_display_images": [r["filename"] for r in rows if not r["models"][model]["display_candidates"]],
        "images_with_GT_but_no_displayed_boxes": sum(bool(r["truth"]) and not r["models"][model]["display_candidates"] for r in rows),
        "GT_but_no_display_images": [r["filename"] for r in rows if r["truth"] and not r["models"][model]["display_candidates"]],
        "unmatched_displayed_boxes": fp,
        "unmatched_truth_boxes": fn,
        "rule": "同一张图内按检测分数降序一对一匹配；仅IoU≥0.5的bird框算匹配；阈值严格>0.30；先按网站原规则IoU≥0.5抑制较低分框。FP/FN均相对现有XML，不自动等同真实背景误检/真实漏检。",
    }


def run_coco_eval(gt_doc: dict, images: list[dict], candidate_key: str, preds_by_name: dict[str, list[dict]], pr_rows: list[dict], label: str) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    temp_gt = REPORTS / f"_temp_gt_{label}.json"
    temp_gt.write_text(json.dumps(gt_doc), encoding="utf-8")
    coco_gt = COCO(str(temp_gt))
    records = []
    for im in images:
        for pred in preds_by_name[im["file_name"]]:
            x1, y1, x2, y2 = pred["bbox_xyxy"]
            records.append({"image_id": im["id"], "category_id": 0, "bbox": [x1, y1, x2 - x1, y2 - y1], "score": pred["detector_score"]})
    coco_dt = coco_gt.loadRes(records)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.params.imgIds = [im["id"] for im in images]
    ev.params.catIds = [0]
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    idx = int(np.argmin(np.abs(ev.params.iouThrs - 0.5)))
    precision = ev.eval["precision"][idx, :, 0, 0, -1]
    recall = ev.params.recThrs
    for rr, pp in zip(recall, precision):
        pr_rows.append({"model": label.split("_")[0], "candidate_stage": candidate_key, "iou": 0.5,
                        "recall": float(rr), "precision": float(pp) if pp >= 0 else None})
    result = {"AP50_95_COCO": float(ev.stats[0]), "AP50_COCO": float(ev.stats[1]),
              "maxDets": list(ev.params.maxDets), "candidate_stage": candidate_key,
              "candidate_count": len(records), "score_floor_from_native_model": 0.001}
    temp_gt.unlink(missing_ok=True)
    return result


def copy_asset(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        shutil.copy2(src, dst)


def html_page(title: str, payload: dict, mode: str) -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    note = ("验证XML仅用于事后对照；候选详情列出每图、每模型得分前30项，完整原生/隐藏框见同目录分阶段JSON。"
            if mode == "fixed" else
            "外来图无逐实例真值；新增框需要人工判断。候选详情列出每图、每模型得分前30项，完整候选见同目录JSON。")
    return HTML_TEMPLATE.replace("__TITLE__", title).replace("__MODE__", mode).replace("__NOTE__", note).replace("__DATA__", data)


def gallery_view(rows: list[dict]) -> list[dict]:
    """图册只嵌入交互所需字段；完整分阶段候选另存JSON，避免将候选重复嵌入HTML。"""
    candidate_fields = ("candidate_index", "bbox_xyxy", "detector_score", "class_name", "displayed", "suppressed",
                        "xml_match", "matched_instance_id", "matched_iou", "matched_truth_bbox_xyxy",
                        "suppressed_by_candidate_index", "iou_to_winner", "is_added_vs_ppyoloe")
    def slim(items: list[dict]) -> list[dict]:
        return [{k: p[k] for k in candidate_fields if k in p} for p in items]
    out = []
    for row in rows:
        models = {}
        for model in ("ppyoloe", "rfdetr"):
            src = row["models"][model]
            models[model] = {
                "native_candidates": slim(src["native_candidates"]),
                "suppressed_candidates": slim(src["suppressed_candidates"]),
                "display_candidates": slim(src["display_candidates"]),
                "added_vs_ppyoloe": slim(src.get("added_vs_ppyoloe", [])),
            }
        pair = row["model_pair"]
        pair_slim = {"matched_pairs": pair.get("matched_pairs", []),
                     "unmatched_ppyoloe": slim(pair.get("unmatched_ppyoloe", [])),
                     "unmatched_rfdetr": slim(pair.get("unmatched_rfdetr", []))}
        out.append({k: row[k] for k in ("filename", "image_path", "image", "width", "height", "truth", "same_photo",
                                         "related_scene", "expected_category_posthoc", "status_labels", "known_baseline_empty") if k in row} |
                   {"models": models, "model_pair": pair_slim})
    return out


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><style>
body{font-family:"Microsoft YaHei",Arial,sans-serif;margin:0;background:#f4f6f4;color:#1c2923}header{background:#173d32;color:#fff;padding:18px 24px}header a{color:#c9f5dc}main{max-width:1600px;margin:auto;padding:18px}.controls{position:sticky;top:0;background:#fff;padding:12px;z-index:3;border:1px solid #ddd;display:flex;flex-wrap:wrap;gap:10px;align-items:center}input,select,button{padding:7px;border:1px solid #aab7ae;border-radius:5px;background:white}.note{background:#fff7df;padding:12px;margin:12px 0;border-left:4px solid #e1ad35}.card{background:white;margin:16px 0;padding:14px;border:1px solid #dce3de;border-radius:9px}.panels{display:grid;grid-template-columns:repeat(3,minmax(220px,1fr));gap:12px}.panel h3{margin:4px 0 8px;font-size:15px}.stage{position:relative;width:100%;background:#eee;overflow:auto;max-height:620px}.stage img{display:block;max-width:100%;height:auto}.stage canvas{position:absolute;left:0;top:0;pointer-events:none}.opts{display:flex;gap:10px;flex-wrap:wrap;font-size:13px;margin:8px 0}.details{margin-top:12px;border-top:1px solid #ddd;padding-top:8px}.cand{border:1px solid #e3e8e4;margin:6px 0;padding:7px;font-size:12px;display:grid;grid-template-columns:minmax(220px,1fr) auto;gap:8px}.tag{font-weight:bold}.tag.base{color:#1674c8}.tag.rf{color:#dc541e}.tag.gt{color:#168a4d}.muted{color:#627269;font-size:12px}.pill{background:#e6f2e9;border-radius:12px;padding:3px 8px;font-size:12px}.no{color:#a53b2b}.zoom{max-width:100%;max-height:82vh}dialog{border:0;border-radius:10px;max-width:95vw}dialog::backdrop{background:#11281bcc}.summary{display:flex;gap:12px;flex-wrap:wrap}.summary div{background:#edf4ef;padding:9px 12px;border-radius:7px}.hidden{display:none!important}@media(max-width:900px){.panels{grid-template-columns:1fr}.controls{position:static}}
</style></head><body><header><h1>__TITLE__</h1><div><a href="index.html">返回汇总首页</a></div></header><main>
<div class="note">__NOTE__</div><section id="summary" class="summary"></section>
<div class="controls"><input id="search" placeholder="按文件名搜索"><select id="filter"></select><label>显示阶段 <select id="stage"><option value="final">最终显示（IoU去重后，分数&gt;0.30）</option><option value="dedup">额外IoU去重后（含低分候选）</option><option value="native">模型原生候选</option></select></label><label><input id="showXml" type="checkbox" checked>验证集XML框</label><label><input id="showPpy" type="checkbox" checked>PP-YOLOE+</label><label><input id="showRf" type="checkbox" checked>RF-DETR</label><button id="exportJson" class="review-only">导出复核JSON</button><button id="exportCsv" class="review-only">导出复核CSV</button><span id="count"></span></div>
<div id="cards"></div></main><dialog id="zoomDialog"><button onclick="document.getElementById('zoomDialog').close()">关闭</button><p id="zoomTitle"></p><canvas id="zoomCanvas" class="zoom"></canvas></dialog>
<script>const DATA=__DATA__,MODE='__MODE__';const $=s=>document.querySelector(s);const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const CLASS_BASE='#1674c8',CLASS_RF='#dc541e',CLASS_GT='#168a4d';
const filters=MODE==='fixed'?['全部','无显示框','RF新增框候选','PP有框_RF无框','定位变化候选','多鸟图']:['全部','无框','新增框候选','PP有框_RF无框','定位变化候选','同源回放','相关场景候选','原基准9张无框图'];for(const v of filters){const o=document.createElement('option');o.value=v;o.textContent=v;$('#filter').appendChild(o)}if(MODE==='fixed')document.querySelectorAll('.review-only').forEach(x=>x.classList.add('hidden'));
function candidates(row,model){let m=row.models[model];if($('#stage').value==='dedup')return m.native_candidates.filter(p=>!p.suppressed);return m[$('#stage').value==='native'?'native_candidates':'display_candidates']};
function draw(canvas,img,boxes,color,labelPrefix){const ctx=canvas.getContext('2d'),w=img.naturalWidth,h=img.naturalHeight;if(!w||!h)return;canvas.width=w;canvas.height=h;canvas.style.width=img.clientWidth+'px';canvas.style.height=(img.clientWidth*h/w)+'px';ctx.clearRect(0,0,w,h);for(const b of boxes){const q=b.bbox_xyxy;if(!q)continue;ctx.strokeStyle=color;ctx.lineWidth=Math.max(2,w/500);ctx.strokeRect(q[0],q[1],q[2]-q[0],q[3]-q[1]);ctx.font='bold '+Math.max(14,w/75)+'px Arial';ctx.fillStyle=color;ctx.fillText(labelPrefix+' '+Number(b.detector_score).toFixed(3),Math.max(0,q[0]),Math.max(15,q[1]-3))}}
function cropZoom(imagePath,box,name,score){const img=new Image();img.onload=()=>{const c=$('#zoomCanvas'),ctx=c.getContext('2d');let [x1,y1,x2,y2]=box,pad=Math.max(20,Math.max(x2-x1,y2-y1)*.25);x1=Math.max(0,x1-pad);y1=Math.max(0,y1-pad);x2=Math.min(img.naturalWidth,x2+pad);y2=Math.min(img.naturalHeight,y2+pad);let w=x2-x1,h=y2-y1,s=Math.min(1100/w,800/h,2);c.width=w*s;c.height=h*s;ctx.drawImage(img,x1,y1,w,h,0,0,c.width,c.height);ctx.strokeStyle='#ff2b00';ctx.lineWidth=3;ctx.strokeRect((box[0]-x1)*s,(box[1]-y1)*s,(box[2]-box[0])*s,(box[3]-box[1])*s);$('#zoomTitle').textContent=name+' · 检测分数 '+Number(score).toFixed(4)+' · 局部放大';$('#zoomDialog').showModal()};img.src=imagePath}
function reviewKey(row,p){return 'rfdetr-review:'+row.filename+':'+p.candidate_index};function getReview(row,p){return localStorage.getItem(reviewKey(row,p))||'暂不能确定'}
function candidateList(row,model){let m=row.models[model],hiddenById=new Map(m.suppressed_candidates.map(p=>[p.candidate_index,p]));return m.native_candidates.map(p=>({...p,...(hiddenById.get(p.candidate_index)||{}),stage_status:p.displayed?'最终显示':p.suppressed?'IoU被抑制':'阈值未通过'})).sort((a,b)=>b.detector_score-a.detector_score).slice(0,30)}
function cardHtml(r){const p=r.models.ppyoloe.display_candidates, f=r.models.rfdetr.display_candidates;const summary=r.status_labels.join('、')||'无显著框数变化';const panels=[['原图',r.image,null,''],['PP-YOLOE+',r.image,p,CLASS_BASE],['RF-DETR',r.image,f,CLASS_RF]];let html='<section class="card" data-file="'+esc(r.filename)+'"><h2>'+esc(r.filename)+(r.expected_category_posthoc?'　<span class="muted">目录事后预期：'+esc(r.expected_category_posthoc)+'（不是真值）</span>':'')+' <span class="pill">'+esc(summary)+'</span> <span class="muted">'+r.width+'×'+r.height+'　XML目标 '+r.truth.length+(r.same_photo?'　同源回放':'')+(r.related_scene?'　相关场景':'')+'</span></h2><div class="panels">';for(let i=0;i<panels.length;i++){const [title,src,boxes,color]=panels[i],id='c'+Math.random().toString(36).slice(2);html+='<div class="panel"><h3>'+title+(i===1||i===2?'　显示框 '+boxes.length:'')+'</h3><div class="stage"><img data-image="'+esc(src)+'" id="i'+id+'" src="'+esc(src)+'"><canvas id="'+id+'"></canvas></div></div>'}html+='</div>';
if(MODE==='external'){const added=r.models.rfdetr.added_vs_ppyoloe;html+='<div class="details"><b>RF-DETR相对基准新增候选</b>：'+added.length+'（是跨模型未匹配候选，未确认是否真鸟）</div>'}
html+='<details class="details"><summary>完整候选、去重抑制链和匹配详情（低分框也保留）</summary>';
if(MODE==='fixed'&&r.truth.length)html+='<div class="cand"><span class="tag gt">XML逐实例框（仅事后评估）</span><span>'+esc(JSON.stringify(r.truth.map(t=>({instance_id:t.instance_id,bbox_xyxy:t.bbox_xyxy}))))+'</span></div>';
for(const model of ['ppyoloe','rfdetr']){html+='<h3 class="tag '+(model==='ppyoloe'?'base':'rf')+'">'+(model==='ppyoloe'?'PP-YOLOE+':'RF-DETR')+'：模型原生候选 '+r.models[model].native_candidates.length+'；候选详情仅列前30项；额外IoU去重显示组 '+r.models[model].native_candidates.filter(p=>!p.suppressed).length+'；最终阈值框 '+r.models[model].display_candidates.length+'</h3>';for(const p of candidateList(r,model)){let decision='';if(MODE==='external'&&model==='rfdetr'&&p.is_added_vs_ppyoloe){decision='<label>人工复核 <select data-review="'+esc(reviewKey(r,p))+'" data-row="'+esc(r.filename)+'" data-ci="'+p.candidate_index+'"><option>暂不能确定</option><option>确实是鸟</option><option>背景</option><option>同鸟重复框</option></select></label>'}html+='<div class="cand"><span><b class="tag '+(model==='ppyoloe'?'base':'rf')+'">'+p.stage_status+'</b>　检测分数 '+Number(p.detector_score).toFixed(5)+'　'+(p.xml_match?('现有XML匹配 '+p.xml_match+' IoU '+Number(p.matched_iou||0).toFixed(3)):'')+'<br>框 '+esc(JSON.stringify(p.bbox_xyxy))+(p.suppressed_by_candidate_index!==undefined?'<br>被候选 #'+p.suppressed_by_candidate_index+' 抑制，IoU '+Number(p.iou_to_winner).toFixed(3):'')+(p.is_added_vs_ppyoloe?'<br><b>RF-DETR新增框候选：需要人工核实</b>':'')+'</span><span><button onclick="cropZoom('+JSON.stringify(r.image)+','+JSON.stringify(p.bbox_xyxy)+','+JSON.stringify(r.filename)+','+Number(p.detector_score)+')">局部放大</button> '+decision+'</span></div>'}}html+='</details></section>';return html}
function matchesFilter(r,f){if(!f||f==='全部')return true;if(f==='无显示框'||f==='无框')return r.models.ppyoloe.display_candidates.length===0&&r.models.rfdetr.display_candidates.length===0;if(f==='RF新增框候选'||f==='新增框候选')return r.models.rfdetr.added_vs_ppyoloe.length>0;if(f==='PP有框_RF无框')return r.models.ppyoloe.display_candidates.length>0&&r.models.rfdetr.display_candidates.length===0;if(f==='定位变化候选')return r.model_pair.unmatched_ppyoloe.length>0||r.model_pair.unmatched_rfdetr.length>0;if(f==='多鸟图')return r.truth.length>1;if(f==='同源回放')return r.same_photo;if(f==='相关场景候选')return r.related_scene;if(f==='原基准9张无框图')return r.known_baseline_empty;return true}
function render(){const q=$('#search').value.toLowerCase(),f=$('#filter').value;const rows=DATA.rows.filter(r=>r.filename.toLowerCase().includes(q)&&matchesFilter(r,f));$('#count').textContent='当前 '+rows.length+' / '+DATA.rows.length+' 张';$('#cards').innerHTML=rows.map(cardHtml).join('');for(const card of document.querySelectorAll('.card')){const row=DATA.rows.find(x=>x.filename===card.dataset.file);const imgs=card.querySelectorAll('img');imgs.forEach((img,i)=>{img.onload=()=>{const can=card.querySelectorAll('canvas')[i];if(MODE==='fixed'&&i===0&&$('#showXml').checked)draw(can,img,row.truth,CLASS_GT,'XML');else if(i===1&&$('#showPpy').checked)draw(can,img,candidates(row,'ppyoloe'),CLASS_BASE,'bird');else if(i===2&&$('#showRf').checked)draw(can,img,candidates(row,'rfdetr'),CLASS_RF,'bird');else {can.width=img.naturalWidth;can.height=img.naturalHeight;can.style.width=img.clientWidth+'px';can.style.height=(img.clientWidth*img.naturalHeight/img.naturalWidth)+'px';can.getContext('2d').clearRect(0,0,can.width,can.height)}};if(img.complete)img.onload()});card.querySelectorAll('select[data-review]').forEach(s=>{s.value=getReview(row,row.models.rfdetr.native_candidates.find(x=>String(x.candidate_index)===s.dataset.ci)||{});s.onchange=()=>localStorage.setItem(s.dataset.review,s.value)})}}
function exportReviews(kind){const out=[];for(const r of DATA.rows)for(const p of r.models.rfdetr.added_vs_ppyoloe){out.push({filename:r.filename,image_path:r.image_path,candidate_index:p.candidate_index,bbox_xyxy:p.bbox_xyxy,detector_score:p.detector_score,review:getReview(r,p),review_note:'新增框仅表示与PP-YOLOE+显示框空间未匹配，不代表已确认是鸟'})}const rows=kind==='json'?JSON.stringify(out,null,2):[['filename','image_path','candidate_index','bbox_xyxy','detector_score','review','review_note'],...out.map(x=>[x.filename,x.image_path,x.candidate_index,JSON.stringify(x.bbox_xyxy),x.detector_score,x.review,x.review_note])].map(r=>r.map(x=>'"'+String(x??'').replaceAll('"','""')+'"').join(',')).join('\r\n');const blob=new Blob([rows],{type:kind==='json'?'application/json':'text/csv'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='rfdetr_external80_manual_review.'+kind;a.click();URL.revokeObjectURL(a.href)}
$('#search').oninput=render;$('#filter').onchange=render;$('#stage').onchange=render;$('#showXml').onchange=render;$('#showPpy').onchange=render;$('#showRf').onchange=render;$('#exportJson').onclick=()=>exportReviews('json');$('#exportCsv').onclick=()=>exportReviews('csv');
for(const [key,val] of Object.entries(DATA.summary||{})){if(typeof val==='number')$('#summary').innerHTML+='<div><b>'+esc(key)+'</b><br>'+val+'</div>'}render();
</script></body></html>'''


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    (ASSETS / "fixed_val").mkdir(parents=True, exist_ok=True)
    (ASSETS / "external80").mkdir(parents=True, exist_ok=True)
    gt = load_json(Path(CFG["fixed_validation_json"]))
    if len(gt["images"]) != 172 or len(gt["annotations"]) != 204 or gt["categories"] != [{"id": 0, "name": "bird", "supercategory": "animal"}]:
        raise RuntimeError("固定验证集版本/目标数/单类定义与核对值不一致")
    ppy_val = load_json(ROOT / "runs/baseline_ppyoloe/fixed_val_raw.json")
    rf_val = load_json(ROOT / "runs/formal/rfdetr_fixed_val_raw.json")
    ppy_ext = load_json(ROOT / "runs/baseline_ppyoloe/external80_raw.json")
    rf_ext = load_json(ROOT / "runs/formal/rfdetr_external80_raw.json")
    ext_flags_doc = load_json(Path(CFG["external_flags_report"]))
    ext_flags = {r["filename"]: r for r in ext_flags_doc["images"]}
    ppy_maps = {split: {r["filename"]: r for r in doc["images"]} for split, doc in (("val", ppy_val), ("ext", ppy_ext))}
    rf_maps = {split: {r["filename"]: r for r in doc["images"]} for split, doc in (("val", rf_val), ("ext", rf_ext))}
    truths_by_name: dict[str, list[dict]] = {row["file_name"]: [] for row in gt["images"]}
    for ann in gt["annotations"]:
        im = next(x for x in gt["images"] if x["id"] == ann["image_id"])
        x, y, w, h = ann["bbox"]
        truths_by_name[im["file_name"]].append({"instance_id": ann["id"], "bbox_xyxy": [x, y, x + w, y + h]})
    fixed_rows = []
    for im in gt["images"]:
        name = im["file_name"]
        source_path = ROOT / "data/dataset/valid" / name
        asset = ASSETS / "fixed_val" / name
        copy_asset(source_path, asset)
        raw_gt = truths_by_name[name]
        model_rows = {}
        for model, lookup in (("ppyoloe", ppy_maps["val"]), ("rfdetr", rf_maps["val"])):
            candidates = lookup[name]["native_candidates"]
            groups, winners = deduplicate(candidates)
            display = [p for p in winners if p["detector_score"] > THRESHOLD]
            matched, missed = match_predictions(display, raw_gt)
            hidden = [x for g in groups for x in g["hidden"]]
            all_candidates = {c["candidate_index"]: c for c in candidates}
            suppressed_ids = {x["candidate_index"] for x in hidden}
            for candidate in candidates:
                candidate["suppressed"] = candidate["candidate_index"] in suppressed_ids
                candidate["displayed"] = candidate["detector_score"] > THRESHOLD and not candidate["suppressed"]
            # 按candidate id附加XML匹配结果；不反向影响任何预测过程。
            detail_matches = {p["candidate_index"]: p for p in matched}
            detail_unmatched = {p["candidate_index"]: p for p in matched if p["xml_match"] != "matched"}
            for p in display:
                detail = detail_matches[p["candidate_index"]]
                p.update({k: v for k, v in detail.items() if k in {"xml_match", "matched_instance_id", "matched_iou", "matched_truth_bbox_xyxy"}})
            model_rows[model] = {"native_candidates": candidates, "dedup_groups": groups, "dedup_candidates": winners,
                                 "display_candidates": display, "suppressed_candidates": hidden,
                                 "matched_predictions": [p for p in matched if p["xml_match"] == "matched"],
                                 "unmatched_predictions": list(detail_unmatched.values()), "unmatched_truths": missed}
        base_display = model_rows["ppyoloe"]["display_candidates"]
        rf_display = model_rows["rfdetr"]["display_candidates"]
        un_base, un_rf = [], []
        remaining = {i: p for i, p in enumerate(rf_display)}
        pairs = []
        for b in sorted(base_display, key=lambda x: -x["detector_score"]):
            choices = [(iou(b["bbox_xyxy"], r["bbox_xyxy"]), idx, r) for idx, r in remaining.items()]
            ov, idx, r = max(choices, default=(0.0, None, None), key=lambda x: x[0])
            if r is not None and ov >= IOU_MATCH:
                pairs.append({"ppyoloe_candidate_index": b["candidate_index"], "rfdetr_candidate_index": r["candidate_index"], "iou": ov,
                              "position_changed_candidate": ov < 0.75})
                del remaining[idx]
            else:
                un_base.append(b)
        un_rf = list(remaining.values())
        status = []
        if not base_display and not rf_display: status.append("两模型无框")
        if not base_display and rf_display: status.append("RF新增框候选")
        if base_display and not rf_display: status.append("PP有框_RF无框")
        if un_base or un_rf: status.append("定位变化候选")
        if len(raw_gt) > 1: status.append("多鸟图")
        fixed_rows.append({
            "filename": name, "image_path": str(source_path), "image": f"assets/fixed_val/{name}",
            "width": im["width"], "height": im["height"], "truth": raw_gt, "same_photo": False, "related_scene": False,
            "models": model_rows, "model_pair": {"matched_pairs": pairs, "unmatched_ppyoloe": un_base, "unmatched_rfdetr": un_rf},
            "status_labels": status, "known_baseline_empty": False,
        })
    # 用户给出的9张外来无框名单仅用于事后筛选/摘要，不进入模型输入或预测流程。
    external_rows = []
    all_ext_names = sorted(set(ppy_maps["ext"]) & set(rf_maps["ext"]), key=str.casefold)
    if len(all_ext_names) != 80:
        raise RuntimeError(f"两模型共同外来图片应为80张，实际{len(all_ext_names)}")
    for name in all_ext_names:
        flags = ext_flags.get(name, {})
        # 外来图按既有清单递归存储；真实路径只负责复用相同图片，不把父目录名用作预测输入。
        original_path = Path(ppy_maps["ext"][name]["image_path"])
        asset = ASSETS / "external80" / name
        copy_asset(original_path, asset)
        ppy_candidates = ppy_maps["ext"][name]["native_candidates"]
        rf_candidates = rf_maps["ext"][name]["native_candidates"]
        model_rows = {}
        for model, candidates in (("ppyoloe", ppy_candidates), ("rfdetr", rf_candidates)):
            groups, winners = deduplicate(candidates)
            display = [p for p in winners if p["detector_score"] > THRESHOLD]
            hidden = [x for g in groups for x in g["hidden"]]
            suppressed_ids = {x["candidate_index"] for x in hidden}
            for candidate in candidates:
                candidate["suppressed"] = candidate["candidate_index"] in suppressed_ids
                candidate["displayed"] = candidate["detector_score"] > THRESHOLD and not candidate["suppressed"]
            model_rows[model] = {"native_candidates": candidates, "dedup_groups": groups, "dedup_candidates": winners,
                                 "display_candidates": display, "suppressed_candidates": hidden}
        base_display, rf_display = model_rows["ppyoloe"]["display_candidates"], model_rows["rfdetr"]["display_candidates"]
        rem = {i: p for i, p in enumerate(rf_display)}
        pairs, un_base = [], []
        for b in sorted(base_display, key=lambda x: -x["detector_score"]):
            ov, j, r = max(((iou(b["bbox_xyxy"], q["bbox_xyxy"]), j, q) for j, q in rem.items()), default=(0.0, None, None), key=lambda x: x[0])
            if r is not None and ov >= IOU_MATCH:
                pairs.append({"ppyoloe_candidate_index": b["candidate_index"], "rfdetr_candidate_index": r["candidate_index"], "iou": ov,
                              "position_changed_candidate": ov < 0.75})
                del rem[j]
            else:
                un_base.append(b)
        un_rf = list(rem.values())
        native_rf_by_id = {p["candidate_index"]: p for p in model_rows["rfdetr"]["native_candidates"]}
        for p in un_rf:
            p["is_added_vs_ppyoloe"] = True
            if p["candidate_index"] in native_rf_by_id:
                native_rf_by_id[p["candidate_index"]]["is_added_vs_ppyoloe"] = True
        statuses = []
        if not base_display and not rf_display: statuses.append("两模型无框")
        if not base_display and rf_display: statuses.append("RF新增框候选")
        if base_display and not rf_display: statuses.append("PP有框_RF无框")
        if un_base or un_rf: statuses.append("定位变化候选")
        if flags.get("same_photo"): statuses.append("同源回放")
        if flags.get("related_scene"): statuses.append("相关场景候选")
        if name in NOBOX_BASELINE: statuses.append("原基准9张无框图")
        external_rows.append({
            "filename": name, "image_path": str(original_path), "image": f"assets/external80/{name}",
            "width": ppy_maps["ext"][name]["width"], "height": ppy_maps["ext"][name]["height"],
            "truth": [], "same_photo": bool(flags.get("same_photo", False)), "related_scene": bool(flags.get("related_scene", False)),
            "expected_category_posthoc": flags.get("expected"),
            "models": model_rows,
            "model_pair": {"matched_pairs": pairs, "unmatched_ppyoloe": un_base, "unmatched_rfdetr": un_rf},
            "status_labels": statuses, "known_baseline_empty": name in NOBOX_BASELINE,
        })

    fixed_summary = {m: metrics_for_rows(fixed_rows, m) for m in ("ppyoloe", "rfdetr")}
    candidate_flow = {}
    for model in ("ppyoloe", "rfdetr"):
        mi = [x["matched_iou"] for row in fixed_rows for x in row["models"][model]["matched_predictions"]]
        multi = [row for row in fixed_rows if len(row["truth"]) > 1]
        missed_multi = [row for row in multi if row["models"][model]["unmatched_truths"]]
        candidate_flow[model] = {
            "native_candidates_before_common_iou": sum(len(row["models"][model]["native_candidates"]) for row in fixed_rows),
            "candidates_suppressed_by_common_iou_rule": sum(len(row["models"][model]["suppressed_candidates"]) for row in fixed_rows),
            "deduplicated_low_score_candidates": sum(len(row["models"][model]["dedup_candidates"]) for row in fixed_rows),
            "final_display_boxes_score_gt_0_30": fixed_summary[model]["displayed_boxes"],
            "mean_iou_of_one_to_one_matched_display_boxes": float(np.mean(mi)) if mi else None,
            "median_iou_of_one_to_one_matched_display_boxes": float(np.median(mi)) if mi else None,
            "matched_boxes_iou_0_50_to_0_75_review_band_count": sum(0.5 <= x < 0.75 for x in mi),
            "annotated_multi_instance_images": len(multi),
            "multi_instance_images_with_unmatched_xml_truth": [row["filename"] for row in missed_multi],
            "unmatched_xml_truth_count_on_annotated_multi_instance_images": sum(len(row["models"][model]["unmatched_truths"]) for row in missed_multi),
            "note": "IoU抑制数是规则隐藏的候选数，不等同于已确认同鸟重复框；匹配IoU 0.50-0.75仅作为定位复核提示。多鸟统计按现有XML实例数。",
        }
    gt_by_name = {r["file_name"]: r for r in gt["images"]}
    ap_by_model = {}
    pr_rows = []
    for model in ("ppyoloe", "rfdetr"):
        native = {r["filename"]: r["models"][model]["native_candidates"] for r in fixed_rows}
        deduped = {r["filename"]: r["models"][model]["dedup_candidates"] for r in fixed_rows}
        ap_by_model[model] = {
            "native_model_output": run_coco_eval(gt, gt["images"], "native_low_score_candidates", native, pr_rows, f"{model}_native"),
            "common_iou_deduplicated_low_score": run_coco_eval(gt, gt["images"], "common_iou_dedup_low_score", deduped, pr_rows, f"{model}_dedup"),
        }
    per_class = []
    for model, summ in fixed_summary.items():
        per_class.append({"model": model, "class": "bird", "TP": summ["TP"], "FP_against_current_XML": summ["FP_against_current_XML"],
                          "FN_against_current_XML": summ["FN_against_current_XML"], "Precision": summ["Precision"], "Recall": summ["Recall"], "F1": summ["F1"]})
    fixed_report = {
        "metadata": {
            "validation_images": 172, "ground_truth_instances": 204, "validation_json": CFG["fixed_validation_json"],
            "threshold_strictly_greater_than": THRESHOLD, "matching_iou": IOU_MATCH, "extra_iou_dedup": SUPPRESS_IOU,
            "native_low_candidate_floor": 0.001, "class_agnostic": True, "teacher_test_used": False,
            "interpretation": "本验证集经多轮探索，不是独立测试。表中FP/FN均相对当前XML；XML漏标/框差异可能影响匹配。AP用完整低分候选，在阈值过滤前计算。",
        },
        "summary": fixed_summary, "candidate_flow": candidate_flow, "ap": ap_by_model, "per_class_single_class": per_class,
        "images": fixed_rows,
    }
    fixed_path = REPORTS / "fixed_validation_comparison.json"
    fixed_path.write_text(json.dumps(fixed_report, ensure_ascii=False), encoding="utf-8")
    # 完整分阶段候选是独立交付，含原生候选、隐藏抑制框、去重赢家及阈值结果。
    for model, source_doc in (("ppyoloe", ppy_val), ("rfdetr", rf_val)):
        rows = []
        for r in fixed_rows:
            m = r["models"][model]
            rows.append({"filename": r["filename"], "native_candidates": m["native_candidates"],
                         "dedup_groups": m["dedup_groups"], "dedup_candidates": m["dedup_candidates"],
                         "display_candidates": m["display_candidates"], "suppressed_candidates": m["suppressed_candidates"]})
        (REPORTS / f"{model}_fixed_val_stages.json").write_text(json.dumps({"metadata": source_doc["metadata"], "images": rows}, ensure_ascii=False), encoding="utf-8")

    external_summary = {}
    for scope, selected in (("all_80", external_rows),
                            ("excluding_9_same_source", [r for r in external_rows if not r["same_photo"]]),
                            ("excluding_9_same_source_and_related_scene", [r for r in external_rows if not r["same_photo"] and not r["related_scene"]])):
        counts = {m: {"images": len(selected), "images_with_displayed_boxes": sum(bool(r["models"][m]["display_candidates"]) for r in selected),
                       "no_box_images": [r["filename"] for r in selected if not r["models"][m]["display_candidates"]],
                       "displayed_boxes": sum(len(r["models"][m]["display_candidates"]) for r in selected)} for m in ("ppyoloe", "rfdetr")}
        new_rows = [r for r in selected if r["model_pair"]["unmatched_rfdetr"]]
        counts["cross_model_candidate_review"] = {
            "rfdetr_unmatched_candidate_images": [r["filename"] for r in new_rows],
            "rfdetr_unmatched_candidate_image_count": len(new_rows),
            "rfdetr_unmatched_candidate_box_count": sum(len(r["model_pair"]["unmatched_rfdetr"]) for r in new_rows),
            "ppyoloe_had_display_but_rfdetr_none": [r["filename"] for r in selected if r["models"]["ppyoloe"]["display_candidates"] and not r["models"]["rfdetr"]["display_candidates"]],
            "same_source_replay_count": sum(r["same_photo"] for r in selected),
            "related_scene_candidate_count": sum(r["related_scene"] for r in selected),
            "note": "跨模型未匹配仅表示通过阈值的框在IoU<0.5下未配对；新增候选不是确认真鸟。",
        }
        external_summary[scope] = counts
    empty_recovery = []
    for name in sorted(NOBOX_BASELINE):
        row = next(r for r in external_rows if r["filename"] == name)
        empty_recovery.append({"filename": name, "rfdetr_display_boxes": row["models"]["rfdetr"]["display_candidates"],
                               "status": "new_unverified_candidates" if row["models"]["rfdetr"]["display_candidates"] else "still_no_display_box"})
    regressions = [r["filename"] for r in external_rows if r["models"]["ppyoloe"]["display_candidates"] and not r["models"]["rfdetr"]["display_candidates"]]
    # 重新计算逐图RF新增框候选列表（与XML无关，只做跨模型IoU空间比对）。
    for r in external_rows:
        if "added_vs_ppyoloe" not in r["models"]["rfdetr"]:
            r["models"]["rfdetr"]["added_vs_ppyoloe"] = r["model_pair"]["unmatched_rfdetr"]
            for p in r["models"]["rfdetr"]["added_vs_ppyoloe"]: p["is_added_vs_ppyoloe"] = True
    added = [{"filename": r["filename"], "same_photo": r["same_photo"], "related_scene": r["related_scene"],
              "candidates": r["models"]["rfdetr"]["added_vs_ppyoloe"]} for r in external_rows if r["models"]["rfdetr"]["added_vs_ppyoloe"]]
    ext_report = {
        "metadata": {"images": 80, "threshold_strictly_greater_than": THRESHOLD, "common_iou_dedup": SUPPRESS_IOU,
                     "no_per_instance_ground_truth": True, "same_source_flags_from": CFG["external_flags_report"],
                     "teacher_test_used": False,
                     "note": "外来80图不计算准确率/AP/召回率；新增框仅表示与基准显示框空间未匹配，不能自动称为找到鸟，所有新增框待人工复核。"},
        "summary": external_summary, "original_baseline_empty_9": empty_recovery,
        "ppyoe_had_box_rfdetr_empty_regressions": regressions, "rfdetr_added_unmatched_candidates": added,
        "images": external_rows,
    }
    (REPORTS / "external80_comparison.json").write_text(json.dumps(ext_report, ensure_ascii=False), encoding="utf-8")

    # 表格输出：逐图、逐框、PR点和汇总。
    with (REPORTS / "summary_metrics.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f); w.writerow(["模型", "阶段", "AP50", "AP50:95", "TP", "FP(XML未匹配)", "FN(XML未匹配)", "Precision", "Recall", "F1", "无框图数", "显示框数"])
        for model in ("ppyoloe", "rfdetr"):
            s = fixed_summary[model]
            for stage in ("native_model_output", "common_iou_deduplicated_low_score"):
                ap = ap_by_model[model][stage]
                w.writerow([model, stage, ap["AP50_COCO"], ap["AP50_95_COCO"], s["TP"], s["FP_against_current_XML"], s["FN_against_current_XML"], s["Precision"], s["Recall"], s["F1"], s["images_with_no_displayed_boxes"], s["displayed_boxes"]])
    with (REPORTS / "fixed_validation_per_image.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f); w.writerow(["filename", "XML目标数", "PP显示框", "PP匹配", "PP未匹配", "RF显示框", "RF匹配", "RF未匹配", "状态"])
        for r in fixed_rows:
            w.writerow([r["filename"], len(r["truth"]), len(r["models"]["ppyoloe"]["display_candidates"]), len(r["models"]["ppyoloe"]["matched_predictions"]), len(r["models"]["ppyoloe"]["unmatched_predictions"]), len(r["models"]["rfdetr"]["display_candidates"]), len(r["models"]["rfdetr"]["matched_predictions"]), len(r["models"]["rfdetr"]["unmatched_predictions"]), "|".join(r["status_labels"])])
    with (REPORTS / "fixed_validation_per_box.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f); w.writerow(["filename", "model", "candidate_index", "stage", "detector_score", "x1", "y1", "x2", "y2", "suppressed", "displayed", "xml_match", "matched_instance_id", "matched_iou"])
        for r in fixed_rows:
            for model in ("ppyoloe", "rfdetr"):
                for p in r["models"][model]["native_candidates"]:
                    q = p["bbox_xyxy"]; w.writerow([r["filename"], model, p["candidate_index"], "native", p["detector_score"], *q, p["suppressed"], p["displayed"], p.get("xml_match", "not_displayed"), p.get("matched_instance_id"), p.get("matched_iou")])
    with (REPORTS / "external80_counts.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f); w.writerow(["filename", "same_photo", "related_scene", "PP显示框数", "RF显示框数", "RF新增候选数", "PP有框_RF无框"])
        for r in external_rows:
            w.writerow([r["filename"], r["same_photo"], r["related_scene"], len(r["models"]["ppyoloe"]["display_candidates"]), len(r["models"]["rfdetr"]["display_candidates"]), len(r["models"]["rfdetr"]["added_vs_ppyoloe"]), bool(r["models"]["ppyoloe"]["display_candidates"] and not r["models"]["rfdetr"]["display_candidates"])])
    with (REPORTS / "pr_curve.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "candidate_stage", "iou", "recall", "precision"]); w.writeheader(); w.writerows(pr_rows)
    with (REPORTS / "candidate_flow.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "native_candidates", "suppressed_by_common_iou", "deduplicated_low_score", "displayed_gt_0.30", "matched_mean_iou", "matched_median_iou", "matched_iou_0.50_to_0.75", "annotated_multi_instance_images", "multi_images_with_unmatched_truth", "unmatched_truths_in_multi_images"])
        for model, row in candidate_flow.items():
            w.writerow([model, row["native_candidates_before_common_iou"], row["candidates_suppressed_by_common_iou_rule"], row["deduplicated_low_score_candidates"], row["final_display_boxes_score_gt_0_30"], row["mean_iou_of_one_to_one_matched_display_boxes"], row["median_iou_of_one_to_one_matched_display_boxes"], row["matched_boxes_iou_0_50_to_0_75_review_band_count"], row["annotated_multi_instance_images"], len(row["multi_instance_images_with_unmatched_xml_truth"]), row["unmatched_xml_truth_count_on_annotated_multi_instance_images"]])

    # PR图和训练曲线都从真实评估日志绘制。
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        for model, color in (("ppyoloe", "#1477bf"), ("rfdetr", "#df5a2a")):
            points = [r for r in pr_rows if r["model"] == model and r["candidate_stage"] == "native_low_score_candidates" and r["precision"] is not None]
            ax.plot([r["recall"] for r in points], [r["precision"] for r in points], label=model, color=color)
        ax.set(xlabel="Recall", ylabel="Precision", title="Native low-score COCO PR curve at IoU=0.50"); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(REPORTS / "pr_curve.png", dpi=150); plt.close(fig)
    except Exception as exc:
        (REPORTS / "plot_warning.txt").write_text(str(exc), encoding="utf-8")

    val_meta = {"ppyoloe": ppy_val["metadata"], "rfdetr": rf_val["metadata"]}
    ext_meta = {"ppyoloe": ppy_ext["metadata"], "rfdetr": rf_ext["metadata"]}
    index = f'''<!doctype html><meta charset="utf-8"><title>RF-DETR与PP-YOLOE+对照</title><style>body{{font-family:Arial,'Microsoft YaHei';max-width:1000px;margin:36px auto;background:#f3f7f4;color:#24352c}}article{{background:white;padding:20px;margin:14px 0;border-radius:10px}}a{{color:#176b45}}table{{border-collapse:collapse}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}</style><h1>单类鸟检测器对照</h1><p>只检测 bird，不运行鸟种分类。固定阈值严格&gt;0.30，展示阶段使用同一原IoU≥0.5最高分winner规则；AP使用模型原生低分候选。</p><article><h2>固定验证集：172图/204目标</h2><p><a href="fixed_validation_comparison.html">打开验证集逐图图册</a>　<a href="fixed_validation_comparison.json">完整逐图JSON</a>　<a href="fixed_validation_per_image.csv">逐图CSV</a>　<a href="fixed_validation_per_box.csv">逐框CSV</a></p></article><article><h2>外来80张：无逐实例真值</h2><p><a href="external80_comparison.html">打开外来图对照及新增框人工复核页</a>　<a href="external80_comparison.json">完整逐图JSON</a>　<a href="external80_counts.csv">逐图计数CSV</a></p></article><article><h2>汇总</h2><p><a href="summary_metrics.csv">AP/P/R/F1汇总CSV</a>　<a href="candidate_flow.csv">候选后处理与定位摘要</a>　<a href="pr_curve.png">PR曲线</a>　<a href="pr_curve.csv">PR点CSV</a>　<a href="training_curve.png">训练曲线</a></p><p>验证集此前多次探索，不是独立测试；XML未匹配不自动视为真实误检/漏检。外来80图不计算准确率或mAP。</p></article>'''
    (REPORTS / "index.html").write_text(index, encoding="utf-8")
    (REPORTS / "fixed_validation_comparison.html").write_text(html_page("固定172张验证集：PP-YOLOE+ vs RF-DETR", {"rows": gallery_view(fixed_rows), "summary": {"AP": ap_by_model, **fixed_summary}}, "fixed"), encoding="utf-8")
    (REPORTS / "external80_comparison.html").write_text(html_page("外来80张鸟图：PP-YOLOE+ vs RF-DETR", {"rows": gallery_view(external_rows), "summary": {k: v for k, v in external_summary["all_80"].items()}}, "external"), encoding="utf-8")
    result = {"fixed_validation": {"summary": fixed_summary, "candidate_flow": candidate_flow, "AP": ap_by_model},
              "external80": external_summary, "baseline_empty_9": empty_recovery,
              "external_ppyoloe_box_to_rfdetr_empty": regressions,
              "external_unmatched_new_box_images": [r["filename"] for r in external_rows if r["models"]["rfdetr"]["added_vs_ppyoloe"]],
              "inference_metadata": {"fixed_val": val_meta, "external80": ext_meta}}
    (REPORTS / "metrics_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "completed", "metrics": result["fixed_validation"], "external_summary": external_summary,
                      "baseline_empty_recovery": empty_recovery, "rfdetr_to_empty_regressions": regressions}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()


