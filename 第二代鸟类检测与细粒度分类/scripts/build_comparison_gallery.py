"""生成人工标注、第一代和第二代预测的本地交互HTML核查图册。"""
from __future__ import annotations
import argparse,csv,json,xml.etree.ElementTree as ET
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'configs/paths.json').read_text(encoding='utf-8'))
FIRST=Path(CFG['first_generation']); CLASSMAP=json.loads((ROOT/'configs/class_mapping.json').read_text(encoding='utf-8'))
LABELS=CLASSMAP['xml_label_to_standard']

def readj(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--gen2-resnet',type=Path,default=ROOT/'runs/two_stage_smoke/predictions.json')
    ap.add_argument('--gen2-bilinear',type=Path,default=ROOT/'runs/two_stage_smoke_bilinear/predictions.json')
    ap.add_argument('--output',type=Path,default=ROOT/'reports/smoke_comparison_gallery.html')
    ap.add_argument('--title',default='第一代与第二代两阶段推理对照')
    ap.add_argument('--max-candidates',type=int,default=25,help='HTML每图显示最高检测分数的前N个候选；完整候选仍保存在预测JSON/CSV')
    args=ap.parse_args()
    first=readj(Path(CFG['first_generation_review_archive'])/'严格阈值全量逐图候选/逐图原始候选_隐藏候选_最终分组.json')
    byfirst={}
    # 对照第一代最终展示口径：跨类别分组每组留最高分框，再严格要求score>0.5。
    for rec in first['images']:
        byfirst[rec['filename']]=[g['winner'] for g in rec.get('candidate_groups',[])
          if float(g['winner']['score'])>0.5]
    res=readj(args.gen2_resnet);bil=readj(args.gen2_bilinear) if args.gen2_bilinear.exists() else {'metadata':{},'images':[]}
    bybil={r['filename']:r for r in bil['images']};rows=[]
    xml_dir=FIRST/'data/annotations_raw'
    for gen2 in res['images']:
        filename=gen2['filename'];image=Path(gen2['image'])
        xml=xml_dir/(image.stem+'.xml');truth=[]
        if xml.exists():
            root=ET.parse(xml).getroot()
            for i,o in enumerate(root.findall('object'),1):
                raw=(o.findtext('name') or '').strip();b=o.find('bndbox')
                if raw in LABELS:
                    truth.append({'instance':i,'class_name':LABELS[raw],'raw_label':raw,
                      'bbox_xyxy':[float(b.findtext(k)) for k in ('xmin','ymin','xmax','ymax')]})
        def top_candidates(records):
            ordered=sorted(records,key=lambda x:float(x.get('detector_score',0)),reverse=True)
            chosen=ordered[:args.max_candidates]
            chosen_ids={id(x) for x in chosen}
            chosen.extend(x for x in ordered if x.get('displayed') and id(x) not in chosen_ids)
            return chosen
        rows.append({'filename':filename,'image':image.as_uri(),'width':gen2['width'],'height':gen2['height'],
          'truth':truth,'gen1':byfirst.get(filename,[]),
          'resnet':top_candidates(gen2.get('candidates',[])),'bilinear':top_candidates(bybil.get(filename,{}).get('candidates',[]))})
    payload=json.dumps(rows,ensure_ascii=False).replace('</','<\\/')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>@@TITLE@@</title>
<style>body{font:15px system-ui,"Microsoft YaHei",sans-serif;margin:18px;background:#f4f6f8;color:#18212a}header{position:sticky;top:0;background:#f4f6f8;padding:12px;z-index:3;border-bottom:1px solid #ccd3da}.controls{display:flex;gap:16px;flex-wrap:wrap;align-items:center}input,select{padding:8px}.card{background:white;margin:18px 0;padding:14px;border-radius:10px;box-shadow:0 1px 5px #ccd3da}.layout{display:grid;grid-template-columns:minmax(400px,1.1fr) minmax(360px,1fr);gap:16px}.stage{position:relative;width:100%;align-self:start}.stage img{display:block;width:100%;height:auto}.stage canvas{position:absolute;inset:0;width:100%;height:100%}.legend{font-size:13px;color:#384650}.items{max-height:70vh;overflow:auto}.item{border-top:1px solid #ddd;padding:7px}.small{font-size:12px;color:#58636d}.g1{color:#1769aa}.r{color:#bd3e12}.b{color:#7746a8}@media(max-width:850px){.layout{grid-template-columns:1fr}.items{max-height:none}}</style>
<header><h2>@@TITLE@@</h2><p>本页每图列出第二代最高检测分数的至多@@LIMIT@@个候选和所有满足显示阈值的框；完整候选另存在预测JSON/CSV。当前默认数据是短跑演示，正式评估请用固定验证集预测文件生成正式图册。</p><div class="controls"><label>文件筛选 <input id="q" placeholder="输入文件名"></label><label>第二代分类器 <select id="model"><option value="resnet">普通ResNet18</option><option value="bilinear">Bilinear CNN</option></select></label><label><input type="checkbox" id="gt" checked>人工XML框</label><label><input type="checkbox" id="g1" checked>第一代预测</label><label><input type="checkbox" id="g2" checked>第二代预测候选</label><label><input type="checkbox" id="high">只画第二代显示框（det&gt;0.5）</label></div><p class="legend">颜色：绿色=人工XML；蓝色=第一代预测；橙色=第二代候选。第二代列出检测分数、分类最高分、乘积排序分及八类分类分数。乘积不保证是校准概率。</p><div id="count"></div></header><main id="main"></main>
<script>const data=@@DATA@@;const $=id=>document.getElementById(id);function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}function draw(card,row){const img=card.querySelector('img'),c=card.querySelector('canvas'),ctx=c.getContext('2d');if(!img.naturalWidth)return;c.width=img.naturalWidth;c.height=img.naturalHeight;ctx.clearRect(0,0,c.width,c.height);const mode=$('model').value;let boxes=[];if($('gt').checked)boxes.push(...row.truth.map(x=>({...x,color:'#21a34a',tag:'XML #'+x.instance+' '+x.class_name})));if($('g1').checked)boxes.push(...row.gen1.map(x=>({...x,color:'#1674c8',tag:'G1 '+x.class_name+' '+Number(x.score).toFixed(3)})));if($('g2').checked){let preds=row[mode]||[];if($('high').checked)preds=preds.filter(x=>x.displayed);boxes.push(...preds.map(x=>({...x,color:'#e45618',tag:'G2 det='+Number(x.detector_score).toFixed(3)+' '+x.class_name})));}for(const b of boxes){const q=b.bbox_xyxy;if(!q||q.length<4)continue;const x=Math.max(0,q[0]),y=Math.max(0,q[1]),w=Math.max(0,Math.min(c.width,q[2])-x),h=Math.max(0,Math.min(c.height,q[3])-y);ctx.strokeStyle=b.color;ctx.lineWidth=Math.max(2,c.width/500);ctx.strokeRect(x,y,w,h);ctx.font='bold '+Math.max(13,c.width/65)+'px sans-serif';ctx.fillStyle=b.color;ctx.fillText(b.tag,x,Math.max(14,y-4));}}
function render(){const q=$('q').value.toLowerCase(),show=data.filter(x=>x.filename.toLowerCase().includes(q));$('count').textContent='当前 '+show.length+' / '+data.length+' 张；第二代真值不会参与推理。';$('main').innerHTML='';for(const row of show){const card=document.createElement('section');card.className='card';const preds=row[$('model').value];const list=[...row.truth.map(x=>({type:'XML',cl:x.class_name,box:x.bbox_xyxy,more:'实例 #'+x.instance+' 原标签 '+x.raw_label})),...row.gen1.map(x=>({type:'第一代',cl:x.class_name,box:x.bbox_xyxy,more:'score='+Number(x.score).toFixed(4)})),...preds.map(x=>({type:'第二代',cl:x.class_name,box:x.bbox_xyxy,more:'det='+Number(x.detector_score).toFixed(4)+' cls='+Number(x.classifier_top_score).toFixed(4)+' det×cls='+Number(x.final_score_product).toFixed(5)+' displayed='+x.displayed+'<br>八类分数：'+Object.entries(x.classifier_scores||{}).map(([k,v])=>esc(k)+' '+Number(v).toFixed(3)).join(' · ')}))];card.innerHTML='<h3>'+esc(row.filename)+' <span class="small">'+row.width+'×'+row.height+'</span></h3><div class="layout"><div class="stage"><img src="'+esc(row.image)+'"><canvas></canvas></div><div class="items">'+list.map(x=>'<div class="item"><b class="'+(x.type==='XML'?'':x.type==='第一代'?'g1':'r')+'">'+esc(x.type)+' '+esc(x.cl)+'</b><div class="small">框 '+esc(JSON.stringify(x.box))+'<br>'+x.more+'</div></div>').join('')+'</div></div>';const img=card.querySelector('img');img.onload=()=>draw(card,row);if(img.complete)draw(card,row);$('main').appendChild(card);}}
['q','model','gt','g1','g2','high'].forEach(id=>$(id).addEventListener(id==='q'?'input':'change',render));render();</script></html>'''
    page=page.replace('@@TITLE@@',args.title).replace('@@DATA@@',payload).replace('@@LIMIT@@',str(args.max_candidates))
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(page,encoding='utf-8')
    print(json.dumps({'output':str(args.output),'images':len(rows),'truth_objects':sum(len(x['truth']) for x in rows),
      'gen1_candidates':sum(len(x['gen1']) for x in rows),'gen2_resnet_candidates':sum(len(x['resnet']) for x in rows),
      'gen2_bilinear_candidates':sum(len(x['bilinear']) for x in rows),'second_generation_status':res['metadata'].get('classifier_weights')},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
