from __future__ import annotations
import json, os, subprocess, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports'; EDGE=Path(r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe')
QA=REPORTS/'browser_qa'; QA.mkdir(parents=True,exist_ok=True)
JS_COMMON=r'''<script>
(async()=>{const wait=ms=>new Promise(r=>setTimeout(r,ms));const imgs=[...document.querySelectorAll('.card img')];await Promise.all(imgs.slice(0,3).map(i=>i.complete?Promise.resolve():new Promise(r=>{i.onload=r;i.onerror=r})));let out={mode:MODE,totalRows:DATA.rows.length,initialCards:document.querySelectorAll('.card').length,firstImageLoaded:imgs[0]?.naturalWidth>0,canvasWidth:document.querySelector('.card canvas')?.width||0,canvasHeight:document.querySelector('.card canvas')?.height||0};
if(MODE==='fixed'){document.querySelector('#filter').value='多鸟图';document.querySelector('#filter').onchange();out.multiBirdFilterCards=document.querySelectorAll('.card').length;document.querySelector('#filter').value='全部';document.querySelector('#filter').onchange();document.querySelector('#search').value='baitoubei_084.jpg';document.querySelector('#search').oninput();out.searchCards=document.querySelectorAll('.card').length;const row=DATA.rows.find(x=>x.filename==='baitoubei_084.jpg');if(row){document.querySelector('#stage').value='dedup';document.querySelector('#stage').onchange();out.dedupStageCandidateCount=candidates(row,'rfdetr').length;document.querySelector('#stage').value='native';document.querySelector('#stage').onchange();out.nativeStageCandidateCount=candidates(row,'rfdetr').length;document.querySelector('#stage').value='final';document.querySelector('#stage').onchange();document.querySelector('#search').value='';document.querySelector('#search').oninput();if(row.truth.length){cropZoom(row.image,row.truth[0].bbox_xyxy,row.filename,.9);await wait(250);out.zoomOpened=document.querySelector('#zoomDialog').open;out.zoomCanvasWidth=document.querySelector('#zoomCanvas').width}}document.querySelector('#showXml').checked=false;document.querySelector('#showXml').onchange();out.xmlTogglePresent=!!document.querySelector('#showXml');}
else{out.posthocExpectedLabelPresent=document.body.innerText.includes('目录事后预期：');document.querySelector('#filter').value='新增框候选';document.querySelector('#filter').onchange();out.additionFilterCards=document.querySelectorAll('.card').length;document.querySelector('#filter').value='原基准9张无框图';document.querySelector('#filter').onchange();out.baselineEmptyFilterCards=document.querySelectorAll('.card').length;document.querySelector('#filter').value='全部';document.querySelector('#filter').onchange();const sel=document.querySelector('select[data-review]');out.reviewControlPresent=!!sel;if(sel){sel.value='背景';sel.dispatchEvent(new Event('change',{bubbles:true}));const create=URL.createObjectURL.bind(URL),blobs=[];URL.createObjectURL=b=>{blobs.push(b);return create(b)};URL.revokeObjectURL=()=>{};exportReviews('json');exportReviews('csv');await wait(100);out.exportJsonWorks=blobs.length>0&&(await blobs[0].text()).startsWith('[');out.exportCsvWorks=blobs.length>1&&(await blobs[1].text()).includes('review');out.exportJsonRows=out.exportJsonWorks?JSON.parse(await blobs[0].text()).length:0}}
document.documentElement.dataset.qa=JSON.stringify(out);})();
</script>'''
for mode,src in [('fixed','fixed_validation_comparison.html'),('external','external80_comparison.html')]:
    source=(REPORTS/src).read_text(encoding='utf-8')
    qa_html=source.replace('</body>',JS_COMMON.replace("if(MODE==='fixed')",f"if({str(mode=='fixed').lower()})")+'</body>')
    qa_path=REPORTS/f'_browser_qa_{mode}.html';qa_path.write_text(qa_html,encoding='utf-8')
    dump=QA/f'{mode}_dom.html';shot=QA/f'{mode}_preview.png';log=QA/f'{mode}_edge.log'
    env=os.environ.copy();env['TEMP']=str(ROOT/'runtime/tmp');env['TMP']=env['TEMP']
    for p in [env['TEMP'],str(ROOT/'runtime'/f'edge_profile_{mode}'),str(ROOT/'runtime'/f'edge_cache_{mode}')]:Path(p).mkdir(parents=True,exist_ok=True)
    url=qa_path.resolve().as_uri()
    args=[str(EDGE),'--headless=new','--no-sandbox','--disable-gpu','--disable-gpu-sandbox','--disable-gpu-compositing','--disable-gpu-rasterization','--use-angle=swiftshader','--enable-unsafe-swiftshader','--disable-features=VizDisplayCompositor,UseSkiaRenderer','--no-first-run','--no-default-browser-check','--allow-file-access-from-files',
          f'--user-data-dir={ROOT/"runtime"/f"edge_profile_{mode}"}',f'--disk-cache-dir={ROOT/"runtime"/f"edge_cache_{mode}"}',
          f'--crash-dumps-dir={ROOT/"runtime"}', '--window-size=1440,1000','--virtual-time-budget=18000',f'--screenshot={shot}', '--dump-dom',url]
    with dump.open('wb') as out,log.open('wb') as err:
        result=subprocess.run(args,stdout=out,stderr=err,env=env,timeout=180)
    dom=dump.read_text(encoding='utf-8',errors='replace')
    import re
    match=re.search(r'<html[^>]*data-qa="([^"]+)"',dom)
    qa=json.loads(match.group(1).replace('&quot;','"').replace('&amp;','&')) if match else None
    (QA/f'{mode}_result.json').write_text(json.dumps({'exit_code':result.returncode,'qa':qa,'dump_bytes':dump.stat().st_size,'screenshot':str(shot),'console_log':str(log)},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'mode':mode,'exit_code':result.returncode,'qa':qa,'dump_bytes':dump.stat().st_size,'screenshot':str(shot)},ensure_ascii=False),flush=True)
    qa_path.unlink(missing_ok=True)


