import json, os, re, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; R=ROOT/"reports"; QA=R/"browser_qa"; QA.mkdir(exist_ok=True)
EDGE=Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
INJECT=r'''<script>(async()=>{const wait=t=>new Promise(r=>setTimeout(r,t));await wait(1500);let fi=document.querySelector('.card img');let o={mode:MODE,rows:D.rows.length,cards:document.querySelectorAll('.card').length,loadedImages:[...document.querySelectorAll('.card img')].slice(0,3).map(x=>x.naturalWidth>0),firstSrc:fi?.src,firstComplete:fi?.complete,firstCanvas:[...document.querySelectorAll('.card canvas')].slice(0,3).map(x=>x.width),scoreSpans:document.querySelectorAll('.scores span').length};
if(MODE==='fixed'){document.querySelector('#f').value='退步';document.querySelector('#f').onchange();o.regressionCards=document.querySelectorAll('.card').length;document.querySelector('#f').value='类别分歧';document.querySelector('#f').onchange();o.disagreementCards=document.querySelectorAll('.card').length;document.querySelector('#f').value='多鸟漏检';document.querySelector('#f').onchange();o.multiBirdOmissionCards=document.querySelectorAll('.card').length;document.querySelector('#f').value='全部';document.querySelector('#f').onchange();document.querySelector('#q').value='shumaque_043.jpg';document.querySelector('#q').oninput();o.searchCards=document.querySelectorAll('.card').length;document.querySelector('#xml').checked=false;document.querySelector('#xml').onchange();o.xmlTogglePresent=!!document.querySelector('#xml');let r=D.rows[0],p=r.models.ppyoloe.display_candidates[0];if(p){zoom(r.asset,p.bbox_xyxy,'QA');await wait(200);o.zoomOpened=document.querySelector('#dlg').open}}
else{document.querySelector('#f').value='新增鸟框';document.querySelector('#f').onchange();o.confirmedNewFrameCards=document.querySelectorAll('.card').length;document.querySelector('#f').value='待复核';document.querySelector('#f').onchange();o.pendingCards=document.querySelectorAll('.card').length;document.querySelector('#f').value='全部';document.querySelector('#f').onchange();let reviews=D.rows.flatMap(r=>r.models.rfdetr.display_candidates.map(p=>p.user_detection_review?.status).filter(Boolean));o.confirmedReviewCount=reviews.filter(x=>x==='确实是鸟').length;o.pendingReviewCount=reviews.filter(x=>x==='暂不能确定').length}
document.documentElement.dataset.qa=JSON.stringify(o)})().catch(e=>{document.documentElement.dataset.qaError=String(e)})</script>'''
for mode,page in [("fixed","fixed_val_end_to_end.html"),("external","external80_end_to_end.html")]:
    qpath=R/("_qa_"+page); source=(R/page).read_text(encoding="utf-8")
    source=source.replace("<script>const D=","<script>window.__errs=[];window.addEventListener('error',e=>{window.__errs.push(String(e.message)+' @ '+e.lineno);document.documentElement.dataset.jsError=String(e.message)+' @ '+e.lineno});const D=")
    qpath.write_text(source.replace("</body>",INJECT.replace("let o={mode:MODE,","let o={errors:window.__errs,mode:MODE,")+"</body>"),encoding="utf-8")
    dump=QA/(mode+"_dom.html"); shot=QA/(mode+"_preview.png"); log=QA/(mode+"_edge.log")
    env=os.environ.copy();env["TEMP"]=str(ROOT/"cache"/"tmp");env["TMP"]=env["TEMP"]
    for p in [env["TEMP"],str(ROOT/"cache"/("edge_"+mode)),str(ROOT/"cache"/("edgecache_"+mode))]:Path(p).mkdir(parents=True,exist_ok=True)
    args=[str(EDGE),"--headless=new","--no-sandbox","--disable-gpu","--disable-gpu-sandbox","--disable-gpu-rasterization","--use-angle=swiftshader","--enable-unsafe-swiftshader","--no-first-run","--no-default-browser-check","--allow-file-access-from-files",f"--user-data-dir={ROOT/'cache'/('edge_'+mode)}",f"--disk-cache-dir={ROOT/'cache'/('edgecache_'+mode)}","--window-size=1440,1000","--virtual-time-budget=18000",f"--screenshot={shot}","--dump-dom",qpath.resolve().as_uri()]
    with dump.open("wb") as fo,log.open("wb") as fe: proc=subprocess.run(args,stdout=fo,stderr=fe,env=env,timeout=120)
    text=dump.read_text(encoding="utf-8",errors="replace"); m=re.search(r'<html[^>]*data-qa="([^"]+)"',text)
    qa=json.loads(m.group(1).replace("&quot;",'"').replace("&amp;","&")) if m else None
    js_error=re.search(r'data-js-error="([^"]+)"',text); qa_error=re.search(r'data-qa-error="([^"]+)"',text)
    (QA/(mode+"_result.json")).write_text(json.dumps({"exit_code":proc.returncode,"qa":qa,"js_error":js_error.group(1) if js_error else None,"qa_error":qa_error.group(1) if qa_error else None,"screenshot":str(shot),"console_log":str(log)},ensure_ascii=False,indent=2),encoding="utf-8")
    qpath.unlink(missing_ok=True)
    try: dump.unlink(missing_ok=True)
    except PermissionError: pass
    print(json.dumps({"mode":mode,"exit_code":proc.returncode,"qa":qa},ensure_ascii=False))
