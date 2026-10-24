// Local Edge/CDP acceptance check using Node's native WebSocket, no dependencies.
const fs=require('node:fs'),path=require('node:path'),cp=require('node:child_process');
const root=__dirname,port=Number(process.argv[2]||8766),debug=9237;
const edge='C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
const profile=path.join(root,'.cache','edge_qa');fs.mkdirSync(profile,{recursive:true});
const processHandle=cp.spawn(edge,['--headless=new','--disable-gpu','--no-first-run','--disable-background-networking','--remote-debugging-port='+debug,'--remote-allow-origins=http://localhost:'+debug,'--user-data-dir='+profile,'about:blank'],{windowsHide:true,stdio:'ignore'});
const pause=ms=>new Promise(r=>setTimeout(r,ms));
let socket,id=0,pending=new Map(),errors=[];
function send(method,params={}){return new Promise((resolve,reject)=>{const n=++id;pending.set(n,{resolve,reject});socket.send(JSON.stringify({id:n,method,params}));});}
async function evaluate(expression){const r=await send('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;}
async function waitFor(expression,limit=15000){const deadline=Date.now()+limit;while(Date.now()<deadline){if(await evaluate(expression))return;await pause(300);}throw Error('Timed out: '+expression);}
(async()=>{try{
 let targets;for(let i=0;i<50;i++){try{targets=await (await fetch('http://127.0.0.1:'+debug+'/json')).json();if(targets.length)break;}catch{}await pause(200);}
 socket=new WebSocket(targets.find(x=>x.type==='page').webSocketDebuggerUrl);
 await new Promise((resolve,reject)=>{socket.onopen=resolve;socket.onerror=reject;});
 socket.onmessage=event=>{const m=JSON.parse(event.data);if(m.id){const p=pending.get(m.id);if(p){pending.delete(m.id);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result);}}else if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails);else if(m.method==='Log.entryAdded'&&m.params.entry.level==='error')errors.push(m.params.entry);};
 await send('Runtime.enable');await send('Log.enable');await send('Page.enable');
 await send('Emulation.setDeviceMetricsOverride',{width:1550,height:1000,deviceScaleFactor:1,mobile:false});
 await send('Page.navigate',{url:'http://127.0.0.1:'+port});
 await waitFor("document.querySelector('#connection')?.textContent.includes('本地运行')");
 await waitFor("document.querySelectorAll('.history-item').length > 0");
 await evaluate("document.querySelector('nav button[data-page=compare]').click()");
 const compare=await evaluate("document.querySelector('#compare').classList.contains('active')");
 await evaluate("document.querySelector('nav button[data-page=project]').click()");
 const metrics=await evaluate("document.querySelectorAll('.metric').length");
 await evaluate("document.querySelector('nav button[data-page=recognize]').click()");
 const history=await evaluate("state.history.filter(x=>x.status==='done' && x.model==='gen3' && x.result.variants.bioclip2.length>0).map(x=>x.id)");
 if(!history.length)throw Error('No successful BioCLIP bird result available');
 const documentTree=await send('DOM.getDocument');
 const fileInput=await send('DOM.querySelector',{nodeId:documentTree.root.nodeId,selector:'#file'});
 await send('DOM.setFileInputFiles',{nodeId:fileInput.nodeId,files:['E:/机器学习课设/自然场景鸟类检测PPYOLOE/data/prepared/images/val/baitoubei_009.jpg']});
 await waitFor("state.file !== null && !document.querySelector('#recognizeButton').disabled");
 const uploadWorks=await evaluate("state.file.name==='baitoubei_009.jpg'");
 await evaluate("document.querySelector('[name=generation][value=gen1]').click();document.querySelector('#recognizeButton').click()");
 await waitFor("state.busy === false && state.current?.status==='done' && state.current?.model==='gen1'",60000);
 const recognitionButtonWorks=await evaluate("document.querySelector('#resultTable tbody tr') !== null");
 await evaluate("document.querySelector('[name=generation][value=gen2]').click();document.querySelector('[data-cnn=both]').click()");
 const secondHistory=await evaluate("state.history.find(x=>x.status==='done'&&x.model==='gen2'&&x.result.variants.bilinear.length>0)?.id");
 if(!secondHistory)throw Error('No completed CNN comparison');
 await evaluate(`document.querySelector('.history-item[data-id="${secondHistory}"]').click()`);
 await waitFor("document.querySelectorAll('#variantTabs button').length === 2");
 await evaluate("document.querySelector('#variantTabs button[data-variant=bilinear]').click()");
 const cnnCompareWorks=await evaluate("state.variant==='bilinear'&&document.querySelector('#scores details')!==null");
 await evaluate("document.querySelector('[name=generation][value=gen3]').click()");
 await evaluate(`document.querySelector('.history-item[data-id="${history[0]}"]').click()`);
 await waitFor("document.querySelector('#resultTable tbody tr') !== null");
 const before=await evaluate("document.querySelector('#canvas').toDataURL()");
 await evaluate("document.querySelector('#showBoxes').click()");
 const after=await evaluate("document.querySelector('#canvas').toDataURL()");
 await evaluate("document.querySelector('#showBoxes').click();document.querySelector('#zoomButton').click()");
 const zoom=await evaluate("document.querySelector('#zoomDialog').open");
 await evaluate("document.querySelector('#closeZoom').click()");
 const shot=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:true});fs.writeFileSync(path.join(root,'网站实机预览.png'),Buffer.from(shot.data,'base64'));
 const rows=await evaluate("document.querySelectorAll('#resultTable tbody tr').length");
 const small=await send('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:false});
 await pause(300);const overflow=await evaluate("document.documentElement.scrollWidth > innerWidth");
 const report={browser:'Microsoft Edge headless, actual CDP interaction',port,fileSelectionEnablesRecognition:uploadWorks,recognitionButtonCompletesRealInference:recognitionButtonWorks,cnnCompareSwitchWorks:cnnCompareWorks,compareTab:compare,metricCards:metrics,historyCount:await evaluate('state.history.length'),displayRows:rows,boxToggleChangesCanvas:before!==after,zoomDialogWorks:zoom,mobileHorizontalOverflow:overflow,consoleErrors:errors};
 fs.writeFileSync(path.join(root,'浏览器验收.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));
 if(errors.length||overflow||!zoom||before===after)process.exitCode=1;
 }catch(e){console.error(e.stack);process.exitCode=1;}finally{if(socket)socket.close();processHandle.kill();}})();
