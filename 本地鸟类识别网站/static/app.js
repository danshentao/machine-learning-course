const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const state={file:null,url:null,generation:'gen4',classifier:'bilinear',variant:null,current:null,image:null,busy:false,inventory:null,history:[]};
const titles={ppyoloe:'PP-YOLOE-s',resnet18:'ResNet18',bilinear:'Bilinear CNN',bioclip2:'BioCLIP 2',rfdetr_bioclip2:'RF-DETR + BioCLIP 2'};
const modelNames={gen1:'第一代',gen2:'第二代',gen3:'第三代',gen4:'第四代'};
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const percent=x=>x==null?'—':(x*100).toFixed(1)+'%';
async function api(path,options){const r=await fetch(path,options);const d=await r.json();if(!r.ok)throw Error(d.error||'请求失败');return d;}
function message(text,error=false){$('#message').textContent=text;$('#message').classList.toggle('error',error);}
function updateButton(){const ready=state.inventory?.[state.generation]?.ready;$('#recognizeButton').disabled=!state.file||state.busy||!ready;$('#recognizeButton').textContent=state.busy?'识别中，请稍候…':'▶ 识别图片';}
function loadImage(src){return new Promise((resolve,reject)=>{const im=new Image();im.onload=()=>resolve(im);im.onerror=()=>reject(Error('图片无法显示'));im.src=src;});}
function clearResult(){state.current=null;state.variant=null;$('#variantTabs').classList.add('hidden');$('#resultTable').innerHTML='';$('#scores').innerHTML='';$('.downloads').querySelectorAll('a').forEach(x=>x.classList.add('hidden'));$('#resultMeta').textContent='待识别图片';$('#resultBadge').textContent='尚未识别';}
async function selectFile(file){
 if(state.busy){message('请等待本次识别完成后再添加图片。');return;}
 if(!file)return;
 if(!/^image\/(jpeg|png|webp|bmp|x-ms-bmp)$/.test(file.type)&&!(/\.(jpe?g|png|webp|bmp)$/i.test(file.name))){message('请选择 JPG、PNG、WEBP 或 BMP 图片。',true);return;}
 if(file.size>20*1024*1024){message('图片超过20MB，请先缩小。',true);return;}
 const newUrl=URL.createObjectURL(file);
 try{const im=await loadImage(newUrl);if(im.naturalWidth*im.naturalHeight>20_000_000)throw Error('图片超过2000万像素，请先缩小。');
  if(state.url)URL.revokeObjectURL(state.url);state.url=newUrl;state.file=file;state.image=im;clearResult();
  $('#selectedFile').innerHTML=`<img src="${esc(newUrl)}" alt="上传预览"><span>${esc(file.name)}</span>`;$('#selectedFile').classList.remove('hidden');
  draw();message('图片已添加，点击“识别图片”开始。');updateButton();
 }catch(e){URL.revokeObjectURL(newUrl);message(e.message,true);}
}
function currentDetections(){return state.current?.result?.variants?.[state.variant]||[];}
function draw(){
 if(!state.image)return;
 const canvas=$('#canvas'),im=state.image,scale=Math.min(1800/im.naturalWidth,1200/im.naturalHeight,1);
 canvas.width=Math.round(im.naturalWidth*scale);canvas.height=Math.round(im.naturalHeight*scale);
 const ctx=canvas.getContext('2d');ctx.drawImage(im,0,0,canvas.width,canvas.height);
 if($('#showBoxes').checked){for(const [index,p] of currentDetections().entries()){
  const b=p.bbox_xyxy.map(v=>v*scale);ctx.strokeStyle='#2e8753';ctx.lineWidth=Math.max(2,canvas.width/350);ctx.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1]);
  const fs=Math.max(15,Math.round(canvas.width/60));ctx.font=`600 ${fs}px "Microsoft YaHei",sans-serif`;
  const cls=p.classifier_top_score==null?'':` · 分类 ${percent(p.classifier_top_score)}`;
  const label=`${index+1} ${p.class_name} · 检测 ${percent(p.detector_score)}${cls}`;const tw=ctx.measureText(label).width+12,th=fs+12;
  const x=Math.max(0,Math.min(b[0],canvas.width-tw)),y=Math.max(0,b[1]-th);
  ctx.fillStyle='#2e8753';ctx.fillRect(x,y,tw,th);ctx.fillStyle='#fff';ctx.fillText(label,x+6,y+fs+2);
 }}
 canvas.classList.remove('hidden');$('#emptyView').classList.add('hidden');$('#zoomButton').classList.remove('hidden');
}
function displayVariant(key){
 state.variant=key;const result=state.current.result,items=currentDetections();
 $$('#variantTabs button').forEach(b=>b.classList.toggle('active',b.dataset.variant===key));draw();
 $('#resultTable').innerHTML=items.length?`<table><thead><tr><th>目标</th><th>预测类别</th><th>检测分数</th><th>分类最高得分</th></tr></thead><tbody>${items.map((p,i)=>`<tr data-target="${i}"><td>鸟 ${i+1}</td><td>${esc(p.class_name)}</td><td>${percent(p.detector_score)}</td><td>${percent(p.classifier_top_score)}</td></tr>`).join('')}</tbody></table>`:'<p class="notice">当前阈值下没有显示框。这不证明图片中没有鸟；复杂背景、遮挡、小目标或模型适用范围都可能影响检测。</p>';
 $('#scores').innerHTML=items.map((p,i)=>p.classifier_top_score==null?'':`<details><summary>鸟 ${i+1} · 查看八类分类得分</summary><div class="score-bars">${Object.entries(p.classifier_scores).sort((a,b)=>b[1]-a[1]).map(([n,v])=>`<div class="score-row"><span>${esc(n)}</span><span class="bar"><i style="width:${v*100}%"></i></span><span>${percent(v)}</span></div>`).join('')}</div></details>`).join('');
 $('#resultMeta').textContent=`${result.width} × ${result.height} · ${titles[key]} · 阈值 > ${percent(result.threshold)} · ${items.length} 个显示框 · ${result.elapsed_seconds} 秒（含模型加载）`;
 $('#downloadImage').href=`/files/${state.current.id}/result_${key}.jpg`;$('#downloadImage').download=`${state.current.filename}_${key}.jpg`;
 $('#downloadJson').href=`/files/${state.current.id}/result.json`;$('.downloads').querySelectorAll('a').forEach(x=>x.classList.remove('hidden'));
}
async function showRecord(row){
 state.current=row;state.image=await loadImage(`/files/${row.id}/input.png`);$('#resultBadge').textContent=row.status==='done'?'真实模型结果':row.stage;
 if(row.status==='done'){
  const keys=Object.keys(row.result.variants);state.variant=keys.includes(state.variant)?state.variant:keys[0];
  $('#variantTabs').innerHTML=keys.map(k=>`<button data-variant="${k}">${titles[k]}</button>`).join('');$('#variantTabs').classList.toggle('hidden',keys.length<2);
  $$('#variantTabs button').forEach(b=>b.onclick=()=>displayVariant(b.dataset.variant));displayVariant(state.variant);
 }else{state.variant=null;$('#variantTabs').classList.add('hidden');$('#resultTable').innerHTML='';$('#scores').innerHTML='';$('.downloads').querySelectorAll('a').forEach(x=>x.classList.add('hidden'));draw();message(row.error||row.stage,row.status==='failed');}
 $$('.history-item').forEach(x=>x.classList.toggle('active',x.dataset.id===row.id));
}
async function refreshHistory(){
 try{state.history=await api('/api/history');$('#historyList').innerHTML=state.history.length?state.history.map(row=>{
  const result=row.result;const species=result?[...new Set(Object.values(result.variants).flat().map(x=>x.class_name))].join('、')||'当前阈值下无框':row.status==='failed'?'识别失败':row.stage;
  const time=new Date(row.created).toLocaleString('zh-CN',{month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit'});
  return `<button class="history-item ${row.id===state.current?.id?'active':''}" data-id="${row.id}"><img src="/files/${row.id}/input.png" alt="历史图片"><div class="description"><strong title="${esc(row.filename)}">${esc(row.filename)}</strong><span class="species">${esc(species)}</span><small>${modelNames[row.model]||row.model} · >${percent(row.threshold)} · ${time}</small></div></button>`;
 }).join(''):'<p class="muted">还没有识别记录</p>';
 $$('.history-item').forEach(b=>b.onclick=async()=>{if(state.busy){message('识别完成后可以查看其他历史记录。');return;}try{await showRecord(await api('/api/job/'+b.dataset.id));}catch(e){message(e.message,true);}});
 }catch(e){message('历史记录加载失败：'+e.message,true);}
}
async function recognize(){
 if(state.busy||!state.file)return;
 state.busy=true;updateButton();clearResult();draw();message('图片正在上传…');
 try{
  const query=new URLSearchParams({model:state.generation,classifier:state.classifier,threshold:$('#threshold').value/100,filename:state.file.name});
  const {id}=await api('/api/recognize?'+query,{method:'POST',headers:{'Content-Type':state.file.type||'application/octet-stream'},body:state.file});
  await refreshHistory();
  for(;;){const row=await api('/api/job/'+id);message(row.stage);$('#resultBadge').textContent=row.stage;
   if(row.status==='done'){await showRecord(row);message('识别完成。可以查看结果，或切换模型再次识别。');break;}
   if(row.status==='failed'){await showRecord(row);throw Error(row.error);}
   await new Promise(resolve=>setTimeout(resolve,1500));
  }
 }catch(e){message(e.message,true);}finally{state.busy=false;updateButton();await refreshHistory();}
}
async function loadProject(){
 try{const d=await api('/api/project');$('#classPills').innerHTML=d.classes.map(n=>`<span>${esc(n)}</span>`).join('');
  const s=d.gen4_summary;$('#metrics').innerHTML=[[percent(s.Precision),'Precision'],[percent(s.Recall),'Recall'],[s.F1.toFixed(4),'F1'],[s.TP+'/'+s.GT,'正确匹配 / 标注实例']].map(([value,label])=>`<div class="metric"><strong>${value}</strong><small>${label}</small></div>`).join('');
  $('#metricScope').textContent=d.scope+' 检测分数严格大于30%，同类且IoU≥0.5一对一匹配。';
  $('#documents').innerHTML=d.documents.filter(x=>x.available).map(x=>`<a href="/api/document/${x.id}" target="_blank" rel="noopener">${esc(x.name)} ↗</a>`).join('');
 }catch(e){$('#metricScope').textContent='项目资料读取失败：'+e.message;}
}
$$('nav button').forEach(b=>b.onclick=()=>{$$('nav button').forEach(x=>x.classList.toggle('active',x===b));$$('.page').forEach(x=>x.classList.toggle('active',x.id===b.dataset.page));});
$$('[name=generation]').forEach(r=>r.onchange=()=>{state.generation=r.value;$$('.model').forEach(x=>x.classList.toggle('selected',x.querySelector('input').checked));$('#cnnOptions').classList.toggle('hidden',r.value!=='gen2');$('#threshold').value=r.value==='gen1'?50:30;$('#thresholdValue').textContent=$('#threshold').value+'%';message('已切换模型；已显示的历史结果不会改变，点击识别生成新结果。');updateButton();});
$$('[data-cnn]').forEach(b=>b.onclick=()=>{state.classifier=b.dataset.cnn;$$('[data-cnn]').forEach(x=>x.classList.toggle('active',x===b));});
$('#threshold').oninput=()=>{$('#thresholdValue').textContent=$('#threshold').value+'%';};
$('#showBoxes').onchange=draw;$('#file').onchange=e=>selectFile(e.target.files[0]);$('#dropzone').onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();$('#file').click();}};
for(const event of ['dragenter','dragover'])$('#dropzone').addEventListener(event,e=>{e.preventDefault();$('#dropzone').classList.add('drag');});
for(const event of ['dragleave','drop'])$('#dropzone').addEventListener(event,e=>{e.preventDefault();$('#dropzone').classList.remove('drag');});
$('#dropzone').addEventListener('drop',e=>selectFile(e.dataTransfer.files[0]));$('#recognizeButton').onclick=recognize;$('#refreshHistory').onclick=refreshHistory;
$('#zoomButton').onclick=()=>{$('#zoomImage').src=$('#canvas').toDataURL('image/png');$('#zoomDialog').showModal();};$('#closeZoom').onclick=()=>$('#zoomDialog').close();
async function boot(){try{const d=await api('/api/status');state.inventory=d.models;$('#connection').textContent='● 本地运行';for(const [key,info] of Object.entries(d.models)){const input=$(`[name=generation][value=${key}]`);input.disabled=!info.ready;if(!info.ready)input.closest('label').title='模型文件缺失';}updateButton();await refreshHistory();await loadProject();}catch(e){$('#connection').textContent='● 连接失败';message('请启动本地服务后刷新网页。',true);}}
boot();
