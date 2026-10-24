const fs=require('fs'),vm=require('vm');
const html=fs.readFileSync('C组包含阈值09逐框人工复核.html','utf8');
const payload=(html.match(/<script id="payload" type="application\/json">([\s\S]*?)<\/script>/)||[])[1];
const js=(html.match(/<script>([\s\S]*?)<\/script>/)||[])[1];
if(!payload||!js)throw Error('页面数据/脚本块缺失');
const store={};let clickCount=0,errors=[];
class Ctx{drawImage(){} strokeRect(){} fillText(){}}
class El{constructor(tag='div'){this.tag=tag;this.dataset={};this.children=[];this.checked=true;this.value='';this.textContent='';this._html='';}
 set innerHTML(v){this._html=v;if(v==='')this.children=[];if(this.tag==='article'||this.tag==='section'){this.children=[];for(const match of v.matchAll(/<canvas class='(view|eventZoom)'([^>]*)><\/canvas>/g)){let c=new El('canvas');c.className=match[1];for(const x of match[2].matchAll(/([a-zA-Z-]+)='([^']*)'/g))c.dataset[x[1].replace(/^data-/,'').replace(/-([a-z])/g,(_,c)=>c.toUpperCase())]=x[2];this.children.push(c)}for(const match of v.matchAll(/<(select|textarea) class='(choice|note)' data-id='([^']*)'/g)){let e=new El(match[1]);e.className=match[2];e.dataset.id=match[3];this.children.push(e)}}}
 get innerHTML(){return this._html}
 querySelectorAll(sel){if(sel==='canvas.view')return this.children.filter(x=>x.tag==='canvas'&&x.className==='view');if(sel==='.eventZoom')return this.children.filter(x=>x.tag==='canvas'&&x.className==='eventZoom');if(sel==='.choice')return this.children.filter(x=>x.className==='choice');if(sel==='.note')return this.children.filter(x=>x.className==='note');return []}
 append(x){this.children.push(x)} appendChild(x){this.children.push(x)}
 getContext(){return new Ctx()} click(){clickCount++}
}
const ids={};for(const id of ['summary','cards','model','filter','search','xml','pred','supp','count','exportj','exportc','payload'])ids[id]=new El(id==='cards'?'div':'input');
ids.model.value='C';ids.filter.value='priority';ids.search.value='';
ids.xml.checked=ids.pred.checked=ids.supp.checked=true;ids.payload.textContent=payload;
const document={getElementById(id){return ids[id]},createElement(tag){return new El(tag)}};
class Img{constructor(){this.naturalWidth=1280;this.naturalHeight=900}set src(v){this._src=v;if(this.onload)this.onload()}get src(){return this._src}}
const localStorage={getItem(k){return store[k]||null},setItem(k,v){store[k]=v}};
const URL={createObjectURL(){return 'blob:test'},revokeObjectURL(){}};
const context={document,Image:Img,localStorage,URL,Blob,console:{log(){},error(...x){errors.push(x)},warn(){}},setTimeout,JSON,Math,Number,String,Object,Array,Map,Set};
vm.runInNewContext(js,context,{timeout:20000});
const initial={cards:ids.cards.children.length,count:ids.count.textContent,summary:ids.summary.innerHTML.includes('176/18/28')};
ids.filter.value='all_events';ids.filter.onchange();
const allC={cards:ids.cards.children.length,count:ids.count.textContent};
ids.model.value='baseline';ids.model.onchange();
const allBase={cards:ids.cards.children.length,count:ids.count.textContent};
ids.model.value='C';ids.filter.value='lost_tp';ids.filter.onchange();
const lost={cards:ids.cards.children.length,zoom:ids.cards.children.reduce((n,c)=>n+c.children.filter(x=>x.className==='eventZoom').length,0)};
ids.exportj.onclick();ids.exportc.onclick();
const result={initial,allC,allBase,lost,clickCount,errors};
console.log(JSON.stringify(result));
if(!initial.summary||allC.cards!==10||allBase.cards!==2||lost.cards!==2||lost.zoom<4||clickCount!==2||errors.length)process.exit(3);

