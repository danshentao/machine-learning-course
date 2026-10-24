import {createRequire} from 'node:module';import {pathToFileURL} from 'node:url';import fs from 'node:fs';
const require=createRequire('E:/机器学习课设/第二代鸟类检测与细粒度分类/work/html_review/check.mjs');const {chromium}=require('playwright');
const out='E:/机器学习课设/第三代鸟类识别现代模型对照/reports/640与960检测输入对照';
const browser=await chromium.launch({executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true});const page=await browser.newPage({viewport:{width:1500,height:1000}});let errors=[];page.on('pageerror',e=>errors.push(String(e)));
await page.goto(pathToFileURL(out+'/640与960逐图对照.html').href);await page.waitForFunction(()=>document.querySelector('canvas')?.width>300);const first=await page.locator('#count').innerText();if(!first.includes('筛选 5 张'))throw Error(first);
await page.selectOption('#filter','all');const all=await page.locator('#count').innerText();if(!all.includes('172'))throw Error(all);
await page.fill('#search','jiamaque_045');await page.locator('#gt').check();await page.waitForTimeout(300);await page.screenshot({path:out+'/找回家麻雀检查.png',fullPage:true});
await page.fill('#search','baitoubei_009');await page.waitForTimeout(300);await page.screenshot({path:out+'/定位退步检查.png',fullPage:true});
if(errors.length)throw Error(errors.join('\n'));fs.writeFileSync(out+'/页面检查.json',JSON.stringify({first,all,errors},null,2));console.log(JSON.stringify({first,all,errors}));await browser.close();
