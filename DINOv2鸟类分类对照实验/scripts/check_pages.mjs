import fs from 'node:fs';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
const ROOT='E:/机器学习课设/DINOv2鸟类分类对照实验';
const require=createRequire('E:/机器学习课设/第二代鸟类检测与细粒度分类/work/html_review/check.mjs');
const {chromium}=require('playwright');
const profile=ROOT+'/cache/browser_check_profile';
const browser=await chromium.launchPersistentContext(profile,{executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true,viewport:{width:1500,height:1050}});
const errors=[];
async function checkPage(file,expectedCount,filters,shot){
 const page=await browser.newPage();page.on('pageerror',e=>errors.push(String(e)));
 await page.goto(pathToFileURL(ROOT+'/reports/'+file).href);
 await page.waitForFunction(()=>{const d=JSON.parse(document.querySelector('#data').textContent);const a=[...document.querySelectorAll('article')];return a.length===d.length&&a.every((article,i)=>[...article.querySelectorAll('canvas')].slice(0,3).every(c=>c.width===d[i].width&&c.height===d[i].height))},null,{timeout:60000});
 const all=await page.locator('article').count();if(all!==expectedCount)throw new Error(`${file}: expected ${expectedCount}, got ${all}`);
 const canvasState=await page.locator('article:first-child canvas').evaluateAll(cs=>cs.map(c=>({width:c.width,height:c.height})));
 for(const [filter,expected] of filters){await page.locator('#filter').selectOption({label:filter});const got=await page.locator('article').count();if(got!==expected)throw new Error(`${file} filter ${filter}: expected ${expected}, got ${got}`)}
 await page.locator('#filter').selectOption({label:'all'});
 if(await page.locator('#bio').isVisible()){await page.locator('#bio').uncheck();await page.locator('#bio').check()}
 await page.screenshot({path:ROOT+'/reports/'+shot});
 await page.close();return {file,all,first_three_canvas_sizes:canvasState,filters:Object.fromEntries(filters)};
}
const fixed=await checkPage('fixed_validation_comparison.html',172,[['类别分歧',3],['无显示框',5],['all',172]],'fixed_gallery_browser_check.png');
const external=await checkPage('external80_comparison.html',80,[['模型分歧',3],['无框',9],['同源回放',9],['相关场景',1],['麻雀三类',30],['all',80]],'external_gallery_browser_check.png');
if(errors.length)throw new Error('Browser page errors: '+errors.join(' | '));
const result={status:'passed',browser:'Microsoft Edge headless via Playwright',page_errors:errors,fixed,external,checked:'local image loads, natural-size canvas overlays, counts, search filters, and BioCLIP visibility toggle'};
fs.writeFileSync(ROOT+'/reports/browser_check.json',JSON.stringify(result,null,2));console.log(JSON.stringify(result));await browser.close();
