"""Loopback-only website, durable local history, serial real-model inference."""
import argparse, concurrent.futures, datetime, hashlib, io, json, os, sqlite3, sys
import subprocess, threading, time, uuid, mimetypes, urllib.parse
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from settings import *
from PIL import Image, ImageOps, ImageDraw, ImageFont, UnidentifiedImageError

DATA=ROOT/'data';JOBS=DATA/'jobs';STATIC=ROOT/'static'
Image.MAX_IMAGE_PIXELS=20_000_000
POOL=concurrent.futures.ThreadPoolExecutor(max_workers=1)
RUNNING=threading.Event()

def connection():
    con=sqlite3.connect(DATA/'history.sqlite',timeout=20)
    con.row_factory=sqlite3.Row
    return con

def update(job_id, **values):
    with connection() as con:
        keys=list(values)
        con.execute('UPDATE history SET '+','.join(k+'=?' for k in keys)+' WHERE id=?',[values[k] for k in keys]+[job_id])

def environment():
    env=os.environ.copy()
    cache=ROOT/'.cache';cache.mkdir(exist_ok=True)
    for k,folder in {'TEMP':'tmp','TMP':'tmp','HF_HOME':'huggingface','HF_HUB_CACHE':'huggingface/hub',
                     'TORCH_HOME':'torch','MPLCONFIGDIR':'matplotlib','PIP_CACHE_DIR':'pip'}.items():
        p=cache/folder;p.mkdir(parents=True,exist_ok=True);env[k]=str(p)
    env.update(PYTHONIOENCODING='utf-8',PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
               HF_HUB_DISABLE_XET='1',OMP_NUM_THREADS='4')
    rf_cache=RF_PROJECT/'cache'
    for folder in ['huggingface','huggingface/hub','torch','tmp']:(rf_cache/folder).mkdir(parents=True,exist_ok=True)
    env.update(RF_HOME=str(RF_PROJECT/'weights/cache'),ROBOFLOW_HOME=str(RF_PROJECT/'weights/cache'),
               HF_HOME=str(rf_cache/'huggingface'),HUGGINGFACE_HUB_CACHE=str(rf_cache/'huggingface/hub'),
               TORCH_HOME=str(rf_cache/'torch'),TEMP=str(rf_cache/'tmp'),TMP=str(rf_cache/'tmp'))
    return env

def run_worker(python, role, file, log):
    env=environment()
    if Path(python)==RF_PYTHON:
        extra=[str(RF_PACKAGE),str(THIRD/'.venv/Lib/site-packages')]
        env['PYTHONPATH']=os.pathsep.join(extra+([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
    with log.open('ab') as stream:
        proc=subprocess.run([str(python),str(ROOT/'worker.py'),role,str(file)],cwd=str(ROOT),env=env,
                            stdout=stream,stderr=subprocess.STDOUT,timeout=360,creationflags=subprocess.CREATE_NO_WINDOW)
    if proc.returncode:
        raise RuntimeError(f'{role} 推理失败，退出码 {proc.returncode}；请查看本次日志。')

def annotated(image_path, detections, output):
    with Image.open(image_path) as fp:im=fp.convert('RGB')
    draw=ImageDraw.Draw(im);size=max(16,im.width//55)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',size)
    for p in detections:
        x,y,x2,y2=p['bbox_xyxy'];color='#278658'
        draw.rectangle((x,y,x2,y2),outline=color,width=max(2,im.width//300))
        text=f"{p['class_name']} 检测{p['detector_score']:.1%}"
        if p.get('classifier_top_score') is not None:text+=f" 分类{p['classifier_top_score']:.1%}"
        box=draw.textbbox((0,0),text,font=font);tw=box[2]-box[0]+12;th=box[3]-box[1]+12
        tx=min(max(x,0),max(0,im.width-tw));ty=max(0,y-th)
        draw.rectangle((tx,ty,tx+tw,ty+th),fill=color)
        draw.text((tx+6,ty+2),text,font=font,fill='white')
    im.save(output,quality=94)

def execute(job):
    start=time.perf_counter();folder=Path(job['directory']);file=folder/'job.json';log=folder/'inference.log'
    update(job['id'],status='running',stage='正在检测鸟的位置')
    try:
        detector_python=RF_PYTHON if job['model']=='gen4' else PADDLE_PYTHON
        detector_role='detect-rf' if job['model']=='gen4' else 'detect'
        run_worker(detector_python,detector_role,file,log)
        prop=json.loads((folder/'proposals.json').read_text(encoding='utf-8'))
        if job['model']=='gen1':variants={'ppyoloe':prop['displayed']}
        elif not prop['displayed']:
            keys=(['resnet18','bilinear'] if job['classifier']=='both' else [job['classifier']]) if job['model']=='gen2' else ['rfdetr_bioclip2' if job['model']=='gen4' else 'bioclip2']
            variants={key:[] for key in keys}
        else:
            update(job['id'],stage='正在判断鸟的种类')
            run_worker(PADDLE_PYTHON if job['model']=='gen2' else TORCH_PYTHON,'cnn' if job['model']=='gen2' else 'bio',file,log)
            variants=json.loads((folder/'classifications.json').read_text(encoding='utf-8'))
        result={'id':job['id'],'model':job['model'],'classifier':job['classifier'],'filename':job['filename'],
                'width':prop['width'],'height':prop['height'],'threshold':job['threshold'],'variants':variants,
                'groups':prop['groups'],'detector_weights':prop['weights'],'device':prop['device'],
            'elapsed_seconds':round(time.perf_counter()-start,2),'input_size':prop.get('input_size',640),'crop_margin':.15,
                'dedup':'IoU >= 0.5, highest detector score, no containment or merging',
            'classifier_weights':{key:str(CNN_WEIGHTS[key]) if key in CNN_WEIGHTS else str(BIO_HEAD) if key in ['bioclip2','rfdetr_bioclip2'] else None for key in variants},
                'bioclip_encoder_weights':str(BIO_WEIGHTS) if job['model'] in ['gen3','gen4'] else None,
                'detector_backbone':prop.get('backbone'),'detector_input_size':prop.get('input_size',640),
                'detector_candidate_threshold':prop.get('candidate_threshold'),'detector_inference_seconds':prop.get('inference_seconds'),
                'detector_peak_cuda_memory_gib':prop.get('peak_cuda_memory_gib'),
                'detector_weights_sha256':prop.get('weights_sha256'),
                'truth_used':False,'note':'只在八类范围内给出判断；分数不是正确率，未显示框不证明图中没有鸟。'}
        for key,items in variants.items():annotated(job['image'],items,folder/f'result_{key}.jpg')
        (folder/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        update(job['id'],status='done',stage='识别完成',result=json.dumps(result,ensure_ascii=False))
    except Exception as exc:
        update(job['id'],status='failed',stage='识别失败',error=str(exc))

def details(row):
    out=dict(row)
    if out['result']:out['result']=json.loads(out['result'])
    return out

def model_inventory():
    paths={'gen1':[PADDLE_PYTHON,PADDLE_SITE_PACKAGES/'paddle/__init__.py',FIRST_WEIGHTS,FIRST/'configs/ppyoloe_s_bird_640.yml'],
           'gen2':[PADDLE_PYTHON,PADDLE_SITE_PACKAGES/'paddle/__init__.py',DETECTOR_WEIGHTS,*CNN_WEIGHTS.values()],
           'gen3':[PADDLE_PYTHON,TORCH_PYTHON,DETECTOR_WEIGHTS,BIO_WEIGHTS,BIO_HEAD],
           'gen4':[RF_PYTHON,RF_CHECKPOINT,BIO_WEIGHTS,BIO_HEAD,RF_PACKAGE/'rfdetr/__init__.py']}
    return {key:{'name':MODELS[key],'ready':all(p.is_file() for p in ps),'missing':[str(p) for p in ps if not p.is_file()]} for key,ps in paths.items()}

def project_info():
    docs={
      'gen1':WORK/'第一代结果归档/README.md',
      'gen2':WORK/'第二代结果归档/20261003_211135/README.md',
      'gen3':THIRD/'README.md',
      'abcreport':THIRD/'experiments/detector_training_improvements_norm_aligned/reports/实验收尾与结果摘要.md',
      'rollback':THIRD/'experiments/detector_training_improvements_norm_aligned/reports/包含去重回退记录/默认规则核查.json',
      'rfdetr_bioclip':RF_PROJECT/'端到端BioCLIP2对照/reports/实验说明与结论.md',
      'user_review_tree08':ROOT/'data/人工复核意见/树麻雀_08_RF-DETR分类复核_20261012.json'}
    return {'classes':NAMES,'models':model_inventory(),'documents':[{'id':key,'name':{'gen1':'第一代归档说明','gen2':'第二代归档说明','gen3':'第三代工程说明','abcreport':'A/B/C 实验摘要','rollback':'默认去重规则核查','rfdetr_bioclip':'第四代RF-DETR＋BioCLIP 2实验记录','user_review_tree08':'树麻雀_08用户复核记录'}[key],'available':p.exists()} for key,p in docs.items()],
      'bio_summary':json.loads((THIRD/'reports/bioclip2/固定验证集结果.json').read_text(encoding='utf-8'))['summary'],
      'gen4_summary':json.loads((RF_PROJECT/'端到端BioCLIP2对照/reports/end_to_end_results.json').read_text(encoding='utf-8'))['fixed_validation']['summary']['rfdetr'],
      'scope':'172张固定验证图、204个标注实例；验证集已用于选权重和探索，不是独立测试。',
      'source_paths':{key:str(p) for key,p in docs.items()}}

class Handler(BaseHTTPRequestHandler):
    server_version='BirdLab/1.0'
    def log_message(self, fmt, *args):print(datetime.datetime.now().strftime('%H:%M:%S'),fmt%args,flush=True)
    def send(self, data, code=200, content='application/json; charset=utf-8'):
        if not isinstance(data,bytes):data=json.dumps(data,ensure_ascii=False).encode('utf-8')
        self.send_response(code);self.send_header('Content-Type',content);self.send_header('Content-Length',str(len(data)))
        self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
        self.end_headers();self.wfile.write(data)
    def local_request(self):
        host=self.headers.get('Host','')
        if host not in [f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}']:return False
        origin=self.headers.get('Origin')
        return not origin or origin in ['http://'+host]
    def do_GET(self):
        if not self.local_request():self.send({'error':'仅允许本地访问'},403);return
        url=urllib.parse.urlsplit(self.path);path=urllib.parse.unquote(url.path)
        try:
            if path=='/api/status':self.send({'models':model_inventory(),'port':self.server.server_port,'classes':NAMES});return
            if path=='/api/project':self.send(project_info());return
            if path=='/api/history':
                with connection() as con:rows=con.execute('SELECT * FROM history ORDER BY created DESC LIMIT 200').fetchall()
                self.send([details(r) for r in rows]);return
            if path.startswith('/api/job/'):
                key=path.split('/')[-1]
                with connection() as con:row=con.execute('SELECT * FROM history WHERE id=?',[key]).fetchone()
                if row:self.send(details(row))
                else:self.send({'error':'记录不存在'},404)
                return
            if path.startswith('/api/document/'):
                info=project_info();key=path.split('/')[-1];target=info['source_paths'].get(key)
                if not target or not Path(target).is_file():self.send({'error':'资料不存在'},404);return
                self.send(Path(target).read_bytes(),content='text/plain; charset=utf-8');return
            if path.startswith('/files/'):
                parts=path.split('/')
                if len(parts)!=4 or len(parts[2])!=32 or any(c not in '0123456789abcdef' for c in parts[2]):raise ValueError('无效路径')
                if parts[3] not in ['input.png','result.json','result_ppyoloe.jpg','result_resnet18.jpg','result_bilinear.jpg','result_bioclip2.jpg','result_rfdetr_bioclip2.jpg','inference.log']:raise ValueError('无效文件')
                target=JOBS/parts[2]/parts[3]
            else:
                if path not in ['/','/index.html','/app.js','/style.css']:self.send({'error':'页面不存在'},404);return
                target=STATIC/('index.html' if path=='/' else path[1:])
            if not target.is_file():self.send({'error':'文件尚未生成'},404);return
            kind=mimetypes.guess_type(str(target))[0] or 'text/plain'
            self.send(target.read_bytes(),content=kind+('; charset=utf-8' if kind.startswith('text/') or kind in ['application/javascript','application/json'] else ''))
        except Exception as exc:self.send({'error':str(exc)},400)
    def do_POST(self):
        if not self.local_request():self.send({'error':'仅允许本地提交'},403);return
        url=urllib.parse.urlsplit(self.path)
        try:
            if url.path!='/api/recognize':self.send({'error':'接口不存在'},404);return
            length=int(self.headers.get('Content-Length','0'))
            if length<=0 or length>20*1024*1024:raise ValueError('请选择小于20MB的图片')
            q=urllib.parse.parse_qs(url.query);model=q.get('model',['gen4'])[0];classifier=q.get('classifier',['bilinear'])[0]
            threshold=float(q.get('threshold',['0.30'])[0])
            if model not in MODELS or classifier not in ['resnet18','bilinear','both'] or not (.05<=threshold<=.95):raise ValueError('模型或阈值无效')
            if model in ['gen3','gen4']:classifier='bioclip2'
            if not model_inventory()[model]['ready']:raise ValueError('所选模型的权重或环境缺失')
            with connection() as con:
                pending=con.execute("SELECT count(*) FROM history WHERE status IN ('queued','running')").fetchone()[0]
            if pending>=5:raise ValueError('识别队列已满，请稍后再试')
            payload=self.rfile.read(length)
            try:
                with Image.open(io.BytesIO(payload)) as src:
                    if src.format not in ['JPEG','PNG','WEBP','BMP']:raise ValueError('请使用JPG、PNG、WEBP或BMP图片')
                    if src.width*src.height>20_000_000:raise ValueError('图片过大，请缩小至2000万像素以内')
                    image=ImageOps.exif_transpose(src).convert('RGB');image.load()
            except (UnidentifiedImageError, OSError):
                raise ValueError('图片无法读取，文件可能已损坏或不是有效图片。')
            key=uuid.uuid4().hex;folder=JOBS/key;folder.mkdir()
            image.save(folder/'input.png')
            filename=q.get('filename',['上传图片'])[0].replace('\\','/').split('/')[-1][:180]
            created=datetime.datetime.now().astimezone().isoformat(timespec='seconds')
            job={'id':key,'model':model,'classifier':classifier,'threshold':threshold,'filename':filename,
                 'image':str(folder/'input.png'),'directory':str(folder),'created':created}
            (folder/'job.json').write_text(json.dumps(job,ensure_ascii=False,indent=2),encoding='utf-8')
            with connection() as con:con.execute('INSERT INTO history VALUES (?,?,?,?,?,?,?,?,?,?)',
                [key,created,filename,model,classifier,threshold,'queued','等待识别',None,None])
            POOL.submit(execute,job);self.send({'id':key},202)
        except Exception as exc:self.send({'error':str(exc)},400)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--port',type=int,default=8765);args=ap.parse_args()
    DATA.mkdir(exist_ok=True);JOBS.mkdir(exist_ok=True)
    environment()
    with connection() as con:
        con.execute('CREATE TABLE IF NOT EXISTS history(id TEXT PRIMARY KEY,created TEXT,filename TEXT,model TEXT,classifier TEXT,threshold REAL,status TEXT,stage TEXT,result TEXT,error TEXT)')
        con.execute("UPDATE history SET status='failed',stage='运行已中断',error='服务上次退出时识别尚未完成，请重新上传识别。' WHERE status IN ('queued','running')")
    (ROOT/'模型接入清单.json').write_text(json.dumps(model_inventory(),ensure_ascii=False,indent=2),encoding='utf-8')
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    pid_file=DATA/'server.pid';pid_file.write_text(str(os.getpid()),encoding='ascii')
    print(f'本地鸟类识别网站 http://127.0.0.1:{args.port}  |  Ctrl+C 停止服务',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:
        server.server_close();POOL.shutdown(wait=True)
        try:
            if pid_file.read_text(encoding='ascii').strip()==str(os.getpid()):pid_file.unlink()
        except OSError:pass

if __name__=='__main__':main()
