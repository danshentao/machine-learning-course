"""Monitor the running FET experiment and publish its completed results."""
from common import *
import time,subprocess,sys,traceback
import psutil

def main():
    status=ROOT/'reports/当前实验状态.json'
    started=time.time();last_epoch=0
    while True:
        running=[]
        for p in psutil.process_iter(['name','cmdline']):
            try:
                cmd=' '.join(p.info['cmdline'] or [])
                if p.info['name'] and p.info['name'].lower().startswith('python') and 'fet_classifier.py' in cmd and '--smoke' not in cmd:running.append(p.pid)
            except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        data=json.loads(status.read_text(encoding='utf-8')) if status.exists() else {}
        log=ROOT/'runs/fet_fgvc/training.jsonl';lines=log.read_text().splitlines() if log.exists() else []
        if lines:
            record=json.loads(lines[-1]);data['fet_last_completed_epoch']=record;last_epoch=record['epoch']
        data['fet_process_ids']=running
        if not running:break
        write(status,data);time.sleep(20)
        if time.time()-started>4*3600:raise RuntimeError('FET monitor exceeded 4h; training process was not terminated')
    result=ROOT/'reports/fet_fgvc/固定验证集结果.json'
    if not result.exists():raise RuntimeError('FET process ended without fixed-validation result; last completed epoch '+str(last_epoch))
    d=json.loads(result.read_text(encoding='utf-8'));data['fet_fgvc']='completed: full training and fixed validation';data['fet_summary']=d['summary'];write(status,data)
    subprocess.run([sys.executable,str(ROOT/'scripts/build_gallery.py')],check=True)
    # User deferred DINOv3 to generation 4 on 2026-10-04; never auto-run it here.
    if data.get('dinov3_deferred_to_generation4'):
        print('FET results published. DINOv3 reserved for generation 4.',flush=True)
        return
    # Pick up official user-downloaded weights only if already present.
    weights=Path('E:/机器学习课设/DINOv3权重');local=None
    if all((weights/f).exists() for f in ['model.safetensors','config.json','preprocessor_config.json']):local=str(weights)
    else:
        matches=list(weights.glob('*vits16*.pth')) if weights.exists() else []
        if len(matches)==1:local=str(matches[0])
    if local:
        data['dinov3']='official local weights found; frozen-head experiment running';write(status,data)
        subprocess.run([sys.executable,str(ROOT/'scripts/frozen_encoder.py'),'--model','dinov3','--model-path',local],check=True)
        data['dinov3']='completed';write(status,data);subprocess.run([sys.executable,str(ROOT/'scripts/build_gallery.py')],check=True)
    print('Finished available generation3 experiments and gallery.',flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        write(ROOT/'reports/后台收尾失败记录.json',{'error':traceback.format_exc(),'time':time.time()});raise
