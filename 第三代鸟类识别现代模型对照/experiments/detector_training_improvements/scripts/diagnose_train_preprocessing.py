"""诊断微调输入归一化：沿用原代训练reader归一化，1轮/小LR作独立短跑。"""
from pathlib import Path
import json, os, re, subprocess, sys, time

ROOT=Path(__file__).resolve().parents[1]
P=json.loads((ROOT/'paths.json').read_text(encoding='utf-8'))
PY=Path(P['paddle_python']); PD=Path(P['paddledetection'])
SOURCE=ROOT/'configs'/'A_phase1.yml'
CFG=ROOT/'configs'/'diagnostics'/'train_imagenet_norm_1epoch.yml'
RUN=ROOT/'runs'/'diagnostic_imagenet_norm'
LOG=ROOT/'logs'/'diagnostic_imagenet_norm_train.log'

def main():
    CFG.parent.mkdir(parents=True,exist_ok=True);RUN.mkdir(parents=True,exist_ok=True)
    s=SOURCE.read_text(encoding='gb18030')
    s=s.replace('epoch: 30\nsnapshot_epoch: 5','epoch: 1\nsnapshot_epoch: 1',1)
    s=s.replace('save_dir: '+str(ROOT/'runs'/'A'), 'save_dir: '+str(RUN),1)
    s=s.replace('base_lr: 0.0001','base_lr: 0.00001',1)
    old=('''      mean:\n      - 0.0\n      - 0.0\n      - 0.0\n      std:\n      - 1.0\n      - 1.0\n      - 1.0\n      is_scale: false\n      norm_type: none''')
    new=('''      mean:\n      - 0.485\n      - 0.456\n      - 0.406\n      std:\n      - 0.229\n      - 0.224\n      - 0.225\n      is_scale: true\n      norm_type: mean_std''')
    if s.count(old)!=1:raise RuntimeError(f'expected one TrainReader normalization block, got {s.count(old)}')
    s=s.replace(old,new,1);CFG.write_text(s,encoding='gb18030')
    env=os.environ.copy();env.update({'PADDLE_HOME':str(ROOT/'.cache'/'paddle'),'TEMP':str(ROOT/'.cache'/'tmp'),'TMP':str(ROOT/'.cache'/'tmp'),'PYTHONUNBUFFERED':'1'})
    lib=PY.parent.parent/'Lib'/'site-packages'/'nvidia';env['PATH']=str(lib/'cudnn'/'bin')+';'+str(lib/'cublas'/'bin')+';'+env.get('PATH','');env['PYTHONPATH']=str(PD)+os.pathsep+env.get('PYTHONPATH','')
    cmd=[str(PY),str(ROOT/'scripts'/'train_seeded_official.py'),'-c',str(CFG),'--enable_ce=True']
    t=time.perf_counter()
    with LOG.open('w',encoding='utf-8') as f:p=subprocess.run(cmd,cwd=PD,env=env,stdout=f,stderr=subprocess.STDOUT)
    if p.returncode:raise RuntimeError(f'diagnostic short-run failed ({p.returncode}); see {LOG}')
    record={'status':'completed','purpose':'one-epoch diagnostic only; not a formal comparison','train_normalization':'ImageNet mean/std with /255, matching archived fine-tune TrainReader','evaluation_normalization':'official inherited EvalReader raw pixels, unchanged','base_lr':1e-5,'epochs':1,'seconds':time.perf_counter()-t,'config':str(CFG),'weight':str(RUN/'model_final.pdparams'),'weight_sha256':__import__('hashlib').sha256((RUN/'model_final.pdparams').read_bytes()).hexdigest(),'log':str(LOG)}
    (ROOT/'reports'/'diagnostic_imagenet_norm.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(record,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
