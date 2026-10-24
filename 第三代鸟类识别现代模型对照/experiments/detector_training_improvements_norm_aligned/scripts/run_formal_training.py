"""A/B/C顺序正式训练器。阶段日志、PID、退出码与最新检查点均落盘，重复运行可恢复。"""
from __future__ import annotations
import json,os,subprocess,sys,time,traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
P=json.loads((ROOT/'paths.json').read_text(encoding='utf-8'))
PY=Path(P['paddle_python']);PD=Path(P['paddledetection'])
STATUS=ROOT/'reports'/'formal_training_status.json'

def save(obj):
    STATUS.parent.mkdir(parents=True,exist_ok=True);tmp=STATUS.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(STATUS)

def latest_checkpoint(run):
    found=[]
    # PaddleDetection 2.9 用0起始epoch文件名，例如4.pdparams代表完成第5轮。
    for p in run.glob('*.pdparams'):
        if p.name=='model_final.pdparams':continue
        try:
            if p.stem.isdigit():ep=int(p.stem)+1
            elif p.stem.startswith('epoch_'):ep=int(p.stem.split('_')[1])
            else:continue
            if not p.with_suffix('.pdopt').exists() or not p.with_suffix('.pdema').exists():continue
            found.append((ep,p))
        except Exception:pass
    return max(found,key=lambda x:x[0]) if found else None

def run_phase(group,phase,state):
    run=ROOT/'runs'/group;run.mkdir(parents=True,exist_ok=True)
    stage_key=f'{group}_phase{phase}'
    if state.get('stages',{}).get(stage_key,{}).get('status')=='completed':return state
    cfg=ROOT/'configs'/f'{group}_phase{phase}.yml'
    args=[str(PY),str(ROOT/'scripts'/'train_seeded_official.py'),'-c',str(cfg),'--enable_ce=True']
    latest=latest_checkpoint(run)
    resume=None
    if phase==1:
        resume=latest[1] if latest else None
    else:
        # 第2阶段应从第1阶段最终状态继续；只有出现完成>30轮的阶段2快照后，才恢复阶段2断点。
        if latest and latest[0]>30:resume=latest[1]
        else:
            resume=run/'model_final.pdparams'
            if not resume.exists():raise RuntimeError(f'{group} phase2 requested, but phase1 final checkpoint is missing')
    if resume:args += ['-r',str(resume)]
    log=ROOT/'logs'/f'{group}_phase{phase}.log';log.parent.mkdir(parents=True,exist_ok=True)
    stage={'status':'running','group':group,'phase':phase,'pid':None,'started_at':time.strftime('%Y-%m-%d %H:%M:%S %z'),
           'command':args,'resume_from':str(resume) if resume else None,'log':str(log)}
    state.setdefault('stages',{})[stage_key]=stage;state.update({'status':'running','current_stage':stage_key});save(state)
    env=os.environ.copy();env.update({'PADDLE_HOME':str(ROOT/'.cache'/'paddle'),'TEMP':str(ROOT/'.cache'/'tmp'),'TMP':str(ROOT/'.cache'/'tmp')})
    for d in [env['PADDLE_HOME'],env['TEMP']]:Path(d).mkdir(parents=True,exist_ok=True)
    lib=Path(P['paddle_python']).parent.parent/'Lib'/'site-packages'/'nvidia'
    env['PATH']=str(lib/'cudnn'/'bin')+';'+str(lib/'cublas'/'bin')+';'+env.get('PATH','')
    env['PYTHONUNBUFFERED']='1';env['PYTHONPATH']=str(PD)+os.pathsep+env.get('PYTHONPATH','')
    with log.open('a',encoding='utf-8',buffering=1) as f:
        f.write('\n\n===== START '+stage['started_at']+' =====\n')
        f.write('COMMAND: '+subprocess.list2cmdline(args)+'\n')
        if stage['resume_from']:f.write('RESUME: '+stage['resume_from']+'\n')
        proc=subprocess.Popen(args,cwd=PD,env=env,stdout=f,stderr=subprocess.STDOUT)
        stage['pid']=proc.pid;save(state)
        code=proc.wait()
    stage['exit_code']=code;stage['finished_at']=time.strftime('%Y-%m-%d %H:%M:%S %z')
    if code:
        stage['status']='failed';state['status']='failed';save(state);raise RuntimeError(f'{stage_key} failed with exit code {code}; see {log}')
    stage['status']='completed';stage['final_checkpoint']=str(run/'model_final.pdparams');save(state)
    return state

def main():
    state=json.loads(STATUS.read_text(encoding='utf-8')) if STATUS.exists() else {'stages':{}}
    for group in 'ABC':
        for phase in (1,2):state=run_phase(group,phase,state)
    state.update({'status':'training_completed','current_stage':None,'training_finished_at':time.strftime('%Y-%m-%d %H:%M:%S %z')})
    save(state)
    print(json.dumps({'status':state['status'],'status_file':str(STATUS)},ensure_ascii=False),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        state=json.loads(STATUS.read_text(encoding='utf-8')) if STATUS.exists() else {}
        state.update({'status':'failed','last_error':str(e),'traceback':traceback.format_exc()});save(state);raise
