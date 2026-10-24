"""后台等待A/B/C正式训练结束后，顺序自动完成官方评估、预测比较与困难样本页。"""
from __future__ import annotations
import argparse,json,os,re,subprocess,sys,time,traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];P=json.loads((ROOT/'paths.json').read_text(encoding='utf-8'))
STATUS=ROOT/'reports'/'post_training_closeout.json';TRAIN=ROOT/'reports'/'formal_training_status.json'
PADDLE=Path(P['paddle_python']);TORCH=Path(P['generation3'])/'.venv'/'Scripts'/'python.exe';PD=Path(P['paddledetection'])
def save(d):
 STATUS.parent.mkdir(parents=True,exist_ok=True);tmp=STATUS.with_suffix('.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(STATUS)
def env_for(kind):
 e=os.environ.copy();e.update({'PYTHONUNBUFFERED':'1','TEMP':str(ROOT/'.cache'/'tmp'),'TMP':str(ROOT/'.cache'/'tmp')})
 for d in [e['TEMP']]:Path(d).mkdir(parents=True,exist_ok=True)
 if kind=='paddle':
  e['PADDLE_HOME']=str(ROOT/'.cache'/'paddle');Path(e['PADDLE_HOME']).mkdir(parents=True,exist_ok=True)
  lib=PADDLE.parent.parent/'Lib'/'site-packages'/'nvidia';e['PATH']=str(lib/'cudnn'/'bin')+';'+str(lib/'cublas'/'bin')+';'+e.get('PATH','')
  e['PYTHONPATH']=str(PD)+os.pathsep+e.get('PYTHONPATH','')
 return e
def stage(state,name,python,script,args,kind,cwd):
 if state.get('stages',{}).get(name,{}).get('status')=='completed':return state
 log=ROOT/'logs'/f'closeout_{name}.log';log.parent.mkdir(parents=True,exist_ok=True)
 cmd=[str(python),str(script),*args];record={'status':'running','command':cmd,'log':str(log),'started_at':time.strftime('%Y-%m-%d %H:%M:%S %z')}
 state.setdefault('stages',{})[name]=record;state['status']='running';state['current_stage']=name;save(state)
 with log.open('a',encoding='utf-8',buffering=1) as f:
  f.write('\nCOMMAND: '+subprocess.list2cmdline(cmd)+'\n');p=subprocess.Popen(cmd,cwd=str(cwd),env=env_for(kind),stdout=f,stderr=subprocess.STDOUT);record['pid']=p.pid;save(state);code=p.wait()
 record['exit_code']=code;record['finished_at']=time.strftime('%Y-%m-%d %H:%M:%S %z');record['status']='completed' if code==0 else 'failed';save(state)
 if code:raise RuntimeError(f'{name} failed with exit {code}; inspect {log}')
 return state
def training_log_summary():
 rows=[];bad=[];pat=re.compile(r'Epoch: \[([^\]]+)\] \[\s*(\d+)/(\d+)\].*?learning_rate:\s*([0-9.eE+-]+) loss:\s*([0-9.eE+-]+).*?max_mem_reserved:\s*(\d+) MB, max_mem_allocated:\s*(\d+) MB')
 for p in sorted((ROOT/'logs').glob('[ABC]_phase*.log')):
  text=p.read_text(encoding='utf-8',errors='replace')
  for line in text.splitlines():
   m=pat.search(line)
   if m:rows.append({'log':p.name,'epoch':m.group(1),'step':int(m.group(2)),'steps':int(m.group(3)),'lr':float(m.group(4)),'loss':float(m.group(5)),'max_mem_reserved_mb':int(m.group(6)),'max_mem_allocated_mb':int(m.group(7))})
  for ln,line in enumerate(text.splitlines(),1):
   if re.search(r'\b(nan|inf)\b',line,re.I):bad.append({'log':p.name,'line':ln,'text':line})
 (ROOT/'reports'/'training_log_samples.json').write_text(json.dumps({'records_at_log_interval':len(rows),'samples':rows,'nan_or_inf_lines':bad,'interpretation':'Logs contain periodic training samples, not per-image or full-epoch averaged loss. GPU memory fields are PaddleDetection max observed values.'},ensure_ascii=False,indent=2),encoding='utf-8')
 return {'loss_samples':len(rows),'nan_inf_lines':len(bad),'max_reserved_mb':max((r['max_mem_reserved_mb'] for r in rows),default=None)}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--no-wait',action='store_true');a=ap.parse_args()
 state=json.loads(STATUS.read_text(encoding='utf-8')) if STATUS.exists() else {'stages':{}}
 while not a.no_wait:
  if not TRAIN.exists():time.sleep(30);continue
  train=json.loads(TRAIN.read_text(encoding='utf-8'))
  if train.get('status')=='training_completed':break
  # status.json 会保留早期已恢复事件；仅当当前阶段本身失败且没有活跃阶段时才停止等待。
  active=any(v.get('status')=='running' for v in train.get('stages',{}).values())
  current=train.get('current_stage');current_record=train.get('stages',{}).get(current,{}) if current else {}
  if train.get('status')=='failed' and not active and current_record.get('status')!='running':
   raise RuntimeError('formal detector training failed: '+str(train.get('last_error')))
  time.sleep(30)
 if a.no_wait:
  train=json.loads(TRAIN.read_text(encoding='utf-8'))
  if train.get('status')!='training_completed':raise RuntimeError('training is not completed')
 # 早期恢复错误独立保存在training_incidents.json；最终运行状态清除已恢复的失败标记。
 train=json.loads(TRAIN.read_text(encoding='utf-8'))
 if train.get('status')=='training_completed':
  train.pop('last_error',None);train.pop('traceback',None);TRAIN.write_text(json.dumps(train,ensure_ascii=False,indent=2),encoding='utf-8')
 state.update({'status':'running','awaited_training_status':str(TRAIN),'stages':state.get('stages',{})});save(state)
 state=stage(state,'final_preflight_audit',PADDLE,ROOT/'scripts'/'preflight_audit.py',[],'paddle',PD)
 state=stage(state,'environment_and_dependency_record',TORCH,ROOT/'scripts'/'record_environment.py',[],'torch',ROOT)
 state=stage(state,'official_checkpoint_ap',PADDLE,ROOT/'scripts'/'evaluate_checkpoints.py',[],'paddle',PD)
 for g in 'ABC':state=stage(state,f'val_predict_{g}',PADDLE,ROOT/'scripts'/'predict_val_candidates.py',['--group',g],'paddle',PD)
 state=stage(state,'fixed_validation_end_to_end',TORCH,ROOT/'scripts'/'compare_end_to_end.py',[],'torch',ROOT)
 state=stage(state,'train_hard_sample_review',PADDLE,ROOT/'scripts'/'generate_hard_training_review.py',[],'paddle',PD)
 state['training_log_summary']=training_log_summary();state.update({'status':'completed','current_stage':None,'finished_at':time.strftime('%Y-%m-%d %H:%M:%S %z'),'D_resampling':'not run; waiting for user confirmation of train-only review candidates'});save(state)
 print(json.dumps({'status':state['status'],'summary':state['training_log_summary'],'report':str(STATUS)},ensure_ascii=False),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  state=json.loads(STATUS.read_text(encoding='utf-8')) if STATUS.exists() else {'stages':{}}
  state.update({'status':'failed','current_error':str(e),'traceback':traceback.format_exc()});save(state);raise
