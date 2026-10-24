"""监视正式训练；只对已保存检查点后的Windows原生0xC0000005崩溃自动尝试有限恢复。"""
from __future__ import annotations
import json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];STATUS=ROOT/'reports'/'formal_training_status.json';LOG=ROOT/'reports'/'training_recovery_attempts.jsonl'
PS=str(ROOT/'start_training.ps1')
def append(d):
 LOG.parent.mkdir(parents=True,exist_ok=True)
 with LOG.open('a',encoding='utf-8') as f:f.write(json.dumps(d,ensure_ascii=False)+'\n')
def latest(group):
 a=[]
 for p in (ROOT/'runs'/group).glob('*.pdparams'):
  if p.stem.isdigit() and p.with_suffix('.pdopt').exists() and p.with_suffix('.pdema').exists():a.append((int(p.stem),p))
 return str(max(a,key=lambda x:x[0])[1]) if a else None
def main():
 attempts={}
 while True:
  if not STATUS.exists():time.sleep(15);continue
  s=json.loads(STATUS.read_text(encoding='utf-8'));state=s.get('status')
  if state=='training_completed':append({'time':time.strftime('%Y-%m-%d %H:%M:%S %z'),'event':'training_completed','status_file':str(STATUS)});return
  if state=='failed':
   stage=s.get('stages',{}).get(s.get('current_stage'),{});code=stage.get('exit_code');group=stage.get('group');ckpt=latest(group) if group else None
   if code!=3221225501 or not ckpt:
    append({'time':time.strftime('%Y-%m-%d %H:%M:%S %z'),'event':'not_auto_recovered','stage':s.get('current_stage'),'exit_code':code,'latest_checkpoint':ckpt,'reason':'only the observed native exception code with a complete checkpoint is retried'});return
   key=(s.get('current_stage'),ckpt);attempts[key]=attempts.get(key,0)+1
   if attempts[key]>3:
    append({'time':time.strftime('%Y-%m-%d %H:%M:%S %z'),'event':'retry_limit_reached','stage':s.get('current_stage'),'latest_checkpoint':ckpt,'attempts':attempts[key]-1});return
   append({'time':time.strftime('%Y-%m-%d %H:%M:%S %z'),'event':'resume_started','stage':s.get('current_stage'),'exit_code':code,'checkpoint':ckpt,'attempt':attempts[key]})
   p=subprocess.run(['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',PS,'-Resume'],capture_output=True,text=True,timeout=30)
   append({'time':time.strftime('%Y-%m-%d %H:%M:%S %z'),'event':'resume_launcher_result','exit_code':p.returncode,'stdout':p.stdout.strip(),'stderr':p.stderr.strip()})
   if p.returncode:return
   time.sleep(20);continue
  time.sleep(15)
if __name__=='__main__':main()
