"""记录用户请求的可恢复暂停状态，不改动任何权重或标注。"""
from pathlib import Path
import json,time

ROOT=Path(__file__).resolve().parents[1]
STATUS=ROOT/'reports'/'formal_training_status.json'
def main():
    s=json.loads(STATUS.read_text(encoding='utf-8'))
    key=s.get('current_stage')
    if not key or key not in s.get('stages',{}):raise RuntimeError('no active stage to pause')
    stage=s['stages'][key]
    group=stage['group'];run=ROOT/'runs'/group
    snaps=[]
    for w in run.glob('*.pdparams'):
        if w.stem.isdigit() and all(w.with_suffix(x).exists() for x in ('.pdopt','.pdema')):
            snaps.append((int(w.stem),w))
    if not snaps:raise RuntimeError('no complete model/optimizer/EMA checkpoint; refusing to mark resumable pause')
    _,w=max(snaps,key=lambda x:x[0])
    stage.update({'status':'paused','paused_at':time.strftime('%Y-%m-%d %H:%M:%S %z'),
                  'pause_reason':'user requested pause after next complete five-epoch checkpoint',
                  'resume_checkpoint':str(w),'resume_checkpoint_epoch':int(w.stem)+1})
    s.update({'status':'paused','current_stage':key,'paused_at':stage['paused_at'],
              'resume_checkpoint':str(w),'resume_checkpoint_epoch':int(w.stem)+1})
    STATUS.write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':'paused','stage':key,'checkpoint':str(w),'completed_epoch':int(w.stem)+1},ensure_ascii=False))
if __name__=='__main__':main()
