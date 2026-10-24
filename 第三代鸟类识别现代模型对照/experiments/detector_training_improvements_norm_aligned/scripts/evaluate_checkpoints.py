"""使用PaddleDetection官方COCO evaluator在同一172张验证集上评估各快照，并按AP50选权重。"""
from __future__ import annotations
import hashlib,json,re,shutil,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];P=json.loads((ROOT/'paths.json').read_text(encoding='utf-8'))
PY=Path(P['paddle_python']);PD=Path(P['paddledetection']);EVAL=PD/'tools'/'eval.py'
PAT=re.compile(r'Average Precision\s+\(AP\)\s+@\[ IoU=([0-9.]+):([0-9.]+) \| area=\s*all \| maxDets=\s*100 \]\s*=\s*([0-9.]+)')
PAT50=re.compile(r'Average Precision\s+\(AP\)\s+@\[ IoU=([0-9.]+)\s+\| area=\s*all \| maxDets=\s*100 \]\s*=\s*([0-9.]+)')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    report={'scope':'固定172张验证图；COCO官方bbox evaluator；单类bird；仅探索验证集，不是独立测试','groups':{}}
    env=__import__('os').environ.copy();env.update({'PADDLE_HOME':str(ROOT/'.cache'/'paddle'),'TEMP':str(ROOT/'.cache'/'tmp'),'TMP':str(ROOT/'.cache'/'tmp')})
    lib=PY.parent.parent/'Lib'/'site-packages'/'nvidia';env['PATH']=str(lib/'cudnn'/'bin')+';'+str(lib/'cublas'/'bin')+';'+env.get('PATH','')
    for group in 'ABC':
        run=ROOT/'runs'/group;evaldir=run/'official_eval';evaldir.mkdir(parents=True,exist_ok=True)
        cfg=ROOT/'configs'/f'{group}_phase2.yml';snaps=[]
        for w in run.glob('*.pdparams'):
            if w.name=='model_final.pdparams':continue
            m=re.fullmatch(r'(\d+)\.pdparams',w.name)
            old=re.fullmatch(r'epoch_(\d+)\.pdparams',w.name)
            if m:snaps.append((int(m.group(1))+1,w))
            elif old:snaps.append((int(old.group(1)),w))
        if not snaps and (run/'model_final.pdparams').exists():snaps=[(40,run/'model_final.pdparams')]
        snaps=sorted({ep:w for ep,w in snaps}.items())
        records=[]
        for ep,w in snaps:
            out=evaldir/f'epoch_{ep:02d}';out.mkdir(exist_ok=True)
            log=evaldir/f'epoch_{ep:02d}.log'
            cmd=[str(PY),str(EVAL),'-c',str(cfg),'--classwise','--output_eval',str(out),'-o',f'weights={w}']
            t=time.perf_counter()
            with log.open('w',encoding='utf-8') as f:
                p=subprocess.run(cmd,cwd=out,env=env,stdout=f,stderr=subprocess.STDOUT,text=True)
            sec=time.perf_counter()-t;txt=log.read_text(encoding='utf-8',errors='replace')
            vals={}
            for ioulo,iouhi,v in PAT.findall(txt):vals[f'AP_{ioulo}_{iouhi}']=float(v)
            # COCO格式把AP@0.50打印成单IoU行，不带“:0.50”，需独立解析。
            for iou,v in PAT50.findall(txt):
                if iou=='0.50':vals['AP_0.50_0.50']=float(v)
            r={'epoch':ep,'checkpoint':str(w),'checkpoint_sha256':sha(w),'exit_code':p.returncode,'seconds':sec,'metrics':vals,'log':str(log)}
            records.append(r)
            if p.returncode or 'AP_0.50_0.95' not in vals or 'AP_0.50_0.50' not in vals:
                (ROOT/'reports'/f'{group}_eval_error.txt').write_text(txt[-12000:],encoding='utf-8')
                raise RuntimeError(f'official eval failed or AP metrics missing: {group} epoch{ep}, inspect {log}')
        if not records:raise RuntimeError(f'No checkpoints found for group {group}')
        # 单类 COCO AP50 最大者为best；并列时选较早轮，降低无收益时的额外训练偏好。
        best=max(records,key=lambda r:(r['metrics']['AP_0.50_0.50'],-r['epoch']))
        bestfile=run/'best_model_ap50.pdparams';shutil.copy2(best['checkpoint'],bestfile)
        report['groups'][group]={'checkpoints':records,'best_epoch':best['epoch'],'best_ap50':best['metrics']['AP_0.50_0.50'],
          'best_ap50_95':best['metrics']['AP_0.50_0.95'],'best_weight':str(bestfile),'best_weight_sha256':sha(bestfile),
          'final_weight':str(run/'model_final.pdparams'),'selection':'maximum official COCO AP50; ties choose earliest epoch'}
        (ROOT/'reports'/f'{group}_official_eval.json').write_text(json.dumps(report['groups'][group],ensure_ascii=False,indent=2),encoding='utf-8')
    report['generated_at']=time.strftime('%Y-%m-%d %H:%M:%S %z')
    (ROOT/'reports'/'official_checkpoint_evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({g:{'best_epoch':d['best_epoch'],'AP50':d['best_ap50'],'AP50_95':d['best_ap50_95']} for g,d in report['groups'].items()},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
