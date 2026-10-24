"""记录复现实验所用仓库提交、依赖版本与GPU环境。"""
from __future__ import annotations
import json,os,platform,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];P=json.loads((ROOT/'paths.json').read_text(encoding='utf-8'))
def run(args,cwd=None):
 try:return subprocess.run(args,cwd=cwd,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=40).stdout.strip()
 except Exception as e:return f'ERROR: {e}'
def main():
 pd=Path(P['paddledetection']);
 d={'recorded_at':time.strftime('%Y-%m-%d %H:%M:%S %z'),'platform':platform.platform(),'python_running_record_script':sys.version,
  'repositories':{'PaddleDetection':{'path':str(pd),'commit':run(['git','rev-parse','HEAD'],pd),'dirty_files':run(['git','status','--short'],pd)},
    'generation3_project':{'path':P['generation3'],'commit':run(['git','rev-parse','HEAD'],P['generation3'])}},
  'gpu':{'summary':run(['nvidia-smi'])},
  'environments':{}}
 for label,py in [('paddle',P['paddle_python']),('torch',str(Path(P['generation3'])/'.venv'/'Scripts'/'python.exe'))]:
  code="import sys,json;d={'python':sys.version};\ntry:\n import paddle;d.update(paddle=paddle.__version__,paddle_cuda=paddle.is_compiled_with_cuda())\nexcept Exception as e:d['paddle_error']=str(e)\ntry:\n import torch;d.update(torch=torch.__version__,torch_cuda=torch.version.cuda,torch_cuda_available=torch.cuda.is_available())\nexcept Exception as e:d['torch_error']=str(e)\nprint(json.dumps(d))"
  d['environments'][label]={'python_executable':py,'runtime':run([py,'-c',code]),'pip_freeze':run([py,'-m','pip','freeze'])}
 out=ROOT/'reports'/'environment_and_source_versions.json';out.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8');print(str(out))
if __name__=='__main__':main()
