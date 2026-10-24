"""从官方来源下载PP-YOLOE+ s COCO权重和Paddle ResNet18 ImageNet权重到E盘。"""
import hashlib,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'weights';OUT.mkdir(exist_ok=True)
SOURCES={
 'ppyoloe_plus_crn_s_80e_coco.pdparams':'https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_s_80e_coco.pdparams',
 'resnet18.pdparams':'https://paddle-hapi.bj.bcebos.com/models/resnet18.pdparams'
}
records=[]
for name,url in SOURCES.items():
 dst=OUT/name
 if not dst.exists() or dst.stat().st_size==0:
  tmp=dst.with_suffix(dst.suffix+'.download')
  print('downloading',name,flush=True)
  urllib.request.urlretrieve(url,tmp);tmp.replace(dst)
 h=hashlib.sha256(dst.read_bytes()).hexdigest()
 records.append({'filename':name,'url':url,'path':str(dst),'bytes':dst.stat().st_size,'sha256':h})
meta={'weights':records,'note':'SHA-256是在下载完成后对本地文件实算；预训练来源使用官方链接。'}
(OUT/'sources_sha256.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(meta,ensure_ascii=False,indent=2))
