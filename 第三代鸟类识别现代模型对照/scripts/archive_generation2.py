"""Non-destructive, byte-verified generation 2 snapshot."""
from pathlib import Path
import hashlib, json, shutil, datetime

BASE = Path('E:/机器学习课设')
SRC = BASE / '第二代鸟类检测与细粒度分类'
OUT = BASE / '第二代结果归档' / datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
OUT.mkdir(parents=True, exist_ok=False)
records = []
def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()
def copy(p, rel):
    q = OUT / rel
    q.parent.mkdir(parents=True, exist_ok=True)
    before = digest(p)
    shutil.copy2(p, q)
    assert before == digest(q) == digest(p), str(p)
    records.append({'source': str(p), 'archive': str(rel), 'bytes': p.stat().st_size, 'sha256': before})
for folder in ['configs', 'scripts', 'reports']:
    for p in (SRC/folder).rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts: copy(p, Path('工程')/p.relative_to(SRC))
copy(SRC/'README.md', Path('工程/README.md'))
for p in (SRC/'data').rglob('*'):
    if p.is_file() and p.suffix.lower() in ['.json','.csv','.txt','.md','.jsonl','.yaml','.yml']:
        copy(p, Path('工程')/p.relative_to(SRC))
for run in ['classifier_resnet_v2','classifier_bilinear_v2','detector','two_stage_resnet_fixed_val','two_stage_bilinear_fixed_val']:
    for p in (SRC/'runs'/run).rglob('*'):
        if not p.is_file(): continue
        if p.suffix.lower() in ['.json','.jsonl','.csv','.txt','.log','.yaml','.yml','.md'] or p.name in ['best_model.pdparams','best_model_ap50.pdparams','model_final.pdparams']:
            copy(p, Path('工程')/p.relative_to(SRC))
DIAG=BASE/'模型检查报告/第二代实验'
for p in DIAG.rglob('*'):
    if p.is_file() and '__pycache__' not in p.parts:
        copy(p, Path('第二代实验报告')/p.relative_to(DIAG))
baseline=DIAG/'最终与最佳分类权重对照/最终与最佳权重比较结果.json'
b=json.loads(baseline.read_text(encoding='utf-8'))
summary={'archived_at':datetime.datetime.now().isoformat(),'source_preserved':str(SRC),
    'formal_detector':'PP-YOLOE+ s, best AP50 checkpoint',
    'comparison_protocol':'172 fixed validation images; same IoU>=0.5 greedy detector-score dedup; detector score strictly >0.30; crop margin 0.15; class selection uses 146 eligible validation crops',
    'generation2_summary':b['summary'],
    'excluded':'shanmaque_090.jpg, dashanque_119.jpg, two missing-XML originals remain excluded; no label changes',
    'limitations':'Not an independent test score. Displayed-class disagreement is not proven model error. Raw data, environment and intermediate epoch snapshots referenced in original project, not duplicated.',
    'files':len(records),'bytes':sum(x['bytes'] for x in records)}
(OUT/'归档清单.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'冻结实验口径与结果.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'README.md').write_text('# 第二代结果冻结归档\n\n原工程保持原样。归档包括正式最佳/最终模型、代码、配置、划分与复核记录、逐图预测及实验报告。全部副本逐字节 SHA-256 核验。原图、裁剪图、环境及中间轮次权重仍由原工程提供。\n\n第三代以第二代最佳 Bilinear CNN（epoch 47）作为基准：固定172张验证图、177个最终显示框、检测分数严格大于0.30，类别与目录不一致13张，无显示框5张；TP159、FP18、FN45、F1约0.83465。13张为类别分歧，不能直接称为13张真实模型错误。未使用教师测试图。\n',encoding='utf-8')
(BASE/'第三代鸟类识别现代模型对照/reports/第二代归档位置.json').write_text(json.dumps({'archive':str(OUT),'summary':summary},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'archive':str(OUT),'files':len(records),'bytes':summary['bytes']},ensure_ascii=False))
