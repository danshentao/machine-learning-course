import json
from pathlib import Path
p=Path(r'E:\机器学习课设\RFDETR与PPYOLOE鸟检测对照\端到端BioCLIP2对照\reports\end_to_end_results.json');d=json.loads(p.read_text(encoding='utf-8'))
print('identity',d['metadata']['image_identity_audit'])
print('RF weight',d['metadata']['source_files']['fixed_detection_comparison'],d['metadata']['source_files']['external_detection_comparison'])
print('bio hashes',d['metadata']['encoder_weight_sha256'],d['metadata']['classifier_head_sha256'])
print('multi bird misses')
for r in d['fixed_validation']['rows']:
 if '多鸟漏检' in r['status_labels']:
  print(r['filename'],'GT',len(r['truth']),'RF_FN',r['per_model_end_to_end_metrics']['rfdetr']['FN_end_to_end'],'unmatched_truth',r['per_model_end_to_end_metrics']['rfdetr']['unmatched_truth_instances'])
print('wrong class remaining')
for r in d['fixed_validation']['rows']:
 if r['end_to_end_change']['rfdetr_spatial_class_errors']:print(r['filename'],r['end_to_end_change']['rfdetr_spatial_class_errors'])
print('timing',d['metadata']['classifier_timing'])
print('report sizes')
