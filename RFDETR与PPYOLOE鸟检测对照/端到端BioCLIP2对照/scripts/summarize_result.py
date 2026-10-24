import json
from pathlib import Path
d=json.loads(Path(r'E:\机器学习课设\RFDETR与PPYOLOE鸟检测对照\端到端BioCLIP2对照\reports\end_to_end_results.json').read_text(encoding='utf-8'))
print('PERCLASS')
for i,name in enumerate(d['metadata']['classes']):
 a=d['fixed_validation']['per_class']['ppyoloe'][i];b=d['fixed_validation']['per_class']['rfdetr'][i];print(name,f"base TP{a['TP']} FP{a['FP']} FN{a['FN']} F1={a['F1']:.3f}",f"RF TP{b['TP']} FP{b['FP']} FN{b['FN']} F1={b['F1']:.3f}")
for r in d['fixed_validation']['rows']:
 ch=r['end_to_end_change']
 if ch['regressed_instances'] or ch['new_class_disagreements'] or ch['rfdetr_spatial_class_errors']:print('DETAIL',r['filename'],json.dumps(ch,ensure_ascii=False))
print('CONFIRMED')
for r in d['external80']['rows']:
 for x in r['models']['rfdetr']['display_candidates']:
  status=x.get('user_detection_review',{}).get('status')
  if status=='确实是鸟':print(r['filename'],x['candidate_index'],x['class_name'],f"det={x['detector_score']:.4f}",f"cls={x['classifier_top_score']:.4f}",json.dumps(x['classifier_scores'],ensure_ascii=False))
print('PENDING')
for r in d['external80']['rows']:
 for x in r['models']['rfdetr']['display_candidates']:
  if x.get('user_detection_review',{}).get('status')=='暂不能确定':print(r['filename'],x['candidate_index'],x['class_name'],round(x['detector_score'],4),round(x['classifier_top_score'],4))
print('EXT')
for scope,x in d['external80']['scopes'].items():print(scope,x['images'],[(m,y['images_with_any_box'],y['displayed_boxes'],len(y['no_box_images']),y['box_class_equal_to_directory_marker_descriptive_only'],y['boxes_in_scope_with_directory_marker']) for m,y in x['models'].items()])
print('TIMING',json.dumps(d['metadata']['classifier_timing'],ensure_ascii=False))
print('REPLAY',[(k,v['matched_candidate_pairs'],v['per_image_count_mismatch_count'],v['top_class_changes'],round(v['max_coordinate_delta_px'],4),round(v['max_classifier_score_delta'],6)) for k,v in d['metadata']['pp_formal_replay_audit'].items()])
