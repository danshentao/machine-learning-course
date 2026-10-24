"""从工作区现有输出汇总可核验的第二代运行记录，不生成课程提交报告。"""
from __future__ import annotations
import csv, hashlib, json, datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT.parent
OUT=WORK/'模型检查报告'/'第二代实验'
CFG=json.loads((ROOT/'configs/paths.json').read_text(encoding='utf-8'))

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    logs=[ROOT/'runs/detector/train.log',ROOT/'runs/detector/train.stdout.log',ROOT/'runs/detector/train.stderr.log',
          ROOT/'runs/eval_smoke/eval.log',ROOT/'runs/data_prepare.log',ROOT/'runs/data_prepare_after_review.log',
          ROOT/'runs/classifier_resnet_smoke/training.jsonl',ROOT/'runs/classifier_bilinear_smoke/training.jsonl']
    logs += [ROOT/'runs/classifier_resnet_v2/training.jsonl',ROOT/'runs/classifier_bilinear_v2/training.jsonl']
    logs += [ROOT/'runs/two_stage_resnet_fixed_val/eval/metrics.json',ROOT/'runs/two_stage_bilinear_fixed_val/eval/metrics.json']
    artifacts=[ROOT/'runs/smoke_detector/model_final.pdparams',
      ROOT/'runs/classifier_resnet_smoke/model_final.pdparams',ROOT/'runs/classifier_bilinear_smoke/model_final.pdparams',
      ROOT/'weights/ppyoloe_plus_crn_s_80e_coco.pdparams',ROOT/'weights/resnet18.pdparams']
    artifacts += [ROOT/'runs/classifier_resnet_v2/model_final.pdparams',ROOT/'runs/classifier_resnet_v2/best_model.pdparams',
                  ROOT/'runs/classifier_bilinear_v2/model_final.pdparams',ROOT/'runs/classifier_bilinear_v2/best_model.pdparams',
                  ROOT/'runs/detector/model_final.pdparams',ROOT/'runs/detector/final_model.pdparams',ROOT/'runs/detector/best_model_ap50.pdparams',
                  ROOT/'runs/two_stage_resnet_fixed_val/predictions.json',ROOT/'runs/two_stage_bilinear_fixed_val/predictions.json']
    configs=[ROOT/'configs/ppyoloe_plus_bird_640.yml',ROOT/'configs/class_mapping.json',ROOT/'configs/paths.json',
             ROOT/'data/annotation_corrections_v2.json',ROOT/'reports/annotation_version.json',ROOT/'reports/manual_review_confirmation_log.csv',
             ROOT/'data/annotations/instances_train_bird.json',ROOT/'data/annotations/instances_val_bird.json',
             Path(CFG['first_generation_split_manifest'])]
    configs += [ROOT/'runs/detector/checkpoint_selection.json',ROOT/'runs/detector/posthoc_checkpoint_ap50.csv',
                ROOT/'runs/two_stage_resnet_fixed_val/eval/metrics_by_class.csv',ROOT/'runs/two_stage_bilinear_fixed_val/eval/metrics_by_class.csv',
                ROOT/'runs/two_stage_resnet_fixed_val/eval/confusion_matrix.csv',ROOT/'runs/two_stage_bilinear_fixed_val/eval/confusion_matrix.csv',
                ROOT/'reports/fixed_val_comparison_gallery.html',ROOT/'reports/user_confirmation_split_check.json',
                ROOT/'scripts/verify_user_confirmation_split.py',ROOT/'scripts/infer_two_stage.py',ROOT/'scripts/evaluate_two_stage.py',ROOT/'README.md']
    rows=[]
    for p in logs+artifacts+configs:
        if p.exists():
            kind='log' if p in logs else ('weight' if p in artifacts else 'config_or_data_manifest')
            rows.append({'path':str(p),'kind':kind,'bytes':p.stat().st_size,
                         'sha256':sha256(p),'modified_local':datetime.datetime.fromtimestamp(p.stat().st_mtime).astimezone().isoformat(timespec='seconds')})
    with (OUT/'run_artifact_inventory.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['path','kind','bytes','sha256','modified_local']);w.writeheader();w.writerows(rows)
    prep=json.loads((ROOT/'reports/data_preparation_summary.json').read_text(encoding='utf-8'))
    env=json.loads((WORK/'第一代结果归档'/'环境与代码版本'/'environment_record.json').read_text(encoding='utf-8'))
    commit=(WORK/'第一代结果归档'/'环境与代码版本'/'paddledetection_commit.txt').read_text(encoding='utf-8').strip()
    excluded=list(csv.DictReader((ROOT/'reports/training_exclusions_pending_review.csv').open(encoding='utf-8-sig',newline='')))
    review=list(csv.DictReader((ROOT/'reports/manual_review_instances.csv').open(encoding='utf-8-sig',newline='')))
    confirmed={r['filename'] for r in review if r.get('status','').startswith('人工已确认')}
    train_review=sorted(({r['filename'] for r in review if r.get('split')=='train'}|{r['filename'] for r in excluded})-confirmed)
    pred=json.loads((ROOT/'runs/two_stage_smoke/predictions.json').read_text(encoding='utf-8'))
    props=json.loads((ROOT/'runs/two_stage_smoke/proposals.json').read_text(encoding='utf-8'))
    bilinear=json.loads((ROOT/'runs/two_stage_smoke_bilinear/predictions.json').read_text(encoding='utf-8'))
    resnet_metrics=json.loads((ROOT/'runs/two_stage_resnet_fixed_val/eval/metrics.json').read_text(encoding='utf-8'))
    bilinear_metrics=json.loads((ROOT/'runs/two_stage_bilinear_fixed_val/eval/metrics.json').read_text(encoding='utf-8'))
    resnet_full=json.loads((ROOT/'runs/two_stage_resnet_fixed_val/predictions.json').read_text(encoding='utf-8'))
    bilinear_full=json.loads((ROOT/'runs/two_stage_bilinear_fixed_val/predictions.json').read_text(encoding='utf-8'))
    detector_selection=json.loads((ROOT/'runs/detector/checkpoint_selection.json').read_text(encoding='utf-8'))
    split_check=json.loads((ROOT/'reports/user_confirmation_split_check.json').read_text(encoding='utf-8'))
    clipped=sum(d.get('box_clipped',False) for im in pred['images'] for d in im['candidates'])
    run={
      'generated_from_actual_files':True,'formal_training_started':True,
      'formal_training_status':{'detector_complete':(ROOT/'runs/detector/model_final.pdparams').exists(),
        'resnet18_complete':(ROOT/'runs/classifier_resnet_v2/model_final.pdparams').exists(),
        'bilinear_complete':(ROOT/'runs/classifier_bilinear_v2/model_final.pdparams').exists(),
        'note':'完整训练、固定验证评估等状态按对应权重和指标文件是否存在记录。'},
      'reused_environment':{'python':env['python'],'platform':env['platform'],'paddle':env['paddle'],'numpy':env['numpy'],'opencv':env['opencv'],'pillow':env['pillow'],'gpu':env['nvidia_smi_hardware'],'paddledetection_commit':commit,'note':'沿用第一代E盘虚拟环境和源码，不覆盖第一代环境。'},
      'smoke_detector':{'config':'configs/ppyoloe_plus_bird_640_smoke.yml','architecture':'PaddleDetection 2.9 official PP-YOLOE+ s','epochs':1,'train_images':12,'batch_size':2,'input':640,'device':'GPU','result':'runs/smoke_detector/model_final.pdparams','max_reserved_memory_mb':941,'max_allocated_memory_mb':798},
      'formal_detector_config':{'path':'configs/ppyoloe_plus_bird_640.yml','model':'official PP-YOLOE+ s','epochs_requested':100,'batch_size':2,'input':640,'learning_rate':0.0003125,'optimizer':'official inherited PP-YOLOE+ optimizer','device':'GPU','seed':0,'seed_mechanism':'PaddleDetection tools/train.py --enable_ce=True calls set_random_seed(0)','snapshot_interval_epochs':5,'in_training_eval':False,'checkpoint_selection':'post-hoc official evaluation of every periodic checkpoint and final model on the fixed validation set'},
      'smoke_detector_official_eval':{'images':8,'AP50':0.211,'AP50_95':0.156,'FPS_logged':18.922796166088062,'caution':'8张冒烟验证子集，只检查数据、权重和官方评估流程；不能代表固定验证集或正式模型指标。'},
      'classifier_resnet_smoke':json.loads((ROOT/'runs/classifier_resnet_smoke/metadata.json').read_text(encoding='utf-8')),
      'classifier_bilinear_smoke':json.loads((ROOT/'runs/classifier_bilinear_smoke/metadata.json').read_text(encoding='utf-8')),
      'formal_classifier_runs':{name:json.loads((ROOT/'runs'/name/'metadata.json').read_text(encoding='utf-8'))
        for name in ('classifier_resnet_v2','classifier_bilinear_v2') if (ROOT/'runs'/name/'metadata.json').exists()},
      'formal_results':{'detector_best_checkpoint_epoch':detector_selection['selected_epoch'],'detector_COCO_AP50':detector_selection['selected_AP50'],
        'detector_checkpoint_selection':'official PaddleDetection evaluation of every 5-epoch snapshot and final checkpoint on fixed validation set',
        'resnet18_end_to_end':resnet_metrics,'bilinear_end_to_end':bilinear_metrics,
        'resnet18_end_to_end_seconds':resnet_full['metadata']['elapsed_seconds'],'bilinear_end_to_end_seconds':bilinear_full['metadata']['elapsed_seconds'],
        'resnet18_candidate_count':sum(len(x['candidates']) for x in resnet_full['images']),'bilinear_candidate_count':sum(len(x['candidates']) for x in bilinear_full['images']),
        'comparison_note':'Both classifiers use the identical top-100-per-image detector proposal file; P/R/F1 use detector score strictly >0.5 and IoU>=0.5 one-to-one class-correct matching. AP uses confidence ranking with detector_score*classifier_top_score, 101-point interpolation.'},
      'user_review_split_check':split_check,
      'two_stage_arbitrary_directory_smoke':{'images':len(pred['images']),'detector_proposals_per_image':[len(x['detections']) for x in props['images']], 'classified_candidates':sum(len(x['candidates']) for x in pred['images']),'clipped_candidates':clipped,'invalid_boxes_after_clipping':sum(not d.get('valid_after_clipping',True) for im in props['images'] for d in im['detections']),'display_threshold_strictly_greater_than':0.5,'displayed_detections':sum(len(x['displayed']) for x in pred['images']),'elapsed_seconds':pred['metadata']['elapsed_seconds'],'note':'短跑权重候选质量很差；框已裁回原图边界后才裁剪分类，结果仅证明推理代码路径可运行。'},
      'same_proposals_comparison':{'resnet_and_bilinear_reused_same_detector_proposals':bilinear['metadata']['proposals_reused'],'proposal_file':'runs/two_stage_smoke/proposals.json','candidate_counts_match':sum(len(x['candidates']) for x in pred['images'])==sum(len(x['candidates']) for x in bilinear['images']),'note':'两个分类器消费同一候选框文件；分类器仅为冒烟权重。'},
      'data_preparation':prep,'excluded_train_images_pending_review':excluded,
      'train_review_image_names':train_review,
      'formal_training_gate':'已收到用户对7张训练图 current_labels_confirmed 的确认；shanmaque_090与dashanque_119仍排除。正式训练已完成。',
      'weight_and_log_hashes':rows
    }
    (OUT/'run_inventory.json').write_text(json.dumps(run,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'gen1_gen2_comparison_status.csv').write_text(
      '方案,固定验证集图片,划分与真值,检测mAP50,端到端八类mAP50,端到端micro_P,端到端micro_R,端到端micro_F1,状态\n'
      f"第一代PP-YOLOE-s,172,Gen1当前XML v1,0.8614556,,0.4564103,0.8725490,0.5993266,已运行；P/R/F1阈值=0.25；非同阈值直接比较\n"
      f"第二代ResNet18分类器,172,固定验证集v1标签；用户确认仅作用于训练图,,{resnet_metrics['mAP50_macro_101point']:.7f},{resnet_metrics['micro_precision_at_detector_threshold']:.7f},{resnet_metrics['micro_recall_at_detector_threshold']:.7f},{resnet_metrics['micro_F1']:.7f},正式已运行；检测器第{detector_selection['selected_epoch']}轮；det>0.5\n"
      f"第二代Bilinear CNN分类器,172,固定验证集v1标签；用户确认仅作用于训练图,,{bilinear_metrics['mAP50_macro_101point']:.7f},{bilinear_metrics['micro_precision_at_detector_threshold']:.7f},{bilinear_metrics['micro_recall_at_detector_threshold']:.7f},{bilinear_metrics['micro_F1']:.7f},正式已运行；同候选同检测器；det>0.5\n",encoding='utf-8-sig')
    md='''# 第二代实验运行记录与复核入口

本目录用于保存可核验运行状态，不是可直接提交的课程报告。所有指标和时间只取自工程中的实际日志或结果文件。

## 当前进度

- 已运行：PP-YOLOE+ s 单类检测器完成100轮GPU训练；训练后对每5轮快照及最终权重做官方固定验证集AP评估，按AP@0.5选第25轮（0.876）。ResNet18与Bilinear CNN各完成60轮/12000步GPU训练，并完成同一固定172图验证集的整图两阶段评估。
- 已核对：两种分类器实际读取了相同的鸟检测候选框文件；本次短跑检测器在两张演示图上生成268和230个低阈值候选，但严格检测分数>0.5的显示框为0，说明推理链路运行成功，不代表模型有效。短跑预测有333个框越出图边，现已在图像坐标下裁回边界后再裁鸟图；两类分类器比较仍共享完全相同的498个有效候选。
- 端到端结果：ResNet18 mAP@0.5=0.8107799、micro P/R/F1=0.8810/0.7255/0.7957；Bilinear CNN=0.7902665、0.8690/0.7157/0.7849。双线性结果略低，当前实验没有显示改善。
- 两种分类器共用第25轮检测器和同一批最高分100候选/图，分类输入224；P/R/F1检测分数严格>0.5、类别正确且IoU≥0.5一对一匹配。AP使用检测分×分类最高分排序及101点插值；教师29张测试图未读取、未参与调参。
- 已记录用户对7张训练图片原标签正确的确认；`shanmaque_090.jpg`仍隔离，`dashanque_119.jpg`仍排除。此次只记录确认，没有改动原XML。

## 已实际运行的短跑结果

PP-YOLOE+ s 使用官方COCO预训练权重。1轮训练只取12张训练图、batch=2、640输入，官方日志记录峰值保留显存941 MB、已分配798 MB。官方8张冒烟验证的AP50=0.211、AP50:0.95=0.156、记录FPS约18.92。这个子集和训练量很小，只证明配置、前向、损失、保存和评估工具可以执行；不能当固定验证指标。

ResNet18 分类器短跑为1轮配置、实际2步、CPU、batch=2、224输入，耗时1.478秒；Bilinear CNN短跑实际1步、CPU、batch=1、224输入，耗时0.455秒。二者使用ImageNet预训练ResNet18，Bilinear版本共享特征图并做空间位置外积均值、signed-square-root、L2归一化和512²到8类线性头。这些是管线检查，不是可比较精度结果。

## 准备数据的实际数量

沿用第一代固定拆分，训练784张原图、固定验证172张原图；鸟检测训练实例1594（训练来源含合格brightness衍生图），验证204个实例。分类训练裁剪1594行，人工框验证诊断裁剪146行。亮度衍生图不是独立人工标注图片。数据来源和实例统计见工程 `reports/data_preparation_summary.json`。

## 已确认及仍隔离的训练标签

以下7张训练图已由用户确认现有标签和框正确，并记录于 `manual_review_confirmation_log.csv`；`shanmaque_090.jpg`仍未检查并隔离，`dashanque_119.jpg`继续永久排除：

| 图片 | 当前问题/需要确认 |
|---|---|
| `dashanque_095.jpg` | 用户已确认原标注大山雀正确；119继续排除。确认日志已记录，未改XML。 |
| `dashanque_106.jpg` | 用户已确认当前标签/框正确。 |
| `heihouhongtunbei_003.jpg`、`heihouhongtunbei_004.jpg`、`heihouhongtunbei_086.jpg` | 用户已确认原标注正确；004依据可见黑喉和红臀。 |
| `shanmaque_028.jpg` | 用户已确认当前标签/框正确。`shanmaque_090.jpg`仍未检查并隔离。 |
| `shumaque_112.jpg` | 用户已确认当前标签/框正确。 |

可用第一代全量逐图HTML与跨类别重复审查图册查看原图、XML框和模型框。完整逐实例坐标与分歧记录在工程 `reports/manual_review_instances.csv`；缺XML的 `meishanque_012.jpg`、`meishanque_101.jpg` 均继续排除，不能作为负样本。

此外，固定验证集有多张第一代“与现有XML不一致”图片。若发现真值需修正，应先更新新版标注，再在同一新版真值上重算第一代固定验证指标和第二代指标，避免版本不一致的比较。详情见 `manual_review_queue.csv`；目前这类条目是待复核分歧，不能称为确认的模型错误。

## 人工确认后的录入格式

后续如需记录新意见，只编辑工程 `data/annotation_corrections_v2.json`，原XML保持只读。每张图填写复核人、确认说明和状态；状态为 `current_labels_confirmed` 或 `corrected`。更正按XML一基实例序号写入，标准类名严格采用固定八类；框按原XML端点坐标，不自行加减像素。可添加实例、修改类别/框或删除已确认误标实例。完成后运行：

```powershell
cd E:\机器学习课设\第二代鸟类检测与细粒度分类
.\scripts\prepare_data.ps1
```

脚本会生成新标注版本记录和修改前后日志，并保持既有图片划分。未确认的疑似训练图会继续隔离；正式训练已完成，新增标签更正后需另建模型版本再训练。

## 文件

- `run_inventory.json`：短跑参数、结果路径、权重/日志SHA-256、数据数量和训练门槛。
- `run_artifact_inventory.csv`：实际关键权重和日志的路径、大小、SHA-256、修改时间。
- `gen1_gen2_comparison_status.csv`：第一代与两种第二代方案的固定验证结果；P/R/F1阈值不同，直接对比时须注明。
- `第二代鸟类检测与细粒度分类/reports/smoke_comparison_gallery.html`：示例HTML，展示人工XML、第一代严格阈值最终框和第二代两种分类器的框/八类得分；当前内容明确标记为短跑演示，不是正式验证比较。
- `第二代鸟类检测与细粒度分类/reports/manual_review_queue.csv`：需复核图片汇总。
- `第二代鸟类检测与细粒度分类/reports/manual_review_instances.csv`：逐实例候选分歧，不是错误真值。
'''
    (OUT/'运行记录与复核入口.md').write_text(md,encoding='utf-8')
    print(f'已写入 {OUT}; hashed artifacts={len(rows)}; train review images={len(train_review)}')

if __name__=='__main__':main()
