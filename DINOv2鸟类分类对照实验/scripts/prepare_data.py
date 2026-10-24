"""复制小型固定清单并审计DINOv2可用训练/验证裁剪；不复制或修改源图/XML。"""
from __future__ import annotations
import csv, hashlib, json, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / 'configs' / 'experiment.json').read_text(encoding='utf-8'))
ARCHIVE = Path(CFG['gen2_archive'])
SOURCE_MANIFEST = ARCHIVE / 'reports' / 'classification_crop_manifest.csv'
SOURCE_MAP = ARCHIVE / 'configs' / 'class_mapping.json'
EXCLUSIONS = ARCHIVE / 'reports' / 'training_exclusions_pending_review.csv'
CONFIRMATIONS = ARCHIVE / 'reports' / 'manual_review_confirmation_log.csv'
ANNOTATION_VERSION = ARCHIVE / 'reports' / 'annotation_version.json'
TRAIN_ONLY_PENDING = {'jiamaque_087.jpg'}  # 用户要求等待补标/排除决定，不能给新实验引入未确认实例

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def main() -> None:
    out = ROOT / 'data'
    out.mkdir(parents=True, exist_ok=True)
    mapping = json.loads(SOURCE_MAP.read_text(encoding='utf-8-sig'))
    classes = mapping['classes']
    names = [c['name'] for c in classes]
    assert names == ['树麻雀','家麻雀','山麻雀','白头鹎','红耳鹎','黑喉红臀鹎','大山雀','煤山雀'], names
    with SOURCE_MANIFEST.open(encoding='utf-8-sig', newline='') as f:
        all_rows = list(csv.DictReader(f))
    with EXCLUSIONS.open(encoding='utf-8-sig', newline='') as f:
        exclusion_rows = list(csv.DictReader(f))
    source_exclusions = {r['filename'] for r in exclusion_rows}
    ann_version = json.loads(ANNOTATION_VERSION.read_text(encoding='utf-8-sig'))
    permanent = set(ann_version.get('permanently_excluded', []))
    excluded = source_exclusions | permanent | TRAIN_ONLY_PENDING

    train = [r for r in all_rows if r['split'] == 'train' and r['use_for_classifier_train'].lower() == 'true'
             and r['source_image'] not in excluded]
    valid = [r for r in all_rows if r['split'] == 'val' and r['review_pending'].lower() != 'true'
             and r['source_image'] not in excluded]
    assert len(train) == 1586, f'train crop count unexpectedly changed: {len(train)}'
    assert len(valid) == 146, f'validation crop count unexpectedly changed: {len(valid)}'
    assert all(r['class_name'] in names and int(r['class_id']) == names.index(r['class_name']) for r in train + valid)
    missing = [r['crop_path'] for r in train + valid if not Path(r['crop_path']).is_file()]
    assert not missing, f'{len(missing)} source crop files missing; first={missing[:3]}'
    train_sources = {r['source_image'] for r in train}
    val_sources = {r['source_image'] for r in valid}
    overlap = train_sources & val_sources
    assert not overlap, f'origin images cross train/val: {sorted(overlap)[:10]}'
    assert not any(r['source_image'] in excluded for r in train + valid)

    for name, rows in [('train_crops.csv', train), ('val_crops.csv', valid)]:
        dst = out / name
        with dst.open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    shutil.copy2(SOURCE_MAP, out / 'class_mapping.json')
    shutil.copy2(SOURCE_MANIFEST, out / 'source_classification_crop_manifest.csv')
    shutil.copy2(EXCLUSIONS, out / 'source_training_exclusions_pending_review.csv')
    shutil.copy2(ANNOTATION_VERSION, out / 'source_annotation_version.json')
    if CONFIRMATIONS.exists(): shutil.copy2(CONFIRMATIONS, out / 'source_manual_review_confirmation_log.csv')
    audit = {
        'status': 'passed', 'source_manifest': str(SOURCE_MANIFEST), 'source_manifest_sha256': sha(SOURCE_MANIFEST),
        'class_mapping_source': str(SOURCE_MAP), 'class_mapping_sha256': sha(SOURCE_MAP), 'class_names_order': names,
        'source_train_rows_before_extra_pending_exclusion': sum(r['split']=='train' and r['use_for_classifier_train'].lower()=='true' for r in all_rows),
        'used_train_crops': len(train), 'used_validation_crops': len(valid),
        'train_source_images': len(train_sources), 'validation_source_images': len(val_sources),
        'review_pending_validation_crops_excluded': sum(r['split']=='val' and r['review_pending'].lower()=='true' for r in all_rows),
        'excluded_training_images': sorted(excluded),
        'jiamaque_087_rows_excluded': sum(r['split']=='train' and r['source_image']=='jiamaque_087.jpg' for r in all_rows),
        'cross_split_source_images': sorted(overlap), 'missing_crop_files': len(missing),
        'brightness_derivatives_keep_manifest_source_split': True,
        'annotation_changes_made': False, 'teacher_test_used': False,
        'class_counts_train_crop_rows': {n: sum(r['class_name']==n for r in train) for n in names},
        'class_counts_validation_crop_rows': {n: sum(r['class_name']==n for r in valid) for n in names},
    }
    (ROOT/'reports'/'data_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(audit, ensure_ascii=False, indent=2))

if __name__ == '__main__': main()
