"""核对用户刚确认图片仍在原训练划分、亮度衍生图和分类裁剪中。"""
from __future__ import annotations
import csv,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'configs/paths.json').read_text(encoding='utf-8'))
GEN1=Path(CFG['first_generation'])
CONF=ROOT/'reports/manual_review_confirmation_log.csv'
OUT=ROOT/'reports/user_confirmation_split_check.json'

def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
    confirmed=[r for r in read_csv(CONF) if r['status']=='current_labels_confirmed']
    split={r['source_image']:r['split'] for r in read_csv(Path(CFG['first_generation_split_manifest']))}
    bright={r['original']:r for r in read_csv(Path(CFG['first_generation_brightness_manifest']))}
    crops=read_csv(ROOT/'reports/classification_crop_manifest.csv')
    crop_sources={(r['source_image'],r['source_variant'],r['split']) for r in crops}
    train_dir=GEN1/'data/prepared/images/train';val_dir=GEN1/'data/prepared/images/val'
    train_names={p.name.lower() for p in train_dir.iterdir() if p.is_file()}
    val_names={p.name.lower() for p in val_dir.iterdir() if p.is_file()}
    details=[]
    for row in confirmed:
        name=row['filename'];stem=Path(name).stem;brightness_name=f'{stem}_brightness.jpg'
        rec=bright.get(name,{})
        train_has_original=name.lower() in train_names
        train_has_brightness=brightness_name.lower() in train_names
        val_has_original=name.lower() in val_names
        source_original_crop=(name,'original','train') in crop_sources
        source_brightness_crop=(name,'brightness','train') in crop_sources and rec.get('included_in_training')=='True'
        details.append({**row,'manifest_split':split.get(name),'train_original_present':train_has_original,
          'brightness_included_by_source_manifest':rec.get('included_in_training')=='True',
          'train_brightness_present':train_has_brightness,'validation_original_present':val_has_original,
          'classifier_original_crop_present':source_original_crop,'classifier_brightness_crop_present':source_brightness_crop,
          'verified':split.get(name)=='train' and train_has_original and not val_has_original and source_original_crop})
    val_manifest={r['source_image'] for r in read_csv(Path(CFG['first_generation_split_manifest'])) if r['split']=='val'}
    val_prepared={p.name for p in val_dir.iterdir() if p.is_file()}
    doc={'confirmation_source':'manual_review_confirmation_log.csv; user confirmed current labels; raw XML unchanged',
      'confirmed_image_count':len(details),'fixed_validation_manifest_images':len(val_manifest),
      'fixed_validation_prepared_images':len(val_prepared),'validation_filename_set_unchanged':val_manifest==val_prepared,
      'all_confirmed_images_restored_to_original_train_split':all(x['verified'] for x in details),
      'images':details,'unreviewed_shanmaque_090_remains_excluded':
        any(r['filename']=='shanmaque_090.jpg' for r in read_csv(ROOT/'reports/training_exclusions_pending_review.csv')),
      'dashanque_119_remains_permanently_excluded':json.loads((ROOT/'reports/annotation_version.json').read_text(encoding='utf-8'))['permanently_excluded']==['dashanque_119.jpg']}
    OUT.write_text(json.dumps(doc,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(OUT),'confirmed':len(details),'all_train':doc['all_confirmed_images_restored_to_original_train_split'],
      'validation_unchanged':doc['validation_filename_set_unchanged'],'brightness_rows_train':sum(x['train_brightness_present'] for x in details)},ensure_ascii=False,indent=2))
    if not doc['all_confirmed_images_restored_to_original_train_split'] or not doc['validation_filename_set_unchanged']:
        raise SystemExit('确认图片或固定验证清单核对失败')

if __name__=='__main__':main()
