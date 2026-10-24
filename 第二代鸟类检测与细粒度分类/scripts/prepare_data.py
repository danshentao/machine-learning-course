"""沿用第一代固定划分，生成单类鸟COCO标签、逐实例分类裁剪及复核队列。"""
from __future__ import annotations
import argparse, csv, json, random, shutil
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'configs/paths.json').read_text(encoding='utf-8'))
MAP=json.loads((ROOT/'configs/class_mapping.json').read_text(encoding='utf-8'))
GEN1=Path(CFG['first_generation']); SOURCE=Path(CFG['source_images'])
PREP=Path(CFG['first_generation_prepared_data']); DATA=ROOT/'data'; REPORTS=ROOT/'reports'
ID={x['name']:x['id'] for x in MAP['classes']}; LABEL_MAP=MAP['xml_label_to_standard']
PERMANENTLY_EXCLUDED={'dashanque_119.jpg'}
CORRECTIONS_PATH=DATA/'annotation_corrections_v2.json'

def load_review_confirmations(doc):
    """只有填写复核人和说明的逐图确认，才可解除复核隔离。"""
    confirmed={}
    for row in doc.get('review_confirmations',[]):
        name=row.get('filename');status=row.get('status')
        if not name or status not in ('current_labels_confirmed','corrected'):
            raise ValueError(f'复核确认需filename及有效status: {row}')
        if not row.get('confirmed_by') or not row.get('note'):
            raise ValueError(f'复核确认缺少confirmed_by/note: {row}')
        if name in confirmed:raise ValueError(f'重复的复核确认: {name}')
        confirmed[name]=row
    return confirmed

def apply_human_corrections(filename, objects, doc, split):
    """只执行由用户明确确认且附原因的逐实例修改；原XML只读保留。"""
    changes=[r for r in doc.get('changes',[]) if r.get('filename')==filename]
    by_instance={int(o['instance']):o for o in objects};log=[]
    for change in changes:
        if not change.get('confirmed_by') or not change.get('reason'):
            raise ValueError(f'人工更正缺少confirmed_by/reason: {filename} {change}')
        action=change.get('action');iid=int(change.get('instance_id',0))
        if action=='add':
            if iid in by_instance or iid<1:raise ValueError(f'新增实例编号冲突: {filename} #{iid}')
            name=change.get('new_standard_class');box=change.get('new_bbox_xyxy')
            if name not in ID or not box or len(box)!=4:raise ValueError(f'新增实例需标准类别和xyxy框: {filename} #{iid}')
            obj={'instance':iid,'raw_label':'人工补标','name':name,'class_id':ID[name],'xyxy':[float(v) for v in box]};by_instance[iid]=obj
            log.append({'filename':filename,'split':split,'instance_id':iid,'action':action,'old_label':'','new_label':name,'old_bbox_xyxy':'','new_bbox_xyxy':obj['xyxy'],'reason':change['reason'],'confirmed_by':change['confirmed_by']})
        elif action in ('update','delete'):
            if iid not in by_instance:raise ValueError(f'更正实例编号不存在: {filename} #{iid}')
            obj=by_instance[iid];old_name=obj['name'];old_box=list(obj['xyxy'])
            if action=='update':
                name=change.get('new_standard_class',old_name);box=change.get('new_bbox_xyxy',old_box)
                if name not in ID or len(box)!=4:raise ValueError(f'更正类别或框格式非法: {filename} #{iid}')
                box=[float(v) for v in box]
                if box[2]<=box[0] or box[3]<=box[1]:raise ValueError(f'更正框退化: {filename} #{iid}')
                obj.update({'name':name,'class_id':ID[name],'xyxy':box})
                log.append({'filename':filename,'split':split,'instance_id':iid,'action':action,'old_label':old_name,'new_label':name,'old_bbox_xyxy':old_box,'new_bbox_xyxy':box,'reason':change['reason'],'confirmed_by':change['confirmed_by']})
            else:
                del by_instance[iid]
                log.append({'filename':filename,'split':split,'instance_id':iid,'action':action,'old_label':old_name,'new_label':'删除已确认误标实例','old_bbox_xyxy':old_box,'new_bbox_xyxy':'','reason':change['reason'],'confirmed_by':change['confirmed_by']})
    return [by_instance[k] for k in sorted(by_instance)],log
SPLIT_CSV=Path(CFG['first_generation_split_manifest'])

def read_csv(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write_csv(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def parse(xml):
    root=ET.parse(xml).getroot();size=root.find('size')
    w=int(size.findtext('width'));h=int(size.findtext('height')); objects=[]
    for n,obj in enumerate(root.findall('object'),1):
        raw=(obj.findtext('name') or '').strip()
        if raw not in LABEL_MAP:raise ValueError(f'未知XML标签: {xml.name} instance={n} label={raw}')
        name=LABEL_MAP[raw];box=obj.find('bndbox')
        xy=[float(box.findtext(k)) for k in ('xmin','ymin','xmax','ymax')]
        if not xy[2]>xy[0] or not xy[3]>xy[1]:raise ValueError(f'退化框: {xml.name} #{n}: {xy}')
        if xy[0]<0 or xy[1]<0 or xy[2]>w or xy[3]>h:raise ValueError(f'越界框: {xml.name} #{n}: {xy} size={w}x{h}')
        objects.append({'instance':n,'raw_label':raw,'name':name,'class_id':ID[name],'xyxy':xy})
    return w,h,objects
def crop_with_margin(image_path, xyxy, margin=.15):
    with Image.open(image_path) as im: im=im.convert('RGB'); w,h=im.size
    x1,y1,x2,y2=xyxy; dx=(x2-x1)*margin;dy=(y2-y1)*margin
    # 不改变源XML端点约定；裁剪只向框外加背景并裁在图像边界内。
    box=(max(0,int(x1-dx)),max(0,int(y1-dy)),min(w,int(x2+dx)),min(h,int(y2+dy)))
    with Image.open(image_path) as im:return im.convert('RGB').crop(box),box
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--smoke',action='store_true',help='只生成少量无已知复核标记样本供短跑')
    ap.add_argument('--max-train-images',type=int,default=0)
    args=ap.parse_args(); DATA.mkdir(exist_ok=True);REPORTS.mkdir(exist_ok=True)
    split=read_csv(SPLIT_CSV); byname={r['source_image']:r for r in split}
    doc=json.loads(CORRECTIONS_PATH.read_text(encoding='utf-8')) if CORRECTIONS_PATH.exists() else {'version':'v2','review_confirmations':[],'changes':[]}
    confirmations=load_review_confirmations(doc)
    if doc.get('version')!='v2':raise ValueError('annotation_corrections_v2.json 的 version 必须为 v2')
    change_names={r.get('filename') for r in doc.get('changes',[])}
    unconfirmed_changes=change_names-set(confirmations)
    if unconfirmed_changes:raise ValueError(f'逐实例更正必须有对应逐图人工确认: {sorted(unconfirmed_changes)}')
    unknown_reviews=(set(confirmations)|change_names)-set(byname)
    if unknown_reviews:raise ValueError(f'复核文件名不在固定划分清单中，禁止静默忽略: {sorted(unknown_reviews)}')
    if PERMANENTLY_EXCLUDED & change_names:raise ValueError(f'永久排除图片不能通过标注更正重新加入: {sorted(PERMANENTLY_EXCLUDED & change_names)}')
    xml_dir=GEN1/'data/annotations_raw';
    bright=read_csv(Path(CFG['first_generation_brightness_manifest']))
    bright_ok={r['brightness'] for r in bright if r['included_in_training']=='True'}
    disagreement=read_csv(Path(CFG['first_generation_review_archive'])/'复算依据/逐图一致性复算.csv')
    queue={r['filename'] for r in disagreement if r['annotation_consistency_disagreement']=='True'}
    dupfile=Path(CFG['workspace'])/'模型检查报告/跨类别图片重复审查/pair_evidence.csv'
    duplicate_rows=read_csv(dupfile) if dupfile.exists() else []
    dup_instances=[]
    for r in duplicate_rows:
        if r.get('同鸟标签冲突','[]') not in ('[]',''): dup_instances.extend([r['文件A'],r['文件B']])
        if r.get('近阈值框对应','[]') not in ('[]',''): dup_instances.extend([r['文件A'],r['文件B']])
    queue.update(dup_instances)
    # 用户明确确认当前标签或已给出更正后，才解除该图的保守隔离。
    queue.difference_update(confirmations)
    known_labels=[]; crop_rows=[]; images={}; det_anns={'train':[],'val':[]}; det_images={'train':[],'val':[]}
    ann_id=1; img_id=1; issue_rows=[]; train_count=Counter();val_count=Counter(); exclusions=[]
    records=[]
    change_log=[]
    for name,r in byname.items():
        if name in PERMANENTLY_EXCLUDED:
            exclusions.append({'filename':name,'split':r['split'],'reason':'用户确认排除的同源冲突图；不进入训练、验证或负样本'})
            continue
        path=Path(r['source_path']);xml=xml_dir/(Path(name).stem+'.xml')
        if not path.exists() or not xml.exists():
            issue_rows.append({'filename':name,'issue':'manifest image or XML missing','path':str(path)});continue
        w,h,objs=parse(xml)
        objs,logged=apply_human_corrections(name,objs,doc,r['split'])
        for obj in objs:
            x1,y1,x2,y2=obj['xyxy']
            if x1<0 or y1<0 or x2>w or y2>h or x2<=x1 or y2<=y1:
                raise ValueError(f'更正后框越界或退化: {name} instance={obj["instance"]} box={obj["xyxy"]} image={w}x{h}')
        change_log.extend(logged)
        # 训练图疑似与现有标注不一致的样本隔离，避免复核前进入正式拟训练清单。
        if r['split']=='train' and name in queue:
            exclusions.append({'filename':name,'split':'train','reason':'第一代预测/重复审查标记待人工复核，先不纳入正式候选训练集'})
            continue
        if args.smoke and r['split']=='train' and len([x for x in records if x['split']=='train'])>=12:continue
        if args.smoke and r['split']=='val' and len([x for x in records if x['split']=='val'])>=8:continue
        # 仅复用第一代已筛选且与原图逐实例匹配的亮度派生图；验证目录永不加入增强图。
        variants=[(path,name,'original')]
        if r['split']=='train':
            bright_name=Path(name).stem+'_brightness.jpg'
            if bright_name in bright_ok:
                bp=PREP/'images/train'/bright_name
                if bp.exists():variants.append((bp,bright_name,'brightness'))
        records.append({'name':name,'split':r['split'],'path':path,'w':w,'h':h,'objects':objs,'variants':variants})

    for rec in records:
        part=rec['split'];
        for image_path,image_name,variant in rec['variants']:
            if not image_path.exists():continue
            with Image.open(image_path) as im:
                actual=im.size
            if actual!=(rec['w'],rec['h']):
                issue_rows.append({'filename':image_name,'issue':f'image size {actual} != xml size {(rec["w"],rec["h"])}','path':str(image_path)});continue
            image_id=img_id;img_id+=1
            det_images[part].append({'id':image_id,'file_name':image_name,'width':rec['w'],'height':rec['h']})
            for obj in rec['objects']:
                x1,y1,x2,y2=obj['xyxy'];bw=x2-x1;bh=y2-y1
                det_anns[part].append({'id':ann_id,'image_id':image_id,'category_id':0,'bbox':[x1,y1,bw,bh],
                    'area':bw*bh,'iscrowd':0});ann_id+=1
                if part=='train':train_count[obj['name']]+=1
                else:val_count[obj['name']]+=1
                crop,cb=crop_with_margin(image_path,obj['xyxy'])
                crop_dir=DATA/'crops'/part;crop_dir.mkdir(parents=True,exist_ok=True)
                crop_name=f'{Path(image_name).stem}__inst{obj["instance"]:02d}.jpg'
                crop_path=crop_dir/crop_name;crop.save(crop_path,quality=95)
                crop_rows.append({'crop_path':str(crop_path),'source_image':rec['name'],'source_variant':variant,
                    'split':part,'instance':obj['instance'],'raw_xml_label':obj['raw_label'],'class_name':obj['name'],
                    'class_id':obj['class_id'],'gt_xyxy':','.join(map(str,obj['xyxy'])),'crop_xyxy':','.join(map(str,cb)),
                    'margin_fraction':0.15,'review_pending':rec['name'] in queue,'use_for_classifier_train':part=='train' and rec['name'] not in queue})
    categories=[{'id':0,'name':'bird','supercategory':'animal'}]
    out=DATA/'annotations';out.mkdir(parents=True,exist_ok=True)
    for part in ('train','val'):
        payload={'images':det_images[part],'annotations':det_anns[part],'categories':categories}
        (out/f'instances_{part}_bird.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    write_csv(REPORTS/'classification_crop_manifest.csv',crop_rows)
    write_csv(REPORTS/'training_exclusions_pending_review.csv',exclusions)
    write_csv(REPORTS/'conversion_issues.csv',issue_rows)
    write_csv(REPORTS/'annotation_change_log.csv',change_log)
    confirmation_rows=[]
    for name,row in sorted(confirmations.items()):
        source_row=byname[name]
        confirmation_rows.append({'filename':name,'split':source_row['split'],'status':row['status'],
          'confirmed_by':row['confirmed_by'],'user_note':row['note'],'source':'用户本轮明确人工复核意见',
          'xml_modified':False,'effective_for_training':source_row['split']=='train' and name not in PERMANENTLY_EXCLUDED})
    write_csv(REPORTS/'manual_review_confirmation_log.csv',confirmation_rows)
    queue_rows=[]
    for r in disagreement:
        if r['annotation_consistency_disagreement']=='True':
            confirmation=confirmations.get(r['filename'])
            queue_rows.append({'filename':r['filename'],'split':r['split'],'reason':'第一代严格>0.5最终显示与现有XML有逐实例匹配分歧；原因待看原图确认',
                'unmatched_xml_instances':r['unmatched_xml_instances'],'unmatched_final_display_boxes':r['unmatched_final_display_boxes'],
                'status':f"人工已确认:{confirmation['status']} ({confirmation['confirmed_by']})" if confirmation else '待人工复核'})
    if 'dashanque_095.jpg' not in confirmations:
        queue_rows.append({'filename':'dashanque_095.jpg','split':'train','reason':'与dashanque_119.jpg确认同源；同一只鸟的逐实例类别不同。095标大山雀；真实类别未确认。','unmatched_xml_instances':'','unmatched_final_display_boxes':'','status':'待人工确认，排除训练'})
    queue_rows.append({'filename':'dashanque_119.jpg','split':'excluded','reason':'同源冲突图；原第一代已排除训练并继续排除。XML标签黑喉红臀鹎，真实类别未确认。','unmatched_xml_instances':'','unmatched_final_display_boxes':'','status':'继续排除'})
    queue_rows.extend([
        {'filename':'shanmaque_028.jpg | shanmaque_090.jpg','split':'near-duplicate review','reason':'同源候选标注同类但框对应IoU=0.48479，低于自动冲突阈值。用户已确认028原标注正确，090仍隔离待复核。','status':'部分确认：shanmaque_090.jpg待复核'},
        {'filename':'heihouhongtunbei_004.jpg | heihouhongtunbei_086.jpg','split':'near-duplicate review','reason':'同源候选标注同类但框对应IoU=0.49158，低于自动匹配阈值。用户已确认两图原标注正确。','status':'用户已确认两图原标注正确'},
    ])
    write_csv(REPORTS/'manual_review_queue.csv',queue_rows)
    # 以原图路径、实例号和框坐标列出可直接打开的逐实例复核工作表。
    detail_rows=[]
    review_root=Path(CFG['first_generation_review_archive'])/'复算依据'
    for r in read_csv(review_root/'疑似同位置类别分歧.csv'):
        confirmed=confirmations.get(r['filename'])
        detail_rows.append({'filename':r['filename'],'split':r['split'],'issue_type':'同一位置类别不一致（预测仅为复核提示）',
          'instance_id':r['xml_instance'],'xml_class':r['xml_class'],'xml_bbox_xyxy':r['xml_bbox_xyxy'],
          'prediction_class':r['display_class'],'prediction_score':r['display_score'],'prediction_bbox_xyxy':r['display_bbox_xyxy'],
          'IoU':r['IoU'],'review_question':'看原图确认XML类别和框；不要按模型输出自动改类。',
          'status':f"人工已确认:{confirmed['status']} ({confirmed['confirmed_by']})" if confirmed else '待人工确认'})
    for r in read_csv(review_root/'未匹配XML实例.csv'):
        detail_rows.append({'filename':r['filename'],'split':r['split'],'issue_type':'XML实例未与最终显示框同类匹配',
          'instance_id':r['gt_index'],'xml_class':r['class_name'],'xml_bbox_xyxy':r['bbox_xyxy'],
          'review_question':'确认框和类别是否正确；检查是否漏标其他鸟或属于漏检/定位差。','status':'待人工复核'})
    for r in read_csv(review_root/'未匹配最终显示框.csv'):
        detail_rows.append({'filename':r['filename'],'split':r['split'],'issue_type':'最终显示预测框未与XML实例同类匹配',
          'prediction_instance_id':r['pred_index'],'prediction_class':r['class_name'],'prediction_score':r['score'],
          'prediction_bbox_xyxy':r['bbox_xyxy'],'review_question':'检查原图该鸟是否漏标、标签不同、误检或定位偏差；预测不作真值。','status':'待人工复核'})
    review_pairs=[('dashanque_119.jpg','同源同鸟类别冲突','和dashanque_095.jpg确认同源；原第一代已排除训练并继续排除。')]
    if 'dashanque_095.jpg' not in confirmations:
        review_pairs.insert(0,('dashanque_095.jpg','同源同鸟类别冲突','和dashanque_119.jpg确认同源；095 XML标大山雀、119 XML标黑喉红臀鹎。真实类未确认，095排除训练。'))
    for file_name,kind,reason in review_pairs:
        detail_rows.append({'filename':file_name,'split':'train' if file_name.endswith('095.jpg') else 'excluded','issue_type':kind,
          'review_question':reason+' 请人工确认真实类别或给出更正后的逐实例标签。','status':'待人工确认/排除训练'})
    detail_rows.extend([
        {'filename':'shanmaque_028.jpg | shanmaque_090.jpg','split':'pair','issue_type':'同源对框低于自动匹配阈值','IoU':'0.48479','review_question':'确认两框是否对应同一只鸟，必要时更正框。','status':'部分确认：shanmaque_028已确认；shanmaque_090仍待复核'},
        {'filename':'heihouhongtunbei_004.jpg | heihouhongtunbei_086.jpg','split':'pair','issue_type':'同源对框低于自动匹配阈值','IoU':'0.49158','review_question':'确认两框是否对应同一只鸟，必要时更正框。','status':'用户已分别确认两图原标注正确'},
        {'filename':'meishanque_012.jpg | meishanque_101.jpg','split':'excluded','issue_type':'图片缺XML','review_question':'已从训练排除；不要当作负样本。确认是否补标。','status':'缺标待处理'},
    ])
    write_csv(REPORTS/'manual_review_instances.csv',detail_rows)
    (REPORTS/'annotation_version.json').write_text(json.dumps({
       'source':'第一代只读原始XML','derived_version':'v2' if doc.get('changes') else 'v1 labels with review confirmations',
       'corrections_file':str(CORRECTIONS_PATH),'correction_count':len(change_log),
       'confirmed_images':len(confirmations),'permanently_excluded':sorted(PERMANENTLY_EXCLUDED),
       'confirmed_image_names':sorted(confirmations),
       'confirmation_source':'用户本轮明确提交的人工复核意见；7图均保留原XML类别和框，未改XML',
       'pending_named_images':['shanmaque_090.jpg','dashanque_119.jpg'],
       'note':'未填写确认前沿用原XML标签且将复核图隔离；模型预测不自动写入标签。'},ensure_ascii=False,indent=2),encoding='utf-8')
    split_summary={'source_split':'第一代 reports/split_manifest.csv；划分沿用，不重分','teacher_examples_used':False,
       'train_images_without_current_review_flags':sum(x['split']=='train' for x in records),
       'validation_images_fixed':sum(x['split']=='val' for x in records),
       'detector_instances':{'train':len(det_anns['train']),'val':len(det_anns['val'])},
       'classification_crop_rows':len(crop_rows),'classifier_train_rows_eligible_after_current_flags':sum(x['use_for_classifier_train'] in (True,'True') for x in crop_rows),
       'train_instance_by_class_including_brightness':dict(train_count),'val_instance_by_class':dict(val_count),
       'raw_image_count_does_not_include_brightness_as_independent':True,
       'formal_training_status':'ready for formal training; unresolved named images remain excluded',
       'pending_train_exclusions':[x['filename'] for x in exclusions],
       'xml_coordinate_note':'原始XML端点原值保留；COCO bbox宽高=xmax-xmin,ymax-ymin，不进行+1/-1平移。'}
    (REPORTS/'data_preparation_summary.json').write_text(json.dumps(split_summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(split_summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
