"""将同一个冻结 BioCLIP 2 八类头应用于 PP-YOLOE+ 和 RF-DETR 显示框。"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, shutil, time, copy
from pathlib import Path
import numpy as np
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs" / "experiment.json").read_text(encoding="utf-8"))
OUT = Path(CFG["output_root"])
for key, value in {"HF_HOME": OUT/"cache"/"huggingface", "TORCH_HOME": OUT/"cache"/"torch",
                   "TORCHINDUCTOR_CACHE_DIR": OUT/"cache"/"inductor", "TEMP": OUT/"cache"/"tmp",
                   "TMP": OUT/"cache"/"tmp", "MPLCONFIGDIR": OUT/"cache"/"matplotlib"}.items():
    Path(value).mkdir(parents=True, exist_ok=True); os.environ[key] = str(value)
os.environ["HF_HUB_DISABLE_XET"] = "1"

DET = Path(CFG["detector_project"]); GEN3 = Path(CFG["gen3_project"])
MAPPING_PATH = Path(CFG["class_mapping"])
CLASSIFIER = CFG["classifier"]
NAMES = [x["name"] for x in json.loads(MAPPING_PATH.read_text(encoding="utf-8-sig"))["classes"]]
THRESHOLD = float(CFG["threshold_strictly_greater"])
MATCH_IOU = float(CFG["match_iou"])
EXPAND = float(CFG["crop_expansion"])

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()

def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0., x2-x1)*max(0., y2-y1)
    area_a = max(0., a[2]-a[0])*max(0., a[3]-a[1]); area_b = max(0., b[2]-b[0])*max(0., b[3]-b[1])
    return inter/(area_a+area_b-inter) if area_a+area_b-inter > 0 else 0.

def crop_image(im, box):
    x1, y1, x2, y2 = map(float, box); dx=(x2-x1)*EXPAND; dy=(y2-y1)*EXPAND
    l=max(0, math.floor(x1-dx)); t=max(0, math.floor(y1-dy)); r=min(im.width, math.ceil(x2+dx)); b=min(im.height, math.ceil(y2+dy))
    if r <= l or b <= t: raise ValueError(f"Degenerate crop {box} for image {im.size}")
    bird=im.crop((l,t,r,b)); w,h=bird.size; side=max(w,h)
    square=Image.new("RGB", (side,side), (114,114,114)); square.paste(bird, ((side-w)//2, (side-h)//2))
    return square

def load_inputs():
    source={}
    for key in ("fixed_detection_comparison", "external_detection_comparison", "fixed_bioclip_formal", "fixed_end_to_end_formal", "external_bioclip_formal"):
        path=Path(CFG["sources"][key]); source[key]={"path":str(path),"sha256":sha(path)}
    fixed_doc=json.loads(Path(CFG["sources"]["fixed_detection_comparison"]).read_text(encoding="utf-8"))
    ext_doc=json.loads(Path(CFG["sources"]["external_detection_comparison"]).read_text(encoding="utf-8"))
    fixed_bio=json.loads(Path(CFG["sources"]["fixed_bioclip_formal"]).read_text(encoding="utf-8"))
    external_bio=json.loads(Path(CFG["sources"]["external_bioclip_formal"]).read_text(encoding="utf-8"))
    image_identity={"fixed_validation_pixel_identical":0,"external80_pixel_identical":0,"fixed_images":0,"external_images":0}
    manual_path=Path(CFG["external_review"]); manual=json.loads(manual_path.read_text(encoding="utf-8"))
    manual_sha=sha(manual_path)
    if manual.get("source_report_sha256") != source["external_detection_comparison"]["sha256"]:
        raise RuntimeError("用户复核记录引用的外来候选报告哈希与现有文件不一致，停止套用复核状态。")
    if manual.get("scope",{}).get("confirmed_image_count") != 6 or manual.get("scope",{}).get("confirmed_candidate_count") != 7 or manual.get("scope",{}).get("other_pending_candidate_count") != 4:
        raise RuntimeError("人工复核记录范围与用户说明不一致。")
    g2=json.loads(MAPPING_PATH.read_text(encoding="utf-8-sig"))
    if [x["name"] for x in g2["classes"]] != NAMES: raise RuntimeError("类别映射顺序变化。")

    # 固定验证集真值从正式 BioCLIP 结果取逐实例类别；对照单类检测报告的实例ID和框。
    bio_rows={r["filename"]:r for r in fixed_bio["rows"]}
    fixed_images=fixed_doc.get("images",fixed_doc.get("rows",[]))
    if len(fixed_images)!=172 or len(bio_rows)!=172: raise RuntimeError("固定验证集必须是同一172张清单。")
    for row in fixed_images:
        old=bio_rows.get(row["filename"])
        if old is None: raise RuntimeError("固定验证清单缺图："+row["filename"])
        with Image.open(row["image_path"]) as cur, Image.open(old["image_path"]) as prev:
            cur=cur.convert("RGB");prev=prev.convert("RGB")
            if cur.size!=prev.size or ImageChops.difference(cur,prev).getbbox() is not None: raise RuntimeError("固定集PP/Bio正式图片像素不一致："+row["filename"])
            image_identity["fixed_validation_pixel_identical"]+=1
        image_identity["fixed_images"]+=1
        gt1=sorted(tuple(map(float,x["bbox_xyxy"])) for x in row["truth"])
        gt2=sorted(tuple(map(float,x["bbox_xyxy"])) for x in old["truth"])
        if gt1 != gt2: raise RuntimeError("真值框/实例ID版本不一致："+row["filename"])
        if len(gt1)!=len(gt2): raise RuntimeError("验证实例数不一致："+row["filename"])
        # 单类检测报告只保存框，不含species和instance_id；固定真值版本按完全相同的逐实例坐标校验后，从正式BioCLIP报告补回类别。
        for i,gt in enumerate(old["truth"],1): gt.setdefault("instance_id",i)
        for gt in old["truth"]:
            if int(gt["class_id"])<0 or int(gt["class_id"])>=len(NAMES): raise RuntimeError("未知验证类标签")
        row["truth"]=old["truth"]
        row["formal_bioclip_predictions"]=copy.deepcopy(old["models"]["bioclip2"]["displayed"])
        for i,p in enumerate(row["formal_bioclip_predictions"]): p.setdefault("candidate_index",i)
        for model in ("ppyoloe","rfdetr"):
            row["models"][model]["display_candidates"]=[p for p in row["models"][model]["display_candidates"] if float(p["detector_score"])>THRESHOLD]
    external_images=ext_doc.get("images",ext_doc.get("rows",[]))
    if len(external_images)!=80: raise RuntimeError(f"外来图清单应80张，实际{len(external_images)}")
    ext_bio_rows={r["filename"]:r for r in external_bio["images"]}
    if len(ext_bio_rows)!=80: raise RuntimeError("既有BioCLIP外来图结果不是80张")
    for row in external_images:
        old=ext_bio_rows.get(row["filename"])
        if old is None: raise RuntimeError("既有BioCLIP外来图缺图："+row["filename"])
        old_asset=(Path(CFG["sources"]["external_bioclip_formal"]).parent/old["asset"]).resolve()
        with Image.open(row["image_path"]) as cur, Image.open(old_asset) as prev:
            cur=cur.convert("RGB");prev=prev.convert("RGB")
            if cur.size!=prev.size or ImageChops.difference(cur,prev).getbbox() is not None: raise RuntimeError("外来图BioCLIP/检测报告图片像素不一致："+row["filename"])
            image_identity["external80_pixel_identical"]+=1
        image_identity["external_images"]+=1
        row["formal_bioclip_predictions"]=copy.deepcopy(old["bio_predictions"])
        for i,p in enumerate(row["formal_bioclip_predictions"]): p.setdefault("candidate_index",i)
        # 外来标记和目录预期沿用现有同源核查表；不作为推理输入。
        row["same_photo"]=bool(old["same_photo"]);row["related_scene"]=bool(old["related_scene"])
        row["expected_category_posthoc"]=old.get("expected")
        for model in ("ppyoloe","rfdetr"):
            row["models"][model]["display_candidates"]=[p for p in row["models"][model]["display_candidates"] if float(p["detector_score"])>THRESHOLD]
    # 既有PP+BioCLIP输出是主方案基准；新采集的PP候选只用于核验和耗时复放。
    for row in fixed_images+external_images:
        row["models"]["ppyoloe"]["current_detector_candidates_for_audit"]=row["models"]["ppyoloe"]["display_candidates"]
        row["models"]["ppyoloe"]["display_candidates"]=row["formal_bioclip_predictions"]
        del row["formal_bioclip_predictions"]
    if sum(len(r["models"]["ppyoloe"]["display_candidates"]) for r in fixed_images)!=177: raise RuntimeError("正式PP+BioCLIP固定候选应177")
    return fixed_images,external_images,bio_rows,manual,source,manual_sha,image_identity

def model_and_transform(device):
    import open_clip, torch
    local=Path(CLASSIFIER["weights"]); head_path=Path(CLASSIFIER["head"]); cfg_path=Path(CLASSIFIER["config"])
    if sha(local)!="b7b2bf6fbc95799e42630e394cf95803892ab447c1a8ab629dbc82fbeaf7dfef": raise RuntimeError("BioCLIP encoder权重哈希与已核验版本不同")
    if sha(head_path)!="7d199f954bf7a125fe50d1f11f632268bbfa2d0c7b1bfa1bcbdf8a681b4b4ece": raise RuntimeError("BioCLIP最佳分类头哈希与正式版本不同")
    cfg=json.loads(cfg_path.read_text(encoding="utf-8")); pcfg=cfg["preprocess_cfg"]
    model,_,transform=open_clip.create_model_and_transforms("ViT-L-14",pretrained=str(local),image_mean=pcfg["mean"],image_std=pcfg["std"],image_interpolation=pcfg["interpolation"])
    model=model.visual.to(device).eval()
    for p in model.parameters(): p.requires_grad_(False)
    with torch.no_grad(): dim=int(model(transform(Image.new("RGB",(224,224),(114,114,114))).unsqueeze(0).to(device)).shape[1])
    if dim!=768: raise RuntimeError(f"BioCLIP特征维度预期768，实际{dim}")
    head=torch.nn.Linear(dim,len(NAMES)).to(device)
    head.load_state_dict(torch.load(head_path,map_location=device,weights_only=True)); head.eval()
    return model,transform,head

def classify_rows(rows,model_name,model,transform,head,device,asset_dir):
    import torch
    jobs=[]; image_map={}
    for row in rows:
        image_path=Path(row["image_path"])
        if not image_path.is_file(): raise FileNotFoundError(image_path)
        if image_path not in image_map:
            with Image.open(image_path) as im:
                rgb=im.convert("RGB")
                if rgb.size!=(int(row["width"]),int(row["height"])): raise RuntimeError("预测尺寸与原图不一致："+row["filename"])
                dest=asset_dir/row["filename"]; dest.parent.mkdir(parents=True,exist_ok=True)
                if not dest.exists(): shutil.copy2(image_path,dest)
                image_map[image_path]=dest
        row["asset"]=str(image_map[image_path].relative_to(OUT)).replace("\\","/")
        for p in row["models"][model_name]["display_candidates"]:
            jobs.append((row,p,image_path))
    start=time.perf_counter(); outputs=[]; batch_size=8
    for start_i in range(0,len(jobs),batch_size):
        chunk=jobs[start_i:start_i+batch_size]; images=[]
        for row,p,path in chunk:
            with Image.open(path) as im: images.append(transform(crop_image(im.convert("RGB"),p["bbox_xyxy"])))
        x=torch.stack(images).to(device)
        with torch.inference_mode(), torch.autocast(device_type=device,enabled=device=="cuda",dtype=torch.float16):
            z=model(x).float(); z=torch.nn.functional.normalize(z,dim=1); prob=head(z).softmax(1).float().cpu().numpy()
        for (_,p,_),scores in zip(chunk,prob):
            top=int(np.argmax(scores)); p.update({"class_id":top,"class_name":NAMES[top],"classifier_top_score":float(scores[top]),
                "classifier_scores":{n:float(v) for n,v in zip(NAMES,scores)},
                "final_score_product":float(p["detector_score"]*scores[top]),
                "score_note":"检测分数与分类softmax分别保存；乘积仅作排序参考，不是校准概率。"})
            outputs.append(scores)
    if device=="cuda": torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    return {"boxes":len(jobs),"seconds_crop_preprocess_and_encode":elapsed,"seconds_per_box":elapsed/len(jobs) if jobs else 0}

def spatial_pairs(truth,preds):
    remaining=set(range(len(truth))); pairs=[]; unpred=[]
    for pi,p in sorted(enumerate(preds),key=lambda z:-float(z[1]["detector_score"])):
        options=sorted(((iou(p["bbox_xyxy"],truth[gi]["bbox_xyxy"]),gi) for gi in remaining),reverse=True)
        if options and options[0][0]>=MATCH_IOU:
            ov,gi=options[0]; remaining.remove(gi); pairs.append((gi,pi,ov))
        else: unpred.append(pi)
    return remaining,unpred,pairs

def metrics_for(rows,model_name):
    cm=np.zeros((len(NAMES)+1,len(NAMES)+1),dtype=np.int64); per=[]
    totals={"TP":0,"FP":0,"FN":0,"displayed_boxes":0,"unmatched_predictions":0,"unmatched_truths":0,"spatial_wrong_class":0}
    perclass=[]
    for cid,name in enumerate(NAMES):
        tp=fp=fn=gt_count=pred_count=0
        for row in rows:
            gt=[g for g in row.get("truth",[]) if int(g["class_id"])==cid]
            ps=[p for p in row["models"][model_name]["display_candidates"] if int(p["class_id"])==cid]
            gt_count+=len(gt);pred_count+=len(ps);unused=set(range(len(gt)))
            for p in sorted(ps,key=lambda x:-float(x["detector_score"])):
                opts=sorted(((iou(p["bbox_xyxy"],g["bbox_xyxy"]),gi) for gi,g in enumerate(gt) if gi in unused),reverse=True)
                if opts and opts[0][0]>=MATCH_IOU: unused.remove(opts[0][1]);tp+=1
                else: fp+=1
            fn+=len(unused)
        precision=tp/(tp+fp) if tp+fp else 0.; recall=tp/(tp+fn) if tp+fn else 0.
        perclass.append({"class_id":cid,"class_name":name,"GT":gt_count,"predictions":pred_count,"TP":tp,"FP":fp,"FN":fn,
                         "Precision":precision,"Recall":recall,"F1":2*precision*recall/(precision+recall) if precision+recall else 0.})
    image_rows=[]
    for row in rows:
        truth=row.get("truth",[]); preds=row["models"][model_name]["display_candidates"]
        ug,up,pairs=spatial_pairs(truth,preds)
        wrong=sum(int(truth[gi]["class_id"])!=int(preds[pi]["class_id"]) for gi,pi,_ in pairs)
        cmrow=cm
        for gi,pi,_ in pairs: cmrow[int(truth[gi]["class_id"]),int(preds[pi]["class_id"])]+=1
        for gi in ug: cmrow[int(truth[gi]["class_id"]),len(NAMES)]+=1
        for pi in up: cmrow[len(NAMES),int(preds[pi]["class_id"])]+=1
        correct_pairs=[]; wrong_pairs=[]
        for gi,pi,ov in pairs:
            item={"instance_id":truth[gi].get("instance_id"),"truth_class":truth[gi]["class_name"],"predicted_class":preds[pi]["class_name"],"iou":ov,
                  "candidate_index":preds[pi]["candidate_index"],"detection_score":preds[pi]["detector_score"],"classifier_score":preds[pi]["classifier_top_score"]}
            (correct_pairs if int(truth[gi]["class_id"])==int(preds[pi]["class_id"]) else wrong_pairs).append(item)
        totals["TP"]+=len(correct_pairs); totals["FP"]+=len(preds)-len(correct_pairs); totals["FN"]+=len(truth)-len(correct_pairs)
        totals["displayed_boxes"]+=len(preds); totals["unmatched_predictions"]+=len(up); totals["unmatched_truths"]+=len(ug); totals["spatial_wrong_class"]+=wrong
        image_rows.append({"filename":row["filename"],"TP":len(correct_pairs),"FP_end_to_end":len(preds)-len(correct_pairs),"FN_end_to_end":len(truth)-len(correct_pairs),
            "spatial_wrong_class":wrong,"unmatched_displayed_boxes":len(up),"unmatched_truths":len(ug),"no_boxes":not preds,
            "correct_instances":correct_pairs,"class_disagreements":wrong_pairs,
            "unmatched_prediction_candidates":[{"candidate_index":preds[pi].get("candidate_index",pi),"bbox_xyxy":preds[pi]["bbox_xyxy"],"class_name":preds[pi]["class_name"],"detector_score":preds[pi]["detector_score"]} for pi in up],
            "unmatched_truth_instances":[{"instance_id":truth[gi].get("instance_id"),"class_name":truth[gi]["class_name"],"bbox_xyxy":truth[gi]["bbox_xyxy"]} for gi in sorted(ug)]})
    p=totals["TP"]/(totals["TP"]+totals["FP"]) if totals["TP"]+totals["FP"] else 0
    r=totals["TP"]/(totals["TP"]+totals["FN"]) if totals["TP"]+totals["FN"] else 0
    totals.update({"images":len(rows),"GT":totals["TP"]+totals["FN"],"Precision":p,"Recall":r,"F1":2*p*r/(p+r) if p+r else 0,
        "images_with_no_boxes":sum(x["no_boxes"] for x in image_rows),"class_disagreement_images":sum(bool(x["class_disagreements"]) for x in image_rows),
        "matching_rule":"每个类别内按检测分数降序，以IoU>=0.5做一对一匹配。错误类别使预测类计FP、真值类计FN；重复预测为FP。混淆矩阵另用忽略类别的分数优先IoU匹配。"})
    return totals,perclass,cm,image_rows

def attach_review(external_rows,manual):
    decisions={(x["filename"],int(x["candidate_index"])):x for x in manual["decisions"]}
    confirmed=0;pending=0
    for row in external_rows:
        for p in row["models"]["rfdetr"]["display_candidates"]:
            d=decisions.get((row["filename"],int(p["candidate_index"])))
            if d:
                if max(abs(float(x)-float(y)) for x,y in zip(p["bbox_xyxy"],d["bbox_xyxy"]))>1e-4 or abs(float(p["detector_score"])-float(d["detector_score"]))>1e-5:
                    raise RuntimeError(f"人工确认框与当前RF候选不匹配：{row['filename']} #{p['candidate_index']}")
                p["user_detection_review"]={"status":d["review"],"review_note":d["review_note"],"reviewer":d.get("reviewer"),"review_date":d.get("review_date")}
                if d["review"]=="确实是鸟": confirmed+=1
                else: pending+=1
        row["known_added_candidates"]=[p for p in row["models"]["rfdetr"]["display_candidates"] if p.get("candidate_index") in {d["candidate_index"] for d in decisions.values() if d["filename"]==row["filename"]}]
    if confirmed!=7 or pending!=4: raise RuntimeError(f"人工复核逐框核对异常：confirmed={confirmed}, pending={pending}")
    return {"confirmed_images":6,"confirmed_boxes":confirmed,"pending_boxes":pending,"confirmed_detection_only":True,"source_record_sha256":sha(Path(CFG["external_review"]))}

def replay_audit(reference_rows, replay_rows):
    """对本轮PP候选复放与原PP+BioCLIP正式预测做空间配对，只核验不替换主方案。"""
    by_ref={r["filename"]:r for r in reference_rows}; by_new={r["filename"]:r for r in replay_rows}
    pairs=[]; count_mismatch=[]; top_changes=0; max_score_delta=0.; max_coord_delta=0.
    for name,oldrow in by_ref.items():
        newrow=by_new[name]; old=oldrow["models"]["ppyoloe"]["display_candidates"]
        new=newrow["models"]["ppyoloe"]["display_candidates"]
        if len(old)!=len(new): count_mismatch.append({"filename":name,"formal":len(old),"current_pp_candidate_count":len(new)})
        unused=set(range(len(new)))
        for oi,op in enumerate(old):
            opts=sorted(((iou(op["bbox_xyxy"],new[ni]["bbox_xyxy"]),ni) for ni in unused),reverse=True)
            if not opts or opts[0][0]<.5: continue
            ov,ni=opts[0];unused.remove(ni);npred=new[ni]
            dc=max(abs(float(a)-float(b)) for a,b in zip(op["bbox_xyxy"],npred["bbox_xyxy"]))
            ds=max(abs(float(op["classifier_scores"][c])-float(npred["classifier_scores"][c])) for c in NAMES)
            max_coord_delta=max(max_coord_delta,dc);max_score_delta=max(max_score_delta,ds)
            top_changes+=int(op["class_name"]!=npred["class_name"])
            pairs.append({"filename":name,"formal_candidate_index":op.get("candidate_index",oi),"current_candidate_index":npred["candidate_index"],"iou":ov,"max_coordinate_delta_px":dc,"max_class_score_delta":ds,"formal_class":op["class_name"],"replay_class":npred["class_name"]})
    return {"formal_display_boxes":sum(len(r["models"]["ppyoloe"]["display_candidates"]) for r in reference_rows),
        "current_pp_candidates":sum(len(r["models"]["ppyoloe"]["display_candidates"]) for r in replay_rows),"matched_candidate_pairs":len(pairs),
        "per_image_count_mismatch_count":len(count_mismatch),"per_image_count_mismatches":count_mismatch,"top_class_changes":top_changes,
        "max_coordinate_delta_px":max_coord_delta,"max_classifier_score_delta":max_score_delta,"paired_examples":pairs[:10],
        "use_in_metrics":False,"reason":"正式主方案直接使用原保存PP+BioCLIP结果；复放仅校验本地BioCLIP实现/耗时，防止数值微差改写历史主方案。"}

def add_pair_analysis(rows,external=False):
    for row in rows:
        base=row["models"]["ppyoloe"]["display_candidates"]; rf=row["models"]["rfdetr"]["display_candidates"]
        tags=[]
        # 两模型候选按IoU空间配对，类别比较仅是结果展示，不用于挑选或改写模型框。
        left=set(range(len(base))); matched=[]
        for ri,rp in sorted(enumerate(rf),key=lambda z:-float(z[1]["detector_score"])):
            options=sorted(((iou(rp["bbox_xyxy"],base[bi]["bbox_xyxy"]),bi) for bi in left),reverse=True)
            if options and options[0][0]>=MATCH_IOU:
                ov,bi=options[0];left.remove(bi);matched.append({"ppyoloe_candidate_index":base[bi].get("candidate_index",bi),"rfdetr_candidate_index":rp.get("candidate_index",ri),"iou":ov,
                    "classes_equal":base[bi]["class_id"]==rp["class_id"],"ppyoloe_class":base[bi]["class_name"],"rfdetr_class":rp["class_name"]})
        row["classifier_cross_model_pairs"]=matched
        if external:
            if row.get("known_added_candidates"):
                if any(p.get("user_detection_review",{}).get("status")=="确实是鸟" for p in row["known_added_candidates"]): tags.append("新增鸟框（用户已确认框是鸟）")
                if any(p.get("user_detection_review",{}).get("status")=="暂不能确定" for p in row["known_added_candidates"]): tags.append("待复核")
            if not base and not rf: tags.append("无框")
            if base and not rf: tags.append("退步（PP有框、RF无框；外来图无真值）")
            if any(not x["classes_equal"] for x in matched): tags.append("类别分歧（模型框空间配对）")
            if row.get("expected_category_posthoc") and any(p["class_name"]!=row["expected_category_posthoc"] for p in rf): tags.append("与目录事后预期不一致")
        else:
            base_by_gt={}
            rf_by_gt={}
            for model_name,dest in (("ppyoloe",base_by_gt),("rfdetr",rf_by_gt)):
                _,_,pairs=spatial_pairs(row["truth"],row["models"][model_name]["display_candidates"])
                for gi,pi,ov in pairs: dest[gi]=(row["models"][model_name]["display_candidates"][pi],ov)
            errors=[]
            newly_wrong=[]; regressed=[]; improved=[]
            for gi,truth in enumerate(row["truth"]):
                b=base_by_gt.get(gi);f=rf_by_gt.get(gi); cls=int(truth["class_id"])
                bcorrect=bool(b and int(b[0]["class_id"])==cls); fcorrect=bool(f and int(f[0]["class_id"])==cls)
                if f and int(f[0]["class_id"])!=cls: errors.append({"instance_id":truth.get("instance_id"),"truth_class":truth["class_name"],"rfdetr_class":f[0]["class_name"],"rfdetr_iou":f[1],"candidate_index":f[0]["candidate_index"]})
                if bcorrect and f and not fcorrect: newly_wrong.append(errors[-1])
                if bcorrect and not fcorrect: regressed.append({"instance_id":truth.get("instance_id"),"truth_class":truth["class_name"],"baseline_status":"正确匹配","rfdetr_status":"类别错误" if f else "无空间匹配", "rfdetr_prediction":f[0]["class_name"] if f else None})
                if fcorrect and not bcorrect: improved.append({"instance_id":truth.get("instance_id"),"truth_class":truth["class_name"],"baseline_status":"正确匹配" if b else "类别错误/无空间匹配","rfdetr_status":"正确匹配"})
            row["end_to_end_change"]={"baseline_correct_instances":len([1 for gi in range(len(row["truth"])) if gi in base_by_gt and int(base_by_gt[gi][0]["class_id"])==int(row["truth"][gi]["class_id"])]),
                "rfdetr_correct_instances":len([1 for gi in range(len(row["truth"])) if gi in rf_by_gt and int(rf_by_gt[gi][0]["class_id"])==int(row["truth"][gi]["class_id"])]),
                "new_class_disagreements":newly_wrong,"regressed_instances":regressed,"improved_instances":improved,"rfdetr_spatial_class_errors":errors}
            base_errors=[gi for gi,g in enumerate(row["truth"]) if gi in base_by_gt and int(base_by_gt[gi][0]["class_id"])!=int(g["class_id"])]
            if errors or base_errors: tags.append("类别分歧")
            if regressed: tags.append("退步")
            if improved: tags.append("改善")
            if not base and rf: tags.append("RF新增框")
            if not base and not rf: tags.append("无框")
        row["status_labels"]=tags

def write_csv(path,header,rows):
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=header,extrasaction="ignore");w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--device",default="cuda",choices=["cuda","cpu"]);a=ap.parse_args()
    import torch, open_clip
    torch.set_num_threads(4)
    if a.device=="cuda" and not torch.cuda.is_available(): raise RuntimeError("Requested CUDA but CUDA is unavailable")
    device=a.device
    fixed,external,bio_rows,manual,sources,manual_sha,image_identity=load_inputs()
    encoder,transform,head=model_and_transform(device)
    timings={}
    # 原正式PP+BioCLIP预测是比较基准；本轮PP候选只做重算校验，RF候选才用于新结果。
    replay_fixed=copy.deepcopy(fixed);replay_external=copy.deepcopy(external)
    for rows in (replay_fixed,replay_external):
        for row in rows: row["models"]["ppyoloe"]["display_candidates"]=row["models"]["ppyoloe"]["current_detector_candidates_for_audit"]
    timings["fixed_val_pp_replay_audit"]=classify_rows(replay_fixed,"ppyoloe",encoder,transform,head,device,OUT/"assets"/"fixed_val")
    timings["fixed_val_rfdetr"]=classify_rows(fixed,"rfdetr",encoder,transform,head,device,OUT/"assets"/"fixed_val")
    timings["external80_pp_replay_audit"]=classify_rows(replay_external,"ppyoloe",encoder,transform,head,device,OUT/"assets"/"external80")
    timings["external80_rfdetr"]=classify_rows(external,"rfdetr",encoder,transform,head,device,OUT/"assets"/"external80")
    replay_audits={"fixed_val":replay_audit(fixed,replay_fixed),"external80":replay_audit(external,replay_external)}
    review=attach_review(external,manual)
    add_pair_analysis(fixed,False);add_pair_analysis(external,True)
    fixed_metrics={};fixed_classes={};fixed_matrices={};fixed_image_metrics={}
    for model_name in ("ppyoloe","rfdetr"):
        m,c,cm,im=metrics_for(fixed,model_name);fixed_metrics[model_name]=m;fixed_classes[model_name]=c;fixed_matrices[model_name]={'labels':NAMES+["背景"],"matrix":cm.tolist()};fixed_image_metrics[model_name]=im
    per_file={mn:{x["filename"]:x for x in fixed_image_metrics[mn]} for mn in ("ppyoloe","rfdetr")}
    for row in fixed:
        row["per_model_end_to_end_metrics"]={mn:per_file[mn][row["filename"]] for mn in ("ppyoloe","rfdetr")}
        if len(row["truth"])>1 and row["per_model_end_to_end_metrics"]["rfdetr"]["FN_end_to_end"]>0:
            row["status_labels"].append("多鸟漏检")
    external_summary={}
    scopes={"all_80":external,"exclude_9_same_source":[r for r in external if not r.get("same_photo")],
        "exclude_9_same_source_and_related_scene":[r for r in external if not r.get("same_photo") and not r.get("related_scene")]}
    for scope,selected in scopes.items():
        external_summary[scope]={"images":len(selected),"same_source_count":sum(bool(r.get("same_photo")) for r in selected),"related_scene_count":sum(bool(r.get("related_scene")) for r in selected),"models":{}}
        for mn in ("ppyoloe","rfdetr"):
            preds=[p for row in selected for p in row["models"][mn]["display_candidates"]]
            expected_pairs=[p for row in selected if row.get("expected_category_posthoc") for p in row["models"][mn]["display_candidates"]]
            external_summary[scope]["models"][mn]={"images_with_any_box":sum(bool(r["models"][mn]["display_candidates"]) for r in selected),
                "no_box_images":[r["filename"] for r in selected if not r["models"][mn]["display_candidates"]],"displayed_boxes":len(preds),
                "box_class_equal_to_directory_marker_descriptive_only":sum(p["class_name"]==row["expected_category_posthoc"] for row in selected if row.get("expected_category_posthoc") for p in row["models"][mn]["display_candidates"]),
                "boxes_in_scope_with_directory_marker":len(expected_pairs)}
    change_summary={
      "new_class_disagreement_instances":sum(len(r["end_to_end_change"]["new_class_disagreements"]) for r in fixed),
      "regressed_instances":sum(len(r["end_to_end_change"]["regressed_instances"]) for r in fixed),
      "improved_instances":sum(len(r["end_to_end_change"]["improved_instances"]) for r in fixed),
      "regression_images":[r["filename"] for r in fixed if r["end_to_end_change"]["regressed_instances"]],
      "new_disagreement_images":[r["filename"] for r in fixed if r["end_to_end_change"]["new_class_disagreements"]],
      "improvement_images":[r["filename"] for r in fixed if r["end_to_end_change"]["improved_instances"]]}
    metadata={"run_date":"2026-10-12","device":device,"torch":torch.__version__,"open_clip":open_clip.__version__,
      "encoder":"BioCLIP 2 ViT-L-14 visual encoder, frozen","encoder_weight":CLASSIFIER["weights"],"encoder_weight_sha256":sha(Path(CLASSIFIER["weights"])),
      "classifier_head":CLASSIFIER["head"],"classifier_head_sha256":sha(Path(CLASSIFIER["head"])),"classifier_head_source":"现有BioCLIP 2最佳八类线性分类头；未重训",
      "classes":NAMES,"threshold_strictly_greater":THRESHOLD,"dedup":"继承固定检测预测：分数优先、非传递IoU>=0.5，未添加新规则",
      "crop":"预测框外扩15%，裁到图像边界，灰色RGB114正方形补边，再用BioCLIP配置bicubic预处理；特征L2归一化，原best_head softmax",
      "score_rule":"检测分数与分类top score分列；乘积为分析排序值，不是校准概率",
      "truth_version":"物种真值采用第三代正式BioCLIP固定验证报告；逐图真值框数量和坐标与本轮检测报告逐一相同；检测报告不含物种标签，类别仅来自正式物种标注版本。",
      "image_identity_audit":image_identity,
      "user_review_record":str(CFG["external_review"]),"user_review_record_sha256":manual_sha,"user_review_scope":review,
      "source_files":sources,"pp_formal_replay_audit":replay_audits,
      "training_or_teacher_test_used":False,"validation_limitation":"固定172张验证图已被反复探索，不是独立测试集。外来80张没有逐实例真值，不计算准确率或mAP。",
      "classifier_timing":timings,
      "detector_timing_from_prior_measured_runs":{"PP-YOLOE+":{"fixed_val_seconds":105.5322547,"device":"CPU; legacy Paddle GPU unavailable due cuDNN8 runtime mismatch","input":"640 square stretch"},
        "RF-DETR":{"fixed_val_seconds":5.467158,"device":"CUDA","input":"512 native official preprocessing"},
        "warning":"检测设备和输入预处理不同，不能据此作同硬件速度对照；分类器裁剪/编码计时分别记录于classifier_timing。"}}
    def slim(rows):
        packed=[]
        for row in rows:
            models={}
            for mn,mr in row["models"].items():
                native=mr.get("native_candidates",[])
                dedup=mr.get("dedup_candidates",[])
                suppressed=mr.get("suppressed_candidates",[])
                models[mn]={"candidate_source_counts":{"native":len(native),"after_existing_dedup":len(dedup),"suppressed":len(suppressed),
                    "displayed_after_threshold":len(mr["display_candidates"])},
                    "display_candidates":mr["display_candidates"]}
            x={k:row[k] for k in ("filename","image_path","asset","width","height","truth","same_photo","related_scene",
                "expected_category_posthoc","known_baseline_empty","status_labels","classifier_cross_model_pairs","end_to_end_change","per_model_end_to_end_metrics") if k in row}
            x["models"]=models
            x["candidate_source"]="完整原生/抑制候选及抑制链见 metadata.source_files 中带SHA-256的原始检测对照JSON；此处保存该图候选数和最终物种输出。"
            packed.append(x)
        return packed
    result={"metadata":metadata,"fixed_validation":{"summary":fixed_metrics,"per_class":fixed_classes,"confusion_matrix":fixed_matrices,
        "per_image":fixed_image_metrics,"model_change_summary":change_summary,"rows":slim(fixed)},"external80":{"scopes":external_summary,"rows":slim(external)},
        "rule":{"end_to_end_tp":"预测框与真值IoU>=0.5，且类别正确，一对一；错误类别对预测类记FP、真值类记FN。","external":"无逐实例真值，仅输出模型预测及用户检测框复核状态；不计算准确率/mAP。"}}
    (OUT/"reports"/"end_to_end_results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    # 逐框CSV: 每个展示框一行，检测/分类得分独立保存。
    for split,rows in (("fixed_val",fixed),("external80",external)):
        flat=[]
        for row in rows:
            for mn in ("ppyoloe","rfdetr"):
                for pi,p in enumerate(row["models"][mn]["display_candidates"]):
                    flat.append({"filename":row["filename"],"model":mn,"candidate_index":p["candidate_index"],"bbox_xyxy":json.dumps(p["bbox_xyxy"]),
                        "detector_score":p["detector_score"],"predicted_species":p["class_name"],"classifier_top_score":p["classifier_top_score"],
                        "final_score_product_not_calibrated":p.get("final_score_product",float(p["detector_score"])*float(p["classifier_top_score"])),"eight_class_scores":json.dumps(p["classifier_scores"],ensure_ascii=False),
                        "directory_expected_posthoc":row.get("expected_category_posthoc",""),"detection_review":p.get("user_detection_review",{}).get("status","")})
        write_csv(OUT/"reports"/(split+"_per_box.csv"),list(flat[0].keys()) if flat else ["filename"],flat)
    write_csv(OUT/"reports"/"fixed_val_per_class.csv",["model","class_id","class_name","GT","predictions","TP","FP","FN","Precision","Recall","F1"],
        [{"model":mn,**x} for mn in fixed_classes for x in fixed_classes[mn]])
    write_csv(OUT/"reports"/"fixed_val_per_image.csv",["filename","model","TP","FP_end_to_end","FN_end_to_end","spatial_wrong_class","unmatched_displayed_boxes","unmatched_truths","no_boxes"],
        [{"model":mn,**x} for mn in fixed_image_metrics for x in fixed_image_metrics[mn]])
    with (OUT/"reports"/"fixed_val_confusion_matrix.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.writer(f);w.writerow(["model","true\\pred",*NAMES,"背景"])
        for mn,mx in fixed_matrices.items():
            for name,values in zip(mx["labels"],mx["matrix"]): w.writerow([mn,name,*values])
    # 复核清单仅保留用户已有决定；这里不覆盖原意见。
    (OUT/"reports"/"manual_review_scope.json").write_text(json.dumps({"record":CFG["external_review"],"sha256":manual_sha,"confirmed":review,
        "remaining_pending":[{"filename":d["filename"],"candidate_index":d["candidate_index"],"detector_score":d["detector_score"],"review":"待复核"} for d in manual["decisions"] if d["review"]!="确实是鸟"]},ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"fixed":fixed_metrics,"external":external_summary,"change_summary":change_summary,"timing":timings,"manual_review":review},ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__": main()
