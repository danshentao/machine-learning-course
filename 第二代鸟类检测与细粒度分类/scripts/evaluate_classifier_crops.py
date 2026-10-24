"""人工框鸟框裁剪分类诊断；该值不代表检测器预测框下端到端效果。"""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np,paddle
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from train_classifier import BirdClassifier,CropSet,read_rows,WEIGHTS
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--weights',type=Path,required=True);ap.add_argument('--classifier',choices=['resnet18','bilinear'],required=True);ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--include-pending-review',action='store_true');a=ap.parse_args()
 paddle.set_device('gpu' if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() else 'cpu')
 rows=read_rows(ROOT/'reports/classification_crop_manifest.csv','val')
 if not a.include_pending_review:rows=[r for r in rows if r.get('review_pending')!='True']
 ds=CropSet(rows,False,224);dl=paddle.io.DataLoader(ds,batch_size=16,shuffle=False,num_workers=0)
 model=BirdClassifier(a.classifier=='bilinear',WEIGHTS/'resnet18.pdparams');model.set_state_dict(paddle.load(str(a.weights)));model.eval()
 classes=json.loads((ROOT/'configs/class_mapping.json').read_text(encoding='utf-8'))['classes'];names=[x['name'] for x in classes];cm=np.zeros((8,8),dtype=np.int64);correct=total=0
 with paddle.no_grad():
  for x,y in dl:
   pred=model(x).argmax(axis=1).numpy();truth=y.numpy()
   for t,p in zip(truth,pred):cm[int(t),int(p)]+=1;correct+=int(t==p);total+=1
 a.output_dir.mkdir(parents=True,exist_ok=True);doc={'diagnostic':'人工框裁剪分類結果，不包含檢測漏檢或定位誤差；不得替代正式整圖端到端評估','classifier':a.classifier,'checkpoint':str(a.weights),'validation_crops':total,'accuracy':correct/total if total else None,'pending_review_crops_included':a.include_pending_review,'confusion_matrix_labels':names,'confusion_matrix':cm.tolist()}
 (a.output_dir/'classifier_crop_diagnostic.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2),encoding='utf-8')
 with (a.output_dir/'classifier_crop_confusion.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.writer(f);w.writerow(['true\\pred',*names]);
  for name,row in zip(names,cm):w.writerow([name,*row.tolist()])
 print(json.dumps({'validation_crops':total,'accuracy':doc['accuracy'],'pending_review_crops_included':a.include_pending_review},ensure_ascii=False))
if __name__=='__main__':main()
