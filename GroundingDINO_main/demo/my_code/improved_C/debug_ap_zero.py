# debug_ap_zero.py
"""
快速诊断为什么AP为0
"""
import os
import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

BASE_DIR = "D:/groundingdino_work/GroundingDINO-main"
ANNO_PATH = os.path.join(BASE_DIR, "data/coco/annotations/instances_val2017.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")

def debug_person_class():
    print("=" * 60)
    print("诊断person类AP=0的问题")
    print("=" * 60)
    
    # 加载数据
    print("\n1. 加载COCO标注...")
    coco_gt = COCO(ANNO_PATH)
    
    print("\n2. 加载原始预测结果...")
    raw_path = os.path.join(OUTPUT_DIR, 'raw_predictions_val.json')
    with open(raw_path, 'r') as f:
        raw_data = json.load(f)
    
    raw_dets = raw_data['detections']
    print(f"   总检测框数: {len(raw_dets)}")
    
    # 筛选person类 (ID=1)
    person_dets = [d for d in raw_dets if d['category_id'] == 1]
    print(f"\n3. person类检测框: {len(person_dets)}")
    
    # 按阈值筛选
    thr = 0.25
    filtered = [d for d in person_dets if d['score'] >= thr]
    print(f"   阈值{thr}以上的检测框: {len(filtered)}")
    
    if len(filtered) == 0:
        print("   ❌ 没有检测框达到阈值")
        return
    
    # 检查第一个检测框的格式
    print("\n4. 检查检测框格式:")
    sample = filtered[0]
    print(f"   image_id: {sample['image_id']} (类型: {type(sample['image_id'])})")
    print(f"   category_id: {sample['category_id']}")
    print(f"   bbox: {sample['bbox']}")
    print(f"   score: {sample['score']}")
    
    # 验证这个image_id是否存在于GT中
    print("\n5. 验证image_id:")
    gt_img_ids = set(coco_gt.getImgIds())
    if sample['image_id'] in gt_img_ids:
        print(f"   ✅ image_id {sample['image_id']} 存在于GT中")
    else:
        print(f"   ❌ image_id {sample['image_id']} 不在GT中")
    
    # 尝试手动评估
    print("\n6. 尝试COCO评估:")
    try:
        coco_dt = coco_gt.loadRes(filtered)
        print(f"   ✅ 成功加载检测结果")
        
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.params.catIds = [1]
        coco_eval.params.imgIds = list(set([d['image_id'] for d in filtered]))
        print(f"   评估图片数: {len(coco_eval.params.imgIds)}")
        
        coco_eval.evaluate()
        print(f"   ✅ evaluate() 成功")
        
        coco_eval.accumulate()
        print(f"   ✅ accumulate() 成功")
        
        # 查看precision数组
        prec = coco_eval.eval['precision']
        print(f"   precision数组形状: {prec.shape}")
        
        # 提取person类的precision
        person_prec = prec[0, :, 0, 0, 2]  # [iouThr, recall, catId, areaRng, maxDet]
        valid_prec = person_prec[person_prec > -1]
        print(f"   有效precision值个数: {len(valid_prec)}")
        
        if len(valid_prec) > 0:
            print(f"   precision值: {valid_prec[:10]}")  # 显示前10个
            print(f"   平均AP: {np.mean(valid_prec):.4f}")
        else:
            print("   ❌ 没有有效的precision值")
            
    except Exception as e:
        print(f"   ❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_person_class()