# check_optimizable_classes.py
"""
快速检查哪些类别会被优化（AP>0.1）
"""
import os
import json
import numpy as np
from collections import defaultdict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

BASE_DIR = "D:/groundingdino_work/GroundingDINO-main"
ANNO_PATH = os.path.join(BASE_DIR, "data/coco/annotations/instances_val2017.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")

CLASS_NAMES = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep",
    21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe",
    27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase",
    34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite",
    39: "baseball bat", 40: "baseball glove", 41: "skateboard", 42: "surfboard",
    43: "tennis racket", 44: "bottle", 46: "wine glass", 47: "cup", 48: "fork",
    49: "knife", 50: "spoon", 51: "bowl", 62: "chair", 63: "couch",
    64: "potted plant", 65: "bed", 67: "dining table", 70: "toilet",
    72: "tv", 73: "laptop", 74: "mouse", 75: "remote", 76: "keyboard",
    77: "cell phone", 78: "microwave", 79: "oven", 80: "toaster", 81: "sink",
    82: "refrigerator", 84: "book", 85: "clock", 86: "vase", 87: "scissors",
    88: "teddy bear", 89: "hair drier", 90: "toothbrush"
}

def evaluate_at_025(coco_gt, raw_dets, cls_id):
    """评估在阈值0.25下的AP"""
    filtered_dets = [d for d in raw_dets if d['score'] >= 0.25]
    
    if len(filtered_dets) == 0:
        return 0.0
    
    try:
        coco_dt = coco_gt.loadRes(filtered_dets)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.params.catIds = [cls_id]
        coco_eval.evaluate()
        coco_eval.accumulate()
        
        prec = coco_eval.eval['precision'][0, :, 0, 0, 2]
        prec = prec[prec > -1]
        
        if len(prec) > 0:
            return float(np.mean(prec))  # 确保返回Python float
        return 0.0
    except Exception as e:
        print(f"  评估类别 {cls_id} 时出错: {e}")
        return 0.0

def main():
    print("=" * 60)
    print("检查可优化的类别（AP>0.1）")
    print("=" * 60)
    
    # 检查文件是否存在
    raw_path = os.path.join(OUTPUT_DIR, 'raw_predictions_val.json')
    if not os.path.exists(raw_path):
        print(f"❌ 文件不存在: {raw_path}")
        print("请先运行 step2_collect_raw_predictions.py")
        return
    
    # 加载数据
    print("\n加载COCO标注...")
    coco_gt = COCO(ANNO_PATH)
    
    print("加载原始预测结果...")
    with open(raw_path, 'r') as f:
        raw_data = json.load(f)
    
    raw_dets = raw_data['detections']
    print(f"原始检测框总数: {len(raw_dets)}")
    
    # 按类别分组
    cls_dets = defaultdict(list)
    for det in raw_dets:
        cls_dets[det['category_id']].append(det)
    
    print(f"有检测结果的类别数: {len(cls_dets)}")
    
    print("\n" + "=" * 60)
    print("各类别在阈值0.25下的AP:")
    print("=" * 60)
    print(f"{'ID':<5} {'类别':<15} {'框数':<10} {'AP@0.25':<10} {'可优化'}")
    print("-" * 55)
    
    optimizable = []
    not_optimizable = []
    
    # 按类别ID排序
    for cls_id in sorted(CLASS_NAMES.keys()):
        if cls_id not in cls_dets:
            print(f"{cls_id:<5} {CLASS_NAMES[cls_id]:<15} {'0':<10} {'0.0000':<10}    {'❌ (无检测)'}")
            not_optimizable.append((cls_id, CLASS_NAMES[cls_id], 0.0))
            continue
            
        dets = cls_dets[cls_id]
        cls_name = CLASS_NAMES[cls_id]
        ap = evaluate_at_025(coco_gt, dets, cls_id)
        
        can_optimize = ap > 0.1
        status = "✅" if can_optimize else "❌"
        
        print(f"{cls_id:<5} {cls_name:<15} {len(dets):<10} {ap:.4f}    {status}")
        
        if can_optimize:
            optimizable.append((cls_id, cls_name, ap))
        else:
            not_optimizable.append((cls_id, cls_name, ap))
    
    print("-" * 55)
    
    # 统计结果
    print(f"\n📊 统计结果:")
    print(f"  总类别数: {len(CLASS_NAMES)}")
    print(f"  有检测结果的类别数: {len(cls_dets)}")
    print(f"  可优化类别数 (AP>0.1): {len(optimizable)}")
    print(f"  不可优化类别数: {len(not_optimizable)}")
    
    if optimizable:
        print(f"\n✅ 可优化的类别 (将进行阈值优化):")
        # 按AP从高到低排序
        optimizable.sort(key=lambda x: x[2], reverse=True)
        for cls_id, cls_name, ap in optimizable:
            print(f"  {cls_name} (ID:{cls_id}): AP={ap:.4f}")
    
    if not_optimizable:
        print(f"\n❌ 不可优化的类别 (将保持默认阈值0.25):")
        # 按AP从高到低排序，显示前10个
        not_optimizable.sort(key=lambda x: x[2], reverse=True)
        for cls_id, cls_name, ap in not_optimizable[:15]:
            if ap > 0:
                print(f"  {cls_name} (ID:{cls_id}): AP={ap:.4f}")
            else:
                print(f"  {cls_name} (ID:{cls_id}): AP=0.0000")
        if len(not_optimizable) > 15:
            print(f"  ... 共 {len(not_optimizable)} 个")
    
    # 建议
    print("\n" + "=" * 60)
    print("建议:")
    if len(optimizable) == 0:
        print("⚠️ 没有类别达到AP>0.1的标准，建议降低 AP_THRESHOLD 到 0.05")
    elif len(optimizable) < 5:
        print(f"ℹ️ 只有 {len(optimizable)} 个类别可优化，可以接受")
    else:
        print(f"✅ 有 {len(optimizable)} 个类别可优化，运行方案A会有较好效果")

if __name__ == "__main__":
    main()