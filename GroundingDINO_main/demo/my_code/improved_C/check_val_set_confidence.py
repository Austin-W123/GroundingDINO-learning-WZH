# check_val_set_confidence.py
"""
检查验证集上各类别的置信度分布
"""
import os
import json
import numpy as np
from collections import defaultdict

BASE_DIR = "D:/groundingdino_work/GroundingDINO-main"
OUTPUT_DIR = os.path.join(BASE_DIR, "results")

def check_confidence_distribution():
    print("=" * 60)
    print("检查验证集上各类别的置信度分布")
    print("=" * 60)
    
    # 加载原始预测结果
    raw_path = os.path.join(OUTPUT_DIR, 'raw_predictions_val.json')
    with open(raw_path, 'r') as f:
        raw_data = json.load(f)
    
    raw_dets = raw_data['detections']
    print(f"总检测框数: {len(raw_dets)}")
    
    # 按类别分组
    cls_dets = defaultdict(list)
    for det in raw_dets:
        cls_dets[det['category_id']].append(det)
    
    print(f"\n各类别最高置信度:")
    print("-" * 50)
    print(f"{'ID':<5} {'类别名':<15} {'框数':<8} {'最高置信度':<12} {'≥0.2框数'}")
    print("-" * 50)
    
    # 类别名称映射（简化版）
    class_names = {
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
        77: "cell phone", 78: "microwave", 79: "oven", 80: "toaster",
        87: "scissors", 88: "teddy bear", 89: "hair drier", 90: "toothbrush"
    }
    
    for cls_id, dets in sorted(cls_dets.items()):
        if cls_id not in class_names:
            continue
            
        scores = [d['score'] for d in dets]
        max_score = max(scores)
        count_ge_02 = len([s for s in scores if s >= 0.2])
        
        status = "✅" if max_score >= 0.25 else "⚠️" if max_score >= 0.2 else "❌"
        
        print(f"{cls_id:<5} {class_names[cls_id]:<15} {len(dets):<8} "
              f"{max_score:.4f}       {count_ge_02:<8} {status}")
    
    print("-" * 50)

if __name__ == "__main__":
    check_confidence_distribution()