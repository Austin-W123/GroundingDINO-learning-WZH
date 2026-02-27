# prompt_ensemble_with_nms.py
"""
多提示集成 + NMS后处理
1. 先用最大置信度融合（保留各位置最高分）
2. 再用NMS去除重叠框
"""
import os
import json
from collections import defaultdict

# 配置
BASE_DIR = "D:/groundingdino_work/GroundingDINO-main"
OUTPUT_DIR = os.path.join(BASE_DIR, "results")

# 三个prompt的结果文件
PROMPT_FILES = {
    'prompt1': {
        'seen': os.path.join(OUTPUT_DIR, 'coco_seen_400imgs_prompt1.json'),
        'unseen': os.path.join(OUTPUT_DIR, 'coco_unseen_100imgs_prompt1.json')
    },
    'prompt2': {
        'seen': os.path.join(OUTPUT_DIR, 'coco_seen_400imgs_prompt2.json'),
        'unseen': os.path.join(OUTPUT_DIR, 'coco_unseen_100imgs_prompt2.json')
    },
    'prompt3': {
        'seen': os.path.join(OUTPUT_DIR, 'coco_seen_400imgs_prompt3.json'),
        'unseen': os.path.join(OUTPUT_DIR, 'coco_unseen_100imgs_prompt3.json')
    }
}

# 输出文件
OUTPUT_SEEN = os.path.join(OUTPUT_DIR, 'coco_seen_400imgs_ensemble_nms.json')
OUTPUT_UNSEEN = os.path.join(OUTPUT_DIR, 'coco_unseen_100imgs_ensemble_nms.json')

def bbox_to_key(bbox):
    """将bbox转换为可哈希的key（用于第一阶段匹配）"""
    return tuple(int(round(x)) for x in bbox)

def compute_iou(bbox1, bbox2):
    """
    计算两个框的IOU
    bbox: [x1, y1, w, h]
    """
    # 转换为[x1,y1,x2,y2]格式
    x1_1, y1_1, w1, h1 = bbox1
    x2_1, y2_1 = x1_1 + w1, y1_1 + h1
    
    x1_2, y1_2, w2, h2 = bbox2
    x2_2, y2_2 = x1_2 + w2, y1_2 + h2
    
    # 交集
    xx1 = max(x1_1, x1_2)
    yy1 = max(y1_1, y1_2)
    xx2 = min(x2_1, x2_2)
    yy2 = min(y2_1, y2_2)
    
    w = max(0, xx2 - xx1)
    h = max(0, yy2 - yy1)
    inter = w * h
    
    # 并集
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0

def nms_per_class(detections, iou_threshold=0.5):
    """
    对单个类别的检测框做NMS
    输入: 同一图片、同一类别的检测框列表
    输出: NMS后的检测框列表
    """
    if len(detections) <= 1:
        return detections
    
    # 按置信度排序
    detections = sorted(detections, key=lambda x: x['score'], reverse=True)
    
    keep = []
    while detections:
        # 取出得分最高的框
        best = detections.pop(0)
        keep.append(best)
        
        # 过滤掉与best重叠过大的框
        remaining = []
        for det in detections:
            iou = compute_iou(best['bbox'], det['bbox'])
            if iou < iou_threshold:  # 不重叠的保留
                remaining.append(det)
            # 重叠的删除（不加入remaining）
        
        detections = remaining
    
    return keep

def max_confidence_fusion(detections_list):
    """
    第一阶段：最大置信度融合
    对精确匹配的框取最高分
    """
    ensemble_dict = {}
    
    for prompt_idx, detections in enumerate(detections_list):
        for det in detections:
            bbox_key = bbox_to_key(det['bbox'])
            key = (det['image_id'], det['category_id'], bbox_key)
            
            if key not in ensemble_dict:
                ensemble_dict[key] = det.copy()
                ensemble_dict[key]['source_prompt'] = prompt_idx
            else:
                if det['score'] > ensemble_dict[key]['score']:
                    ensemble_dict[key] = det.copy()
                    ensemble_dict[key]['source_prompt'] = prompt_idx
    
    return list(ensemble_dict.values())

def apply_nms(detections, iou_threshold=0.5):
    """
    第二阶段：对融合结果应用NMS
    按图片和类别分组，分别做NMS
    """
    # 按图片和类别分组
    grouped = defaultdict(list)
    for det in detections:
        key = (det['image_id'], det['category_id'])
        grouped[key].append(det)
    
    # 对每组做NMS
    nms_results = []
    for (img_id, cat_id), group in grouped.items():
        nms_group = nms_per_class(group, iou_threshold)
        nms_results.extend(nms_group)
    
    return nms_results

def main():
    print("=" * 60)
    print("方向A：多提示集成 + NMS后处理")
    print("=" * 60)
    
    # 检查文件
    all_exist = True
    for prompt_name, files in PROMPT_FILES.items():
        for type_name, file_path in files.items():
            if not os.path.exists(file_path):
                print(f"❌ 文件不存在: {file_path}")
                all_exist = False
    
    if not all_exist:
        print("\n请先完成三轮prompt实验，生成所有结果文件")
        return
    
    # 加载所有结果
    print("\n📂 加载检测结果...")
    
    seen_dets_list = []
    unseen_dets_list = []
    
    for prompt_name, files in PROMPT_FILES.items():
        print(f"\n加载 {prompt_name}:")
        
        with open(files['seen'], 'r') as f:
            seen_dets = json.load(f)
            seen_dets_list.append(seen_dets)
            print(f"  SEEN: {len(seen_dets)} 框")
        
        with open(files['unseen'], 'r') as f:
            unseen_dets = json.load(f)
            unseen_dets_list.append(unseen_dets)
            print(f"  UNSEEN: {len(unseen_dets)} 框")
    
    # 第一阶段：最大置信度融合
    print("\n🔄 第一阶段：最大置信度融合...")
    
    ensemble_seen = max_confidence_fusion(seen_dets_list)
    ensemble_unseen = max_confidence_fusion(unseen_dets_list)
    
    print(f"  SEEN融合后: {len(ensemble_seen)} 框")
    print(f"  UNSEEN融合后: {len(ensemble_unseen)} 框")
    
    # 第二阶段：NMS后处理
    print("\n🔄 第二阶段：NMS去重 (IOU阈值=0.5)...")
    
    nms_seen = apply_nms(ensemble_seen, iou_threshold=0.5)
    nms_unseen = apply_nms(ensemble_unseen, iou_threshold=0.5)
    
    print(f"  SEEN: {len(ensemble_seen)} → {len(nms_seen)} 框 (减少{len(ensemble_seen)-len(nms_seen)}个)")
    print(f"  UNSEEN: {len(ensemble_unseen)} → {len(nms_unseen)} 框 (减少{len(ensemble_unseen)-len(nms_unseen)}个)")
    
    # 保存结果
    print("\n💾 保存最终结果...")
    
    with open(OUTPUT_SEEN, 'w') as f:
        json.dump(nms_seen, f, indent=2)
    print(f"  ✅ SEEN结果: {OUTPUT_SEEN}")
    
    with open(OUTPUT_UNSEEN, 'w') as f:
        json.dump(nms_unseen, f, indent=2)
    print(f"  ✅ UNSEEN结果: {OUTPUT_UNSEEN}")
    
    # 统计
    print("\n📊 各阶段框数对比 (UNSEEN):")
    total_raw = sum(len(d) for d in unseen_dets_list)
    print(f"  原始总框数: {total_raw}")
    print(f"  融合后: {len(ensemble_unseen)} ({len(ensemble_unseen)/total_raw*100:.1f}%)")
    print(f"  NMS后: {len(nms_unseen)} ({len(nms_unseen)/total_raw*100:.1f}%)")
    print(f"  最终/原始比例: {len(nms_unseen)/total_raw*100:.1f}%")
    
    print("\n" + "=" * 60)
    print("✅ 融合完成！")
    print("=" * 60)
    print("\n下一步：运行评测脚本对比效果")
    print("  python scripts/step5_evaluate_ensemble_nms.py")

if __name__ == "__main__":
    main()