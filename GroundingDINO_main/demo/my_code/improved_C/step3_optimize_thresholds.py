# step3_optimize_thresholds.py (修复版)
"""
第3步：基于验证集的原始预测结果，为每个类别统计最优阈值
方案C：根据每个类别的置信度分布动态选择阈值
"""
import os
import json
import numpy as np
from collections import defaultdict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate

# 配置
BASE_DIR = "D:/groundingdino_work/GroundingDINO-main"
ANNO_PATH = os.path.join(BASE_DIR, "data/coco/annotations/instances_val2017.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")

# 类别名称映射
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
    77: "cell phone", 78: "microwave", 79: "oven", 80: "toaster",
    87: "scissors", 88: "teddy bear", 89: "hair drier", 90: "toothbrush"
}

# SEEN类别ID
SEEN_CLS_IDS = list(CLASS_NAMES.keys())

# 配置参数
DEFAULT_THRESHOLD = 0.15  # 默认阈值
OPTIMIZE_THRESHOLD = 0.2  # 最高置信度大于此值才进行优化
MIN_BOXES = 5  # 最少检测框数

def evaluate_at_threshold(coco_gt, raw_dets, cls_id, threshold):
    """在指定阈值下评估某个类别的性能"""
    filtered_dets = [d for d in raw_dets if d['score'] >= threshold]
    
    if len(filtered_dets) == 0:
        return {'ap': 0.0, 'ar': 0.0, 'f1': 0.0}
    
    try:
        coco_dt = coco_gt.loadRes(filtered_dets)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.params.catIds = [cls_id]
        coco_eval.params.imgIds = list(set([d['image_id'] for d in filtered_dets]))
        coco_eval.evaluate()
        coco_eval.accumulate()
        
        prec = coco_eval.eval['precision'][0, :, 0, 0, 2]
        rec = coco_eval.eval['recall'][0, :, 0, 2]
        
        prec = prec[prec > -1]
        rec = rec[rec > -1]
        
        if len(prec) > 0 and len(rec) > 0:
            avg_prec = np.mean(prec)
            avg_rec = np.mean(rec)
            
            if avg_prec + avg_rec > 0:
                f1 = 2 * avg_prec * avg_rec / (avg_prec + avg_rec)
                return {
                    'ap': avg_prec,
                    'ar': avg_rec,
                    'f1': f1
                }
        
        return {'ap': 0.0, 'ar': 0.0, 'f1': 0.0}
    
    except Exception as e:
        print(f"  评估出错: {e}")
        return {'ap': 0.0, 'ar': 0.0, 'f1': 0.0}

def optimize_thresholds_dynamic():
    """动态阈值策略：根据每个类别的置信度分布自动选择"""
    print("=" * 60)
    print("第3步：动态阈值优化（方案C）")
    print("=" * 60)
    print(f"默认阈值: {DEFAULT_THRESHOLD}")
    print(f"优化阈值线: {OPTIMIZE_THRESHOLD}")
    print(f"最少框数: {MIN_BOXES}")
    print("=" * 60)
    
    # 加载COCO GT
    print("\n加载COCO标注...")
    coco_gt = COCO(ANNO_PATH)
    
    # 加载原始预测结果
    raw_path = os.path.join(OUTPUT_DIR, 'raw_predictions_val.json')
    if not os.path.exists(raw_path):
        print(f"❌ 文件不存在: {raw_path}")
        print("请先运行 step2_collect_raw_predictions.py")
        return
    
    with open(raw_path, 'r') as f:
        raw_data = json.load(f)
    
    raw_dets = raw_data['detections']
    print(f"原始检测框总数: {len(raw_dets)}")
    
    # 按类别分组
    cls_dets = defaultdict(list)
    for det in raw_dets:
        cls_dets[det['category_id']].append(det)
    
    print(f"有检测结果的类别数: {len(cls_dets)}")
    
    # 阈值扫描范围
    scan_thresholds = np.arange(0.05, 0.31, 0.02)  # 0.05到0.30，步长0.02
    
    optimal_thresholds = {}
    analysis_results = []
    
    # 统计信息
    optimize_count = 0
    medium_count = 0
    low_count = 0
    no_box_count = 0
    
    high_conf_classes = []  # 记录高置信度类别
    
    for cls_id in SEEN_CLS_IDS:
        cls_name = CLASS_NAMES.get(cls_id, f"Unknown-{cls_id}")
        dets = cls_dets.get(cls_id, [])
        
        print(f"\n处理类别: {cls_name} (ID: {cls_id})")
        print(f"  原始检测框数: {len(dets)}")
        
        # 情况1：没有检测框
        if len(dets) == 0:
            print(f"  ⚠️ 无检测框，使用默认阈值 {DEFAULT_THRESHOLD}")
            optimal_thresholds[str(cls_id)] = DEFAULT_THRESHOLD
            analysis_results.append([
                cls_id, cls_name, 0, DEFAULT_THRESHOLD, 
                0.0, 0.0, 0.0, "无检测框"
            ])
            no_box_count += 1
            continue
        
        # 计算该类别所有检测框的置信度
        scores = [d['score'] for d in dets]
        max_score = max(scores)
        mean_score = np.mean(scores)
        
        print(f"  最高置信度: {max_score:.4f}, 平均置信度: {mean_score:.4f}")
        
        # 情况2：检测框太少
        if len(dets) < MIN_BOXES:
            print(f"  ⚠️ 检测框太少 ({len(dets)} < {MIN_BOXES})，使用默认阈值 {DEFAULT_THRESHOLD}")
            optimal_thresholds[str(cls_id)] = DEFAULT_THRESHOLD
            analysis_results.append([
                cls_id, cls_name, len(dets), DEFAULT_THRESHOLD,
                0.0, 0.0, 0.0, "框太少"
            ])
            low_count += 1
            continue
        
        # 情况3：最高置信度很低 (<0.1)
        if max_score < 0.1:
            print(f"  ⚠️ 最高置信度 < 0.1，使用默认阈值 {DEFAULT_THRESHOLD}")
            optimal_thresholds[str(cls_id)] = DEFAULT_THRESHOLD
            analysis_results.append([
                cls_id, cls_name, len(dets), DEFAULT_THRESHOLD,
                0.0, 0.0, 0.0, "低置信度"
            ])
            low_count += 1
            continue
        
        # 情况4：最高置信度中等 (0.1-0.2)
        if max_score < OPTIMIZE_THRESHOLD:
            # 用0.1作为阈值，可能捕捉到一些有用框
            suggested_thr = 0.1
            print(f"  ℹ️ 最高置信度 {max_score:.3f} 在 0.1-0.2 之间，使用阈值 0.1")
            
            # 验证一下0.1的效果
            metrics_at_01 = evaluate_at_threshold(coco_gt, dets, cls_id, 0.1)
            
            optimal_thresholds[str(cls_id)] = 0.1
            analysis_results.append([
                cls_id, cls_name, len(dets), 0.1,
                metrics_at_01['ap'], metrics_at_01['ar'], metrics_at_01['f1'],
                f"中等置信度(max={max_score:.3f})"
            ])
            medium_count += 1
            continue
        
        # 情况5：最高置信度高 (>0.2)，进行阈值优化
        print(f"  ✅ 最高置信度 > {OPTIMIZE_THRESHOLD}，进行阈值优化")
        print("  扫描阈值: ", end="")
        
        best_f1 = -1  # 初始化为-1，确保能找到更好的
        best_thr = DEFAULT_THRESHOLD
        best_metrics = {'ap': 0.0, 'ar': 0.0, 'f1': 0.0}  # 初始化为默认值
        
        for thr in scan_thresholds:
            thr = round(thr, 2)
            metrics = evaluate_at_threshold(coco_gt, dets, cls_id, thr)
            
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                best_thr = thr
                best_metrics = metrics
            
            print(".", end="", flush=True)
        
        print(f" ✓")
        
        # 确保best_metrics不为None
        if best_metrics is None:
            best_metrics = {'ap': 0.0, 'ar': 0.0, 'f1': 0.0}
        
        optimal_thresholds[str(cls_id)] = best_thr
        analysis_results.append([
            cls_id, cls_name, len(dets), best_thr,
            best_metrics['ap'], best_metrics['ar'], best_metrics['f1'],
            "优化"
        ])
        optimize_count += 1
        high_conf_classes.append(f"{cls_name}(max={max_score:.3f}->thr={best_thr:.2f}, AP={best_metrics['ap']:.3f})")
        
        print(f"  ✅ 最优阈值: {best_thr:.2f} "
              f"(AP={best_metrics['ap']:.3f}, "
              f"AR={best_metrics['ar']:.3f}, "
              f"F1={best_metrics['f1']:.3f})")
    
    # 保存最优阈值
    thresh_path = os.path.join(OUTPUT_DIR, 'optimal_thresholds.json')
    with open(thresh_path, 'w') as f:
        json.dump(optimal_thresholds, f, indent=2)
    
    print(f"\n✅ 最优阈值已保存: {thresh_path}")
    
    # 生成分析报告
    headers = ["ID", "类别", "框数", "阈值", "AP", "AR", "F1", "状态"]
    
    # 按状态排序
    def sort_key(x):
        if x[7] == "优化":
            return (0, -x[3] if isinstance(x[3], (int, float)) else 0)
        elif "中等" in x[7]:
            return (1, -x[3] if isinstance(x[3], (int, float)) else 0)
        else:
            return (2, -x[3] if isinstance(x[3], (int, float)) else 0)
    
    analysis_results.sort(key=sort_key)
    
    print("\n" + "=" * 80)
    print("📊 动态阈值优化结果")
    print("=" * 80)
    print(tabulate(analysis_results, headers=headers, tablefmt="grid", floatfmt=".3f"))
    
    # 统计摘要
    print(f"\n📈 统计摘要:")
    print(f"  优化类别 (>{OPTIMIZE_THRESHOLD}): {optimize_count}")
    print(f"  中等置信度类别 (0.1-0.2): {medium_count}")
    print(f"  低置信度类别 (<0.1/框少): {low_count}")
    print(f"  无检测框类别: {no_box_count}")
    
    if high_conf_classes:
        print(f"\n✅ 优化的类别:")
        for cls_info in high_conf_classes:
            print(f"  {cls_info}")
    
    # 保存分析报告
    report_path = os.path.join(OUTPUT_DIR, 'threshold_optimization_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("阈值优化分析报告 (方案C：动态阈值)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"默认阈值: {DEFAULT_THRESHOLD}\n")
        f.write(f"优化阈值线: {OPTIMIZE_THRESHOLD}\n")
        f.write(f"最少框数: {MIN_BOXES}\n\n")
        f.write(tabulate(analysis_results, headers=headers, tablefmt="grid", floatfmt=".3f"))
        f.write(f"\n\n统计摘要:\n")
        f.write(f"  优化类别: {optimize_count}\n")
        f.write(f"  中等置信度类别: {medium_count}\n")
        f.write(f"  低置信度类别: {low_count}\n")
        f.write(f"  无检测框类别: {no_box_count}\n")
    
    print(f"\n✅ 分析报告已保存: {report_path}")
    
    return optimal_thresholds

if __name__ == "__main__":
    # 确保tabulate已安装
    try:
        from tabulate import tabulate
    except ImportError:
        import subprocess
        subprocess.check_call(["pip", "install", "tabulate"])
        from tabulate import tabulate
    
    optimize_thresholds_dynamic()