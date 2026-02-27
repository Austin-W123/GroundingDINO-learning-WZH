"""
最优阈值统计脚本（简化美观版）
从检测结果中为每个类别找到最优阈值
输出：清晰表格 + JSON文件
"""
import os
import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate  

# 配置
COCO_ANNO_PATH = "D:/groundingdino_work/GroundingDINO-main/data/coco/annotations/instances_val2017.json"
DETECTION_RESULT = "D:/groundingdino_work/GroundingDINO-main/results/coco_seen_400imgs_prompt1.json"
OUTPUT_DIR = "D:/groundingdino_work/GroundingDINO-main/results"
OUTPUT_THRESHOLDS = os.path.join(OUTPUT_DIR, "best_thresholds.json")
OUTPUT_REPORT = os.path.join(OUTPUT_DIR, "threshold_analysis.txt")

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
    49: "knife", 50: "spoon", 51: "bowl", 52: "banana", 53: "apple",
    54: "sandwich", 55: "orange", 56: "broccoli", 57: "carrot", 58: "hot dog",
    59: "pizza", 60: "donut", 61: "cake", 62: "chair", 63: "couch",
    64: "potted plant", 65: "bed", 67: "dining table", 70: "toilet",
    72: "tv", 73: "laptop", 74: "mouse", 75: "remote", 76: "keyboard",
    77: "cell phone", 78: "microwave", 79: "oven", 80: "toaster", 81: "sink",
    82: "refrigerator", 84: "book", 85: "clock", 86: "vase", 87: "scissors",
    88: "teddy bear", 89: "hair drier", 90: "toothbrush"
}

def analyze_thresholds():
    """分析每个类别的最优阈值"""
    print("=" * 80)
    print("最优阈值统计工具 v1.0")
    print("=" * 80)
    
    # 加载数据
    print("\n📂 加载数据中...")
    coco_gt = COCO(COCO_ANNO_PATH)
    with open(DETECTION_RESULT, 'r') as f:
        results = json.load(f)
    print(f"  检测结果文件: {DETECTION_RESULT}")
    print(f"  总检测框数: {len(results)}")
    
    # 按类别分组
    cls_results = {}
    for det in results:
        cls_id = det['category_id']
        if cls_id not in cls_results:
            cls_results[cls_id] = []
        cls_results[cls_id].append(det)
    
    print(f"  有检测结果的类别数: {len(cls_results)}")
    print("\n" + "=" * 80)
    
    # 存储结果
    best_thresholds = {}
    analysis_results = []
    
    # 对每个类别分析
    threshold_range = np.arange(0.1, 0.51, 0.05)
    
    for idx, (cls_id, dets) in enumerate(sorted(cls_results.items()), 1):
        cls_name = CLASS_NAMES.get(cls_id, f"Unknown({cls_id})")
        print(f"\n[{idx:2d}/{len(cls_results)}] 分析类别: {cls_name} (ID: {cls_id})")
        print(f"  检测框数量: {len(dets)}")
        
        best_f1 = 0
        best_thr = 0.25
        best_stats = None
        
        # 进度条
        print("  阈值扫描: ", end="")
        
        for thr in threshold_range:
            thr = round(thr, 2)
            filtered_dets = [d for d in dets if d['score'] >= thr]
            
            if len(filtered_dets) < 3:
                print(".", end="", flush=True)
                continue
            
            try:
                coco_dt = coco_gt.loadRes(filtered_dets)
                coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
                coco_eval.params.catIds = [cls_id]
                coco_eval.evaluate()
                coco_eval.accumulate()
                
                # 获取精确率和召回率
                prec = coco_eval.eval['precision'][0, :, 0, 0, 2]
                rec = coco_eval.eval['recall'][0, :, 0, 2]
                
                prec = prec[prec > -1]
                rec = rec[rec > -1]
                
                if len(prec) > 0 and len(rec) > 0:
                    avg_prec = np.mean(prec)
                    avg_rec = np.mean(rec)
                    
                    if avg_prec + avg_rec > 0:
                        f1 = 2 * avg_prec * avg_rec / (avg_prec + avg_rec)
                        
                        if f1 > best_f1:
                            best_f1 = f1
                            best_thr = thr
                            best_stats = (avg_prec, avg_rec, f1)
                
                print("*", end="", flush=True)
            except:
                print(".", end="", flush=True)
        
        print(" ✓")
        
        # 记录最佳结果
        best_thresholds[str(cls_id)] = best_thr
        if best_stats:
            analysis_results.append([
                cls_id, 
                cls_name, 
                len(dets),
                best_thr,
                f"{best_stats[0]:.3f}",
                f"{best_stats[1]:.3f}",
                f"{best_stats[2]:.3f}"
            ])
            print(f"  ✅ 最优阈值: {best_thr:.2f} (Prec={best_stats[0]:.3f}, Rec={best_stats[1]:.3f}, F1={best_stats[2]:.3f})")
        else:
            analysis_results.append([cls_id, cls_name, len(dets), best_thr, "-", "-", "-"])
            print(f"  ⚠️ 默认阈值: {best_thr:.2f} (无有效数据)")
    
    # 保存JSON
    with open(OUTPUT_THRESHOLDS, 'w') as f:
        json.dump(best_thresholds, f, indent=2)
    print(f"\n✅ 阈值配置文件已保存: {OUTPUT_THRESHOLDS}")
    
    # 生成美观表格
    headers = ["ID", "类别名称", "检测框数", "最优阈值", "精确率", "召回率", "F1分数"]
    
    # 按阈值排序
    analysis_results.sort(key=lambda x: x[3] if isinstance(x[3], (int, float)) else 0, reverse=True)
    
    print("\n" + "=" * 80)
    print("📊 最优阈值统计结果 (按阈值从高到低排序)")
    print("=" * 80)
    print(tabulate(analysis_results, headers=headers, tablefmt="grid", numalign="center"))
    
    # 统计摘要
    thresholds = [r[3] for r in analysis_results if isinstance(r[3], (int, float))]
    print("\n📈 统计摘要:")
    print(f"  平均阈值: {np.mean(thresholds):.3f}")
    print(f"  中位数阈值: {np.median(thresholds):.3f}")
    print(f"  最小阈值: {np.min(thresholds):.3f}")
    print(f"  最大阈值: {np.max(thresholds):.3f}")
    
    # 保存文本报告
    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("最优阈值统计分析报告\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"检测结果文件: {DETECTION_RESULT}\n")
        f.write(f"总检测框数: {len(results)}\n")
        f.write(f"有检测结果的类别数: {len(cls_results)}\n\n")
        
        f.write(tabulate(analysis_results, headers=headers, tablefmt="grid", numalign="center"))
        
        f.write("\n\n📈 统计摘要:\n")
        f.write(f"  平均阈值: {np.mean(thresholds):.3f}\n")
        f.write(f"  中位数阈值: {np.median(thresholds):.3f}\n")
        f.write(f"  最小阈值: {np.min(thresholds):.3f}\n")
        f.write(f"  最大阈值: {np.max(thresholds):.3f}\n")
    
    print(f"\n✅ 详细报告已保存: {OUTPUT_REPORT}")
    print("=" * 80)
    print("\n📁 生成文件汇总:")
    print(f"  1. 阈值配置文件: {OUTPUT_THRESHOLDS}")
    print(f"  2. 分析报告: {OUTPUT_REPORT}")

if __name__ == "__main__":
    import os
    # 安装tabulate（如果没有的话）
    try:
        from tabulate import tabulate
    except ImportError:
        print("正在安装 tabulate...")
        import subprocess
        subprocess.check_call(["pip", "install", "tabulate"])
        from tabulate import tabulate
    
    analyze_thresholds()