# step5_evaluate_improved.py
"""
第5步：对比评测 - 基线与改进版对比
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate

# 配置
BASE_DIR = "D:/groundingdino_work/GroundingDINO-main"
ANNO_PATH = os.path.join(BASE_DIR, "data/coco/annotations/instances_val2017.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")

# 文件路径
BASELINE_SEEN = os.path.join(OUTPUT_DIR, "coco_seen_400imgs_prompt1.json")
BASELINE_UNSEEN = os.path.join(OUTPUT_DIR, "coco_unseen_100imgs_prompt1.json")
IMPROVED_SEEN = os.path.join(OUTPUT_DIR, "coco_seen_400imgs_improved_C.json")
IMPROVED_UNSEEN = os.path.join(OUTPUT_DIR, "coco_unseen_100imgs_improved_C.json")

# 类别划分
SEEN_CLS_IDS = {
    1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,
    21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,
    40,41,42,43,44,46,47,48,49,50,51,62,63,64,65,67,
    70,72,73,74,75,76,77,78,79,80,87,88,89,90
}

UNSEEN_CLS_IDS = {52,53,54,55,56,57,58,59,60,61,81,82,84,85,86}

# 类别名称
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

def evaluate_results(coco_gt, result_file, cat_ids, name):
    """评估检测结果"""
    print(f"\n评估 {name}...")
    
    if not os.path.exists(result_file):
        print(f"  ⚠️ 文件不存在: {result_file}")
        return None
    
    with open(result_file, 'r') as f:
        results = json.load(f)
    
    print(f"  检测框数: {len(results)}")
    
    if len(results) == 0:
        print("  ⚠️ 无检测框")
        return None
    
    try:
        coco_dt = coco_gt.loadRes(results)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.params.catIds = list(cat_ids)
        coco_eval.params.imgIds = list(set([d['image_id'] for d in results]))
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        metrics = {
            'AP@[0.5:0.95]': coco_eval.stats[0],
            'AP@0.5': coco_eval.stats[1],
            'AP@0.75': coco_eval.stats[2],
            'AP_small': coco_eval.stats[3],
            'AP_medium': coco_eval.stats[4],
            'AP_large': coco_eval.stats[5],
            'AR@1': coco_eval.stats[6],
            'AR@10': coco_eval.stats[7],
            'AR@100': coco_eval.stats[8]
        }
        
        return metrics
    
    except Exception as e:
        print(f"  评估出错: {e}")
        return None

def plot_comparison(baseline_metrics, improved_metrics, title):
    """绘制对比柱状图（图表文字为英文）"""
    metrics_names = ['AP@0.5', 'AP@[0.5:0.95]', 'AR@100']
    baseline_vals = [baseline_metrics.get(m, 0) for m in metrics_names]
    improved_vals = [improved_metrics.get(m, 0) for m in metrics_names]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline (Fixed Threshold 0.25)', color='#2E86AB')
    bars2 = ax.bar(x + width/2, improved_vals, width, label='Improved C (Adaptive Threshold)', color='#A23B72')
    
    ax.set_title(f'Baseline vs Improved - {title}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Evaluation Metrics', fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(max(baseline_vals), max(improved_vals)) * 1.2)
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    return fig

def main():
    print("=" * 70)
    print("第5步：基线与改进版对比评测")
    print("=" * 70)
    
    # 加载COCO GT
    print("\n加载COCO标注...")
    coco_gt = COCO(ANNO_PATH)
    
    # 评估基线
    print("\n" + "=" * 50)
    print("基线结果评估")
    print("=" * 50)
    
    baseline_seen = evaluate_results(coco_gt, BASELINE_SEEN, SEEN_CLS_IDS, "基线-SEEN")
    baseline_unseen = evaluate_results(coco_gt, BASELINE_UNSEEN, UNSEEN_CLS_IDS, "基线-UNSEEN")
    
    # 评估改进版
    print("\n" + "=" * 50)
    print("改进版结果评估")
    print("=" * 50)
    
    improved_seen = evaluate_results(coco_gt, IMPROVED_SEEN, SEEN_CLS_IDS, "改进版-SEEN")
    improved_unseen = evaluate_results(coco_gt, IMPROVED_UNSEEN, UNSEEN_CLS_IDS, "改进版-UNSEEN")
    
    # 生成对比报告
    print("\n" + "=" * 70)
    print("对比报告")
    print("=" * 70)
    
    if baseline_unseen and improved_unseen:
        # 表格数据
        table_data = [
            ['UNSEEN AP@0.5', 
             f"{baseline_unseen['AP@0.5']:.4f}", 
             f"{improved_unseen['AP@0.5']:.4f}",
             f"{improved_unseen['AP@0.5'] - baseline_unseen['AP@0.5']:+.4f}"],
            ['UNSEEN AP@[0.5:0.95]', 
             f"{baseline_unseen['AP@[0.5:0.95]']:.4f}", 
             f"{improved_unseen['AP@[0.5:0.95]']:.4f}",
             f"{improved_unseen['AP@[0.5:0.95]'] - baseline_unseen['AP@[0.5:0.95]']:+.4f}"],
            ['UNSEEN AR@100', 
             f"{baseline_unseen['AR@100']:.4f}", 
             f"{improved_unseen['AR@100']:.4f}",
             f"{improved_unseen['AR@100'] - baseline_unseen['AR@100']:+.4f}"]
        ]
        
        print("\n📊 UNSEEN类别性能对比:")
        print(tabulate(table_data, 
                      headers=['指标', '基线', '改进版', '提升'],
                      tablefmt='grid'))
        
        # 绘制对比图（图表文字为英文）
        fig = plot_comparison(baseline_unseen, improved_unseen, 'UNSEEN Classes')
        fig.savefig(os.path.join(OUTPUT_DIR, 'comparison_unseen.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"\n✅ 对比图已保存: {os.path.join(OUTPUT_DIR, 'comparison_unseen.png')}")
    
    # 保存完整报告
    report_path = os.path.join(OUTPUT_DIR, 'improvement_C_comparison_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("小幅改进C - 基线与改进版对比报告\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("改进动机:\n")
        f.write("  不同类别的最优置信度阈值不同，统一阈值无法平衡所有类别。\n")
        f.write("  通过为每个类别设置独立阈值，可以更好地平衡精确率和召回率。\n\n")
        
        f.write("实现细节:\n")
        f.write("  1. 从SEEN类别中划分20%作为验证集\n")
        f.write("  2. 用极低阈值(0.01)收集原始预测结果\n")
        f.write("  3. 对每个类别扫描0.05-0.5的阈值，选择F1最高的作为最优阈值\n")
        f.write("  4. 用优化后的阈值进行正式推理\n\n")
        
        if baseline_unseen and improved_unseen:
            f.write("UNSEEN类别性能对比:\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'指标':<20} {'基线':<15} {'改进版':<15} {'提升':<15}\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'AP@0.5':<20} {baseline_unseen['AP@0.5']:<15.4f} "
                   f"{improved_unseen['AP@0.5']:<15.4f} "
                   f"{improved_unseen['AP@0.5'] - baseline_unseen['AP@0.5']:+.4f}\n")
            f.write(f"{'AP@[0.5:0.95]':<20} {baseline_unseen['AP@[0.5:0.95]']:<15.4f} "
                   f"{improved_unseen['AP@[0.5:0.95]']:<15.4f} "
                   f"{improved_unseen['AP@[0.5:0.95]'] - baseline_unseen['AP@[0.5:0.95]']:+.4f}\n")
            f.write(f"{'AR@100':<20} {baseline_unseen['AR@100']:<15.4f} "
                   f"{improved_unseen['AR@100']:<15.4f} "
                   f"{improved_unseen['AR@100'] - baseline_unseen['AR@100']:+.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        if improved_unseen and baseline_unseen and improved_unseen['AP@0.5'] > baseline_unseen['AP@0.5']:
            f.write("✅ 结论：自适应阈值策略有效，提升了UNSEEN类别检测性能\n")
        else:
            f.write("⚠️ 结论：自适应阈值策略未带来明显提升，需要进一步分析原因\n")
    
    print(f"\n✅ 对比报告已保存: {report_path}")

if __name__ == "__main__":
    main()