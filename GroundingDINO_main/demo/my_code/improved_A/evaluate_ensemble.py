# evaluate_ensemble.py
"""
对比评测：基线(prompt1) vs 集成结果
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

# ===== 修改点：使用集成结果文件 =====
BASELINE_SEEN = os.path.join(OUTPUT_DIR, "coco_seen_400imgs_prompt1.json")
BASELINE_UNSEEN = os.path.join(OUTPUT_DIR, "coco_unseen_100imgs_prompt1.json")
ENSEMBLE_SEEN = os.path.join(OUTPUT_DIR, "coco_seen_400imgs_ensemble.json")  # 改为集成结果
ENSEMBLE_UNSEEN = os.path.join(OUTPUT_DIR, "coco_unseen_100imgs_ensemble.json")  # 改为集成结果

# 类别划分
SEEN_CLS_IDS = {
    1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,
    21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,
    40,41,42,43,44,46,47,48,49,50,51,62,63,64,65,67,
    70,72,73,74,75,76,77,78,79,80,87,88,89,90
}

UNSEEN_CLS_IDS = {52,53,54,55,56,57,58,59,60,61,81,82,84,85,86}

# 类别名称（同上，省略）

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

def plot_comparison(baseline_metrics, ensemble_metrics, title):
    """绘制对比柱状图"""
    metrics_names = ['AP@0.5', 'AP@[0.5:0.95]', 'AR@100']
    baseline_vals = [baseline_metrics.get(m, 0) for m in metrics_names]
    ensemble_vals = [ensemble_metrics.get(m, 0) for m in metrics_names]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline (prompt1)', color='#2E86AB')
    bars2 = ax.bar(x + width/2, ensemble_vals, width, label='Ensemble (3 prompts)', color='#A23B72')
    
    ax.set_title(f'Baseline vs Ensemble - {title}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Evaluation Metrics', fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(max(baseline_vals), max(ensemble_vals)) * 1.2)
    
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
    print("评测：基线(prompt1) vs 集成结果(3 prompts)")
    print("=" * 70)
    
    # 加载COCO GT
    print("\n加载COCO标注...")
    coco_gt = COCO(ANNO_PATH)
    
    # 评估基线
    print("\n" + "=" * 50)
    print("基线结果评估 (prompt1)")
    print("=" * 50)
    
    baseline_seen = evaluate_results(coco_gt, BASELINE_SEEN, SEEN_CLS_IDS, "基线-SEEN")
    baseline_unseen = evaluate_results(coco_gt, BASELINE_UNSEEN, UNSEEN_CLS_IDS, "基线-UNSEEN")
    
    # 评估集成结果
    print("\n" + "=" * 50)
    print("集成结果评估 (3 prompts)")
    print("=" * 50)
    
    ensemble_seen = evaluate_results(coco_gt, ENSEMBLE_SEEN, SEEN_CLS_IDS, "集成-SEEN")
    ensemble_unseen = evaluate_results(coco_gt, ENSEMBLE_UNSEEN, UNSEEN_CLS_IDS, "集成-UNSEEN")
    
    # 生成对比报告
    print("\n" + "=" * 70)
    print("对比报告")
    print("=" * 70)
    
    if baseline_unseen and ensemble_unseen:
        # 表格数据
        table_data = [
            ['UNSEEN AP@0.5', 
             f"{baseline_unseen['AP@0.5']:.4f}", 
             f"{ensemble_unseen['AP@0.5']:.4f}",
             f"{ensemble_unseen['AP@0.5'] - baseline_unseen['AP@0.5']:+.4f}"],
            ['UNSEEN AP@[0.5:0.95]', 
             f"{baseline_unseen['AP@[0.5:0.95]']:.4f}", 
             f"{ensemble_unseen['AP@[0.5:0.95]']:.4f}",
             f"{ensemble_unseen['AP@[0.5:0.95]'] - baseline_unseen['AP@[0.5:0.95]']:+.4f}"],
            ['UNSEEN AR@100', 
             f"{baseline_unseen['AR@100']:.4f}", 
             f"{ensemble_unseen['AR@100']:.4f}",
             f"{ensemble_unseen['AR@100'] - baseline_unseen['AR@100']:+.4f}"]
        ]
        
        print("\n📊 UNSEEN类别性能对比:")
        print(tabulate(table_data, 
                      headers=['指标', '基线(prompt1)', '集成(3 prompts)', '提升'],
                      tablefmt='grid'))
        
        # 绘制对比图
        fig = plot_comparison(baseline_unseen, ensemble_unseen, 'UNSEEN Classes')
        fig.savefig(os.path.join(OUTPUT_DIR, 'comparison_ensemble.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"\n✅ 对比图已保存: {os.path.join(OUTPUT_DIR, 'comparison_ensemble.png')}")
    
    # 保存完整报告
    report_path = os.path.join(OUTPUT_DIR, 'ensemble_comparison_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("方向A：多提示集成 - 基线与集成结果对比报告\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("改进动机:\n")
        f.write("  不同prompt可能擅长检测不同类别的目标，集成可以取长补短。\n\n")
        
        f.write("实现细节:\n")
        f.write("  1. 使用三轮prompt实验的结果：prompt1(纯类名)、prompt2(模板句)、prompt3(细粒度描述)\n")
        f.write("  2. 对同一个(图片,类别,位置)的检测框，取三个prompt中的最高置信度\n")
        f.write("  3. 保留所有不重复的检测框\n\n")
        
        if baseline_unseen and ensemble_unseen:
            f.write("UNSEEN类别性能对比:\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'指标':<20} {'基线(prompt1)':<15} {'集成(3 prompts)':<15} {'提升':<15}\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'AP@0.5':<20} {baseline_unseen['AP@0.5']:<15.4f} "
                   f"{ensemble_unseen['AP@0.5']:<15.4f} "
                   f"{ensemble_unseen['AP@0.5'] - baseline_unseen['AP@0.5']:+.4f}\n")
            f.write(f"{'AP@[0.5:0.95]':<20} {baseline_unseen['AP@[0.5:0.95]']:<15.4f} "
                   f"{ensemble_unseen['AP@[0.5:0.95]']:<15.4f} "
                   f"{ensemble_unseen['AP@[0.5:0.95]'] - baseline_unseen['AP@[0.5:0.95]']:+.4f}\n")
            f.write(f"{'AR@100':<20} {baseline_unseen['AR@100']:<15.4f} "
                   f"{ensemble_unseen['AR@100']:<15.4f} "
                   f"{ensemble_unseen['AR@100'] - baseline_unseen['AR@100']:+.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        if ensemble_unseen and baseline_unseen and ensemble_unseen['AP@0.5'] > baseline_unseen['AP@0.5']:
            f.write("✅ 结论：多提示集成策略有效，提升了UNSEEN类别检测性能\n")
        else:
            f.write("⚠️ 结论：多提示集成策略未带来明显提升，需要进一步分析原因\n")
    
    print(f"\n✅ 对比报告已保存: {report_path}")

if __name__ == "__main__":
    main()