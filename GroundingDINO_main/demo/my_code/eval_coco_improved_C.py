# -*- coding: utf-8 -*-
"""
COCO检测结果评测脚本 - 小幅改进C版对比实验
计算SEEN/UNSEEN类COCO指标，并与基线prompt1进行对比
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# 配置
BASE_DIR = "D:/groundingdino_work/GroundingDINO-main"
ANNO_PATH = os.path.join(BASE_DIR, "data/coco/annotations/instances_val2017.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")

# Prompt配置
BASELINE_PROMPT = "prompt1"  # 基线版本
IMPROVED_PROMPT = "prompt1_improved_C"  # 改进版（自适应阈值）

PROMPT_NAME_MAP = {
    BASELINE_PROMPT: "Baseline (Uniform Threshold 0.25)",
    IMPROVED_PROMPT: "Improved C (Adaptive Threshold)"
}

# 文件路径
BASELINE_SEEN_PATH = os.path.join(OUTPUT_DIR, f"coco_seen_400imgs_{BASELINE_PROMPT}.json")
BASELINE_UNSEEN_PATH = os.path.join(OUTPUT_DIR, f"coco_unseen_100imgs_{BASELINE_PROMPT}.json")
IMPROVED_SEEN_PATH = os.path.join(OUTPUT_DIR, f"coco_seen_400imgs_{IMPROVED_PROMPT}.json")
IMPROVED_UNSEEN_PATH = os.path.join(OUTPUT_DIR, f"coco_unseen_100imgs_{IMPROVED_PROMPT}.json")

# 输出文件
COMPARISON_REPORT_TXT = os.path.join(OUTPUT_DIR, "improvement_C_comparison_report.txt")
COMPARISON_BAR_PNG = os.path.join(OUTPUT_DIR, "improvement_C_comparison_bar.png")
COMPARISON_HEATMAP_PNG = os.path.join(OUTPUT_DIR, "improvement_C_comparison_heatmap.png")
IMPROVEMENT_METRICS_PNG = os.path.join(OUTPUT_DIR, "improvement_C_metrics.png")

# 单轮可视化文件（用于改进版自身）
IMPROVED_METRICS_FULL_PNG = os.path.join(OUTPUT_DIR, "metrics_full_comparison_improved_C.png")
IMPROVED_SIZE_SENSITIVE_PNG = os.path.join(OUTPUT_DIR, "size_sensitive_metrics_improved_C.png")
IMPROVED_BBOX_ANALYSIS_PNG = os.path.join(OUTPUT_DIR, "bbox_analysis_improved_C.png")
IMPROVED_RECALL_CURVE_PNG = os.path.join(OUTPUT_DIR, "recall_curve_improved_C.png")

# SEEN: 65个类, UNSEEN: 15个类
SEEN_CLS_IDS = {1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80}
UNSEEN_CLS_IDS = {52,53,54,55,56,57,58,59,60,61,81,82,84,85,86}

UNSEEN_CLS_NAME = {
    52: "banana", 53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli",
    57: "carrot", 58: "hot dog", 59: "pizza", 60: "donut", 61: "cake",
    81: "sink", 82: "refrigerator", 84: "book", 85: "clock", 86: "vase"
}

def load_detection_results(file_path):
    """加载检测结果文件"""
    if not os.path.exists(file_path):
        print(f"⚠️ 文件不存在：{file_path}")
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    print(f"✅ 加载检测框：{file_path} | 检测框数：{len(results)}")
    return results

def evaluate_coco_metrics(cocoGt, results, cat_ids, eval_name):
    """
    计算COCO全量12项指标
    参数：
        cocoGt: COCO Ground Truth
        results: 检测框列表
        cat_ids: 要评估的类别ID列表
        eval_name: 评测名称（SEEN/UNSEEN）
    返回：
        metrics: 12项指标字典
    """
    print(f"\n===== 开始{eval_name}类别评测 =====")
    
    # 如果没有检测框，直接返回全0指标
    if len(results) == 0:
        print("⚠️ 没有检测框，返回全0指标")
        metrics = {
            "AP@[0.5:0.95]": 0.0,
            "AP@0.5": 0.0,
            "AP@0.75": 0.0,
            "AP_small": 0.0,
            "AP_medium": 0.0,
            "AP_large": 0.0,
            "AR@1": 0.0,
            "AR@10": 0.0,
            "AR@100": 0.0,
            "AR_small": 0.0,
            "AR_medium": 0.0,
            "AR_large": 0.0
        }
        return metrics
    
    try:
        cocoDt = cocoGt.loadRes(results)
        cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
        cocoEval.params.catIds = cat_ids
        cocoEval.params.imgIds = list(set([d['image_id'] for d in results]))
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        
        if hasattr(cocoEval, 'stats') and len(cocoEval.stats) >= 12:
            metrics = {
                "AP@[0.5:0.95]": round(cocoEval.stats[0], 4),
                "AP@0.5": round(cocoEval.stats[1], 4),
                "AP@0.75": round(cocoEval.stats[2], 4),
                "AP_small": round(cocoEval.stats[3], 4),
                "AP_medium": round(cocoEval.stats[4], 4),
                "AP_large": round(cocoEval.stats[5], 4),
                "AR@1": round(cocoEval.stats[6], 4),
                "AR@10": round(cocoEval.stats[7], 4),
                "AR@100": round(cocoEval.stats[8], 4),
                "AR_small": round(cocoEval.stats[9], 4),
                "AR_medium": round(cocoEval.stats[10], 4),
                "AR_large": round(cocoEval.stats[11], 4)
            }
        else:
            print("⚠️ cocoEval.stats 长度不足，返回全0指标")
            metrics = {k: 0.0 for k in ["AP@[0.5:0.95]", "AP@0.5", "AP@0.75", "AP_small", 
                                        "AP_medium", "AP_large", "AR@1", "AR@10", "AR@100",
                                        "AR_small", "AR_medium", "AR_large"]}
    except Exception as e:
        print(f"⚠️ COCO评测出错: {e}")
        metrics = {k: 0.0 for k in ["AP@[0.5:0.95]", "AP@0.5", "AP@0.75", "AP_small", 
                                    "AP_medium", "AP_large", "AR@1", "AR@10", "AR@100",
                                    "AR_small", "AR_medium", "AR_large"]}
    
    return metrics

def analyze_unseen_categories(unseen_results, cocoGt):
    """
    分析UNSEEN类别各指标的详细情况
    返回：{cls_id: {"AP@[0.5:0.95]": value, "AP@0.5": value}}
    """
    if len(unseen_results) == 0:
        return {}
    
    try:
        cocoDt = cocoGt.loadRes(unseen_results)
        cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
        cocoEval.params.catIds = list(UNSEEN_CLS_IDS)
        cocoEval.params.imgIds = list(set([d['image_id'] for d in unseen_results]))
        cocoEval.evaluate()
        cocoEval.accumulate()
        
        cls_metrics = {}
        precisions = cocoEval.eval['precision']
        
        for idx, cat_id in enumerate(cocoEval.params.catIds):
            if cat_id in UNSEEN_CLS_IDS:
                cat_precision = precisions[:, :, idx, 0, -1]
                if cat_precision.size > 0:
                    ap = np.mean(cat_precision[cat_precision > -1])
                    ap50 = np.mean(precisions[0, :, idx, 0, -1][precisions[0, :, idx, 0, -1] > -1])
                else:
                    ap, ap50 = 0.0, 0.0
                
                cls_metrics[cat_id] = {
                    "AP@[0.5:0.95]": round(ap, 4),
                    "AP@0.5": round(ap50, 4)
                }
        
        return cls_metrics
    except Exception as e:
        print(f"⚠️ 分析UNSEEN类别出错: {e}")
        return {}

def plot_improved_metrics(seen_metrics, unseen_metrics, title_suffix):
    """绘制改进版的指标对比图"""
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    
    metrics_names = ["AP@[0.5:0.95]", "AP@0.5", "AP@0.75", "AR@100"]
    seen_vals = [seen_metrics[m] for m in metrics_names]
    unseen_vals = [unseen_metrics[m] for m in metrics_names]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, seen_vals, width, label="SEEN (65 classes)", color="#2E86AB")
    ax.bar(x + width/2, unseen_vals, width, label="UNSEEN (15 classes)", color="#A23B72")
    
    ax.set_title(f"COCO Core Metrics Comparison ({title_suffix})", fontsize=14, fontweight="bold")
    ax.set_xlabel("Metric Name", fontsize=12)
    ax.set_ylabel("Metric Value", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    for i, v in enumerate(seen_vals):
        ax.text(i - width/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    for i, v in enumerate(unseen_vals):
        ax.text(i + width/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    
    plt.tight_layout()
    return fig

def plot_improved_size_sensitive(seen_metrics, unseen_metrics, title_suffix):
    """绘制改进版的尺寸敏感指标图"""
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    
    size_metrics = ["AP_small", "AP_medium", "AP_large"]
    seen_vals = [seen_metrics[m] for m in size_metrics]
    unseen_vals = [unseen_metrics[m] for m in size_metrics]
    
    x = np.arange(len(size_metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, seen_vals, width, label="SEEN", color="#F18F01")
    ax.bar(x + width/2, unseen_vals, width, label="UNSEEN", color="#C73E1D")
    
    ax.set_title(f"Size-sensitive AP Metrics ({title_suffix})", fontsize=14, fontweight="bold")
    ax.set_xlabel("Object Size", fontsize=12)
    ax.set_ylabel("AP Value", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(["Small", "Medium", "Large"])
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    for i, v in enumerate(seen_vals):
        ax.text(i - width/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    for i, v in enumerate(unseen_vals):
        ax.text(i + width/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    
    plt.tight_layout()
    return fig

def plot_improved_bbox_analysis(seen_results, unseen_results, title_suffix):
    """绘制改进版的检测框分析图"""
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    
    seen_scores = [det["score"] for det in seen_results]
    unseen_scores = [det["score"] for det in unseen_results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.hist(seen_scores, bins=20, alpha=0.7, label="SEEN", color="#2E86AB", edgecolor="black")
    ax1.hist(unseen_scores, bins=20, alpha=0.7, label="UNSEEN", color="#A23B72", edgecolor="black")
    ax1.set_title(f"Detection Score Distribution ({title_suffix})", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Detection Score", fontsize=10)
    ax1.set_ylabel("Count", fontsize=10)
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0.8, 1.0)
    
    unseen_cls_ids = [det["category_id"] for det in unseen_results 
                     if det["category_id"] in UNSEEN_CLS_NAME]
    unseen_cls_names = [UNSEEN_CLS_NAME[cid] for cid in unseen_cls_ids]
    
    cls_counts = {}
    for name in unseen_cls_names:
        cls_counts[name] = cls_counts.get(name, 0) + 1
    
    sorted_items = sorted(cls_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    cls_names = [item[0] for item in sorted_items]
    cls_vals = [item[1] for item in sorted_items]
    
    ax2.barh(cls_names, cls_vals, color="#F18F01", edgecolor="black")
    ax2.set_title("UNSEEN Category Detection Box Count (Top 10)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Detection Box Count", fontsize=10)
    ax2.set_ylabel("Category Name", fontsize=10)
    ax2.grid(axis="x", alpha=0.3)
    
    for i, v in enumerate(cls_vals):
        ax2.text(v + 1, i, f"{v}", va="center", fontsize=9, fontweight="bold")
    
    plt.tight_layout()
    return fig

def plot_improved_recall_curve(seen_metrics, unseen_metrics, title_suffix):
    """绘制改进版的召回率曲线"""
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    
    recall_metrics = ["AR@1", "AR@10", "AR@100"]
    seen_vals = [seen_metrics[m] for m in recall_metrics]
    unseen_vals = [unseen_metrics[m] for m in recall_metrics]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(recall_metrics, seen_vals, marker="o", linewidth=3, markersize=8, label="SEEN", color="#2E86AB")
    ax.plot(recall_metrics, unseen_vals, marker="s", linewidth=3, markersize=8, label="UNSEEN", color="#A23B72")
    
    ax_twin = ax.twinx()
    diffs = [s - u for s, u in zip(seen_vals, unseen_vals)]
    ax_twin.bar(recall_metrics, diffs, alpha=0.3, color="#F18F01", label="SEEN-UNSEEN Difference", edgecolor="black")
    
    ax.set_title(f"Recall Curve Comparison ({title_suffix})", fontsize=14, fontweight="bold")
    ax.set_xlabel("Recall Type", fontsize=12)
    ax.set_ylabel("AR Value", fontsize=12, color="#2E86AB")
    ax_twin.set_ylabel("Difference", fontsize=12, color="#F18F01")
    ax.legend(loc="upper left")
    ax_twin.legend(loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    for i, v in enumerate(seen_vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    for i, v in enumerate(unseen_vals):
        ax.text(i, v - 0.02, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    
    plt.tight_layout()
    return fig

def plot_comparison_bar(baseline_metrics, improved_metrics):
    """绘制基线与改进版的对比柱状图"""
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    
    # 只对比UNSEEN类别的AP@0.5和AP@[0.5:0.95]
    metrics_names = ["AP@[0.5:0.95]", "AP@0.5"]
    baseline_vals = [baseline_metrics["unseen"][m] for m in metrics_names]
    improved_vals = [improved_metrics["unseen"][m] for m in metrics_names]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, baseline_vals, width, label="Baseline (Uniform Threshold)", color="#2E86AB")
    ax.bar(x + width/2, improved_vals, width, label="Improved C (Adaptive Threshold)", color="#A23B72")
    
    ax.set_title("UNSEEN Metrics: Baseline vs Improved C", fontsize=14, fontweight="bold")
    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("AP Value", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 0.5)
    
    # 添加数值标签
    for i, v in enumerate(baseline_vals):
        ax.text(i - width/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    for i, v in enumerate(improved_vals):
        ax.text(i + width/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    
    plt.tight_layout()
    return fig

def plot_improvement_heatmap(baseline_cls_metrics, improved_cls_metrics):
    """绘制基线与改进版的类别热力图对比"""
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    
    # 获取所有有数据的UNSEEN类别
    all_cls_ids = set()
    if baseline_cls_metrics:
        all_cls_ids.update(baseline_cls_metrics.keys())
    if improved_cls_metrics:
        all_cls_ids.update(improved_cls_metrics.keys())
    
    if not all_cls_ids:
        print("⚠️ 没有UNSEEN类别数据，跳过热力图")
        return None
    
    all_cls_ids = sorted(list(all_cls_ids))
    cls_names = [UNSEEN_CLS_NAME.get(cid, str(cid)) for cid in all_cls_ids]
    
    # 构建热力图数据
    heatmap_data = np.zeros((2, len(all_cls_ids)))
    versions = ["Baseline", "Improved C"]
    
    for j, cid in enumerate(all_cls_ids):
        if baseline_cls_metrics and cid in baseline_cls_metrics:
            heatmap_data[0, j] = baseline_cls_metrics[cid]["AP@0.5"]
        else:
            heatmap_data[0, j] = 0.0
            
        if improved_cls_metrics and cid in improved_cls_metrics:
            heatmap_data[1, j] = improved_cls_metrics[cid]["AP@0.5"]
        else:
            heatmap_data[1, j] = 0.0
    
    fig, ax = plt.subplots(figsize=(14, 4))
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu",
                xticklabels=cls_names, yticklabels=versions,
                cbar_kws={"label": "AP@0.5"}, ax=ax)
    
    ax.set_title("UNSEEN Category AP@0.5: Baseline vs Improved C", fontsize=14, fontweight="bold")
    ax.set_xlabel("UNSEEN Category", fontsize=12)
    ax.set_ylabel("Version", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig

def save_comparison_report(baseline_metrics, improved_metrics, 
                          baseline_cls_metrics, improved_cls_metrics):
    """保存对比报告"""
    print("\n🔄 保存对比报告...")
    
    with open(COMPARISON_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("小幅改进C - 基线与改进版对比报告\n")
        f.write("=" * 80 + "\n\n")
        
        # 整体指标对比
        f.write("📊 UNSEEN类别整体指标对比:\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Metric':<20} {'Baseline':<15} {'Improved C':<15} {'Improvement':<15}\n")
        f.write("-" * 50 + "\n")
        
        metrics_to_compare = ["AP@[0.5:0.95]", "AP@0.5", "AP@0.75", "AR@100"]
        for metric in metrics_to_compare:
            baseline_val = baseline_metrics["unseen"][metric]
            improved_val = improved_metrics["unseen"][metric]
            improvement = improved_val - baseline_val
            f.write(f"{metric:<20} {baseline_val:<15.4f} {improved_val:<15.4f} {improvement:+.4f}\n")
        
        f.write("\n\n📊 SEEN类别整体指标对比:\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Metric':<20} {'Baseline':<15} {'Improved C':<15} {'Improvement':<15}\n")
        f.write("-" * 50 + "\n")
        
        for metric in metrics_to_compare:
            baseline_val = baseline_metrics["seen"][metric]
            improved_val = improved_metrics["seen"][metric]
            improvement = improved_val - baseline_val
            f.write(f"{metric:<20} {baseline_val:<15.4f} {improved_val:<15.4f} {improvement:+.4f}\n")
        
        # 各UNSEEN类别详细对比
        f.write("\n\n📊 UNSEEN各类别AP@0.5详细对比:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Category':<15} {'Baseline':<15} {'Improved C':<15} {'Improvement':<15}\n")
        f.write("-" * 60 + "\n")
        
        all_cls_ids = set()
        if baseline_cls_metrics:
            all_cls_ids.update(baseline_cls_metrics.keys())
        if improved_cls_metrics:
            all_cls_ids.update(improved_cls_metrics.keys())
        
        for cls_id in sorted(all_cls_ids):
            cls_name = UNSEEN_CLS_NAME.get(cls_id, str(cls_id))
            baseline_val = baseline_cls_metrics.get(cls_id, {}).get("AP@0.5", 0.0) if baseline_cls_metrics else 0.0
            improved_val = improved_cls_metrics.get(cls_id, {}).get("AP@0.5", 0.0) if improved_cls_metrics else 0.0
            improvement = improved_val - baseline_val
            f.write(f"{cls_name:<15} {baseline_val:<15.4f} {improved_val:<15.4f} {improvement:+.4f}\n")
        
        # 改进总结
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("改进总结:\n")
        f.write("=" * 80 + "\n")
        
        unseen_ap_improve = improved_metrics["unseen"]["AP@0.5"] - baseline_metrics["unseen"]["AP@0.5"]
        f.write(f"• UNSEEN AP@0.5 提升: {unseen_ap_improve:+.4f}\n")
        
        if unseen_ap_improve > 0:
            f.write("✅ 自适应阈值策略有效，提升了UNSEEN类别检测性能\n")
        else:
            f.write("⚠️ 自适应阈值策略未带来提升，可能需要进一步优化\n")
    
    print(f"✅ 对比报告已保存：{COMPARISON_REPORT_TXT}")

def main():
    print("=" * 80)
    print("小幅改进C - 基线与改进版对比评测")
    print("=" * 80)
    
    # 检查文件是否存在
    if not os.path.exists(BASELINE_SEEN_PATH):
        print(f"❌ 基线文件不存在：{BASELINE_SEEN_PATH}")
        return
    if not os.path.exists(IMPROVED_SEEN_PATH):
        print(f"❌ 改进版文件不存在：{IMPROVED_SEEN_PATH}")
        return
    
    try:
        # 1. 加载COCO Ground Truth
        print("\n🔄 加载COCO 2017 val标注文件...")
        cocoGt = COCO(ANNO_PATH)
        
        # 2. 加载基线结果
        print("\n📊 加载基线结果...")
        baseline_seen = load_detection_results(BASELINE_SEEN_PATH)
        baseline_unseen = load_detection_results(BASELINE_UNSEEN_PATH)
        
        # 3. 加载改进版结果
        print("\n📊 加载改进版结果...")
        improved_seen = load_detection_results(IMPROVED_SEEN_PATH)
        improved_unseen = load_detection_results(IMPROVED_UNSEEN_PATH)
        
        # 4. 计算指标
        print("\n" + "=" * 80)
        print("评测基线结果...")
        baseline_seen_metrics = evaluate_coco_metrics(cocoGt, baseline_seen, list(SEEN_CLS_IDS), "SEEN")
        baseline_unseen_metrics = evaluate_coco_metrics(cocoGt, baseline_unseen, list(UNSEEN_CLS_IDS), "UNSEEN")
        
        print("\n" + "=" * 80)
        print("评测改进版结果...")
        improved_seen_metrics = evaluate_coco_metrics(cocoGt, improved_seen, list(SEEN_CLS_IDS), "SEEN")
        improved_unseen_metrics = evaluate_coco_metrics(cocoGt, improved_unseen, list(UNSEEN_CLS_IDS), "UNSEEN")
        
        # 5. 分析各类别详细指标
        print("\n" + "=" * 80)
        print("分析UNSEEN各类别详细指标...")
        baseline_cls_metrics = analyze_unseen_categories(baseline_unseen, cocoGt)
        improved_cls_metrics = analyze_unseen_categories(improved_unseen, cocoGt)
        
        # 6. 生成改进版自身的可视化图表
        print("\n🔄 生成改进版可视化图表...")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # 改进版全量指标对比图
        fig = plot_improved_metrics(improved_seen_metrics, improved_unseen_metrics, "Improved C")
        fig.savefig(IMPROVED_METRICS_FULL_PNG, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"✅ 改进版全量指标对比图已保存：{IMPROVED_METRICS_FULL_PNG}")
        
        # 改进版尺寸敏感指标图
        fig = plot_improved_size_sensitive(improved_seen_metrics, improved_unseen_metrics, "Improved C")
        fig.savefig(IMPROVED_SIZE_SENSITIVE_PNG, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"✅ 改进版尺寸敏感指标图已保存：{IMPROVED_SIZE_SENSITIVE_PNG}")
        
        # 改进版检测框分析图
        fig = plot_improved_bbox_analysis(improved_seen, improved_unseen, "Improved C")
        fig.savefig(IMPROVED_BBOX_ANALYSIS_PNG, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"✅ 改进版检测框分析图已保存：{IMPROVED_BBOX_ANALYSIS_PNG}")
        
        # 改进版召回率曲线
        fig = plot_improved_recall_curve(improved_seen_metrics, improved_unseen_metrics, "Improved C")
        fig.savefig(IMPROVED_RECALL_CURVE_PNG, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"✅ 改进版召回率曲线已保存：{IMPROVED_RECALL_CURVE_PNG}")
        
        # 7. 生成对比可视化图表
        print("\n🔄 生成对比可视化图表...")
        
        # 对比柱状图
        baseline_metrics = {"seen": baseline_seen_metrics, "unseen": baseline_unseen_metrics}
        improved_metrics = {"seen": improved_seen_metrics, "unseen": improved_unseen_metrics}
        
        fig = plot_comparison_bar(baseline_metrics, improved_metrics)
        fig.savefig(COMPARISON_BAR_PNG, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"✅ 对比柱状图已保存：{COMPARISON_BAR_PNG}")
        
        # 对比热力图
        fig = plot_improvement_heatmap(baseline_cls_metrics, improved_cls_metrics)
        if fig:
            fig.savefig(COMPARISON_HEATMAP_PNG, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"✅ 对比热力图已保存：{COMPARISON_HEATMAP_PNG}")
        
        # 8. 保存对比报告
        save_comparison_report(baseline_metrics, improved_metrics,
                              baseline_cls_metrics, improved_cls_metrics)
        
        # 9. 最终汇总
        print("\n" + "=" * 80)
        print("评测完成！")
        print("=" * 80)
        print(f"\n基线 UNSEEN AP@0.5: {baseline_unseen_metrics['AP@0.5']:.4f}")
        print(f"改进版 UNSEEN AP@0.5: {improved_unseen_metrics['AP@0.5']:.4f}")
        print(f"提升: {improved_unseen_metrics['AP@0.5'] - baseline_unseen_metrics['AP@0.5']:+.4f}")
        print(f"\n📁 所有结果已保存至：{OUTPUT_DIR}")
        
    except Exception as e:
        print(f"\n❌ 评测失败：{str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()