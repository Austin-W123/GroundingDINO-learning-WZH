# -*- coding: utf-8 -*-
"""
COCO检测结果评测脚本 - 三轮Prompt对比实验（完整对比版）
计算SEEN/UNSEEN类COCO指标，并生成三轮对比可视化
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

# 三轮prompt配置
PROMPT_VERSIONS = ["prompt1", "prompt2", "prompt3"]
PROMPT_NAME_MAP = {
    "prompt1": "Pure Class Name",
    "prompt2": "Template Sentence",
    "prompt3": "Fine-grained Description"
}

# 当前轮次（用于单轮评测）
CURRENT_PROMPT = "prompt3"  # 可根据需要修改
CURRENT_PROMPT_NAME = PROMPT_NAME_MAP.get(CURRENT_PROMPT, "Unknown")

# 文件路径模板
SEEN_RESULT_PATH = os.path.join(OUTPUT_DIR, f"coco_seen_400imgs_{CURRENT_PROMPT}.json")
UNSEEN_RESULT_PATH = os.path.join(OUTPUT_DIR, f"coco_unseen_100imgs_{CURRENT_PROMPT}.json")
EVAL_RESULT_TXT = os.path.join(OUTPUT_DIR, f"coco_eval_result_{CURRENT_PROMPT}.txt")

# 单轮可视化文件
METRICS_FULL_PNG = os.path.join(OUTPUT_DIR, f"metrics_full_comparison_{CURRENT_PROMPT}.png")
SIZE_SENSITIVE_PNG = os.path.join(OUTPUT_DIR, f"size_sensitive_metrics_{CURRENT_PROMPT}.png")
BBOX_ANALYSIS_PNG = os.path.join(OUTPUT_DIR, f"bbox_analysis_{CURRENT_PROMPT}.png")
RECALL_CURVE_PNG = os.path.join(OUTPUT_DIR, f"recall_curve_{CURRENT_PROMPT}.png")

# 三轮对比可视化文件
PROMPT_COMPARISON_BAR_PNG = os.path.join(OUTPUT_DIR, "prompt_comparison_bar.png")
PROMPT_COMPARISON_HEATMAP_PNG = os.path.join(OUTPUT_DIR, "prompt_comparison_heatmap.png")

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
    with open(file_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    print(f"✅ 加载检测框：{file_path} | 检测框数：{len(results)}")
    return results

def load_all_prompt_results():
    """加载所有三轮的检测结果"""
    all_seen_results = {}
    all_unseen_results = {}
    
    for prompt in PROMPT_VERSIONS:
        seen_file = os.path.join(OUTPUT_DIR, f"coco_seen_400imgs_{prompt}.json")
        unseen_file = os.path.join(OUTPUT_DIR, f"coco_unseen_100imgs_{prompt}.json")
        
        if os.path.exists(seen_file):
            with open(seen_file, 'r', encoding='utf-8') as f:
                all_seen_results[prompt] = json.load(f)
            print(f"✅ 加载 {PROMPT_NAME_MAP[prompt]} SEEN 结果: {len(all_seen_results[prompt])} 框")
        
        if os.path.exists(unseen_file):
            with open(unseen_file, 'r', encoding='utf-8') as f:
                all_unseen_results[prompt] = json.load(f)
            print(f"✅ 加载 {PROMPT_NAME_MAP[prompt]} UNSEEN 结果: {len(all_unseen_results[prompt])} 框")
    
    return all_seen_results, all_unseen_results

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
    print(f"\n===== 开始{eval_name}类别评测（COCO全量12项指标） =====")
    
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
        cocoEval.params.imgIds = list(set([d['image_id'] for d in results]))  # 只评估有检测框的图片
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        
        # 确保cocoEval.stats存在且长度足够
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
    except Exception as e:
        print(f"⚠️ COCO评测出错: {e}")
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

def analyze_prompt_impact(unseen_results, cocoGt, prompt_version):
    """
    分析特定Prompt对unseen类的影响
    参数：
        unseen_results: unseen类检测框列表
        cocoGt: COCO Ground Truth
        prompt_version: prompt版本 (prompt1/prompt2/prompt3)
    返回：
        cls_metrics: {cls_id: {"AP@[0.5:0.95]": value, "AP@0.5": value}}
        avg_metrics: {"AP@[0.5:0.95]": value, "AP@0.5": value}
    """
    print(f"\n📊 分析 {PROMPT_NAME_MAP[prompt_version]} 的UNSEEN类别...")
    print(f"    输入检测框总数: {len(unseen_results)}")
    
    # 过滤出UNSEEN类别的检测框
    unseen_results_filtered = [det for det in unseen_results if det["category_id"] in UNSEEN_CLS_IDS]
    print(f"    过滤后UNSEEN框数: {len(unseen_results_filtered)}")
    
    # 如果过滤后没有检测框，直接返回0
    if len(unseen_results_filtered) == 0:
        print("    ⚠️ 没有UNSEEN类别检测框，返回0")
        return {}, {"AP@[0.5:0.95]": 0.0, "AP@0.5": 0.0}
    
    # 直接使用所有UNSEEN检测框进行一次整体评估，而不是按类别分开
    try:
        cocoDt = cocoGt.loadRes(unseen_results_filtered)
        cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
        cocoEval.params.catIds = list(UNSEEN_CLS_IDS)  # 评估所有UNSEEN类别
        cocoEval.params.imgIds = list(set([d['image_id'] for d in unseen_results_filtered]))
        cocoEval.evaluate()
        cocoEval.accumulate()
        
        # 获取每个类别的AP
        cls_metrics = {}
        precisions = cocoEval.eval['precision']  # [TxRxKxAxM] 
        
        # 对于每个UNSEEN类别
        for idx, cat_id in enumerate(cocoEval.params.catIds):
            if cat_id in UNSEEN_CLS_IDS:
                # 取所有IoU阈值和所有面积的均值作为该类别的AP
                cat_precision = precisions[:, :, idx, 0, -1]  # 取所有IoU阈值、所有召回率点
                if cat_precision.size > 0:
                    ap = np.mean(cat_precision[cat_precision > -1])  # 忽略-1
                    ap50 = np.mean(precisions[0, :, idx, 0, -1][precisions[0, :, idx, 0, -1] > -1])  # IoU=0.5
                else:
                    ap, ap50 = 0.0, 0.0
                
                cls_metrics[cat_id] = {
                    "AP@[0.5:0.95]": round(ap, 4),
                    "AP@0.5": round(ap50, 4)
                }
                print(f"      类别 {UNSEEN_CLS_NAME.get(cat_id, cat_id)}: AP@0.5={cls_metrics[cat_id]['AP@0.5']}")
        
        # 计算平均指标
        if cls_metrics:
            ap_all_vals = [m["AP@[0.5:0.95]"] for m in cls_metrics.values()]
            ap05_vals = [m["AP@0.5"] for m in cls_metrics.values()]
            avg_metrics = {
                "AP@[0.5:0.95]": round(np.mean(ap_all_vals), 4),
                "AP@0.5": round(np.mean(ap05_vals), 4)
            }
        else:
            avg_metrics = {"AP@[0.5:0.95]": 0.0, "AP@0.5": 0.0}
        
    except Exception as e:
        print(f"    ⚠️ 整体评估出错: {e}")
        cls_metrics = {}
        avg_metrics = {"AP@[0.5:0.95]": 0.0, "AP@0.5": 0.0}
    
    print(f"    {PROMPT_NAME_MAP[prompt_version]} 平均指标: AP@[0.5:0.95]={avg_metrics['AP@[0.5:0.95]']}, AP@0.5={avg_metrics['AP@0.5']}")
    return cls_metrics, avg_metrics

def save_eval_report(seen_metrics, unseen_metrics, prompt_metrics, prompt_avg):
    """保存评测报告"""
    print("\n🔄 保存全量评测报告...")
    with open(EVAL_RESULT_TXT, "w", encoding="utf-8") as f:
        f.write("===== COCO Detection Full Evaluation Report =====\n")
        f.write(f"Evaluation Time: {os.popen('date /t').read().strip()}\n")
        f.write(f"Prompt Version: {CURRENT_PROMPT} ({CURRENT_PROMPT_NAME})\n\n")
        
        # SEEN指标
        f.write("===== SEEN Categories (65 classes) 12 Metrics =====\n")
        for k, v in seen_metrics.items():
            f.write(f"{k} = {v}\n")
        
        # UNSEEN指标
        f.write("\n===== UNSEEN Categories (15 classes) 12 Metrics =====\n")
        for k, v in unseen_metrics.items():
            f.write(f"{k} = {v}\n")
        
        # Prompt分析
        f.write("\n===== Prompt Impact Analysis =====\n")
        for prompt_name, metrics in prompt_metrics.items():
            f.write(f"\n📌 {prompt_name}:\n")
            f.write(f"   Average AP@[0.5:0.95] = {prompt_avg[prompt_name]['AP@[0.5:0.95]']}\n")
            f.write(f"   Average AP@0.5 = {prompt_avg[prompt_name]['AP@0.5']}\n")
    
    print(f"✅ 评测报告已保存：{EVAL_RESULT_TXT}")

def plot_metrics_comparison(seen_metrics, unseen_metrics):
    """绘制全量指标对比图"""
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
    
    ax.set_title(f"COCO Core Metrics Comparison (Prompt: {CURRENT_PROMPT_NAME})", fontsize=14, fontweight="bold")
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
    plt.savefig(METRICS_FULL_PNG, dpi=300, bbox_inches="tight")
    print(f"✅ 全量指标对比图已保存：{METRICS_FULL_PNG}")

def plot_size_sensitive_metrics(seen_metrics, unseen_metrics):
    """绘制尺寸敏感指标图"""
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
    
    ax.set_title(f"Size-sensitive AP Metrics (Prompt: {CURRENT_PROMPT_NAME})", fontsize=14, fontweight="bold")
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
    plt.savefig(SIZE_SENSITIVE_PNG, dpi=300, bbox_inches="tight")
    print(f"✅ 尺寸敏感指标图已保存：{SIZE_SENSITIVE_PNG}")

def plot_bbox_analysis(seen_results, unseen_results):
    """绘制检测框分析图"""
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    
    seen_scores = [det["score"] for det in seen_results]
    unseen_scores = [det["score"] for det in unseen_results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.hist(seen_scores, bins=20, alpha=0.7, label="SEEN", color="#2E86AB", edgecolor="black")
    ax1.hist(unseen_scores, bins=20, alpha=0.7, label="UNSEEN", color="#A23B72", edgecolor="black")
    ax1.set_title(f"Detection Score Distribution (Prompt: {CURRENT_PROMPT_NAME})", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Detection Score", fontsize=10)
    ax1.set_ylabel("Count", fontsize=10)
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0.8, 1.0)
    
    # 统计UNSEEN类别检测框数量
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
    plt.savefig(BBOX_ANALYSIS_PNG, dpi=300, bbox_inches="tight")
    print(f"✅ 检测框分析图已保存：{BBOX_ANALYSIS_PNG}")

def plot_recall_curve(seen_metrics, unseen_metrics):
    """绘制召回率曲线"""
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
    
    ax.set_title(f"Recall Curve Comparison (Prompt: {CURRENT_PROMPT_NAME})", fontsize=14, fontweight="bold")
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
    plt.savefig(RECALL_CURVE_PNG, dpi=300, bbox_inches="tight")
    print(f"✅ 召回率曲线已保存：{RECALL_CURVE_PNG}")

def plot_prompt_comparison_bar(all_avg_metrics):
    """绘制所有prompt的对比柱状图"""
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    
    prompts = list(all_avg_metrics.keys())
    prompt_names = [PROMPT_NAME_MAP[p] for p in prompts]
    
    ap_all = [all_avg_metrics[p]["AP@[0.5:0.95]"] for p in prompts]
    ap05 = [all_avg_metrics[p]["AP@0.5"] for p in prompts]
    
    # 打印调试信息
    print("\n📊 三轮对比数据:")
    for i, prompt in enumerate(prompts):
        print(f"  {prompt_names[i]}: AP@[0.5:0.95]={ap_all[i]}, AP@0.5={ap05[i]}")
    
    x = np.arange(len(prompts))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    bars1 = ax.bar(x - width/2, ap_all, width, label="AP@[0.5:0.95]", color="#2E86AB", edgecolor="black")
    bars2 = ax.bar(x + width/2, ap05, width, label="AP@0.5", color="#A23B72", edgecolor="black")
    
    ax.set_title("Prompt Impact on UNSEEN Average Metrics (3-Round Comparison)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Prompt Type", fontsize=12)
    ax.set_ylabel("Average AP Value", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(prompt_names, rotation=45, ha="right")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    # 添加数值标签
    for i, (bar, v) in enumerate(zip(bars1, ap_all)):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v:.3f}", 
                   ha="center", fontsize=10, fontweight="bold", color="blue")
    
    for i, (bar, v) in enumerate(zip(bars2, ap05)):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v:.3f}", 
                   ha="center", fontsize=10, fontweight="bold", color="red")
    
    plt.tight_layout()
    plt.savefig(PROMPT_COMPARISON_BAR_PNG, dpi=300, bbox_inches="tight")
    print(f"✅ Prompt对比柱状图已保存：{PROMPT_COMPARISON_BAR_PNG}")

def plot_prompt_comparison_heatmap(all_prompt_metrics):
    """绘制所有prompt的热力图对比"""
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    
    prompts = list(all_prompt_metrics.keys())
    prompt_names = [PROMPT_NAME_MAP[p] for p in prompts]
    
    # 获取所有UNSEEN类别
    all_cls_ids = set()
    for prompt, cls_metrics in all_prompt_metrics.items():
        all_cls_ids.update(cls_metrics.keys())
    all_cls_ids = sorted(list(all_cls_ids))
    cls_names = [UNSEEN_CLS_NAME.get(cid, str(cid)) for cid in all_cls_ids]
    
    if len(all_cls_ids) == 0:
        print("⚠️ 没有UNSEEN类别数据，跳过热力图")
        return
    
    # 构建热力图数据
    heatmap_data = np.zeros((len(prompts), len(all_cls_ids)))
    for i, prompt in enumerate(prompts):
        for j, cid in enumerate(all_cls_ids):
            if cid in all_prompt_metrics[prompt]:
                heatmap_data[i, j] = all_prompt_metrics[prompt][cid]["AP@0.5"]
            else:
                heatmap_data[i, j] = 0.0
    
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu",
                xticklabels=cls_names, yticklabels=prompt_names,
                cbar_kws={"label": "AP@0.5"}, ax=ax)
    
    ax.set_title("Prompt × UNSEEN Category AP@0.5 Heatmap (3-Round Comparison)", fontsize=14, fontweight="bold")
    ax.set_xlabel("UNSEEN Category", fontsize=12)
    ax.set_ylabel("Prompt Type", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(PROMPT_COMPARISON_HEATMAP_PNG, dpi=300, bbox_inches="tight")
    print(f"✅ Prompt热力图对比已保存：{PROMPT_COMPARISON_HEATMAP_PNG}")

def main():
    print("===== COCO检测结果全量评测（支持三轮对比） =====")
    print(f"📌 当前单轮Prompt：{CURRENT_PROMPT}（{CURRENT_PROMPT_NAME}）")
    print(f"📌 输入文件：{SEEN_RESULT_PATH}、{UNSEEN_RESULT_PATH}")
    
    try:
        # 1. 加载COCO Ground Truth
        print("\n🔄 加载COCO 2017 val标注文件...")
        cocoGt = COCO(ANNO_PATH)
        
        # 2. 加载当前轮的检测结果
        seen_results = load_detection_results(SEEN_RESULT_PATH)
        unseen_results = load_detection_results(UNSEEN_RESULT_PATH)
        
        # 3. 计算当前轮的SEEN/UNSEEN指标
        seen_metrics = evaluate_coco_metrics(cocoGt, seen_results, list(SEEN_CLS_IDS), "SEEN")
        unseen_metrics = evaluate_coco_metrics(cocoGt, unseen_results, list(UNSEEN_CLS_IDS), "UNSEEN")
        
        # 4. 分析当前轮的Prompt对unseen类的影响
        current_cls_metrics, current_avg_metrics = analyze_prompt_impact(unseen_results, cocoGt, CURRENT_PROMPT)
        
        # 5. 生成当前轮的可视化图表
        print("\n🔄 生成当前轮可视化图表...")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        plot_metrics_comparison(seen_metrics, unseen_metrics)
        plot_size_sensitive_metrics(seen_metrics, unseen_metrics)
        plot_bbox_analysis(seen_results, unseen_results)
        plot_recall_curve(seen_metrics, unseen_metrics)
        
        # 6. 加载所有三轮结果并生成对比图
        print("\n🔄 加载所有三轮结果生成对比图...")
        all_seen_results, all_unseen_results = load_all_prompt_results()
        
        if len(all_unseen_results) >= 2:
            print("\n📊 开始分析三轮数据...")
            all_prompt_metrics = {}  # {prompt: {cls_id: metrics}}
            all_avg_metrics = {}     # {prompt: avg_metrics}
            
            for prompt, results in all_unseen_results.items():
                print(f"\n🔍 调试 - {PROMPT_NAME_MAP[prompt]} 的检测框数量: {len(results)}")
                if len(results) > 0:
                    print(f"    示例类别ID: {results[0]['category_id']}")
                    print(f"    示例类别名称: {UNSEEN_CLS_NAME.get(results[0]['category_id'], 'unknown')}")
                
                cls_metrics, avg_metrics = analyze_prompt_impact(results, cocoGt, prompt)
                all_prompt_metrics[prompt] = cls_metrics
                all_avg_metrics[prompt] = avg_metrics
                
                print(f"    计算得到的平均AP@0.5: {avg_metrics['AP@0.5']}")
            
            # 保存三轮对比的评测报告
            save_eval_report(seen_metrics, unseen_metrics, 
                           {PROMPT_NAME_MAP[p]: all_prompt_metrics[p] for p in all_prompt_metrics},
                           {PROMPT_NAME_MAP[p]: all_avg_metrics[p] for p in all_avg_metrics})
            
            # 生成对比图
            plot_prompt_comparison_bar(all_avg_metrics)
            plot_prompt_comparison_heatmap(all_prompt_metrics)
        else:
            print("⚠️ 只有一轮数据，保存当前轮报告")
            save_eval_report(seen_metrics, unseen_metrics, 
                           {CURRENT_PROMPT_NAME: current_cls_metrics},
                           {CURRENT_PROMPT_NAME: current_avg_metrics})
        
        # 最终汇总
        print("\n===== 评测完成 =====")
        print(f"📊 SEEN AP@[0.5:0.95]: {seen_metrics['AP@[0.5:0.95]']:.4f}")
        print(f"📊 SEEN AP@0.5: {seen_metrics['AP@0.5']:.4f}")
        print(f"📊 UNSEEN AP@[0.5:0.95]: {unseen_metrics['AP@[0.5:0.95]']:.4f}")
        print(f"📊 UNSEEN AP@0.5: {unseen_metrics['AP@0.5']:.4f}")
        print(f"📁 所有结果已保存至：{OUTPUT_DIR}")
        
    except Exception as e:
        print(f"\n❌ 评测失败：{str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()