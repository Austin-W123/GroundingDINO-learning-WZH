# prompt_ensemble_simple.py
"""
多提示集成方案：
对同一个(图片, 类别, 位置)的检测框，取三个prompt中的最高置信度
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
OUTPUT_SEEN = os.path.join(OUTPUT_DIR, 'coco_seen_400imgs_ensemble.json')
OUTPUT_UNSEEN = os.path.join(OUTPUT_DIR, 'coco_unseen_100imgs_ensemble.json')

def bbox_to_key(bbox):
    """
    将bbox转换为可哈希的key
    四舍五入到整数，避免浮点数精度问题
    """
    return tuple(int(round(x)) for x in bbox)

def ensemble_detections(detections_list):
    """
    融合多个prompt的检测结果
    策略：对同一个(图片,类别,位置)的框，取最高分
    """
    # 使用字典存储：key=(image_id, category_id, bbox_key) -> best_detection
    ensemble_dict = {}
    
    for prompt_idx, detections in enumerate(detections_list):
        print(f"  处理第 {prompt_idx+1} 个prompt，共 {len(detections)} 个框")
        
        for det in detections:
            # 创建唯一键
            bbox_key = bbox_to_key(det['bbox'])
            key = (det['image_id'], det['category_id'], bbox_key)
            
            # 如果这个框还没出现过，直接添加
            if key not in ensemble_dict:
                ensemble_dict[key] = det.copy()
                ensemble_dict[key]['source_prompt'] = prompt_idx
            else:
                # 如果出现过，保留得分更高的
                if det['score'] > ensemble_dict[key]['score']:
                    ensemble_dict[key] = det.copy()
                    ensemble_dict[key]['source_prompt'] = prompt_idx
    
    # 转换为列表
    ensemble_results = list(ensemble_dict.values())
    
    # 按得分排序
    ensemble_results.sort(key=lambda x: x['score'], reverse=True)
    
    return ensemble_results

def main():
    print("=" * 60)
    print("方向A：多提示集成（取最高分）")
    print("=" * 60)
    
    # 检查所有文件是否存在
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
        
        # 加载SEEN
        with open(files['seen'], 'r') as f:
            seen_dets = json.load(f)
            seen_dets_list.append(seen_dets)
            print(f"  SEEN: {len(seen_dets)} 框")
        
        # 加载UNSEEN
        with open(files['unseen'], 'r') as f:
            unseen_dets = json.load(f)
            unseen_dets_list.append(unseen_dets)
            print(f"  UNSEEN: {len(unseen_dets)} 框")
    
    # 融合SEEN
    print("\n🔄 融合SEEN结果...")
    ensemble_seen = ensemble_detections(seen_dets_list)
    print(f"  融合前总框数: {sum(len(d) for d in seen_dets_list)}")
    print(f"  融合后框数: {len(ensemble_seen)}")
    print(f"  去重比例: {(1 - len(ensemble_seen)/sum(len(d) for d in seen_dets_list))*100:.1f}%")
    
    # 融合UNSEEN
    print("\n🔄 融合UNSEEN结果...")
    ensemble_unseen = ensemble_detections(unseen_dets_list)
    print(f"  融合前总框数: {sum(len(d) for d in unseen_dets_list)}")
    print(f"  融合后框数: {len(ensemble_unseen)}")
    print(f"  去重比例: {(1 - len(ensemble_unseen)/sum(len(d) for d in unseen_dets_list))*100:.1f}%")
    
    # 保存结果
    print("\n💾 保存融合结果...")
    
    with open(OUTPUT_SEEN, 'w') as f:
        json.dump(ensemble_seen, f, indent=2)
    print(f"  ✅ SEEN结果已保存: {OUTPUT_SEEN}")
    
    with open(OUTPUT_UNSEEN, 'w') as f:
        json.dump(ensemble_unseen, f, indent=2)
    print(f"  ✅ UNSEEN结果已保存: {OUTPUT_UNSEEN}")
    
    # 统计每个prompt的贡献
    print("\n📊 各prompt贡献统计:")
    source_counts = {0: 0, 1: 2, 2: 0}  # prompt1, prompt2, prompt3
    
    for det in ensemble_seen:
        source = det.get('source_prompt', 0)
        source_counts[source] = source_counts.get(source, 0) + 1
    
    prompt_names = ['prompt1', 'prompt2', 'prompt3']
    print("  SEEN结果来源:")
    for i, name in enumerate(prompt_names):
        count = source_counts.get(i, 0)
        percentage = count / len(ensemble_seen) * 100 if ensemble_seen else 0
        print(f"    {name}: {count} 框 ({percentage:.1f}%)")
    
    print("\n" + "=" * 60)
    print("✅ 融合完成！")
    print("=" * 60)
    print("\n下一步：运行评测脚本对比效果")
    print("  python scripts/step5_evaluate_improved.py")
    print("（记得先修改step5中的文件名）")

if __name__ == "__main__":
    main()