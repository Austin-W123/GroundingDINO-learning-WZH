# step1_split_val_set.py
"""
第1步：从SEEN类别图片中划分验证集
用于后续的阈值优化统计
"""
import os
import json
import random
import shutil
from collections import defaultdict

# 配置
BASE_DIR = "D:/groundingdino_work/GroundingDINO-main"
ANNO_PATH = os.path.join(BASE_DIR, "data/coco/annotations/instances_val2017.json")
IMG_DIR = os.path.join(BASE_DIR, "data/coco/val2017")
VAL_SET_DIR = os.path.join(BASE_DIR, "data/coco/val_set_for_threshold")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(VAL_SET_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# SEEN类别ID（65个常见类）
SEEN_CLS_IDS = {
    1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,
    21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,
    40,41,42,43,44,46,47,48,49,50,51,62,63,64,65,67,
    70,72,73,74,75,76,77,78,79,80,87,88,89,90
}

def split_val_set():
    """从SEEN图片中划分20%作为验证集"""
    print("=" * 60)
    print("第1步：划分验证集")
    print("=" * 60)
    
    # 加载COCO标注
    with open(ANNO_PATH, 'r') as f:
        coco = json.load(f)
    
    # 找出包含SEEN类别的图片
    seen_images = set()
    image_to_cats = defaultdict(set)
    
    for ann in coco['annotations']:
        cat_id = ann['category_id']
        img_id = ann['image_id']
        
        if cat_id in SEEN_CLS_IDS:
            seen_images.add(img_id)
            image_to_cats[img_id].add(cat_id)
    
    print(f"包含SEEN类别的图片总数: {len(seen_images)}")
    
    # 随机划分（20%验证集，80%训练集）
    seen_images = list(seen_images)
    random.seed(42)
    random.shuffle(seen_images)
    
    val_size = int(len(seen_images) * 0.2)
    val_images = seen_images[:val_size]
    train_images = seen_images[val_size:]
    
    print(f"验证集图片数: {len(val_images)}")
    print(f"训练集图片数: {len(train_images)}")
    
    # 保存验证集图片信息
    val_image_info = []
    img_id_to_file = {img['id']: img['file_name'] for img in coco['images']}
    
    for img_id in val_images:
        if img_id in img_id_to_file:
            file_name = img_id_to_file[img_id]
            src_path = os.path.join(IMG_DIR, file_name)
            dst_path = os.path.join(VAL_SET_DIR, file_name)
            
            # 复制图片到验证集目录（可选）
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
            
            val_image_info.append({
                'id': img_id,
                'file_name': file_name,
                'categories': list(image_to_cats[img_id])
            })
    
    # 保存验证集信息
    val_info_path = os.path.join(OUTPUT_DIR, 'val_set_info.json')
    with open(val_info_path, 'w') as f:
        json.dump({
            'val_images': val_image_info,
            'val_image_ids': val_images,
            'split_info': {
                'total_seen_images': len(seen_images),
                'val_size': len(val_images),
                'train_size': len(train_images),
                'val_ratio': 0.2
            }
        }, f, indent=2)
    
    print(f"\n✅ 验证集信息已保存: {val_info_path}")
    print(f"✅ 验证集图片已复制到: {VAL_SET_DIR}")
    
    return val_images

if __name__ == "__main__":
    split_val_set()