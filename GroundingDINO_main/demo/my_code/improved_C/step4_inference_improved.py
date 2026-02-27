# step4_inference_improved.py (修改版)
"""
第4步：使用优化后的阈值进行推理
为每个类别应用独立的阈值
"""
import os
import sys
import json
import random
import torch
from PIL import Image
from tqdm import tqdm

# 添加项目路径
PROJECT_ROOT = "D:/groundingdino_work/GroundingDINO-main"
sys.path.insert(0, PROJECT_ROOT)

try:
    from groundingdino.util.inference import load_model, load_image, predict
except ImportError as e:
    print(f"导入错误: {e}")
    sys.exit(1)

# 配置
BASE_DIR = "D:/groundingdino_work/GroundingDINO-main"
MODEL_CONFIG_PATH = os.path.join(BASE_DIR, "groundingdino/config/GroundingDINO_SwinB_cfg.py")
MODEL_WEIGHTS_PATH = os.path.join(BASE_DIR, "weights/groundingdino_swinb_cogcoor.pth")
ANNO_PATH = os.path.join(BASE_DIR, "data/coco/annotations/instances_val2017.json")
IMG_DIR = os.path.join(BASE_DIR, "data/coco/val2017")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")

# 类别映射
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

# SEEN/UNSEEN划分
SEEN_CLS_IDS = {
    1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,
    21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,
    40,41,42,43,44,46,47,48,49,50,51,62,63,64,65,67,
    70,72,73,74,75,76,77,78,79,80,87,88,89,90
}

UNSEEN_CLS_IDS = {52,53,54,55,56,57,58,59,60,61,81,82,84,85,86}

ALL_CLASSES = {**CLASS_NAMES}
NAME_TO_ID = {v: k for k, v in ALL_CLASSES.items()}

# 配置参数
SEEN_SAMPLE_NUM = 400
UNSEEN_SAMPLE_NUM = 100
RANDOM_SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 修改点1：默认阈值从0.25改为0.15 =====
DEFAULT_THRESHOLD = 0.15  # 原来是0.25

def get_prompt_text(cls_ids):
    """生成Prompt"""
    names = [CLASS_NAMES[cid] for cid in cls_ids if cid in CLASS_NAMES]
    return ", ".join(names) + "."

def convert_bbox(boxes, h, w):
    """(cx,cy,w,h)[0,1] -> (x1,y1,w,h)像素"""
    result = []
    for box in boxes:
        cx, cy, bw, bh = box.tolist() if hasattr(box, 'tolist') else box
        x1 = max(0, (cx - bw/2) * w)
        y1 = max(0, (cy - bh/2) * h)
        width = min(bw * w, w - x1)
        height = min(bh * h, h - y1)
        result.append([float(x1), float(y1), float(width), float(height)])
    return result

def match_category(phrase):
    """将phrase匹配到category_id"""
    phrase_clean = phrase.strip().lower()
    if phrase_clean in NAME_TO_ID:
        return NAME_TO_ID[phrase_clean]
    
    candidates = sorted(CLASS_NAMES.values(), key=len, reverse=True)
    for name in candidates:
        if name.lower() in phrase_clean:
            return NAME_TO_ID.get(name)
    return None

def run_inference(model, img_path, prompt, img_id, cls_ids, thresholds):
    """使用类别自适应阈值进行推理"""
    try:
        image_source, image = load_image(img_path)
        
        # 用较低阈值获取候选框
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=prompt,
            box_threshold=0.05,  # 保持0.05不变
            text_threshold=0.05,
            device=device
        )
        
        if len(boxes) == 0:
            return []
        
        img = Image.open(img_path)
        w, h = img.size
        bboxes = convert_bbox(boxes, h, w)
        
        detections = []
        for bbox, logit, phrase in zip(bboxes, logits, phrases):
            cat_id = match_category(phrase)
            if cat_id is None or cat_id not in cls_ids:
                continue
            
            score = float(logit.item()) if hasattr(logit, 'item') else float(logit)
            
            # ===== 修改点2：默认阈值使用DEFAULT_THRESHOLD =====
            optimal_thr = thresholds.get(str(cat_id), DEFAULT_THRESHOLD)
            if score < optimal_thr:
                continue
            
            detections.append({
                "image_id": int(img_id),
                "category_id": int(cat_id),
                "bbox": [float(x) for x in bbox],
                "score": score,
                "phrase": str(phrase)
            })
        
        return detections
    
    except Exception as e:
        print(f"处理图片 {img_path} 时出错: {e}")
        return []

def main():
    print("=" * 70)
    print("改进版推理：使用类别自适应阈值")
    print("=" * 70)
    print(f"默认阈值: {DEFAULT_THRESHOLD}")
    print("=" * 70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(RANDOM_SEED)
    
    # 加载最优阈值
    thresh_path = os.path.join(OUTPUT_DIR, 'optimal_thresholds.json')
    if not os.path.exists(thresh_path):
        print(f"❌ 未找到阈值配置文件: {thresh_path}")
        print("请先运行 step3_optimize_thresholds.py")
        return
    
    with open(thresh_path, 'r') as f:
        optimal_thresholds = json.load(f)
    
    print(f"✅ 加载阈值配置，共 {len(optimal_thresholds)} 个类别")
    
    # 加载COCO标注，获取图片信息
    with open(ANNO_PATH, 'r') as f:
        coco = json.load(f)
    
    # 构建图片路径映射
    img_paths = {}
    for img in coco['images']:
        path = os.path.join(IMG_DIR, img['file_name'])
        if os.path.exists(path):
            img_paths[img['id']] = path
    
    # 找出包含SEEN/UNSEEN的图片
    img_with_seen = set()
    img_with_unseen = set()
    
    for ann in coco['annotations']:
        img_id = ann['image_id']
        cat_id = ann['category_id']
        
        if img_id not in img_paths:
            continue
        
        if cat_id in SEEN_CLS_IDS:
            img_with_seen.add(img_id)
        elif cat_id in UNSEEN_CLS_IDS:
            img_with_unseen.add(img_id)
    
    print(f"SEEN图片数: {len(img_with_seen)}")
    print(f"UNSEEN图片数: {len(img_with_unseen)}")
    
    # 随机抽样
    sampled_seen = random.sample(list(img_with_seen), min(SEEN_SAMPLE_NUM, len(img_with_seen)))
    sampled_unseen = random.sample(list(img_with_unseen), min(UNSEEN_SAMPLE_NUM, len(img_with_unseen)))
    
    print(f"\n抽样:")
    print(f"  SEEN: {len(sampled_seen)} 张")
    print(f"  UNSEEN: {len(sampled_unseen)} 张")
    
    # 加载模型
    print("\n加载模型...")
    model = load_model(MODEL_CONFIG_PATH, MODEL_WEIGHTS_PATH)
    model.to(device)
    model.eval()
    print("模型加载完成")
    
    # 推理SEEN
    print(f"\n推理SEEN图片...")
    seen_prompt = get_prompt_text(SEEN_CLS_IDS)
    seen_results = []
    
    for img_id in tqdm(sampled_seen, desc="SEEN"):
        dets = run_inference(model, img_paths[img_id], seen_prompt, 
                            img_id, SEEN_CLS_IDS, optimal_thresholds)
        seen_results.extend(dets)
    
    # 推理UNSEEN
    print(f"\n推理UNSEEN图片...")
    unseen_prompt = get_prompt_text(UNSEEN_CLS_IDS)
    unseen_results = []
    
    for img_id in tqdm(sampled_unseen, desc="UNSEEN"):
        dets = run_inference(model, img_paths[img_id], unseen_prompt,
                            img_id, UNSEEN_CLS_IDS, optimal_thresholds)
        unseen_results.extend(dets)
    
    # 保存结果
    seen_file = os.path.join(OUTPUT_DIR, 'coco_seen_400imgs_improved_C.json')
    unseen_file = os.path.join(OUTPUT_DIR, 'coco_unseen_100imgs_improved_C.json')
    
    with open(seen_file, 'w') as f:
        json.dump(seen_results, f, indent=2)
    
    with open(unseen_file, 'w') as f:
        json.dump(unseen_results, f, indent=2)
    
    print(f"\n✅ 完成!")
    print(f"  SEEN检测框: {len(seen_results)} → {seen_file}")
    print(f"  UNSEEN检测框: {len(unseen_results)} → {unseen_file}")

if __name__ == "__main__":
    main()