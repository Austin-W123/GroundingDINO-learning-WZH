# -*- coding: utf-8 -*-
"""
GroundingDINO COCO推理脚本 - 三轮Prompt对比实验
关键原则：完全按照评测要求，严格分离SEEN/UNSEEN
"""
import os
import sys
import json
import random
import warnings
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")

PROJECT_ROOT = "D:/groundingdino_work/GroundingDINO-main"
sys.path.insert(0, PROJECT_ROOT)

try:
    import torch
    from groundingdino.util.inference import load_model, load_image, predict
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[OK] Model device: {DEVICE}")
except ImportError as e:
    raise ImportError(f"缺少GroundingDINO依赖：{e}")

# 配置
BASE_DIR = "D:/groundingdino_work/GroundingDINO-main"
IMG_DIR = os.path.join(BASE_DIR, "data/coco/val2017")
ANNO_PATH = os.path.join(BASE_DIR, "data/coco/annotations/instances_val2017.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")

MODEL_CONFIG_PATH = os.path.join(BASE_DIR, "groundingdino/config/GroundingDINO_SwinB_cfg.py")
MODEL_WEIGHTS_PATH = os.path.join(BASE_DIR, "weights/groundingdino_swinb_cogcoor.pth")

BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
PROMPT_ROUND = "prompt3"
SEEN_SAMPLE_NUM = 400
UNSEEN_SAMPLE_NUM = 100
RANDOM_SEED = 42

# UNSEEN: 15个稀有类（ID 52-61; 81-86）
UNSEEN_CLASSES = {
    52: "banana", 53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli",
    57: "carrot", 58: "hot dog", 59: "pizza", 60: "donut", 61: "cake",
    81: "sink", 82: "refrigerator", 84: "book", 85: "clock", 86: "vase"
}

# SEEN: 65个常见类（除UNSEEN外）
SEEN_CLASSES = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep",
    21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe",
    27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase",
    34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite",
    39: "baseball bat", 40: "baseball glove", 41: "skateboard", 42: "surfboard", 43: "tennis racket",
    44: "bottle", 46: "wine glass", 47: "cup", 48: "fork", 49: "knife",
    50: "spoon", 51: "bowl", 62: "chair", 63: "couch", 64: "potted plant",
    65: "bed", 67: "dining table", 70: "toilet", 72: "tv", 73: "laptop",
    74: "mouse", 75: "remote", 76: "keyboard", 77: "cell phone", 78: "microwave",
    79: "oven", 80: "toaster", 87: "scissors", 88: "teddy bear", 89: "hair drier", 90: "toothbrush"
}

SEEN_CLS_IDS = set(SEEN_CLASSES.keys())
UNSEEN_CLS_IDS = set(UNSEEN_CLASSES.keys())
ALL_CLASSES = {**SEEN_CLASSES, **UNSEEN_CLASSES}
NAME_TO_CLS_ID = {v: k for k, v in ALL_CLASSES.items()}

# 验证类别映射
print("="*50)
print("类别映射验证:")
test_names = ["person", "cat", "dog", "car", "book", "cake"]
for name in test_names:
    if name in NAME_TO_CLS_ID:
        print(f"  {name} -> ID {NAME_TO_CLS_ID[name]}")
    else:
        print(f"  {name} -> 未找到!")
print("="*50)

def get_prompt_text(cls_ids, prompt_type):
    """生成Prompt"""
    names = [ALL_CLASSES[cid] for cid in cls_ids if cid in ALL_CLASSES]
    if prompt_type == "prompt1":
        return ", ".join(names) + ", and other objects"
    elif prompt_type == "prompt2":
        formatted = []
        for n in names:
            article = "an" if n[0] in "aeiou" else "a"
            formatted.append(f"{article} {n}")
        return ", ".join(formatted) + ", and other objects"
    else:  # prompt3
        return ", ".join(names) + " in a scene"

def convert_bbox(boxes, h, w):
    """(cx,cy,w,h)[0,1] -> (x1,y1,w,h)像素，确保返回Python原生类型"""
    result = []
    for cx, cy, bw, bh in boxes:
        cx = float(cx) if hasattr(cx, 'item') else float(cx)
        cy = float(cy) if hasattr(cy, 'item') else float(cy)
        bw = float(bw) if hasattr(bw, 'item') else float(bw)
        bh = float(bh) if hasattr(bh, 'item') else float(bh)
        
        x_center = cx * w
        y_center = cy * h
        width = bw * w
        height = bh * h
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        
        # 确保坐标在图像范围内
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        width = min(width, w - x1)
        height = min(height, h - y1)
        
        # 验证bbox有效性
        if width <= 0 or height <= 0:
            print(f"警告: 无效bbox: x1={x1}, y1={y1}, w={width}, h={height}")
            continue
            
        result.append([float(x1), float(y1), float(width), float(height)])
    
    # # 打印第一个bbox作为示例
    # if result:
    #     print(f"bbox示例: {result[0]}")
    return result

def match_category(phrase, cls_ids):
    """将phrase匹配到category_id（长度优先）"""
    phrase_clean = phrase.strip().lower()
    # print(f"匹配短语: '{phrase_clean}'")
    
    if phrase_clean in NAME_TO_CLS_ID:
        return NAME_TO_CLS_ID[phrase_clean]
    
    # 按长度降序：避免"car"匹配到"carrot"
    candidates = sorted([ALL_CLASSES[cid] for cid in cls_ids if cid in ALL_CLASSES], 
                       key=len, reverse=True)
    for name in candidates:
        if name.lower() in phrase_clean:
            # print(f"  匹配到: {name}")
            return NAME_TO_CLS_ID.get(name)
    return None

def run_inference(img_path, prompt, img_id, cls_ids):
    """推理单张图片，只收集指定类别"""
    global MODEL
    try:
        image_source, image = load_image(img_path)
        boxes, logits, phrases = predict(
            model=MODEL, image=image, caption=prompt,
            box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD, device=DEVICE)
        
        img = Image.open(img_path)
        w, h = img.size
        bboxes = convert_bbox(boxes, h, w)
        
        # # === 调试：加载该图片的GT，查看正确的类别 ===
        # with open(ANNO_PATH) as f:
        #     anno_data = json.load(f)
        
        # # 找出这张图片的GT标注
        # gt_annos = [ann for ann in anno_data['annotations'] if ann['image_id'] == img_id]
        # gt_cats = set([ann['category_id'] for ann in gt_annos])
        # gt_names = [ALL_CLASSES.get(cid, 'unknown') for cid in gt_cats]
        # print(f"\n图片 {img_id} 的GT类别: {gt_names}")
        # # === 调试结束 ===
        
        detections = []
        for bbox, logit, phrase in zip(bboxes, logits, phrases):
            cat_id = match_category(phrase, cls_ids)
            if cat_id is None or cat_id not in cls_ids:
                continue
            score = float(logit.item()) if hasattr(logit, 'item') else float(logit)
            
            # # 打印匹配到的检测框信息
            # print(f"  检测到: {ALL_CLASSES.get(cat_id, 'unknown')} (ID:{cat_id}), score={score:.3f}, bbox={bbox}")
            
            clean_bbox = [float(x) for x in bbox]
            detections.append({
                "image_id": int(img_id),
                "category_id": int(cat_id),
                "bbox": clean_bbox,
                "score": score,
                "prompt": str(phrase).strip().lower()
            })
        
        # # 打印匹配统计
        # print(f"  匹配到 {len(detections)} 个框")
        return detections
    except Exception as e:
        print(f"[ERROR] {os.path.basename(img_path)}: {e}")
        return []

def main():
    global MODEL
    print("="*70)
    print("GroundingDINO推理 - SEEN/UNSEEN严格分离")
    print("="*70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(RANDOM_SEED)
    
    # 加载COCO
    with open(ANNO_PATH) as f:
        anno = json.load(f)
    
    img_info = {}
    for img in anno['images']:
        path = os.path.join(IMG_DIR, img['file_name'])
        if os.path.exists(path):
            img_info[img['id']] = path
    
    # 统计
    img_id_seen = set()
    img_id_unseen = set()
    for ann in anno['annotations']:
        img_id = ann['image_id']
        cat_id = ann['category_id']
        if img_id not in img_info:
            continue
        if cat_id in UNSEEN_CLS_IDS:
            img_id_unseen.add(img_id)
        elif cat_id in SEEN_CLS_IDS:
            img_id_seen.add(img_id)
    
    print(f"SEEN图片: {len(img_id_seen)} | UNSEEN图片: {len(img_id_unseen)}")
    
    # 加载模型
    print("加载模型...")
    MODEL = load_model(MODEL_CONFIG_PATH, MODEL_WEIGHTS_PATH)
    MODEL.to(DEVICE)
    print("模型加载完成\n")
    
    # 抽样
    sampled_seen = random.sample(list(img_id_seen), min(SEEN_SAMPLE_NUM, len(img_id_seen)))
    sampled_unseen = random.sample(list(img_id_unseen), min(UNSEEN_SAMPLE_NUM, len(img_id_unseen)))
    
    # 推理SEEN
    print(f"推理SEEN ({len(sampled_seen)}张)...")
    seen_prompt = get_prompt_text(SEEN_CLS_IDS, PROMPT_ROUND)
    print(f"Prompt: {seen_prompt[:100]}...\n")
    
    seen_results = []
    for img_id in tqdm(sampled_seen):
        dets = run_inference(img_info[img_id], seen_prompt, img_id, SEEN_CLS_IDS)
        seen_results.extend(dets)
    
    # 推理UNSEEN
    print(f"\n推理UNSEEN ({len(sampled_unseen)}张)...")
    unseen_prompt = get_prompt_text(UNSEEN_CLS_IDS, PROMPT_ROUND)
    print(f"Prompt: {unseen_prompt}\n")
    
    unseen_results = []
    for img_id in tqdm(sampled_unseen):
        dets = run_inference(img_info[img_id], unseen_prompt, img_id, UNSEEN_CLS_IDS)
        unseen_results.extend(dets)
    
    # 保存
    seen_file = os.path.join(OUTPUT_DIR, f"coco_seen_400imgs_{PROMPT_ROUND}.json")
    unseen_file = os.path.join(OUTPUT_DIR, f"coco_unseen_100imgs_{PROMPT_ROUND}.json")
    
    with open(seen_file, 'w') as f:
        json.dump(seen_results, f)
    
    with open(unseen_file, 'w') as f:
        json.dump(unseen_results, f)
    
    print(f"\n完成")
    print(f"SEEN: {len(seen_results)} 框 → {seen_file}")
    print(f"UNSEEN: {len(unseen_results)} 框 → {unseen_file}")

if __name__ == "__main__":
    main()