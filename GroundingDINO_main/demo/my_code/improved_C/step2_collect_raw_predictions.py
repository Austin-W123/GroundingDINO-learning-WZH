# step2_collect_raw_predictions.py
"""
第2步：在验证集上运行推理，收集所有原始预测结果
使用极低阈值（0.01）确保不丢失任何候选框
"""
import os
import sys
import json
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
    print("请确保GroundingDINO已正确安装")
    sys.exit(1)

# 配置
BASE_DIR = "D:/groundingdino_work/GroundingDINO-main"
MODEL_CONFIG_PATH = os.path.join(BASE_DIR, "groundingdino/config/GroundingDINO_SwinB_cfg.py")
MODEL_WEIGHTS_PATH = os.path.join(BASE_DIR, "weights/groundingdino_swinb_cogcoor.pth")
ANNO_PATH = os.path.join(BASE_DIR, "data/coco/annotations/instances_val2017.json")
VAL_SET_DIR = os.path.join(BASE_DIR, "data/coco/val_set_for_threshold")
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

# SEEN类别ID
SEEN_CLS_IDS = {
    1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,
    21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,
    40,41,42,43,44,46,47,48,49,50,51,62,63,64,65,67,
    70,72,73,74,75,76,77,78,79,80,87,88,89,90
}

NAME_TO_ID = {v: k for k, v in CLASS_NAMES.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

def get_prompt_text(cls_ids):
    """生成Prompt"""
    names = [CLASS_NAMES[cid] for cid in cls_ids if cid in CLASS_NAMES]
    return ", ".join(names) + "."

def convert_bbox(boxes, h, w):
    """(cx,cy,w,h)[0,1] -> (x1,y1,w,h)像素"""
    result = []
    for box in boxes:
        cx, cy, bw, bh = box.tolist() if hasattr(box, 'tolist') else box
        x1 = (cx - bw/2) * w
        y1 = (cy - bh/2) * h
        width = bw * w
        height = bh * h
        result.append([float(x1), float(y1), float(width), float(height)])
    return result

def match_category(phrase):
    """将phrase匹配到category_id"""
    phrase_clean = phrase.strip().lower()
    if phrase_clean in NAME_TO_ID:
        return NAME_TO_ID[phrase_clean]
    
    # 按长度降序匹配
    candidates = sorted(CLASS_NAMES.values(), key=len, reverse=True)
    for name in candidates:
        if name.lower() in phrase_clean:
            return NAME_TO_ID.get(name)
    return None

def collect_raw_predictions():
    """收集原始预测结果（不设阈值过滤）"""
    print("=" * 60)
    print("第2步：收集原始预测结果")
    print("=" * 60)
    
    # 加载验证集信息
    val_info_path = os.path.join(OUTPUT_DIR, 'val_set_info.json')
    with open(val_info_path, 'r') as f:
        val_info = json.load(f)
    
    val_images = val_info['val_images']
    print(f"验证集图片数: {len(val_images)}")
    
    # 加载模型
    print("加载模型...")
    model = load_model(MODEL_CONFIG_PATH, MODEL_WEIGHTS_PATH)
    model.to(device)
    model.eval()
    print("模型加载完成")
    
    # 生成prompt
    prompt = get_prompt_text(SEEN_CLS_IDS)
    print(f"Prompt: {prompt[:100]}...")
    
    # 收集所有原始预测
    all_raw_detections = []
    
    for img_info in tqdm(val_images, desc="推理验证集"):
        img_id = img_info['id']
        file_name = img_info['file_name']
        img_path = os.path.join(VAL_SET_DIR, file_name)
        
        if not os.path.exists(img_path):
            continue
        
        try:
            # 使用极低阈值0.01，保留几乎所有候选框
            image_source, image = load_image(img_path)
            boxes, logits, phrases = predict(
                model=model,
                image=image,
                caption=prompt,
                box_threshold=0.01,  # 极低阈值
                text_threshold=0.01,
                device=device
            )
            
            if len(boxes) == 0:
                continue
            
            # 获取图像尺寸
            img = Image.open(img_path)
            w, h = img.size
            bboxes = convert_bbox(boxes, h, w)
            
            for bbox, logit, phrase in zip(bboxes, logits, phrases):
                cat_id = match_category(phrase)
                if cat_id is None or cat_id not in SEEN_CLS_IDS:
                    continue
                
                score = float(logit.item()) if hasattr(logit, 'item') else float(logit)
                
                all_raw_detections.append({
                    'image_id': img_id,
                    'category_id': cat_id,
                    'bbox': bbox,
                    'score': score,
                    'file_name': file_name
                })
        
        except Exception as e:
            print(f"处理图片 {file_name} 时出错: {e}")
    
    # 保存原始预测结果
    raw_output_path = os.path.join(OUTPUT_DIR, 'raw_predictions_val.json')
    with open(raw_output_path, 'w') as f:
        json.dump({
            'detections': all_raw_detections,
            'num_detections': len(all_raw_detections),
            'num_images': len(val_images),
            'prompt': prompt
        }, f, indent=2)
    
    print(f"\n✅ 原始预测结果已保存: {raw_output_path}")
    print(f"总检测框数: {len(all_raw_detections)}")
    
    return all_raw_detections

if __name__ == "__main__":
    collect_raw_predictions()