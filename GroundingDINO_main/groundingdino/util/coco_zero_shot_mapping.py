# COCO2017 零样本划分映射表
# 存放路径：D:/groundingdino_work/GroundingDINO-main/groundingdino/util/coco_zero_shot_mapping.py

# Seen类（65类，与Unseen类完全互斥）
COCO_SEEN_CLASSES = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep",
    21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe",
    27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase",
    34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite",
    39: "baseball bat", 40: "baseball glove", 41: "skateboard", 42: "surfboard",
    43: "tennis racket", 44: "bottle", 46: "wine glass", 47: "cup", 48: "fork",
    49: "knife", 50: "spoon", 51: "bowl",
    62: "chair", 63: "couch", 65: "bed", 66: "dining table", 67: "toilet",
    74: "mouse", 75: "remote", 76: "keyboard", 78: "microwave", 79: "oven",
    80: "toaster", 81: "sink", 82: "refrigerator",
    84: "clock", 85: "vase", 86: "scissors", 87: "teddy bear", 88: "hair drier", 89: "toothbrush"
}

# Unseen类（15类，与Seen类完全互斥）
COCO_UNSEEN_CLASSES = {
    52: "banana", 53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli",
    57: "carrot", 58: "hot dog", 59: "pizza", 60: "donut", 61: "cake",
    64: "potted plant", 72: "tv", 73: "laptop", 77: "cell phone", 83: "book"
}

# 反向映射：类别名→ID（推理时匹配文本提示）
COCO_UNSEEN_NAME_TO_ID = {v: k for k, v in COCO_UNSEEN_CLASSES.items()}
COCO_SEEN_NAME_TO_ID = {v: k for k, v in COCO_SEEN_CLASSES.items()}

# 纯Unseen类名称列表（推理时直接调用）
COCO_UNSEEN_CLASS_NAMES = list(COCO_UNSEEN_CLASSES.values())