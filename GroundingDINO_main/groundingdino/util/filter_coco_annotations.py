import json
from coco_zero_shot_mapping import COCO_SEEN_CLASSES, COCO_UNSEEN_CLASSES  

# COCO标注路径
ANNOTATION_PATH = "D:/groundingdino_work/GroundingDINO-main/data/coco/annotations/instances_val2017.json"
OUTPUT_PATH = "D:/groundingdino_work/GroundingDINO-main/data/coco/annotations/instances_val2017_seen_only.json"

# 加载原始标注
with open(ANNOTATION_PATH, "r", encoding="utf-8") as f:
    coco_data = json.load(f)

# 过滤标注：只保留seen类的标注
seen_ids = set(COCO_SEEN_CLASSES.keys())
filtered_annotations = []
for ann in coco_data["annotations"]:
    if ann["category_id"] in seen_ids:
        filtered_annotations.append(ann)

# 过滤类别：只保留seen类的类别信息
filtered_categories = []
for cat in coco_data["categories"]:
    if cat["id"] in seen_ids:
        filtered_categories.append(cat)

# 生成过滤后的标注文件
coco_data["annotations"] = filtered_annotations
coco_data["categories"] = filtered_categories

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(coco_data, f, indent=2)

# 保留你的原始打印，仅新增Unseen类标注验证（核心必要补充，无其他修改）
print(f"过滤完成！生成仅含seen类的标注文件：{OUTPUT_PATH}")
print(f"Seen类数量：{len(filtered_categories)}，标注数量：{len(filtered_annotations)}")
# 新增验证：确保Unseen类无标注（零样本实验核心约束）
unseen_ids = set(COCO_UNSEEN_CLASSES.keys())
unseen_ann_count = sum(1 for ann in filtered_annotations if ann["category_id"] in unseen_ids)
print(f"Unseen类标注数量：{unseen_ann_count}（正常应为0，否则过滤失败）")