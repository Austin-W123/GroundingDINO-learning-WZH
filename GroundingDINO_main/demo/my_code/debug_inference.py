#!/usr/bin/env python3
"""
最小化调试脚本：逐步测试GroundingDINO推理工作流
"""
import os
import sys
import torch
import warnings
warnings.filterwarnings("ignore")

# 添加项目路径
BASE_DIR = "D:/groundingdino_work/GroundingDINO-main"
sys.path.insert(0, BASE_DIR)

# =============================================================================
# 1. 环境和导入检查
# =============================================================================
print("=" * 80)
print("🔍 调试步骤1：环境和导入检查")
print("=" * 80)

try:
    from groundingdino.util.inference import load_model, load_image, predict
    print("✅ GroundingDINO模块导入成功")
except ImportError as e:
    print(f"❌ 导入失败：{e}")
    sys.exit(1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 设备：{DEVICE}")

# =============================================================================
# 2. 模型加载检查
# =============================================================================
print("\n" + "=" * 80)
print("🔍 调试步骤2：模型加载检查")
print("=" * 80)

MODEL_CONFIG_PATH = os.path.join(BASE_DIR, "groundingdino/config/GroundingDINO_SwinB_cfg.py")
MODEL_WEIGHTS_PATH = os.path.join(BASE_DIR, "weights/groundingdino_swinb_cogcoor.pth")

print(f"配置文件：{MODEL_CONFIG_PATH}")
print(f"存在？{os.path.exists(MODEL_CONFIG_PATH)}")
print(f"权重文件：{MODEL_WEIGHTS_PATH}")
print(f"存在？{os.path.exists(MODEL_WEIGHTS_PATH)}")

try:
    model = load_model(MODEL_CONFIG_PATH, MODEL_WEIGHTS_PATH)
    model.to(DEVICE)
    print(f"✅ 模型加载成功，已送至 {DEVICE}")
except Exception as e:
    print(f"❌ 模型加载失败：{type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# =============================================================================
# 3. 图片加载检查
# =============================================================================
print("\n" + "=" * 80)
print("🔍 调试步骤3：图片加载检查")
print("=" * 80)

# 列出val2017目录中的前5张图片
IMG_DIR = os.path.join(BASE_DIR, "data/coco/val2017")
if os.path.exists(IMG_DIR):
    img_files = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))])[:5]
    print(f"✅ 找到 {len(img_files)} 张测试图片（共{len(os.listdir(IMG_DIR))}张）")
    
    # 选择第一张图片进行测试
    test_img_path = os.path.join(IMG_DIR, img_files[0])
    print(f"   测试图片：{img_files[0]}")
    print(f"   完整路径：{test_img_path}")
    print(f"   存在？{os.path.exists(test_img_path)}")
    
    try:
        image_source, image = load_image(test_img_path)
        print(f"✅ 图片加载成功")
        print(f"   原始形状：{image_source.shape if hasattr(image_source, 'shape') else 'N/A'}")
        print(f"   处理后形状：{image.shape if hasattr(image, 'shape') else 'N/A'}")
    except Exception as e:
        print(f"❌ 图片加载失败：{e}")
        sys.exit(1)
else:
    print(f"❌ 数据目录不存在：{IMG_DIR}")
    sys.exit(1)

# =============================================================================
# 4. 推理测试（简单Prompt）
# =============================================================================
print("\n" + "=" * 80)
print("🔍 调试步骤4：推理测试（简单Prompt）")
print("=" * 80)

# 尝试最简单的Prompt
simple_prompt = "person"
print(f"📝 Prompt：'{simple_prompt}'")
print(f"框阈值：0.1")
print(f"文本阈值：0.15")

try:
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=simple_prompt,
        box_threshold=0.1,
        text_threshold=0.15,
        device=DEVICE
    )
    
    print(f"✅ 推理完成")
    print(f"   返回框数：{len(boxes)}")
    print(f"   Logits形状：{logits.shape if hasattr(logits, 'shape') else len(logits)}")
    print(f"   Phrases：{phrases}")
    
    if len(boxes) > 0:
        print(f"   ✅ 检测到 {len(boxes)} 个框！")
        for i, (box, logit, phrase) in enumerate(zip(boxes[:3], logits[:3], phrases[:3])):
            print(f"      [{i}] phrase='{phrase}', logit={float(logit):.4f}, box={[float(x) for x in box[:2]]}")
    else:
        print(f"   ⚠️ 未检测到任何框")
        # 尝试更宽松的阈值
        print(f"\n   尝试更宽松的阈值...")
        boxes2, logits2, phrases2 = predict(
            model=model,
            image=image,
            caption=simple_prompt,
            box_threshold=0.01,
            text_threshold=0.01,
            device=DEVICE
        )
        print(f"   更宽松阈值(0.01, 0.01)：{len(boxes2)} 个框")
        if len(logits2) > 0:
            print(f"   Logits范围：min={float(logits2.min()):.4f}, max={float(logits2.max()):.4f}, mean={float(logits2.mean()):.4f}")
        
except Exception as e:
    print(f"❌ 推理失败：{type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# =============================================================================
# 5. 推理测试（多类别Prompt）
# =============================================================================
print("\n" + "=" * 80)
print("🔍 调试步骤5：推理测试（多类别Prompt）")
print("=" * 80)

multi_prompt = "person, car, dog, cat, bicycle"
print(f"📝 Prompt：'{multi_prompt}'")

try:
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=multi_prompt,
        box_threshold=0.1,
        text_threshold=0.15,
        device=DEVICE
    )
    
    print(f"✅ 推理完成")
    print(f"   返回框数：{len(boxes)}")
    print(f"   Phrases：{set(phrases) if len(phrases) > 0 else '无'}")
    
except Exception as e:
    print(f"❌ 推理失败：{e}")

# =============================================================================
# 总结
# =============================================================================
print("\n" + "=" * 80)
print("🎯 调试总结")
print("=" * 80)
print("""
如果上面所有步骤都成功且返回了检测框，那么问题可能出在：
1. PROMPT_CONFIG 的内容（类别名称格式）
2. Prompt 长度过长
3. 神经网络激活函数或Logit计算方式

如果只有最后几步才失败，问题出在：
1. 模型输出Logits过低（所有检测被阈值过滤）
2. 需要进一步降低阈值

如果前面的步骤失败了，检查：
1. 模型权重/配置路径
2. CUDA/CPU 切换
3. PyTorch 版本
""")
