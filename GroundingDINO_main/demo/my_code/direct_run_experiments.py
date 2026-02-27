"""
直接集成的三轮Prompt对比实验脚本
功能：在一个Python进程中直接运行三轮实验，无需调用子进程
使用方式：python direct_run_experiments.py
"""
import os
import sys
import json

# 将GroundingDINO项目根目录添加到Python路径
PROJECT_ROOT = "D:/groundingdino_work/GroundingDINO-main"
sys.path.append(PROJECT_ROOT)

DEMO_DIR = "D:/groundingdino_work/GroundingDINO-main/demo"
OUTPUT_DIR = "D:/groundingdino_work/GroundingDINO-main/results"

def run_single_prompt_experiment(prompt_round):
    """在同一进程中直接运行单轮实验"""
    print(f"\n{'='*70}")
    print(f"🚀 开始{prompt_round}轮实验（直接集成模式）")
    print(f"{'='*70}\n")
    
    # 设置PROMPT_ROUND
    os.environ["PROMPT_ROUND_OVERRIDE"] = prompt_round
    
    # 修改inference_coco.py
    inference_file = os.path.join(DEMO_DIR, "inference_coco.py")
    
    print(f"📝 正在修改 inference_coco.py...")
    with open(inference_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 替换所有PROMPT_ROUND设置
    for old_round in ["prompt1", "prompt2", "prompt3"]:
        content = content.replace(
            f'PROMPT_ROUND = "{old_round}"',
            f'PROMPT_ROUND = "{prompt_round}"'
        )
    
    with open(inference_file, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"✅ 已设置Prompt轮次为：{prompt_round}")
    
    # 修改eval_coco.py
    eval_file = os.path.join(DEMO_DIR, "eval_coco.py")
    
    print(f"📝 正在修改 eval_coco.py...")
    with open(eval_file, "r", encoding="utf-8") as f:
        eval_content = f.read()
    
    for old_round in ["prompt1", "prompt2", "prompt3"]:
        eval_content = eval_content.replace(
            f'PROMPT_VERSION = "{old_round}"',
            f'PROMPT_VERSION = "{prompt_round}"'
        )
    
    with open(eval_file, "w", encoding="utf-8") as f:
        f.write(eval_content)
    
    print(f"✅ 已设置eval评测Prompt版本为：{prompt_round}")
    
    # Step 1: 运行推理
    print(f"\n📝 运行推理脚本（这可能需要30-120分钟）...")
    print("=" * 70)
    
    # 直接导入并执行inference_coco.py的main函数
    try:
        # 保存当前工作目录
        original_dir = os.getcwd()
        os.chdir(DEMO_DIR)
        
        # 清除所有inference_coco和eval_coco相关的模块缓存
        modules_to_remove = [m for m in list(sys.modules.keys()) 
                            if 'inference_coco' in m or 'eval_coco' in m or 'groundingdino' in m]
        for module in modules_to_remove:
            try:
                del sys.modules[module]
            except:
                pass
        
        # 重新添加项目路径
        PROJECT_ROOT = os.path.join(DEMO_DIR, "..")
        if PROJECT_ROOT not in sys.path:
            sys.path.insert(0, PROJECT_ROOT)
        
        # 导入并执行推理脚本
        print(f"🔄 导入 inference_coco 模块...")
        import importlib
        if 'inference_coco' in sys.modules:
            importlib.reload(sys.modules['inference_coco'])
        else:
            import inference_coco
        
        print(f"🔄 运行推理 main() 函数...")
        inference_coco.main()
        
        # 恢复原工作目录
        os.chdir(original_dir)
        
        print("\n" + "=" * 70)
        print(f"✅ {prompt_round}轮推理完成！")
        
    except Exception as e:
        print(f"\n❌ {prompt_round}轮推理失败：{str(e)}")
        print(f"   错误类型：{type(e).__name__}")
        import traceback
        traceback.print_exc()
        os.chdir(original_dir)
        return False
    
    # Step 2: 运行评测
    print(f"\n📝 运行评测脚本（这可能需要10-20分钟）...")
    print("=" * 70)
    
    try:
        # 保存当前工作目录
        original_dir = os.getcwd()
        os.chdir(DEMO_DIR)
        
        # 清除之前导入的模块缓存
        modules_to_remove = [m for m in list(sys.modules.keys()) 
                            if 'eval_coco' in m]
        for module in modules_to_remove:
            try:
                del sys.modules[module]
            except:
                pass
        
        # 重新添加项目路径
        PROJECT_ROOT = os.path.join(DEMO_DIR, "..")
        if PROJECT_ROOT not in sys.path:
            sys.path.insert(0, PROJECT_ROOT)
        
        # 导入并执行评测脚本
        print(f"🔄 导入 eval_coco 模块...")
        import importlib
        if 'eval_coco' in sys.modules:
            importlib.reload(sys.modules['eval_coco'])
        else:
            import eval_coco
        
        print(f"🔄 运行评测 main() 函数...")
        eval_coco.main()
        
        # 恢复原工作目录
        os.chdir(original_dir)
        
        print("\n" + "=" * 70)
        print(f"✅ {prompt_round}轮评测完成！")
        return True
        
    except Exception as e:
        print(f"\n❌ {prompt_round}轮评测失败：{str(e)}")
        print(f"   错误类型：{type(e).__name__}")
        import traceback
        traceback.print_exc()
        os.chdir(original_dir)
        return False

def check_results():
    """检查三轮结果是否完成"""
    print(f"\n{'='*70}")
    print(f"📊 检查三轮实验结果")
    print(f"{'='*70}\n")
    
    results_status = {}
    
    for prompt_round in ["prompt1", "prompt2", "prompt3"]:
        seen_file = os.path.join(OUTPUT_DIR, f"coco_seen_400imgs_{prompt_round}.json")
        unseen_file = os.path.join(OUTPUT_DIR, f"coco_unseen_100imgs_{prompt_round}.json")
        eval_txt = os.path.join(OUTPUT_DIR, f"coco_eval_result_{prompt_round}.txt")
        
        # 检查文件是否存在且非空
        seen_ok = os.path.exists(seen_file) and os.path.getsize(seen_file) > 100
        unseen_ok = os.path.exists(unseen_file) and os.path.getsize(unseen_file) > 100
        eval_ok = os.path.exists(eval_txt) and os.path.getsize(eval_txt) > 100
        
        if seen_ok:
            with open(seen_file, "r") as f:
                seen_count = len(json.load(f))
            seen_info = f"✅ SEEN检测框: {seen_count}个"
        else:
            seen_info = "❌ SEEN检测框: 缺失"
        
        if unseen_ok:
            with open(unseen_file, "r") as f:
                unseen_count = len(json.load(f))
            unseen_info = f"✅ UNSEEN检测框: {unseen_count}个"
        else:
            unseen_info = "❌ UNSEEN检测框: 缺失"
        
        eval_info = "✅ 评测结果: 完成" if eval_ok else "❌ 评测结果: 缺失"
        
        all_ok = seen_ok and unseen_ok and eval_ok
        status = "✅ 完成" if all_ok else "⏳ 未完成"
        
        print(f"{prompt_round}:")
        print(f"  {status}")
        print(f"  {seen_info}")
        print(f"  {unseen_info}")
        print(f"  {eval_info}\n")
        
        results_status[prompt_round] = all_ok
    
    return results_status

def main():
    """主函数"""
    print("="*70)
    print("🎯 GroundingDINO Prompt工程对比实验（三轮直接集成模式）")
    print("="*70)
    
    print("\n📋 实验计划：")
    print("  Round 1 (prompt1): 纯类名 (Pure Class Names)")
    print("  Round 2 (prompt2): 模板句 (Template Sentences)")
    print("  Round 3 (prompt3): 细粒度描述 (Fine-grained Descriptions)")
    
    print("\n⏱️ 预计耗时：60-180分钟（取决于GPU/CPU性能）")
    print("📌 注意：脚本会显示详细的进度信息，包括进度条和输出日志\n")
    
    user_input = input("是否开始运行三轮实验？(yes/no): ").strip().lower()
    if user_input != "yes":
        print("❌ 已取消实验")
        return
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n📁 结果目录: {OUTPUT_DIR}")
    
    # 运行三轮实验
    print("\n" + "="*70)
    print("开始执行三轮实验...")
    print("="*70)
    
    completed = []
    failed = []
    
    for i, prompt_round in enumerate(["prompt1", "prompt2", "prompt3"], 1):
        print(f"\n[{i}/3] 运行{prompt_round}轮实验")
        try:
            success = run_single_prompt_experiment(prompt_round)
            if success:
                completed.append(prompt_round)
            else:
                failed.append(prompt_round)
        except KeyboardInterrupt:
            print("\n⚠️ 用户中断了实验")
            failed.append(prompt_round)
            break
        except Exception as e:
            print(f"\n❌ {prompt_round}轮实验异常: {str(e)}")
            failed.append(prompt_round)
    
    # 检查并显示结果
    check_results()
    
    # 最终总结
    print("\n" + "="*70)
    print("📌 实验执行总结")
    print("="*70)
    
    if completed:
        print(f"\n✅ 成功完成: {len(completed)}/3 轮")
        print(f"   轮次: {', '.join(completed)}")
    
    if failed:
        print(f"\n❌ 失败: {len(failed)}/3 轮")
        print(f"   轮次: {', '.join(failed)}")
        print("\n   故障排查建议：")
        print("   1. 查看上述错误日志")
        print("   2. 检查磁盘空间是否充足")
        print("   3. 检查COCO数据集是否完整")
        print("   4. 重新运行单个轮次进行调试")
    
    if set(completed) == {"prompt1", "prompt2", "prompt3"}:
        print("\n✅ 所有三轮实验均成功完成！")
        print("\n🎉 下一步：")
        print("   1. 生成对比分析：python compare_prompt_results.py")
        print("   2. 查看对比表格：results/prompt_comparison_summary.csv")
        print("   3. 查看对比报告：results/prompt_comparison_analysis.txt")
    
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
