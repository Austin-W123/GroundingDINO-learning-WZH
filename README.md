# GroundingDINO - 零样本目标检测项目

## 项目概述

本项目基于 [GroundingDINO官方仓库](https://github.com/IDEA-Research/GroundingDINO) 进行二次开发，用于完成 **ComSen2026寒假任务 - 算法方向** 的零样本目标检测任务。 详细任务要求：[ComSen2026_Winter_Break_Assignment](ComSen2026_Winter_Break_Assignment.pdf)

**核心工作**：
- 在COCO数据集上构建seen/unseen类别划分（65/15 split）
- 系统研究不同prompt形式对零样本检测的影响
- 探索并实现改进方案
- 深入分析失败案例，总结错误模式

**项目性质**：本项目是**探索性研究**，包含多阶段的尝试和迭代，而非简单的线性实验。所有探索过程均保留在 `results/` 目录中。

---

## 快速开始

### 环境配置

```bash
# 克隆本仓库
git clone https://github.com/Austin-W123/GroundingDINO-learning-WZH
cd GroundingDINO-learning-WZH
```
我的详细环境配置见：[my_environment](my_environment.txt)、[my_requirements](my_requirements.txt)
```bash
# 一键安装依赖
# 1. 创建新环境（可选）
conda create -n groundingdino python=3.10
conda activate groundingdino

# 2. 一键安装所有依赖
pip install -r my_requirements.txt
```
## 数据集说明
数据集结构
```bash
data/
├── annotations/          # 标注文件
│   ├── instances_train2017.json
│   └── instances_val2017.json
├── train2017/            # 训练集图片
└── val2017/              # 验证集图片
```
### COCO 2017 官方下载链接

| 文件 | 大小 | 下载链接 |
|------|------|----------|
| 训练集图片 (train2017) | 18GB | [下载链接](http://images.cocodataset.org/zips/train2017.zip) |
| 验证集图片 (val2017) | 1GB | [下载链接](http://images.cocodataset.org/zips/val2017.zip) |
| 标注文件 (annotations) | 241MB | [下载链接](http://images.cocodataset.org/annotations/annotations_trainval2017.zip) |

## 项目结构
我首先克隆了groundingDINO官方仓库到本地，随后在官方给出的代码中做了一些微小调整来适应我本地的情况，以便模型可以正常加载。之后编写了代码来实现具体的任务，具体存放在`my_code/`文件夹中。所有的结果均存放在文件夹`results/`中。未作详细说明的均为官方仓库中的代码。以下是我的项目结构。
```bash
GROUNDINGDINO_WORK/
├── GroundingDINO_main/ # 主项目目录
│ ├── demo/ # 官方demo文件夹（包含我的自定义代码）
│ ├── my_code/ # 我的自定义代码
│ │ ├── compare_prompt_results.py # 结果对比代码
│ │ ├── debug_inference.py # 运行过程中寻找bug所在的过程性代码
│ │ ├── direct_run_experiments.py # 一键运行三次prompt推理+评测代码
│ │ ├── eval_coco_improved_C.py # 方向C小幅改进后的具体评测代码
│ │ ├── eval_coco.py # 三轮prompt对比试验的评测代码
│ │ ├── find_best_thresholds.py # 方向C寻找每个类别的最佳阈值代码
│ │ ├── inference_coco_improved_C.py # 方向C小幅改进后的具体推理代码
│ │ ├── inference_coco.py # 三轮prompt对比试验的推理代码
│ │ └── inference_on_a_image.py # 针对单张图片的推理代码
│ ├── create_coco_dataset.py
│ ├── gradio_app.py
│ ├── image_editing_with_groundingdino_gligen.ipynb
│ ├── image_editing_with_groundingdino_stablediffusion.ipynb
│ └── test_on_coco.py
│ ├── groundingdino/ # 官方核心代码
│ ├── groundingdino.egg-info/ # 包信息
│ ├── results/ # 实验结果
│ │ ├── groundingdino_test_results/ # 针对单张图片的评测结果
│ │ ├── results_improved_A/ # 方向A小幅改进后的结果
│ │ ├── results_improved_C/ # 方向C小幅改进后的结果
│ │ ├── results_prompt_experiment/ # 三次prompt对比实验结果
│ │ └── results0/ # 基线复现试运行结果
│ ├── weights/ # 预训练权重
│ ├── .gitignore
│ ├── docker_test.py
│ ├── Dockerfile
│ ├── environment.yaml # Conda环境配置
│ ├── LICENSE
│ ├── README.md
│ ├── requirements.txt
│ ├── setup.py
│ ├── test.ipynb
│ └── vs_BuildTools.exe
├── ComSen2026_Winter_Break_Assignment.pdf # 任务说明
└── groundingdino_learning.md # 学习笔记（过程性记录）

```

## 实验结果
详细的实验过程分析、结果分析和收获体会均在实验报告中。实验报告：[REPORT](REPORT.md)