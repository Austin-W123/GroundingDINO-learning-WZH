# 基础调优参数
1.--box_threshold（默认 0.3）：控制检测框的置信度。

调大（如 0.5）：去掉低置信度的误检框；
调小（如 0.2）：找回漏检的框。

2.--text_threshold（默认 0.25）：控制文本和框的匹配度。

调大（如 0.4）：只有文本匹配度高的框才保留；
调小（如 0.2）：匹配度低的也保留。

pred0：
set PYTHONPATH=D:\groundingdino_work\GroundingDINO-main;%PYTHONPATH% 
&& python D:\groundingdino_work\GroundingDINO-main\demo\inference_on_a_image.py 
--config D:\groundingdino_work\GroundingDINO-main\groundingdino\config\GroundingDINO_SwinB_cfg.py 
--checkpoint D:\groundingdino_work\GroundingDINO-main\weights\groundingdino_swinb_cogcoor.pth 
--image_path D:\groundingdino_test\test.png 
--text_prompt "cat" 
--output_dir D:\groundingdino_test\output

**说明模型可以正常运行，cat检测良好，置信度、检测框等都没有大问题**



pred1（**测试精细描述--"black cat"**）：
set PYTHONPATH=D:\groundingdino_work\GroundingDINO-main;%PYTHONPATH% 
&& python D:\groundingdino_work\GroundingDINO-main\demo\inference_on_a_image.py 
--config D:\groundingdino_work\GroundingDINO-main\groundingdino\config\GroundingDINO_SwinB_cfg.py 
--checkpoint D:\groundingdino_work\GroundingDINO-main\weights\groundingdino_swinb_cogcoor.pth 
--image_path D:\groundingdino_test\test.png 
--text_prompt "black cat" 
--output_dir D:\groundingdino_test\output

**说明模型可以理解精细描述，将黑猫和别的猫进行区分**


pred2（**测试完整句子--"a cat sitting on the floor"**）：
set PYTHONPATH=D:\groundingdino_work\GroundingDINO-main;%PYTHONPATH% 
&& python D:\groundingdino_work\GroundingDINO-main\demo\inference_on_a_image.py 
--config D:\groundingdino_work\GroundingDINO-main\groundingdino\config\GroundingDINO_SwinB_cfg.py 
--checkpoint D:\groundingdino_work\GroundingDINO-main\weights\groundingdino_swinb_cogcoor.pth 
--image_path D:\groundingdino_test\test.png 
--text_prompt "a cat sitting on the floor" 
--output_dir D:\groundingdino_test\output

**说明模型可以理解长句子，但检测出的目标并不全面（漏检一只）**


pred3（**修改参数--**）：
set PYTHONPATH=D:\groundingdino_work\GroundingDINO-main;%PYTHONPATH% 
&& python D:\groundingdino_work\GroundingDINO-main\demo\inference_on_a_image.py 
--config D:\groundingdino_work\GroundingDINO-main\groundingdino\config\GroundingDINO_SwinB_cfg.py 
--checkpoint D:\groundingdino_work\GroundingDINO-main\weights\groundingdino_swinb_cogcoor.pth 
--image_path D:\groundingdino_test\test.png 
--text_prompt "a cat sitting on the floor" 
--output_dir D:\groundingdino_test\output 
--box_threshold 0.2 
--text_threshold 0.2

**接上，降低置信度和匹配度，找回漏检，但出现了误检**


pred4（**修改参数--**）：
set PYTHONPATH=D:\groundingdino_work\GroundingDINO-main;%PYTHONPATH% 
&& python D:\groundingdino_work\GroundingDINO-main\demo\inference_on_a_image.py 
--config D:\groundingdino_work\GroundingDINO-main\groundingdino\config\GroundingDINO_SwinB_cfg.py 
--checkpoint D:\groundingdino_work\GroundingDINO-main\weights\groundingdino_swinb_cogcoor.pth 
--image_path D:\groundingdino_test\test.png 
--text_prompt "a cat sitting on the floor" 
--output_dir D:\groundingdino_test\output 
--box_threshold 0.3 
--text_threshold 0.2

**接上，提升置信度，匹配度不变，出现漏检，误检消失，结果同pred2**


pred5（**修改参数--**）：
set PYTHONPATH=D:\groundingdino_work\GroundingDINO-main;%PYTHONPATH% 
&& python D:\groundingdino_work\GroundingDINO-main\demo\inference_on_a_image.py 
--config D:\groundingdino_work\GroundingDINO-main\groundingdino\config\GroundingDINO_SwinB_cfg.py 
--checkpoint D:\groundingdino_work\GroundingDINO-main\weights\groundingdino_swinb_cogcoor.pth 
--image_path D:\groundingdino_test\test.png 
--text_prompt "a cat sitting on the floor" 
--output_dir D:\groundingdino_test\output 
--box_threshold 0.2 
--text_threshold 0.3

**接上，恢复置信度，提高匹配度，找回漏检，出现误检，结果同pred3**


pred6（**修改参数--**）：
set PYTHONPATH=D:\groundingdino_work\GroundingDINO-main;%PYTHONPATH% 
&& python D:\groundingdino_work\GroundingDINO-main\demo\inference_on_a_image.py 
--config D:\groundingdino_work\GroundingDINO-main\groundingdino\config\GroundingDINO_SwinB_cfg.py 
--checkpoint D:\groundingdino_work\GroundingDINO-main\weights\groundingdino_swinb_cogcoor.pth 
--image_path D:\groundingdino_test\test.png 
--text_prompt "a cat sitting on the floor" 
--output_dir D:\groundingdino_test\output 
--box_threshold 0.2 
--text_threshold 0.25

**接上，置信度不变，降低匹配度，找回漏检，出现误检，结果同pred5**


pred7（**修改参数--**）：
set PYTHONPATH=D:\groundingdino_work\GroundingDINO-main;%PYTHONPATH% 
&& python D:\groundingdino_work\GroundingDINO-main\demo\inference_on_a_image.py 
--config D:\groundingdino_work\GroundingDINO-main\groundingdino\config\GroundingDINO_SwinB_cfg.py 
--checkpoint D:\groundingdino_work\GroundingDINO-main\weights\groundingdino_swinb_cogcoor.pth 
--image_path D:\groundingdino_test\test.png 
--text_prompt "a cat sitting on the floor" 
--output_dir D:\groundingdino_test\output 
--box_threshold 0.25 
--text_threshold 0.25

**接上，提高置信度，匹配度不变，找回漏检，误检消失，结果没有问题，参数完美**


另：**调试参数过程中发现，0.01的变化对结果没有很大影响，故调节过程应该以0.05为步长，且两个参数均调整时结果改变明显。**


# COCO2017 seen/unseen 标准化划分
1.划分依据和内容

seen 类（65 类）：有标注框，可用于模型微调 / 监督（GroundingDINO 预训练已包含，你只需用这些类验证 “已知类” 检测）

|cocoID | 类别|cocoID | 类别|cocoID | 类别|
| :--- | :---: | :--- | :---: | :--- | :---: |
|1	|person	        |2	|bicycle	        |3	|car
|4	|motorcycle	    |5	|airplane	        |6	|bus
|7	|train	        |8	|truck	            |9	|boat
|10	|traffic light	|11	|fire hydrant	    |13	|stop sign
|14	|parking meter	|15	|bench	            |16	|bird
|17	|cat	        |18	|dog	            |19	|horse
|20	|sheep	        |21	|cow	            |22	|elephant
|23	|bear	        |24	|zebra	            |25	|giraffe
|27	|backpack	    |28	|umbrella	        |31	|handbag
|32	|tie	        |33	|suitcase	        |34	|frisbee
|35	|skis	        |36	|snowboard	        |37	|sports ball
|38	|kite	        |39	|baseball bat	    |40	|baseball glove
|41	|skateboard	    |42	|surfboard	        |43	|tennis racket
|44	|bottle	        |46	|wine glass	        |47	|cup
|48	|fork	        |49	|knife	            |50	|spoon
|51	|bowl	        |55	|clock	            |56	|vase
|57	|scissors	    |58	|teddy bear	        |59	|hair drier
|60	|toothbrush	    |61	|chair	            |62	|couch
|63	|bed	        |64	|dining table	    |65	|toilet
|67	|tv	            |70	|mouse	            |71	|remote
|72	|keyboard	    |73	|cell phone	        |74	|microwave
|75	|oven	        |76	|toaster	        |77	|sink
|78	|refrigerator	|79	|book	            |80	|laptop


unseen 类（15 类）：无任何训练 / 调参用的标注框，仅用文本提示检测，验证模型 “零样本泛化能力”

|cocoID | 类别 | 提示词
| :--- | :---: | :---: |
|52	|banana	        |"banana" / "a yellow banana"
|53	|apple	        |"apple" / "a red apple"
|54	|sandwich	    |"sandwich" / "a ham sandwich"
|55	|orange	        |"orange" / "a round orange"
|56	|broccoli	    |"broccoli" / "green broccoli"
|57	|carrot	        |"carrot" / "orange carrot"
|58	|hot dog	    |"hot dog" / "a grilled hot dog"
|59	|pizza	        |"pizza" / "cheese pizza"
|60	|donut	        |"donut" / "chocolate donut"
|61	|cake	        |"cake" / "birthday cake"
|64	|potted plant	|"potted plant" / "green plant in pot"
|67	|tv	            |"tv" / "television screen"
|80	|laptop	        |"laptop" / "silver laptop"
|73	|cell phone	    |"cell phone" / "smartphone"
|79	|book	        |"book" / "thick book"

2.创建类别映射文件coco_zero_shot_mapping ，它的作用是定义 COCO2017 的 seen/unseen 类别列表，方便脚本调用。

3.过滤数据集标注（只保留 seen 类标注），编写脚本filter_coco_annotations.py，过滤instances_val2017.json，确保 unseen 类无标注

4.注意事项：

**严格零样本约束**：训练 / 调参时绝对不能用 unseen 类的标注，仅用文本提示检测。

**文本提示对齐**：检测 unseen 类时，文本提示要和类别名一致（如检测 banana 用 "banana"，不要用 “香蕉”）。

**评测只看 unseen 类**：后续计算 mAP/AP50 时，只统计 unseen 类的检测结果，这是零样本性能的核心指标。

# 推理+评测 结果展示（基础版）
PS D:\User\Microsoft VS Code> & D:\User\Anaconda\anaconda\envs\groundingdino\python.exe d:/groundingdino_work/GroundingDINO-main/demo/eval_coco.py
===== COCO检测结果全量评测（8张子图可视化，真实标注版） =====
📌 输入文件：D:/groundingdino_work/GroundingDINO-main\results\coco_seen_400imgs.json、D:/groundingdino_work/GroundingDINO-main\results\coco_unseen_100imgs.json
📌 输出文件：D:/groundingdino_work/GroundingDINO-main\results\coco_eval_result.txt、D:/groundingdino_work/GroundingDINO-main\results\metrics_full_comparison.png、D:/groundingdino_work/GroundingDINO-main\results\size_sensitive_metrics.png、D:/groundingdino_work/GroundingDINO-main\results\bbox_analysis.png、D:/groundingdino_work/GroundingDINO-main\results\recall_curve.png
📌 覆盖指标：COCO全量12项（AP@[0.5:0.95]、AP@0.5、AP@0.75、AP_small/medium/large、AR@1/10/100、AR_small/medium/large）

===== 开始SEEN类别评测（COCO全量12项指标） =====
loading annotations into memory...
Done (t=0.33s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.01s)
creating index...
index created!
✅ 加载成功：标注数=36781 | SEEN检测框数=3256
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=3.13s).
Accumulating evaluation results...
DONE (t=0.91s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.090
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.090
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.090
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.087
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.088
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.104
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.052
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.085
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.087
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.084
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.085
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.101

📊 SEEN全量评测指标：
   --- AP指标 ---
   AP@[0.5:0.95]   = 0.0905
   AP@0.5          = 0.0905
   AP@0.75         = 0.0905
   AP_small        = 0.0871
   AP_medium       = 0.0885
   AP_large        = 0.1036
   --- AR指标 ---
   AR@1            = 0.0524
   AR@10           = 0.0847
   AR@100          = 0.087
   AR_small        = 0.0839
   AR_medium       = 0.0852
   AR_large        = 0.1007

===== 开始UNSEEN类别评测（COCO全量12项指标） =====
loading annotations into memory...
Done (t=0.26s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.01s)
creating index...
index created!
✅ 加载成功：标注数=36781 | UNSEEN检测框数=655
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=2.84s).
Accumulating evaluation results...
DONE (t=0.69s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.030
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.030
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.030
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.040
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.026
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.025
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.012
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.028
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.029
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.039
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.026
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.024

📊 UNSEEN全量评测指标：
   --- AP指标 ---
   AP@[0.5:0.95]   = 0.0298
   AP@0.5          = 0.0298
   AP@0.75         = 0.0298
   AP_small        = 0.0399
   AP_medium       = 0.0265
   AP_large        = 0.0245
   --- AR指标 ---
   AR@1            = 0.0122
   AR@10           = 0.0277
   AR@100          = 0.0292
   AR_small        = 0.0392
   AR_medium       = 0.0259
   AR_large        = 0.0239

🔄 生成全量AP/AR指标对比图...
✅ 全量指标对比图已保存：D:/groundingdino_work/GroundingDINO-main\results\metrics_full_comparison.png

🔄 生成尺寸敏感指标对比图...
✅ 尺寸敏感指标图已保存：D:/groundingdino_work/GroundingDINO-main\results\size_sensitive_metrics.png

🔄 生成检测框深度分析图...
✅ 检测框分析图已保存：D:/groundingdino_work/GroundingDINO-main\results\bbox_analysis.png

🔄 生成召回率曲线+差值分析图...
✅ 召回率曲线图已保存：D:/groundingdino_work/GroundingDINO-main\results\recall_curve.png

🔄 保存全量评测汇总报告...
✅ 全量评测报告已保存：D:/groundingdino_work/GroundingDINO-main\results\coco_eval_result.txt

===== 全量评测+可视化完成 ======
📄 全量评测报告：D:/groundingdino_work/GroundingDINO-main\results\coco_eval_result.txt
📊 可视化文件：
   - 全量AP/AR指标对比：D:/groundingdino_work/GroundingDINO-main\results\metrics_full_comparison.png
   - 尺寸敏感指标对比：D:/groundingdino_work/GroundingDINO-main\results\size_sensitive_metrics.png
   - 检测框深度分析：D:/groundingdino_work/GroundingDINO-main\results\bbox_analysis.png
   - 召回率曲线+差值：D:/groundingdino_work/GroundingDINO-main\results\recall_curve.png
✅ 关键结论：覆盖COCO全量12项指标，8张子图可视化，所有指标非0，图像文本已改为英文！

# Prompt 工程与对比实验
reference_coco.py、eval_coco.py
## 第一轮
===== COCO检测结果全量评测（支持三轮对比） =====
📌 当前单轮Prompt：prompt1（Pure Class Name）
📌 输入文件：D:/groundingdino_work/GroundingDINO-main\results\coco_seen_400imgs_prompt1.json、D:/groundingdino_work/GroundingDINO-main\results\coco_unseen_100imgs_prompt1.json       

🔄 加载COCO 2017 val标注文件...
loading annotations into memory...
Done (t=0.34s)
creating index...
index created!
✅ 加载检测框：D:/groundingdino_work/GroundingDINO-main\results\coco_seen_400imgs_prompt1.json | 检测框数：1876
✅ 加载检测框：D:/groundingdino_work/GroundingDINO-main\results\coco_unseen_100imgs_prompt1.json | 检测框数：252

===== 开始SEEN类别评测（COCO全量12项指标） =====
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.29s).
Accumulating evaluation results...
DONE (t=0.17s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.286
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.344
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.308
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.113
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.291
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.489
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.313
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.359
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.359
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.136
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.344
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.566

===== 开始UNSEEN类别评测（COCO全量12项指标） =====
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.02s).
Accumulating evaluation results...
DONE (t=0.03s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.141
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.173
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.149
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.042
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.097
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.232
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.164
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.174
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.174
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.044
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.118
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.261

📊 分析 Pure Class Name 的UNSEEN类别...
    输入检测框总数: 252
    过滤后UNSEEN框数: 252
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.09s).
Accumulating evaluation results...
DONE (t=0.03s).
      类别 banana: AP@0.5=0.1089
      类别 apple: AP@0.5=0.2013
      类别 sandwich: AP@0.5=0.2475
      类别 orange: AP@0.5=0.0
      类别 broccoli: AP@0.5=0.139
      类别 carrot: AP@0.5=0.0594
      类别 hot dog: AP@0.5=0.0
      类别 pizza: AP@0.5=0.2591
      类别 donut: AP@0.5=0.0495
      类别 cake: AP@0.5=0.0
      类别 sink: AP@0.5=0.3244
      类别 refrigerator: AP@0.5=0.505
      类别 book: AP@0.5=0.0273
      类别 clock: AP@0.5=0.6139
      类别 vase: AP@0.5=0.0554
    Pure Class Name 平均指标: AP@[0.5:0.95]=0.1409, AP@0.5=0.1727

🔄 生成当前轮可视化图表...
✅ 全量指标对比图已保存：D:/groundingdino_work/GroundingDINO-main\results\metrics_full_comparison_prompt1.png
✅ 尺寸敏感指标图已保存：D:/groundingdino_work/GroundingDINO-main\results\size_sensitive_metrics_prompt1.png
✅ 检测框分析图已保存：D:/groundingdino_work/GroundingDINO-main\results\bbox_analysis_prompt1.png
✅ 召回率曲线已保存：D:/groundingdino_work/GroundingDINO-main\results\recall_curve_prompt1.png



## 第二轮
===== COCO检测结果全量评测（支持三轮对比） =====
📌 当前单轮Prompt：prompt2（Template Sentence）
📌 输入文件：D:/groundingdino_work/GroundingDINO-main\results\coco_seen_400imgs_prompt2.json、D:/groundingdino_work/GroundingDINO-main\results\coco_unseen_100imgs_prompt2.json

🔄 加载COCO 2017 val标注文件...
loading annotations into memory...
Done (t=0.34s)
creating index...
index created!
✅ 加载检测框：D:/groundingdino_work/GroundingDINO-main\results\coco_seen_400imgs_prompt2.json | 检测框数：998
✅ 加载检测框：D:/groundingdino_work/GroundingDINO-main\results\coco_unseen_100imgs_prompt2.json | 检测框数：315

===== 开始SEEN类别评测（COCO全量12项指标） =====
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.24s).
Accumulating evaluation results...
DONE (t=0.15s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.184
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.228
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.202
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.059
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.215
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.350
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.223
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.241
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.241
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.068
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.254
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.405

===== 开始UNSEEN类别评测（COCO全量12项指标） =====
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.03s).
Accumulating evaluation results...
DONE (t=0.03s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.113
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.158
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.117
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.048
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.116
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.158
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.153
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.174
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.174
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.048
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.154
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.230

📊 分析 Template Sentence 的UNSEEN类别...
    输入检测框总数: 315
    过滤后UNSEEN框数: 315
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.03s).
Accumulating evaluation results...
DONE (t=0.03s).
      类别 banana: AP@0.5=0.0801
      类别 apple: AP@0.5=0.1429
      类别 sandwich: AP@0.5=0.2672
      类别 orange: AP@0.5=0.1089
      类别 broccoli: AP@0.5=0.126
      类别 carrot: AP@0.5=0.1064
      类别 hot dog: AP@0.5=0.0
      类别 pizza: AP@0.5=0.0594
      类别 donut: AP@0.5=0.0
      类别 cake: AP@0.5=0.0
      类别 sink: AP@0.5=0.3366
      类别 refrigerator: AP@0.5=0.4555
      类别 book: AP@0.5=0.016
      类别 clock: AP@0.5=0.5014
      类别 vase: AP@0.5=0.177
    Template Sentence 平均指标: AP@[0.5:0.95]=0.1131, AP@0.5=0.1585

🔄 生成当前轮可视化图表...
✅ 全量指标对比图已保存：D:/groundingdino_work/GroundingDINO-main\results\metrics_full_comparison_prompt2.png
✅ 尺寸敏感指标图已保存：D:/groundingdino_work/GroundingDINO-main\results\size_sensitive_metrics_prompt2.png
✅ 检测框分析图已保存：D:/groundingdino_work/GroundingDINO-main\results\bbox_analysis_prompt2.png
✅ 召回率曲线已保存：D:/groundingdino_work/GroundingDINO-main\results\recall_curve_prompt2.png


## 第三轮
===== COCO检测结果全量评测（支持三轮对比） =====
📌 当前单轮Prompt：prompt3（Fine-grained Description）
📌 输入文件：D:/groundingdino_work/GroundingDINO-main\results\coco_seen_400imgs_prompt3.json、D:/groundingdino_work/GroundingDINO-main\results\coco_unseen_100imgs_prompt3.json       

🔄 加载COCO 2017 val标注文件...
loading annotations into memory...
Done (t=0.35s)
creating index...
index created!
✅ 加载检测框：D:/groundingdino_work/GroundingDINO-main\results\coco_seen_400imgs_prompt3.json | 检测框数：1867
✅ 加载检测框：D:/groundingdino_work/GroundingDINO-main\results\coco_unseen_100imgs_prompt3.json | 检测框数：301

===== 开始SEEN类别评测（COCO全量12项指标） =====
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.30s).
Accumulating evaluation results...
DONE (t=0.17s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.286
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.345
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.308
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.114
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.281
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.496
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.310
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.357
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.357
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.138
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.333
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.571

===== 开始UNSEEN类别评测（COCO全量12项指标） =====
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.10s).
Accumulating evaluation results...
DONE (t=0.03s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.166
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.205
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.177
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.045
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.122
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.266
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.186
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.213
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.213
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.047
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.159
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.308

📊 分析 Fine-grained Description 的UNSEEN类别...
    输入检测框总数: 301
    过滤后UNSEEN框数: 301
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.03s).
Accumulating evaluation results...
DONE (t=0.03s).
      类别 banana: AP@0.5=0.1119
      类别 apple: AP@0.5=0.1966
      类别 sandwich: AP@0.5=0.2266
      类别 orange: AP@0.5=0.0
      类别 broccoli: AP@0.5=0.1853
      类别 carrot: AP@0.5=0.1146
      类别 hot dog: AP@0.5=0.0
      类别 pizza: AP@0.5=0.4356
      类别 donut: AP@0.5=0.0495
      类别 cake: AP@0.5=0.0
      类别 sink: AP@0.5=0.4547
      类别 refrigerator: AP@0.5=0.4972
      类别 book: AP@0.5=0.0269
      类别 clock: AP@0.5=0.6787
      类别 vase: AP@0.5=0.0935
    Fine-grained Description 平均指标: AP@[0.5:0.95]=0.1661, AP@0.5=0.2047

🔄 生成当前轮可视化图表...
✅ 全量指标对比图已保存：D:/groundingdino_work/GroundingDINO-main\results\metrics_full_comparison_prompt3.png
✅ 尺寸敏感指标图已保存：D:/groundingdino_work/GroundingDINO-main\results\size_sensitive_metrics_prompt3.png
✅ 检测框分析图已保存：D:/groundingdino_work/GroundingDINO-main\results\bbox_analysis_prompt3.png
✅ 召回率曲线已保存：D:/groundingdino_work/GroundingDINO-main\results\recall_curve_prompt3.png


## 三轮对比
🔄 加载所有三轮结果生成对比图...
✅ 加载 Pure Class Name SEEN 结果: 1876 框
✅ 加载 Pure Class Name UNSEEN 结果: 252 框
✅ 加载 Template Sentence SEEN 结果: 998 框
✅ 加载 Template Sentence UNSEEN 结果: 315 框
✅ 加载 Fine-grained Description SEEN 结果: 1867 框
✅ 加载 Fine-grained Description UNSEEN 结果: 301 框

📊 开始分析三轮数据...

🔍 调试 - Pure Class Name 的检测框数量: 252
    示例类别ID: 85
    示例类别名称: clock

📊 分析 Pure Class Name 的UNSEEN类别...
    输入检测框总数: 252
    过滤后UNSEEN框数: 252
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.02s).
Accumulating evaluation results...
DONE (t=0.03s).
      类别 banana: AP@0.5=0.1089
      类别 apple: AP@0.5=0.2013
      类别 sandwich: AP@0.5=0.2475
      类别 orange: AP@0.5=0.0
      类别 broccoli: AP@0.5=0.139
      类别 carrot: AP@0.5=0.0594
      类别 hot dog: AP@0.5=0.0
      类别 pizza: AP@0.5=0.2591
      类别 donut: AP@0.5=0.0495
      类别 cake: AP@0.5=0.0
      类别 sink: AP@0.5=0.3244
      类别 refrigerator: AP@0.5=0.505
      类别 book: AP@0.5=0.0273
      类别 clock: AP@0.5=0.6139
      类别 vase: AP@0.5=0.0554
    Pure Class Name 平均指标: AP@[0.5:0.95]=0.1409, AP@0.5=0.1727
    计算得到的平均AP@0.5: 0.1727

🔍 调试 - Template Sentence 的检测框数量: 315
    示例类别ID: 84
    示例类别名称: book

📊 分析 Template Sentence 的UNSEEN类别...
    输入检测框总数: 315
    过滤后UNSEEN框数: 315
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.03s).
Accumulating evaluation results...
DONE (t=0.03s).
      类别 banana: AP@0.5=0.0801
      类别 apple: AP@0.5=0.1429
      类别 sandwich: AP@0.5=0.2672
      类别 orange: AP@0.5=0.1089
      类别 broccoli: AP@0.5=0.126
      类别 carrot: AP@0.5=0.1064
      类别 hot dog: AP@0.5=0.0
      类别 pizza: AP@0.5=0.0594
      类别 donut: AP@0.5=0.0
      类别 cake: AP@0.5=0.0
      类别 sink: AP@0.5=0.3366
      类别 refrigerator: AP@0.5=0.4555
      类别 book: AP@0.5=0.016
      类别 clock: AP@0.5=0.5014
      类别 vase: AP@0.5=0.177
    Template Sentence 平均指标: AP@[0.5:0.95]=0.1131, AP@0.5=0.1585
    计算得到的平均AP@0.5: 0.1585

🔍 调试 - Fine-grained Description 的检测框数量: 301
    示例类别ID: 85
    示例类别名称: clock

📊 分析 Fine-grained Description 的UNSEEN类别...
    输入检测框总数: 301
    过滤后UNSEEN框数: 301
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.03s).
Accumulating evaluation results...
DONE (t=0.03s).
      类别 banana: AP@0.5=0.1119
      类别 apple: AP@0.5=0.1966
      类别 sandwich: AP@0.5=0.2266
      类别 orange: AP@0.5=0.0
      类别 broccoli: AP@0.5=0.1853
      类别 carrot: AP@0.5=0.1146
      类别 hot dog: AP@0.5=0.0
      类别 pizza: AP@0.5=0.4356
      类别 donut: AP@0.5=0.0495
      类别 cake: AP@0.5=0.0
      类别 sink: AP@0.5=0.4547
      类别 refrigerator: AP@0.5=0.4972
      类别 book: AP@0.5=0.0269
      类别 clock: AP@0.5=0.6787
      类别 vase: AP@0.5=0.0935
    Fine-grained Description 平均指标: AP@[0.5:0.95]=0.1661, AP@0.5=0.2047
    计算得到的平均AP@0.5: 0.2047

📊 三轮对比数据:
  Pure Class Name: AP@[0.5:0.95]=0.1409, AP@0.5=0.1727
  Template Sentence: AP@[0.5:0.95]=0.1131, AP@0.5=0.1585
  Fine-grained Description: AP@[0.5:0.95]=0.1661, AP@0.5=0.2047
✅ Prompt对比柱状图已保存：D:/groundingdino_work/GroundingDINO-main\results\prompt_comparison_bar.png
✅ Prompt热力图对比已保存：D:/groundingdino_work/GroundingDINO-main\results\prompt_comparison_heatmap.png


结论：

细粒度描述效果最好 (20.5%)，比纯类名提升了约3个百分点

模板句效果反而略差 (15.9%)，可能因为模板过于通用

这说明详细的属性描述确实有助于零样本检测



# 小幅改进（方向C）
## 确定最优阈值

阈值配置文件已保存: D:/groundingdino_work/GroundingDINO-main/results\best_thresholds.json

最优阈值统计结果 (按阈值从高到低排序)

+------+---------------+--------+--------+-------+-------+--------+
|  ID  | 类别名称          |  检测框数  |  最优阈值  | 精确率   | 召回率   | F1分数   |
+======+===============+========+========+=======+=======+========+
|  14  | parking meter |   1    |  0.25  | -     | -     | -      |
+------+---------------+--------+--------+-------+-------+--------+
|  31  | handbag       |   6    |  0.25  | -     | -     | -      |
+------+---------------+--------+--------+-------+-------+--------+
|  34  | frisbee       |   1    |  0.25  | -     | -     | -      |
+------+---------------+--------+--------+-------+-------+--------+
|  36  | snowboard     |   30   |  0.25  | -     | -     | -      |
+------+---------------+--------+--------+-------+-------+--------+
|  38  | kite          |   1    |  0.25  | -     | -     | -      |
+------+---------------+--------+--------+-------+-------+--------+
|  87  | scissors      |   43   |  0.25  | -     | -     | -      |
+------+---------------+--------+--------+-------+-------+--------+
|  8   | truck         |   57   |  0.15  | 0.017 | 0.039 | 0.024  |
+------+---------------+--------+--------+-------+-------+--------+
|  1   | person        |  360   |  0.1   | 0.029 | 0.026 | 0.027  |
+------+---------------+--------+--------+-------+-------+--------+
|  2   | bicycle       |   13   |  0.1   | 0.013 | 0.013 | 0.013  |
+------+---------------+--------+--------+-------+-------+--------+
|  3   | car           |   72   |  0.1   | 0.028 | 0.030 | 0.029  |
+------+---------------+--------+--------+-------+-------+--------+
|  4   | motorcycle    |   36   |  0.1   | 0.036 | 0.046 | 0.041  |
+------+---------------+--------+--------+-------+-------+--------+
|  5   | airplane      |   22   |  0.1   | 0.066 | 0.070 | 0.068  |
+------+---------------+--------+--------+-------+-------+--------+
|  6   | bus           |   20   |  0.1   | 0.027 | 0.028 | 0.027  |
+------+---------------+--------+--------+-------+-------+--------+
|  7   | train         |   22   |  0.1   | 0.034 | 0.047 | 0.040  |
+------+---------------+--------+--------+-------+-------+--------+
|  9   | boat          |   13   |  0.1   | 0.017 | 0.014 | 0.016  |
+------+---------------+--------+--------+-------+-------+--------+
|  10  | traffic light |   27   |  0.1   | 0.010 | 0.019 | 0.013  |
+------+---------------+--------+--------+-------+-------+--------+
|  11  | fire hydrant  |   15   |  0.1   | 0.119 | 0.119 | 0.119  |
+------+---------------+--------+--------+-------+-------+--------+
|  15  | bench         |   90   |  0.1   | 0.025 | 0.041 | 0.031  |
+------+---------------+--------+--------+-------+-------+--------+
|  16  | bird          |   16   |  0.1   | 0.020 | 0.019 | 0.019  |
+------+---------------+--------+--------+-------+-------+--------+
|  17  | cat           |   17   |  0.1   | 0.069 | 0.064 | 0.066  |
+------+---------------+--------+--------+-------+-------+--------+
|  18  | dog           |   25   |  0.1   | 0.069 | 0.064 | 0.067  |
+------+---------------+--------+--------+-------+-------+--------+
|  19  | horse         |   29   |  0.1   | 0.046 | 0.044 | 0.045  |
+------+---------------+--------+--------+-------+-------+--------+
|  20  | sheep         |   12   |  0.1   | 0.028 | 0.028 | 0.028  |
+------+---------------+--------+--------+-------+-------+--------+
|  21  | cow           |   7    |  0.1   | 0.010 | 0.008 | 0.009  |
+------+---------------+--------+--------+-------+-------+--------+
|  22  | elephant      |   28   |  0.1   | 0.073 | 0.075 | 0.074  |
+------+---------------+--------+--------+-------+-------+--------+
|  23  | bear          |   18   |  0.1   | 0.070 | 0.099 | 0.082  |
+------+---------------+--------+--------+-------+-------+--------+
|  24  | zebra         |   13   |  0.1   | 0.024 | 0.030 | 0.027  |
+------+---------------+--------+--------+-------+-------+--------+
|  25  | giraffe       |   13   |  0.1   | 0.049 | 0.047 | 0.048  |
+------+---------------+--------+--------+-------+-------+--------+
|  27  | backpack      |   80   |  0.1   | 0.018 | 0.035 | 0.024  |
+------+---------------+--------+--------+-------+-------+--------+
|  28  | umbrella      |   54   |  0.1   | 0.027 | 0.034 | 0.030  |
+------+---------------+--------+--------+-------+-------+--------+
|  32  | tie           |   9    |  0.1   | 0.016 | 0.020 | 0.017  |
+------+---------------+--------+--------+-------+-------+--------+
|  33  | suitcase      |   14   |  0.1   | 0.008 | 0.017 | 0.011  |
+------+---------------+--------+--------+-------+-------+--------+
|  41  | skateboard    |   23   |  0.1   | 0.065 | 0.073 | 0.069  |
+------+---------------+--------+--------+-------+-------+--------+
|  42  | surfboard     |   14   |  0.1   | 0.001 | 0.004 | 0.001  |
+------+---------------+--------+--------+-------+-------+--------+
|  44  | bottle        |  102   |  0.1   | 0.017 | 0.026 | 0.021  |
+------+---------------+--------+--------+-------+-------+--------+
|  47  | cup           |   75   |  0.1   | 0.034 | 0.035 | 0.034  |
+------+---------------+--------+--------+-------+-------+--------+
|  48  | fork          |   19   |  0.1   | 0.059 | 0.051 | 0.055  |
+------+---------------+--------+--------+-------+-------+--------+
|  49  | knife         |   37   |  0.1   | 0.042 | 0.049 | 0.046  |
+------+---------------+--------+--------+-------+-------+--------+
|  50  | spoon         |   22   |  0.1   | 0.026 | 0.032 | 0.029  |
+------+---------------+--------+--------+-------+-------+--------+
|  51  | bowl          |   77   |  0.1   | 0.024 | 0.030 | 0.027  |
+------+---------------+--------+--------+-------+-------+--------+
|  62  | chair         |   68   |  0.1   | 0.025 | 0.024 | 0.025  |
+------+---------------+--------+--------+-------+-------+--------+
|  63  | couch         |   26   |  0.1   | 0.048 | 0.054 | 0.050  |
+------+---------------+--------+--------+-------+-------+--------+
|  64  | potted plant  |   6    |  0.1   | 0.018 | 0.015 | 0.016  |
+------+---------------+--------+--------+-------+-------+--------+
|  65  | bed           |   5    |  0.1   | 0.026 | 0.025 | 0.025  |
+------+---------------+--------+--------+-------+-------+--------+
|  70  | toilet        |   24   |  0.1   | 0.088 | 0.089 | 0.089  |
+------+---------------+--------+--------+-------+-------+--------+
|  72  | tv            |   18   |  0.1   | 0.040 | 0.038 | 0.039  |
+------+---------------+--------+--------+-------+-------+--------+
|  73  | laptop        |   22   |  0.1   | 0.040 | 0.039 | 0.039  |
+------+---------------+--------+--------+-------+-------+--------+
|  74  | mouse         |   22   |  0.1   | 0.053 | 0.066 | 0.059  |
+------+---------------+--------+--------+-------+-------+--------+
|  76  | keyboard      |   14   |  0.1   | 0.053 | 0.052 | 0.053  |
+------+---------------+--------+--------+-------+-------+--------+
|  77  | cell phone    |   16   |  0.1   | 0.046 | 0.042 | 0.044  |
+------+---------------+--------+--------+-------+-------+--------+
|  78  | microwave     |   24   |  0.1   | 0.129 | 0.127 | 0.128  |
+------+---------------+--------+--------+-------+-------+--------+
|  79  | oven          |   22   |  0.1   | 0.027 | 0.049 | 0.035  |
+------+---------------+--------+--------+-------+-------+--------+
|  80  | toaster       |   25   |  0.1   | 0.191 | 0.222 | 0.206  |
+------+---------------+--------+--------+-------+-------+--------+
|  88  | teddy bear    |   39   |  0.1   | 0.071 | 0.084 | 0.077  |
+------+---------------+--------+--------+-------+-------+--------+
|  90  | toothbrush    |   12   |  0.1   | 0.059 | 0.053 | 0.056  |
+------+---------------+--------+--------+-------+-------+--------+

📈 统计摘要:
  平均阈值: 0.117
  中位数阈值: 0.100
  最小阈值: 0.100
  最大阈值: 0.250

详细报告已保存: D:/groundingdino_work/GroundingDINO-main/results\threshold_analysis.txt


📁 生成文件汇总:
  1. 阈值配置文件: D:/groundingdino_work/GroundingDINO-main/results\best_thresholds.json
  2. 分析报告: D:/groundingdino_work/GroundingDINO-main/results\threshold_analysis.txt

## 利用最优阈值生成的结果(失败版)
小幅改进C - 基线与改进版对比评测


🔄 加载COCO 2017 val标注文件...
loading annotations into memory...
Done (t=0.36s)
creating index...
index created!

📊 加载基线结果...
✅ 加载检测框：D:/groundingdino_work/GroundingDINO-main\results\coco_seen_400imgs_prompt1.json | 检测框数：1876
✅ 加载检测框：D:/groundingdino_work/GroundingDINO-main\results\coco_unseen_100imgs_prompt1.json | 检测框数：252

📊 加载改进版结果...
✅ 加载检测框：D:/groundingdino_work/GroundingDINO-main\results\coco_seen_400imgs_prompt1_improved_C.json | 检测框数：11803
✅ 加载检测框：D:/groundingdino_work/GroundingDINO-main\results\coco_unseen_100imgs_prompt1_improved_C.json | 检测框数：257


评测基线结果...

===== 开始SEEN类别评测 =====
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.39s).
Accumulating evaluation results...
DONE (t=0.18s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.286
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.344
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.308
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.113
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.291
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.489
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.313
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.359
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.359
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.136
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.344
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.566

===== 开始UNSEEN类别评测 =====
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.02s).
Accumulating evaluation results...
DONE (t=0.03s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.141
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.173
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.149
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.042
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.097
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.232
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.164
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.174
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.174
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.044
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.118
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.261


评测改进版结果...

===== 开始SEEN类别评测 =====
Loading and preparing results...
DONE (t=0.01s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.61s).
Accumulating evaluation results...
DONE (t=0.29s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.162
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.214
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.176
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.083
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.224
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.254
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.218
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.305
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.309
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.153
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.342
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.436

===== 开始UNSEEN类别评测 =====
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.02s).
Accumulating evaluation results...
DONE (t=0.03s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.109
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.134
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.119
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.042
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.084
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.183
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.136
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.155
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.155
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.044
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.099
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.254


分析UNSEEN各类别详细指标...
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.02s).
Accumulating evaluation results...
DONE (t=0.03s).
Loading and preparing results...
DONE (t=0.02s).
Accumulating evaluation results...
DONE (t=0.03s).
Loading and preparing results...
DONE (t=0.03s).
Loading and preparing results...
DONE (t=0.00s)
creating index...
Loading and preparing results...
DONE (t=0.00s)
creating index...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.02s).
Accumulating evaluation results...
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.02s).
Accumulating evaluation results...
DONE (t=0.03s).

🔄 生成改进版可视化图表...
DONE (t=0.02s).
Accumulating evaluation results...
DONE (t=0.03s).

🔄 生成改进版可视化图表...
✅ 改进版全量指标对比图已保存：D:/groundingdino_work/GroundingDINO-main\results\metrics_full_comparison_improved_C.png
✅ 改进版尺寸敏感指标图已保存：D:/groundingdino_work/GroundingDINO-main\results\size_sensitive_metrics_improved_C.png
DONE (t=0.03s).

🔄 生成改进版可视化图表...
✅ 改进版全量指标对比图已保存：D:/groundingdino_work/GroundingDINO-main\results\metrics_full_comparison_improved_C.png
✅ 改进版尺寸敏感指标图已保存：D:/groundingdino_work/GroundingDINO-main\results\size_sensitive_metrics_improved_C.png
✅ 改进版检测框分析图已保存：D:/groundingdino_work/GroundingDINO-main\results\bbox_analysis_improved_C.png
✅ 改进版全量指标对比图已保存：D:/groundingdino_work/GroundingDINO-main\results\metrics_full_comparison_improved_C.png
✅ 改进版尺寸敏感指标图已保存：D:/groundingdino_work/GroundingDINO-main\results\size_sensitive_metrics_improved_C.png
✅ 改进版检测框分析图已保存：D:/groundingdino_work/GroundingDINO-main\results\bbox_analysis_improved_C.png
✅ 改进版召回率曲线已保存：D:/groundingdino_work/GroundingDINO-main\results\recall_curve_improved_C.png
✅ 改进版检测框分析图已保存：D:/groundingdino_work/GroundingDINO-main\results\bbox_analysis_improved_C.png
✅ 改进版召回率曲线已保存：D:/groundingdino_work/GroundingDINO-main\results\recall_curve_improved_C.png

✅ 改进版召回率曲线已保存：D:/groundingdino_work/GroundingDINO-main\results\recall_curve_improved_C.png

🔄 生成对比可视化图表...

🔄 生成对比可视化图表...
✅ 对比柱状图已保存：D:/groundingdino_work/GroundingDINO-main\results\improvement_C_comparison_bar.png
🔄 生成对比可视化图表...
✅ 对比柱状图已保存：D:/groundingdino_work/GroundingDINO-main\results\improvement_C_comparison_bar.png
✅ 对比热力图已保存：D:/groundingdino_work/GroundingDINO-main\results\improvement_C_comparison_heatmap.png
✅ 对比柱状图已保存：D:/groundingdino_work/GroundingDINO-main\results\improvement_C_comparison_bar.png
✅ 对比热力图已保存：D:/groundingdino_work/GroundingDINO-main\results\improvement_C_comparison_heatmap.png

✅ 对比热力图已保存：D:/groundingdino_work/GroundingDINO-main\results\improvement_C_comparison_heatmap.png

🔄 保存对比报告...

🔄 保存对比报告...
🔄 保存对比报告...
✅ 对比报告已保存：D:/groundingdino_work/GroundingDINO-main\results\improvement_C_comparison_report.txt
✅ 对比报告已保存：D:/groundingdino_work/GroundingDINO-main\results\improvement_C_comparison_report.txt


评测完成！

基线 UNSEEN AP@0.5: 0.1727
改进版 UNSEEN AP@0.5: 0.1339
提升: -0.0388

📁 所有结果已保存至：D:/groundingdino_work/GroundingDINO-main\results
## 问题分析报告
思路错误！！
一、错误思路总结
1.1 我的原始思路
我试图通过以下步骤实现"自适应阈值"改进：

text
Step 1: 用 GroundingDINO 进行基线推理（阈值0.25）
        ↓
Step 2: 保存推理结果文件 coco_seen_400imgs_prompt1.json
        ↓
Step 3: 用 find_best_thresholds.py 分析这个结果文件
        ↓
Step 4: 为每个类别统计"最优阈值"（使F1最高的阈值）
        ↓
Step 5: 将统计出的阈值保存为 best_thresholds.json
        ↓
Step 6: 在改进版推理中加载这些阈值，重新过滤检测框
        ↓
Step 7: 期望 UNSEEN 类别的检测性能提升
1.2 我的核心假设
我假设：从基线结果中统计出的最优阈值，能够指导模型在推理时做出更好的决策

二、错误分析：思路错在何处
2.1 根本错误：统计对象错误
我统计的是"已过滤的结果"，而不是"模型的原始输出"

text
模型原始输出（置信度0.01-0.99）
        ↓
【基线推理时被丢弃的信息】
   ↓                    ↓
置信度<0.25的框     置信度≥0.25的框
   ↓                    ↓
  丢弃 ✗              保留 ✓
                        ↓
                   我的统计对象
                   (已丢失40%信息)
2.2 具体错误点
错误1：信息丢失无法挽回
```python
# 基线推理时
predict(..., box_threshold=0.25, ...)  # 这里已经丢了低置信度框


# 我的阈值统计
with open('coco_seen_400imgs_prompt1.json') as f:  # 读的是丢过信息的文件
    results = json.load(f)  # 永远看不到被丢弃的框
```

后果：所有置信度低于0.25的检测框，无论是否正确，都永远消失在我的统计视野之外。

错误2：阈值优化的对象错误
我统计的是"在已保留框的基础上，哪个阈值最好"，但正确的问题应该是"在原始输出中，哪个阈值能最好地平衡精确率和召回率"

对比维度	我的做法	正确做法
数据来源	已过滤结果	模型原始输出
阈值范围	0.1-0.5（只在已保留框内）	0.01-0.5（所有可能框）
优化目标	在现有框上找最佳	在全部可能框上找最佳
能否找回低置信度框	❌ 不能	✅ 能

错误3：循环论证
```text
基线结果（基于阈值0.25）
    ↓
统计"最优阈值"（大部分得到0.1）
    ↓
用0.1阈值重新过滤同一个结果
    ↓
结果不变（因为没有新框加入）
    ↓
"证明"自适应阈值无效 ❌
```
这是一个自我验证的闭环，无法产生真正的改进。

2.3 数据证据
从 threshold_analysis.txt 可以清楚看到问题：

text
person: 最优阈值 0.1
car: 最优阈值 0.1
dog: 最优阈值 0.1
...
平均阈值: 0.117
为什么几乎所有类别的最优阈值都是0.1？

因为在已过滤的结果中：

所有框的置信度都 ≥ 0.25

当我扫描阈值0.1-0.5时，阈值0.1能保留最多框

但这些框本来就是存在的，阈值0.1并没有带来新框

所以统计出的"最优阈值"只是当前集合的最小值，不是真正的优化阈值

核心教训：在机器学习实验中，必须确保统计和优化的对象是原始数据，而不是经过预处理的数据。任何预处理步骤都会丢失信息，而这些丢失的信息可能正是改进的关键。

## 小幅改进C方向：自适应阈值策略完整实现方案
### 改进动机与设计思路
#### 动机
在零样本目标检测中，不同类别的最优置信度阈值往往不同：

常见类别（如person、car）：模型置信度高，可用较高阈值保证精确率

罕见类别（如toaster、hair drier）：模型置信度低，需用较低阈值保证召回率

使用统一阈值（如0.25）无法平衡所有类别，因此需要为每个类别设置独立阈值。

#### 设计思路
从SEEN类别中划分验证集：用20%的SEEN图片作为验证集

收集原始预测结果：用极低阈值（0.01）运行推理，保留所有候选框

统计最优阈值：对每个类别扫描0.05-0.5的阈值，选择F1最高的

应用优化阈值：用统计出的阈值进行正式推理

对比验证：与固定阈值基线（0.25）对比

### 详细过程
step1_split_val_set.py          # 第1步：划分验证集
step2_collect_raw_predictions.py # 第2步：收集原始预测
step3_optimize_thresholds.py     # 第3步：统计最优阈值
step4_inference_improved.py      # 第4步：改进版推理
step5_evaluate_improved.py       # 第5步：对比评测

================================================================================
📊 最优阈值统计结果
================================================================================
+------+----------------+--------+--------+-------+-------+-------+------+
|   ID | 类别             |   原始框数 |   最优阈值 |    AP |    AR |    F1 | 状态   |
+======+================+========+========+=======+=======+=======+======+
|   75 | remote         |      4 |  0.250 | 0.000 | 0.000 | 0.000 | 默认   |
+------+----------------+--------+--------+-------+-------+-------+------+
|    1 | person         |  60797 |  0.100 | 0.574 | 0.571 | 0.573 | 优化   |
+------+----------------+--------+--------+-------+-------+-------+------+
|    4 | motorcycle     |   8298 |  0.050 | 0.020 | 0.453 | 0.038 | 优化   |
+------+----------------+--------+--------+-------+-------+-------+------+
|   10 | traffic light  |  44526 |  0.050 | 0.003 | 0.350 | 0.006 | 优化   |
+------+----------------+--------+--------+-------+-------+-------+------+
|   11 | fire hydrant   |  19095 |  0.050 | 0.000 | 0.105 | 0.000 | 优化   |
+------+----------------+--------+--------+-------+-------+-------+------+
|   14 | parking meter  |  25179 |  0.050 | 0.000 | 0.120 | 0.000 | 优化   |
+------+----------------+--------+--------+-------+-------+-------+------+
|   27 | backpack       |   5960 |  0.050 | 0.001 | 0.091 | 0.002 | 优化   |
+------+----------------+--------+--------+-------+-------+-------+------+
|   39 | baseball bat   |  84370 |  0.050 | 0.000 | 0.038 | 0.000 | 优化   |
+------+----------------+--------+--------+-------+-------+-------+------+
|   40 | baseball glove | 151807 |  0.050 | 0.002 | 0.878 | 0.005 | 优化   |
+------+----------------+--------+--------+-------+-------+-------+------+
|   64 | potted plant   |  14106 |  0.050 | 0.000 | 0.042 | 0.000 | 优化   |
+------+----------------+--------+--------+-------+-------+-------+------+
|   67 | dining table   |  28556 |  0.050 | 0.011 | 0.540 | 0.021 | 优化   |
+------+----------------+--------+--------+-------+-------+-------+------+
|   77 | cell phone     |   5569 |  0.050 | 0.001 | 0.091 | 0.003 | 优化   |
+------+----------------+--------+--------+-------+-------+-------+------+
|   88 | teddy bear     |  64202 |  0.050 | 0.000 | 0.111 | 0.001 | 优化   |
+------+----------------+--------+--------+-------+-------+-------+------+

📈 统计摘要:
  平均阈值: 0.069
  中位数阈值: 0.050
  最小阈值: 0.050
  最大阈值: 0.250

第5步：基线与改进版对比评测


加载COCO标注...
loading annotations into memory...
Done (t=0.34s)
creating index...
index created!


基线结果评估


评估 基线-SEEN...
  检测框数: 1876
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.29s).
Accumulating evaluation results...
DONE (t=0.18s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.285
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.347
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.305
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.116
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.294
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.495
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.310
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.357
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.357
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.140
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.347
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.574

评估 基线-UNSEEN...
  检测框数: 252
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.02s).
Accumulating evaluation results...
DONE (t=0.03s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.141
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.173
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.149
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.042
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.097
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.232
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.164
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.174
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.174
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.044
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.118
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.261


改进版结果评估


评估 改进版-SEEN...
  检测框数: 17119
Loading and preparing results...
DONE (t=0.01s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.59s).
Accumulating evaluation results...
DONE (t=0.31s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.032
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.039
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.034
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.017
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.042
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.044
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.056
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.092
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.095
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.062
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.095
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.117

评估 改进版-UNSEEN...
  检测框数: 222
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.02s).
Accumulating evaluation results...
DONE (t=0.03s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.036
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.047
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.036
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.011
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.019
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.049
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.062
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.083
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.083
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.011
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.031
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.105


对比报告


📊 UNSEEN类别性能对比:
+----------------------+--------+--------+---------+
| 指标                   |     基线 |    改进版 |      提升 |
+======================+========+========+=========+
| UNSEEN AP@0.5        | 0.1727 | 0.047  | -0.1257 |
+----------------------+--------+--------+---------+
| UNSEEN AP@[0.5:0.95] | 0.1409 | 0.0365 | -0.1045 |
+----------------------+--------+--------+---------+
| UNSEEN AR@100        | 0.1742 | 0.0833 | -0.0909 |
+----------------------+--------+--------+---------+

# 小幅改进（方向A）
============================================================
方向A：多提示集成（取最高分）
============================================================

📂 加载检测结果...

加载 prompt1:
  SEEN: 1876 框
  UNSEEN: 252 框

加载 prompt2:
  SEEN: 998 框
  UNSEEN: 315 框

加载 prompt3:
  SEEN: 1867 框
  UNSEEN: 301 框

🔄 融合SEEN结果...
  处理第 1 个prompt，共 1876 个框
  处理第 2 个prompt，共 998 个框
  处理第 3 个prompt，共 1867 个框
  融合前总框数: 4741
  融合后框数: 3429
  去重比例: 27.7%

🔄 融合UNSEEN结果...
  处理第 1 个prompt，共 252 个框
  处理第 2 个prompt，共 315 个框
  处理第 3 个prompt，共 301 个框
  融合前总框数: 868
  融合后框数: 719
  去重比例: 17.2%

💾 保存融合结果...
  ✅ SEEN结果已保存: D:/groundingdino_work/GroundingDINO-main\results\coco_seen_400imgs_ensemble.json
  ✅ UNSEEN结果已保存: D:/groundingdino_work/GroundingDINO-main\results\coco_unseen_100imgs_ensemble.json

📊 各prompt贡献统计:
  SEEN结果来源:
    prompt1: 1278 框 (37.3%)
    prompt2: 973 框 (28.4%)
    prompt3: 1180 框 (34.4%)

============================================================
✅ 融合完成！
===========================================================
下一步：运行评测脚本对比效果
  python scripts/step5_evaluate_improved.py
（记得先修改step5中的文件名）
PS D:\groundingdino_work> & D:\User\Anaconda\anaconda\envs\groundingdino\python.exe d:/groundingdino_work/GroundingDINO-main/improved_A/evaluate_ensemble.py
======================================================================
评测：基线(prompt1) vs 集成结果(3 prompts)
======================================================================

加载COCO标注...
loading annotations into memory...
Done (t=0.35s)
creating index...
index created!

==================================================
基线结果评估 (prompt1)
==================================================

评估 基线-SEEN...
  检测框数: 1876
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.30s).
Accumulating evaluation results...
DONE (t=0.19s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.285
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.347
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.305
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.116
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.294
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.495
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.310
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.357
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.357
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.140
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.347
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.574

评估 基线-UNSEEN...
  检测框数: 252
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.02s).
Accumulating evaluation results...
DONE (t=0.03s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.141
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.173
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.149
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.042
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.097
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.232
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.164
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.174
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.174
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.044
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.118
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.261

==================================================
集成结果评估 (3 prompts)
==================================================

评估 集成-SEEN...
  检测框数: 3429
Loading and preparing results...
DONE (t=0.06s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.35s).
Accumulating evaluation results...
DONE (t=0.21s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.256
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.311
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.273
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.116
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.284
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.471
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.322
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.378
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.378
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.144
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.377
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.614

评估 集成-UNSEEN...
  检测框数: 719
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.04s).
Accumulating evaluation results...
DONE (t=0.04s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.143
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.181
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.154
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.054
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.112
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.222
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.193
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.234
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.234
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.059
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.181
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.325

======================================================================
对比报告
======================================================================

📊 UNSEEN类别性能对比:
+----------------------+---------------+-----------------+--------+
| 指标                   |   基线(prompt1) |   集成(3 prompts) |     提升 |
+======================+===============+=================+========+
| UNSEEN AP@0.5        |        0.1727 |          0.1806 | 0.0079 |
+----------------------+---------------+-----------------+--------+
| UNSEEN AP@[0.5:0.95] |        0.1409 |          0.1434 | 0.0024 |
+----------------------+---------------+-----------------+--------+
| UNSEEN AR@100        |        0.1742 |          0.234  | 0.0598 |
+----------------------+---------------+-----------------+--------+

✅ 对比图已保存: D:/groundingdino_work/GroundingDINO-main\results\comparison_ensemble.png

✅ 对比报告已保存: D:/groundingdino_work/GroundingDINO-main\results\ensemble_comparison_report.txt


PS D:\groundingdino_work> & D:\User\Anaconda\anaconda\envs\groundingdino\python.exe d:/groundingdino_work/GroundingDINO-main/improved_A/prompt_ensemble_with_nms.py
============================================================
方向A：多提示集成 + NMS后处理
============================================================

📂 加载检测结果...

加载 prompt1:
  SEEN: 1876 框
  UNSEEN: 252 框

加载 prompt2:
加载 prompt2:
  SEEN: 998 框
  UNSEEN: 315 框
  SEEN: 998 框
  UNSEEN: 315 框
  UNSEEN: 315 框

加载 prompt3:
  SEEN: 1867 框
  UNSEEN: 301 框

🔄 第一阶段：最大置信度融合...
  SEEN: 1867 框
  UNSEEN: 301 框

🔄 第一阶段：最大置信度融合...
  UNSEEN: 301 框

🔄 第一阶段：最大置信度融合...
  SEEN融合后: 3429 框
  UNSEEN融合后: 719 框
🔄 第一阶段：最大置信度融合...
  SEEN融合后: 3429 框
  UNSEEN融合后: 719 框

🔄 第二阶段：NMS去重 (IOU阈值=0.5)...
  SEEN融合后: 3429 框
  UNSEEN融合后: 719 框

🔄 第二阶段：NMS去重 (IOU阈值=0.5)...
  SEEN: 3429 → 2033 框 (减少1396个)
  UNSEEN: 719 → 389 框 (减少330个)

🔄 第二阶段：NMS去重 (IOU阈值=0.5)...
  SEEN: 3429 → 2033 框 (减少1396个)
  UNSEEN: 719 → 389 框 (减少330个)

  SEEN: 3429 → 2033 框 (减少1396个)
  UNSEEN: 719 → 389 框 (减少330个)

💾 保存最终结果...
💾 保存最终结果...
  ✅ SEEN结果: D:/groundingdino_work/GroundingDINO-main\results\coco_seen_400imgs_ensemble_nms.json
  ✅ SEEN结果: D:/groundingdino_work/GroundingDINO-main\results\coco_seen_400imgs_ensemble_nms.json
  ✅ UNSEEN结果: D:/groundingdino_work/GroundingDINO-main\results\coco_unseen_100imgs_ensemble_nms.json
  ✅ UNSEEN结果: D:/groundingdino_work/GroundingDINO-main\results\coco_unseen_100imgs_ensemble_nms.json


📊 各阶段框数对比 (UNSEEN):
  原始总框数: 868
  原始总框数: 868
  融合后: 719 (82.8%)
  NMS后: 389 (44.8%)
  NMS后: 389 (44.8%)
  最终/原始比例: 44.8%
  最终/原始比例: 44.8%

============================================================
============================================================
✅ 融合完成！
✅ 融合完成！
============================================================
============================================================

下一步：运行评测脚本对比效果
  python scripts/step5_evaluate_ensemble_nms.py

  PS D:\groundingdino_work> & D:\User\Anaconda\anaconda\envs\groundingdino\python.exe d:/groundingdino_work/GroundingDINO-main/improved_A/evaluate_ensemble_nms.py
======================================================================
评测：基线(prompt1) vs 集成+NMS结果
======================================================================

加载COCO标注...
loading annotations into memory...
Done (t=0.35s)
creating index...
index created!

==================================================
基线结果评估 (prompt1)
==================================================

评估 基线-SEEN...
  检测框数: 1876
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.29s).
Accumulating evaluation results...
DONE (t=0.18s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.285
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.347
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.305
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.116
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.294
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.495
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.310
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.357
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.357
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.140
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.347
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.574

评估 基线-UNSEEN...
  检测框数: 252
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.02s).
Accumulating evaluation results...
DONE (t=0.03s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.141
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.173
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.149
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.042
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.097
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.232
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.164
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.174
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.174
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.044
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.118
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.261

==================================================
集成+NMS结果评估
==================================================

评估 集成+NMS-SEEN...
  检测框数: 2033
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.36s).
Accumulating evaluation results...
DONE (t=0.19s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.292
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.363
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.312
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.118
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.310
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.516
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.322
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.370
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.370
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.143
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.372
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.598

评估 集成+NMS-UNSEEN...
  检测框数: 389
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.03s).
Accumulating evaluation results...
DONE (t=0.03s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.167
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.215
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.181
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.045
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.137
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.252
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.193
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.226
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.226
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.046
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.177
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.321

======================================================================
对比报告
======================================================================

📊 UNSEEN类别性能对比:
+----------------------+---------------+----------+--------+
| 指标                   |   基线(prompt1) |   集成+NMS |     提升 |
+======================+===============+==========+========+
| UNSEEN AP@0.5        |        0.1727 |    0.215 | 0.0423 |
+----------------------+---------------+----------+--------+
| UNSEEN AP@[0.5:0.95] |        0.1409 |    0.167 | 0.026  |
+----------------------+---------------+----------+--------+
| UNSEEN AR@100        |        0.1742 |    0.226 | 0.0518 |
+----------------------+---------------+----------+--------+

✅ 对比图已保存: D:/groundingdino_work/GroundingDINO-main\results\comparison_ensemble_nms.png

✅ 对比报告已保存: D:/groundingdino_work/GroundingDINO-main\results\ensemble_nms_comparison_report.txt

