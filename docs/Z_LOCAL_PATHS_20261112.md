# Z线局部归属与纹理追踪实验（2026-11-12）

## 目的与实现

继续使用已有Z线预测，不提取新的Z线，也不修改生产模型。
在dev00、dev05两个既有开发样本中，每条长度至少5px的Z线选20%、50%、80%位置，
分别向两侧法向出发，在6/12px双尺度结构张量纹理场中追踪。
不同局部位置可以指向不同目标，而非强制整条Z线只有一个中心归属。

每步3px，最长400px；两尺度方向一致性cos>=0.85，置信度>=0.15，
单步转向<=30°；纹理缺失连续超过18px停止。首先碰到另一Z线拟合线段时停止，
20px以内命中记为too_close_hit，不当作有效邻接。同一步出现近似重合目标记歧义。
目前的交点是Z线拟合直线交点，不是像素级边界相交，未实现曲线Z盘或真实肌原纤维区域分割。

对照两种启动条件：直接读起点纹理，以及先沿原Z线法向离开12px。
后者只避开起点Z线纹理，不恢复丢失的图像信息，且每步仍检查其他Z线交点。
这些阈值均是开发设定，不是物理或医学标准；没有用测试GT选参。

## 实际运行

|条件|样本|既有Z节点|原中心配对|追踪条数|有效命中路径|局部候选配对|双向命中配对|
|---|---|---:|---:|---:|---:|---:|---:|
|直接启动|dev00|56|21|324|0|0|0|
|直接启动|dev05|73|35|438|5|3|1|
|离开12px|dev00|56|21|324|0|0|0|
|离开12px|dev05|73|35|438|14|11|1|

dev05离开12px条件下11组局部候选有7组不在原中心配对集合中。
“新配对”不是“修正了7个错误”；命中增加也不是召回提升，双向命中也不是正确性证明。
原中心配对完整保留，本轮没有用11组候选覆盖原35组。

## 视觉检查与限制

dev05部分路径沿条状组织弯曲到达其他Z线，展示了不通过中心直线的局部路径构造能力。
但dev00完全未得到有效命中，两个条件下大量追踪因支持中断或急转停止。
这说明简单纹理积分目前不够稳健，不能宣称已完成同肌束身份识别或肌节建模。

可能因素包括Z线附近纹理方向、低对比区域、弯曲路径、非肌原纤维的条状纹理及预测Z线误差；
目前没有人工逐路径核查，不能把其中一种归为唯一失败原因。
相邻、同束和分支必须分别判断，不能仅因路径碰到一条Z线就认定是真实相邻肌节。
未计算真实肌节长度、α角或生物学准确率；CSV长度均为像素路径长度。

## 交付与后续

原始追踪JSON保存每条路径及停止原因；candidate_pairs.csv保存命中计数、双向性、
是否出现在中心基线中，以及待人工填写的同束/相邻字段。
local_assignment_audit.csv按Z节点及方向统计不同起点的目标归属。
对照图左为中心基线，右只显示局部命中路径；右图更少的连线不能理解成全部漏连已被纠正。

下一步应将少量明确可见的肌原纤维条带做区域/中心路径参考，再判断哪些停止属于算法漏追，
哪些属于原Z线误检或结构中断。不能靠继续放宽阈值制造完整框架。

## 复现

```sh
python tools/pilot_z_local_paths.py --inputs outputs/research/sarcgraph_development_inputs --nodes outputs/research/sarcgraph_texture_development_v2 --pairs outputs/research/sarcgraph_eight_case_review_v2 --output outputs/z_local_new --departure-px 0
python tools/pilot_z_local_paths.py --inputs outputs/research/sarcgraph_development_inputs --nodes outputs/research/sarcgraph_texture_development_v2 --pairs outputs/research/sarcgraph_eight_case_review_v2 --output outputs/z_local_depart_new --departure-px 12
python tools/test_z_local_streamlines.py
```

依赖numpy、pandas、scipy、scikit-image、Pillow。默认只运行dev00/dev05，可用--cases指定。
5项合成测试通过：先命中相邻线、无支持停止、多局部位置、重合目标歧义、离开起点纹理。
输出总数与逐路径停止状态已做一致性核对；不代表解剖学验证。
