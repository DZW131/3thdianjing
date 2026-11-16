# 修正方向：在原始 SarcGraph 骨架上附加肌原纤维约束

此前绿色四边形版本没有准确对应需求。主对象应是原有的 Z 节点—连接图，
肌原纤维分割应帮助图的连接选择与局部归属，而不是另行绘制闭合四边形作为主要建模结果。

## 本次落实

读取 `sarcgraph_texture_development_v2/<case>/prediction/bundle_audit.csv`，
它来自适配器实际执行 SarcGraph 后导出的原始边，而不是后续的法向邻居替代图。
保留全部原始节点与边，逐边附加区域证据；所有原始边的集合一致性已在运行中断言。
目前还没有执行新的拓扑优化，不能把着色变化当成准确率提升。

对原有连接的中间 15%--85% 区段，在法向 +/-8 原图像素的五条采样线上读取
区域网络三类概率。至少三条采样线判为肌原纤维时，该截面有区域支持。
覆盖率 >=0.6 且最长无支持区段比例 <=0.25 时标为 region_supported；
覆盖率 <0.2 为 weak_region_evidence，其余不确定。
分数为 coverage*(1-max_gap_fraction)，仅是启发式证据，不是解剖学置信度。
弱区域证据不能等同错连，因为区域模型存在漏检和域偏移。

补连建议来源是已有法向邻居和局部路径候选中不在原图内的边。
通过现有共同 Z 投影几何检查且有区域支持的，只作为蓝色虚线建议。
这些尚未完成第三条 Z、交叉连接、分支归属检查，未加入原始图。

## 本次实测

| 图像 | 原始 Z 节点 | 原始边 / 保留边 | 区域支持边 | 区域弱证据 | 区域不确定 | 几何待核查边 | 区域支持的补连建议 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev00 | 56 | 35 / 35 | 1 | 30 | 4 | 12 | 0 |
| dev05 | 73 | 47 / 47 | 31 | 12 | 4 | 19 | 8 |

区域支持与几何待核查并非互斥；比如一条方向不合理的边仍可能经过绿色区域。
图中几何待核查优先标红，其余区域支持标绿，弱证据/不确定标黄。
之前报告的 dev00 21、dev05 42 是替代配对与局部路径的候选并集，
不是本次原始 SarcGraph 图的边数，不能混用。

三联图：左为原始图，中为原图叠加区域，右为全部原始连接的证据标记与蓝色补连建议。
没有绿色框，没有依据预测 mask 删除原始边，也没有重训现有模型。
这属于正确对象上的融合诊断层，不是已经完成的图优化或完整肌节重建。

## 文件与验证

输出：`outputs/research/sarcgraph_region_graph_20261112/`。
`*_original_edges_with_region_evidence.csv` 保留原始边与区域证据；
`*_addition_proposals.csv` 单独保存补连建议；
`*_region_annotated_graph.json` 保存不变的节点/边与尚未加入的建议；
`summary.json` 记录统计、参数、输入和脚本 SHA-256。

```powershell
python -m unittest discover -s tools -p test_sarcgraph_region_evidence.py -v
python tools/annotate_sarcgraph_with_regions.py --inputs outputs/research/sarcgraph_development_inputs --graphs outputs/research/sarcgraph_texture_development_v2 --regions outputs/research/myofibril_region_20261112/region_learning_v2 --alternatives outputs/research/sarcgraph_eight_case_review_v2 --local-pairs outputs/research/z_local_paths_depart12_20261112 --output outputs/research/sarcgraph_region_graph_rerun
```

依赖 numpy、pandas、Pillow、scipy、scikit-image；输出目录必须不存在。
5 项单元测试及语法检查通过；两例运行均确认原始边集合与数量完全保留。
下一阶段在该图上研究局部 Z 归属、边排序、受约束补连、歧义与分支处理，
并用逐边人工核查比较修正前后错连/漏连，不能仅以连线更多证明更完整。
