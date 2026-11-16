# Z 线与肌原纤维区域融合建模（2026-11-12）

## 目标与实际完成内容

复用此前 SarcGraph 适配流程导出的 Z 线对象、法向邻居配对及局部路径候选，
结合新训练的区域网络预测，导出两端 Z 线、内部骨架和闭合四边形组成的局部肌节候选。
本次没有重新训练或替换 Z 线模型，也没有修改原有节点与配对结果。

这不是完整肌节重建：区域预测不是实例标签，骨架是预测 mask 的中轴，
两条侧边由共同 Z 线投影范围推断，并非观测到的肌原纤维边界。
同一连通域可以合并多束肌原纤维，因此“同连通域”只是必要支持之一，不能证明同束。

## 融合步骤

1. 读取原始 Z 线拟合端点；坐标为 row,column，保持原图尺度。
2. 读取上轮区域网络的 myofibril 类 mask；在缩小的网格上做半径 2 的闭运算，
   去除小于 16 网格像素的组件，计算 8 连通区域与中轴骨架。没有把整条 Z 线并入区域，避免沿长 Z 线跨束桥接。
3. 合并“已选法向邻居”和“局部路径命中”候选，按 Z 对去重；不是对全部 Z 线任意两两连边。
4. 取两条 Z 线在平均切向上的共同覆盖范围，形成四边形；要求夹角不超过 30 度、
   中位连接相对法向偏离不超过 35 度、两端中点距离 20--400 原图像素。
5. 在四边形纵向 15%--85% 的内部检查：原始肌原纤维 mask 覆盖不低于 60%，
   非零支持中同一主连通域占比不低于 70%；若其他 Z 线有至少两个采样点穿过内部则拒绝。
6. 在该组件且位于四边形内的骨架上寻找路径。两端位于纵向 15%/85%，
   允许最多 32 原图像素的端点吸附；必须连通，路径长度不超过两端 Z 中点距离的 1.6 倍。
7. 若两个通过候选的四边形重叠超过较小区域的 20%，两者都标为不确定，不强制二选一。

此处路径只是内部骨架段，不是完整 Z--Z 肌节长度。
跨过未检出的 Z 线仍可能发生，因为“没有检测到第三条 Z”不等于实际不存在。

## 实测输出

| 图像 | 原有 Z 对象 | 去重候选对 | 0.25 尺度支持候选 | 0.125 尺度支持候选 |
| --- | ---: | ---: | ---: | ---: |
| dev00 | 56 | 21 | 1 | 1 |
| dev05 | 73 | 42 | 8 | 6 |

0.25 为主试验；0.125 仅为敏感性探索，没有根据数量选择最优模型。
两个尺度都固定使用 2 网格像素闭运算，因此对应的原图修复半径分别为 8/16 像素，
不能把这一对照解释为纯粹的输入分辨率消融。
这些数值是候选数量，不是正确率、召回率或肌节实际总数。
两个尺度的候选集合也不应只凭数量视为相同。

主试验 dev00：9 个候选无区域支持、9 个区域弱/碎裂、1 个骨架断开、1 个骨架端点过远，保留 1 个。
主试验 dev05：14 个端点过远、8 个骨架断开、6 个区域弱/碎裂、4 个无区域、2 个几何冲突，保留 8 个。
可视化中部分保留四边形很窄，只对应 Z 线共同覆盖的局部范围，不代表整束完整横截面。
无肌原纤维实例和肌节边界 GT，未报告 Dice/IoU、微米长度或 α 角。

## 输出与复现

主试验：`outputs/research/fused_sarcomere_regions_v2_20261112/`。
敏感性探索：`outputs/research/fused_sarcomere_regions_scale0125_20261112/`。
最初的 `fused_sarcomere_regions_20261112/` 是使用旧闭运算 API 的首版，候选数量与 v2 一致；以 v2 为交付。

- `*_fusion_comparison.png`：区域、骨架、闭合候选三联图。
- `*_candidate_audit.csv`：每一对的区域支持、连通性、距离和拒绝原因。
- `*_candidate_cells.json`：候选四边形、Z 对、区域 ID、内部路径；顶点/路径为原图 row,column。
- `*_candidate_cells.tif`：缩小网格上的候选 ID，1 开始，背景及共享边界像素为 0。
- `*_region_components.tif` / `*_region_skeleton.png`：缩小网格的区域标签和骨架。
- `summary.json`：结果、阈值、坐标约定、代码与输入 SHA-256。

```powershell
python -m unittest discover -s tools -p test_fused_sarcomere_regions.py -v
python tools/pilot_fused_sarcomere_regions.py --inputs outputs/research/sarcgraph_development_inputs --nodes outputs/research/sarcgraph_texture_development_v2 --pairs outputs/research/sarcgraph_eight_case_review_v2 --regions outputs/research/myofibril_region_20261112/region_learning_v2 --local-pairs outputs/research/z_local_paths_depart12_20261112 --output outputs/research/fused_sarcomere_regions_rerun --scale 0.25
```

依赖 numpy、pandas、Pillow、scipy、scikit-image（本次 0.26 系列）；不需要 GPU 或重新训练。
输出目录必须不存在。7 项测试覆盖正常融合、第三条 Z 拒绝、空区域、交叉 Z 几何、
端点顺序不变性、小组件清除和坐标轴约定。

## 下一步优先级

先对导出的 9 个主试验候选逐一检查同束/相邻关系及侧边界，不把通过自动阈值当成医生确认。
针对“骨架端点过远/断开”检查 Z 带缺口与区域漏检，修复应受区域和方向限制，不能无条件膨胀。
进一步把长 Z 线分解为局部边界片段，实现多个相邻肌原纤维各自的归属；
之后才在确认的同束邻接图上扩展肌节序列、建立可靠长度和角度量化。
