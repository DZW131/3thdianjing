# JIJIE 项目实施清单

## 1. 当前共识

- 现有 [data/jijie](D:/work/fenge/data/jijie) prepared dataset 仍然是当前训练基线。
- 效果不理想的主矛盾不在“数据集整理错了”，而在“任务组织和训练策略不够匹配目标”。
- 版本管理上要把“开发基线”和“最终训练方案”区分开：
  - 开发基线：保留 `default` 和 `folds`
  - 最终训练：使用 `final_200`

## 2. 数据集使用策略

### 2.1 现阶段继续使用 prepared dataset

当前训练、评估、推理入口都继续对齐：

```text
data/jijie/
```

这套数据集与现有 loader、训练脚本和评估脚本是打通的，不需要在本轮里推翻重建。

### 2.2 原始导出库的角色

原始库 `fenge_datasets/` 继续只承担这些职责：

- 数据审计
- 医生补标
- 后续增强版 manifest 重建

### 2.3 split 组织方式

当前保留三套 split：

- `default`：旧版 train/val/test，适合开发基线
- `folds`：group-wise 5-fold，适合调参
- `final_200`：v0.5 最终训练方案，适合最终版模型

`final_200` 的约束：

- 必须按 `group_id` 划分，不能乱切
- 训练集精确使用 `200` 张图
- `val.txt` 为空
- 其余可安全评估的样本进入 `test.txt`
- 剩余但不满足安全评估条件的样本单独列出，不混入 test

## 3. 三条任务线

### 3.1 主线 A：线粒体状态分割

- 标签：`1,2,3,4`
- 目标：四类线粒体分割和对象级统计
- 输入：高分辨率随机 patch
- 输出：像素级指标 + 对象级计数和形态统计

### 3.2 主线 B：线粒体-肌浆网关系

- 标签：`1,2,3,4,5`
- 目标：联合分割和距离量化
- 输入：比 `mito` 更大的 patch
- 输出：像素级指标 + 肌浆网面积 + 最近线粒体边界距离

### 3.3 主线 C：肌节几何结构

- 标签：`10,11,12,13`
- 目标：基础组件级分割与统计
- 输入：大 patch / 整图滑窗
- 输出：基础组件统计，后续再扩展到高级几何量化

## 4. 数据处理与 patch 方案

### 4.1 通用原则

- 不再统一把所有任务都缩到 `512x512`
- 小目标任务优先保分辨率
- 大结构任务优先保上下文

### 4.2 mito

- `train_resize_mode=random_crop`
- `crop_size=640`
- `eval_resize_mode=none`
- `inference_mode=sliding`

### 4.3 mito_sr

- `train_resize_mode=random_crop`
- `crop_size=768`
- `eval_resize_mode=none`
- `inference_mode=sliding`

### 4.4 sarcomere

- `train_resize_mode=random_crop`
- `crop_size=1024`
- `eval_resize_mode=none`
- `inference_mode=sliding`
- 对细线标签允许小半径增粗

## 5. 训练方案

### 5.1 开发阶段

开发阶段优先使用：

- `default`
- 或 `folds`

目的不是追求最终结果，而是稳定比较：

- patch 尺寸
- 学习率
- loss
- 任务拆分方式

### 5.2 最终阶段

最终阶段统一切到：

- `final_200`
- `no_val=true`

原因：

- 训练样本从旧版默认 `101` 左右提升到 `200`
- 不再被每轮整图验证拖慢
- 最终 checkpoint 自动提升为 `model_best.pth.tar`

### 5.3 当前版本推荐配置

- `configs/mito/mito_v0_5_final200.json`
- `configs/mito_sr/mito_sr_v0_5_final200.json`
- `configs/sarcomere/sarcomere_v0_5_final200.json`

## 6. 推理与评估规范

### 6.1 推理模式

- 默认支持滑窗推理
- v0.5 已支持滑窗 `batch_size > 1`

### 6.2 评估层级

- 像素级：IoU、Dice、Precision、Recall、Pixel Accuracy
- 图像级：任务相关统计汇总
- 对象级：计数、面积、周长、圆度、Feret、长宽比等

### 6.3 彩色可视化

推理与评估导图需要满足：

- 不同类别使用固定颜色
- 同一个类别在不同任务子集下颜色保持稳定
- 自动导出 `class_palette.csv`
- 自动导出 `class_legend.png`
- 每张样本至少导出：
  - 彩色 GT mask
  - 彩色预测 mask
  - GT overlay
  - prediction overlay
  - composite 图

## 7. v0.5 已落地的工程修改

- `tools/build_jijie_task_manifests.py`
  - 支持生成 `final_200`
  - 输出 `group_assignment.csv`
- `train.py`
  - 支持无验证集最终训练
  - 最终 epoch 自动生成可评估的 `model_best.pth.tar`
- `utils/saver.py`
  - 支持强制提升最终 checkpoint 为 best
- `utils/jijie/inference.py`
  - 支持滑窗批量推理
- `utils/jijie/visualization.py`
  - 新增任务级颜色映射与图例工具
- `evaluate.py`
  - 导出类别颜色表
  - 导出类别图例
  - 导出更清晰的彩色 mask 与 overlay
- `configs/*/*_v0_5_final200.json`
  - 新增最终版配置

## 8. 实施顺序

### 阶段 1：开发基线

- 用 `default` 或 `folds` 做调参
- 不直接用 test 反复选模型

### 阶段 2：最终训练

- 切到 `final_200`
- 关闭验证
- 训练结束后只在 test 上做正式评估

### 阶段 3：补标并回流

- 医生补标完成后
- 基于原始库重新审计
- 视情况重建增强版 manifest

## 9. 当前最优先动作

1. 服务器拉取 `codex/jijie-v0.5`
2. 使用 v0.5 配置直接训练 `mito`
3. 训练完成后跑一次 `test` 正式评估
4. 确认可视化导图和对象级指标是否满足医生查看需求

## 10. 一句话结论

当前最合理的工程方案不是推翻 prepared dataset，而是在它之上采用“任务化训练 + group-wise `final_200` 最终 split + 无验证集最终训练 + 彩色可视化评估”的 v0.5 方案。
