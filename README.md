# JIJIE v0.5 任务化分割与量化工程

## 项目定位

这个版本把心肌超微结构项目按最终量化目标拆成三条任务线来做，而不是继续用单一的 14 类统一分割：

- `mito`：线粒体状态分割
- `mito_sr`：线粒体-肌浆网关系
- `sarcomere`：肌节几何结构

v0.5 的重点不是再重做 prepared dataset，而是在现有 [data/jijie](data/jijie) 基线之上，把训练组织、最终 split、推理导出和可视化链路做得更适合服务器直接复现。

## v0.5 核心变化

- 新增 `final_200` group-wise manifest：
  - 每个任务都按 `group_id` 精确选出 `200` 张训练图
  - 剩余可安全评估的样本进入 `test`
  - `val.txt` 保留为空文件，用于无验证集最终训练
- 保留原有两套 split：
  - `default`：开发基线
  - `folds`：group-wise 5-fold 调参
- 新增 v0.5 配置文件：
  - `configs/mito/mito_v0_5_final200.json`
  - `configs/mito_sr/mito_sr_v0_5_final200.json`
  - `configs/sarcomere/sarcomere_v0_5_final200.json`
- 无验证集训练闭环补齐：
  - 训练结束时自动把最终 checkpoint 提升为 `model_best.pth.tar`
  - 后续评估命令可直接复用现有入口
- 推理可视化增强：
  - 不同类别使用固定颜色
  - 自动输出 `class_palette.csv`
  - 自动输出 `class_legend.png`
  - 每个样本单独保存彩色 mask 和 overlay 图
- 滑窗推理支持 `batch_size > 1`，评估脚本更稳健

## 仓库结构

```text
.
├── train.py
├── evaluate.py
├── JIJIE_IMPLEMENTATION_PLAN.md
├── train_jijie.txt
├── configs/
│   ├── mito/
│   │   ├── mito_v0_2.json
│   │   └── mito_v0_5_final200.json
│   ├── mito_sr/
│   │   ├── mito_sr_v0_2.json
│   │   └── mito_sr_v0_5_final200.json
│   └── sarcomere/
│       ├── sarcomere_v0_2.json
│       └── sarcomere_v0_5_final200.json
├── dataloaders/
│   ├── __init__.py
│   ├── custom_transforms.py
│   └── datasets/jijie.py
├── tools/
│   ├── build_jijie_task_manifests.py
│   ├── audit_raw_jijie_dataset.py
│   └── prepare_jijie_dataset.py
├── utils/
│   ├── loss.py
│   ├── metrics.py
│   ├── saver.py
│   ├── summaries.py
│   └── jijie/
│       ├── config.py
│       ├── inference.py
│       ├── quantify.py
│       ├── tasks.py
│       └── visualization.py
└── data/
    └── jijie/
        ├── JPEGImages/
        ├── SegmentationClass/
        ├── ImageSets/Segmentation/
        └── metadata/
```

## 数据组织

训练仍然直接使用 [data/jijie](data/jijie)：

```text
data/jijie/
├── JPEGImages/
├── SegmentationClass/
├── ImageSets/Segmentation/
└── metadata/
```

原始导出库 `fenge_datasets/` 的角色没有变，仍然只用于：

- 数据审计
- 医生补标
- 后续增强版 manifest 重建

## 任务 manifest

当前任务 manifest 目录：

```text
data/jijie/ImageSets/Segmentation/tasks/
├── mito/
│   ├── default/
│   ├── final_200/
│   └── folds/
├── mito_sr/
│   ├── default/
│   ├── final_200/
│   └── folds/
└── sarcomere/
    ├── default/
    ├── final_200/
    └── folds/
```

其中：

- `default`：沿用旧版 train/val/test，用于开发基线
- `folds`：group-wise 5-fold，用于调参和稳定性检查
- `final_200`：v0.5 最终训练方案，`200 train / rest test / no val`

`final_200` 是按 `group_id` 选出来的，不会乱切同一病例组。

### 当前 v0.5 final_200 规模

| Task | Train | Test | Test Groups | Holdout 但不纳入 test |
| --- | ---: | ---: | ---: | ---: |
| `mito` | 200 | 36 | 5 | 33 |
| `mito_sr` | 200 | 36 | 5 | 33 |
| `sarcomere` | 200 | 32 | 5 | 49 |

“Holdout 但不纳入 test” 指的是剩余样本里不满足安全评估条件的部分，比如仅部分标注或仅模板样本，它们不会被误当成正式测试集。

### 重新生成 manifest

```bash
python tools/build_jijie_task_manifests.py \
  --prepared-root ./data/jijie \
  --task-registry ./data/jijie_audit/task_registry.csv \
  --output-root ./data/jijie/ImageSets/Segmentation/tasks \
  --num-folds 5 \
  --final-train-count 200 \
  --final-manifest-name final_200
```

## 三条任务线

### 1. mito

- 标签：`1,2,3,4`
- 目标：四类线粒体分割与对象级统计
- 输入策略：高分辨率随机 patch
- 量化：计数、面积、周长、圆度、Feret、长宽比、form factor

### 2. mito_sr

- 标签：`1,2,3,4,5`
- 目标：线粒体和肌浆网联合分割
- 输入策略：更大的随机 patch
- 量化：肌浆网面积、最近线粒体边界距离

### 3. sarcomere

- 标签：训练用 `10,11,12,13`
- 主要量化目标：`10,12`
- 输入策略：大 patch / 整图滑窗
- 量化：基础组件级统计，后续再向高级几何参数扩展

## 训练命令

### v0.5 最终训练

线粒体状态分割：

```bash
python train.py \
  --config configs/mito/mito_v0_5_final200.json \
  --gpu-ids 0 \
  --workers 4
```

线粒体-肌浆网关系：

```bash
python train.py \
  --config configs/mito_sr/mito_sr_v0_5_final200.json \
  --gpu-ids 0 \
  --workers 4
```

肌节结构：

```bash
python train.py \
  --config configs/sarcomere/sarcomere_v0_5_final200.json \
  --gpu-ids 0 \
  --workers 2
```

### v0.2 开发基线

如果你还想保留旧的 train/val/test 方案做对照，可以继续使用 v0.2 配置：

```bash
python train.py --config configs/mito/mito_v0_2.json --gpu-ids 0 --workers 4
python train.py --config configs/mito_sr/mito_sr_v0_2.json --gpu-ids 0 --workers 4
python train.py --config configs/sarcomere/sarcomere_v0_2.json --gpu-ids 0 --workers 2
```

## 评估命令

### v0.5 最终评估

```bash
python evaluate.py \
  --config configs/mito/mito_v0_5_final200.json \
  --split test \
  --resume run/jijie/jijie_mito_v0_5_final200/model_best.pth.tar \
  --save-dir outputs/eval/jijie_mito_v0_5_final200
```

```bash
python evaluate.py \
  --config configs/mito_sr/mito_sr_v0_5_final200.json \
  --split test \
  --resume run/jijie/jijie_mito_sr_v0_5_final200/model_best.pth.tar \
  --save-dir outputs/eval/jijie_mito_sr_v0_5_final200
```

```bash
python evaluate.py \
  --config configs/sarcomere/sarcomere_v0_5_final200.json \
  --split test \
  --resume run/jijie/jijie_sarcomere_v0_5_final200/model_best.pth.tar \
  --save-dir outputs/eval/jijie_sarcomere_v0_5_final200
```

### 保存彩色可视化

```bash
python evaluate.py \
  --config configs/mito/mito_v0_5_final200.json \
  --split test \
  --resume run/jijie/jijie_mito_v0_5_final200/model_best.pth.tar \
  --save-dir outputs/eval/jijie_mito_v0_5_final200_vis \
  --visualize \
  --num-vis-samples 20
```

## 可视化输出

启用 `--visualize` 后，除了常规指标，还会输出：

- `class_palette.csv`
- `visualizations/class_legend.png`
- `visualizations/*_composite.png`
- `visualizations/*_gt_mask.png`
- `visualizations/*_pred_mask.png`
- `visualizations/*_overlay_gt.png`
- `visualizations/*_overlay_pred.png`

这些图里不同类别会使用固定颜色，不同任务子集也会保持稳定映射。

## 评估输出

每次评估目录下会保存：

- `metrics_summary.json`
- `per_class_metrics.csv`
- `class_palette.csv`
- `confusion_matrix.csv`
- `confusion_matrix.png`
- `image_task_metrics.csv`
- `object_metrics.csv`
- `detection_metrics.csv`

## 服务器推荐工作流

### 1. 拉取 v0.5 分支

```bash
cd /root
git clone -b codex/jijie-v0.5 https://github.com/DZW131/3thdianjing.git 3thdianjing_v0.5
```

### 2. 复用旧环境

```bash
conda activate dianjing
cd /root/3thdianjing_v0.5
python -c "import torch, torchvision, numpy, PIL, matplotlib, scipy, tqdm; print('env ok')"
```

### 3. 复用旧数据

推荐直接软链接旧目录里的数据：

```bash
cd /root/3thdianjing_v0.5/data
rm -rf jijie
ln -s /root/3thdianjing/data/jijie jijie
```

如果你已经保留了新版仓库自带的 `ImageSets/Segmentation/tasks`，则不要整目录覆盖，只需要保证：

```text
data/jijie/JPEGImages
data/jijie/SegmentationClass
data/jijie/ImageSets/Segmentation/tasks
```

同时存在。

### 4. 直接启动 v0.5 训练

```bash
python train.py --config configs/mito/mito_v0_5_final200.json --gpu-ids 0 --workers 4
```

### 5. 训练结束后评估

```bash
python evaluate.py \
  --config configs/mito/mito_v0_5_final200.json \
  --split test \
  --resume run/jijie/jijie_mito_v0_5_final200/model_best.pth.tar \
  --save-dir outputs/eval/jijie_mito_v0_5_final200
```

## 已知限制

- 本地工作区当前仍然缺少 `data/jijie/JPEGImages`，因此无法在本机完成完整训练 smoke test。
- `sarcomere` 的高级几何参数仍需要后续规则校准，当前主打的是稳定的任务分割和基础组件统计。
- `um_per_pixel` 还没有完整回写进 prepared dataset，因此当前导出的多数统计默认仍以像素单位为主。

## 推荐实验记录表

| Version | Task | Config | Train Split | Test Split | Output Dir | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| v0.5 | mito | `configs/mito/mito_v0_5_final200.json` | `final_200/train.txt` | `final_200/test.txt` | `outputs/eval/jijie_mito_v0_5_final200` | no val |
| v0.5 | mito_sr | `configs/mito_sr/mito_sr_v0_5_final200.json` | `final_200/train.txt` | `final_200/test.txt` | `outputs/eval/jijie_mito_sr_v0_5_final200` | no val |
| v0.5 | sarcomere | `configs/sarcomere/sarcomere_v0_5_final200.json` | `final_200/train.txt` | `final_200/test.txt` | `outputs/eval/jijie_sarcomere_v0_5_final200` | no val |
| v0.2 | mito | `configs/mito/mito_v0_2.json` | `default/train.txt` | `default/test.txt` | `outputs/eval/jijie_mito_v0_2` | baseline |

## sarcomere v0.6 细长结构实验

本分支新增了 3 个只针对 `sarcomere` 任务的实验配置，用来验证 Z 线、M 线和心肌侧管这类细长、断续、边界模糊结构的训练增强方案。旧的 v0.5 配置默认不启用这些逻辑，因此原始训练和推理流程不会被污染。

### 结构约定

- Z 线和 M 线是肌节几何测量的主要锚点，后续肌节长度、形状和排列指标主要依赖它们的端点与相邻关系。
- T 管是单独类别，本轮实验先不对 T 管做膨胀，也不生成 T 管伪标签。
- 心肌侧管是单独类别，Z/M 端点连接线只作为低权重的心肌侧管候选先验，不能监督到 T 管类别。

### 新增方案

| Experiment | Config | Purpose |
| --- | --- | --- |
| Z/M/side dilation only | `configs/sarcomere/sarcomere_v0_6_zm_side_dilate_only.json` | 训练阶段只对 Z 线、M 线、心肌侧管做 1 像素轻微膨胀，不处理 T 管 |
| side prior only | `configs/sarcomere/sarcomere_v0_6_side_prior_only.json` | 根据 Z/M 线端点生成低权重心肌侧管候选线辅助监督 |
| dilation + side prior | `configs/sarcomere/sarcomere_v0_6_zm_side_dilate_side_prior.json` | 同时启用 Z/M/侧管标签膨胀和心肌侧管结构先验 |

心肌侧管结构先验只在训练阶段生效：代码会在训练 mask 中提取 Z 线、M 线连通域端点，生成候选连接线，并以 `side_tubule_prior_loss_weight` 控制辅助监督强度。默认采用 warm-up，前 20 个 epoch 从 0 逐步增加到最大权重 0.2。候选线默认只落在背景或已有心肌侧管区域，不覆盖 Z 线、M 线和 T 管标注；评估和测试仍然使用原始标注 mask。

早期的 `sarcomere_v0_6_t_prior_only.json` 和 `sarcomere_v0_6_dilate_t_prior.json` 文件名保留为兼容入口，但内部已经改为心肌侧管先验，并关闭 T 管先验。新实验建议优先使用上表中的显式配置名。

### 服务器启动方式

建议在服务器单独 clone 本分支，避免覆盖旧的 baseline 目录：

```bash
git clone -b codex/sarcomere-structure-prior https://github.com/DZW131/3thdianjing.git 3thdianjing_sarcomere_prior
cd 3thdianjing_sarcomere_prior
conda activate dianjing
```

复用已有数据时，保证以下目录存在：

```text
data/jijie/JPEGImages
data/jijie/SegmentationClass
data/jijie/ImageSets/Segmentation/tasks
```

如需软链接旧数据：

```bash
rm -rf data/jijie
ln -s /root/3thdianjing/data/jijie data/jijie
```

启动三组消融实验：

```bash
python train.py --config configs/sarcomere/sarcomere_v0_6_zm_side_dilate_only.json --gpu-ids 0 --workers 2
python train.py --config configs/sarcomere/sarcomere_v0_6_side_prior_only.json --gpu-ids 0 --workers 2
python train.py --config configs/sarcomere/sarcomere_v0_6_zm_side_dilate_side_prior.json --gpu-ids 0 --workers 2
```

训练完成后评估：

```bash
python evaluate.py \
  --config configs/sarcomere/sarcomere_v0_6_zm_side_dilate_only.json \
  --split test \
  --resume run/jijie/jijie_sarcomere_v0_6_zm_side_dilate_only/model_best.pth.tar \
  --save-dir outputs/eval/jijie_sarcomere_v0_6_zm_side_dilate_only

python evaluate.py \
  --config configs/sarcomere/sarcomere_v0_6_side_prior_only.json \
  --split test \
  --resume run/jijie/jijie_sarcomere_v0_6_side_prior_only/model_best.pth.tar \
  --save-dir outputs/eval/jijie_sarcomere_v0_6_side_prior_only

python evaluate.py \
  --config configs/sarcomere/sarcomere_v0_6_zm_side_dilate_side_prior.json \
  --split test \
  --resume run/jijie/jijie_sarcomere_v0_6_zm_side_dilate_side_prior/model_best.pth.tar \
  --save-dir outputs/eval/jijie_sarcomere_v0_6_zm_side_dilate_side_prior
```

建议优先比较 `per_class_metrics.csv` 中 Z 线、M 线、心肌侧管的 IoU/Dice/Recall，并额外观察 T 管的假阳性是否下降；可视化重点看 `visualizations/*_composite.png` 中细长结构连续性和误分到 T 管的情况。
