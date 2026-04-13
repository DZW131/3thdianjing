# JIJIE v0.2 任务化分割与量化工程

## 项目定位

这个版本不再把项目当成“14 类统一语义分割”来做，而是按最终量化目标拆成三条任务线：

- `mito`：线粒体状态分割
- `mito_sr`：线粒体-肌浆网关系
- `sarcomere`：肌节几何结构

v0.2 的目标是让服务器上可以直接按任务配置训练、评估和输出图像级统计，不再依赖手工拼参数或临时脚本。

## 本次版本的核心变化

- 增加任务配置文件，统一管理 `selected_classes`、patch 大小、推理窗口和主指标。
- 增加任务化 manifest 生成器，从已有 metadata 自动构建三条任务线的 split。
- `jijie` 数据加载器支持：
  - 任务 manifest
  - 训练与评估独立的尺寸策略
  - 原分辨率随机裁 patch
  - 可选的细线标签增粗
- `train.py` 支持：
  - `--config` 配置文件驱动
  - 任务主指标选择
  - `ce_dice / focal_dice / tversky_dice`
- `evaluate.py` 支持：
  - 滑窗推理
  - 任务化像素指标
  - 线粒体对象级统计
  - 肌浆网面积与最近边界距离统计
  - 图像级 CSV 导出
- 新增 `group-wise 5-fold` 任务 manifest。

## 环境要求

推荐环境：

- Linux 服务器
- Python 3.10 或 3.11
- NVIDIA GPU

先按你的 CUDA 版本安装 PyTorch 和 torchvision，再安装其余依赖：

```bash
pip install -r requirements.txt
```

## 仓库结构

```text
.
├── train.py
├── evaluate.py
├── JIJIE_IMPLEMENTATION_PLAN.md
├── configs/
│   ├── mito/mito_v0_2.json
│   ├── mito_sr/mito_sr_v0_2.json
│   └── sarcomere/sarcomere_v0_2.json
├── dataloaders/
│   ├── __init__.py
│   ├── custom_transforms.py
│   └── datasets/jijie.py
├── tools/
│   ├── build_jijie_task_manifests.py
│   ├── prepare_jijie_dataset.py
│   └── audit_raw_jijie_dataset.py
├── utils/
│   ├── loss.py
│   ├── metrics.py
│   ├── summaries.py
│   └── jijie/
│       ├── config.py
│       ├── inference.py
│       ├── quantify.py
│       └── tasks.py
└── data/
    └── jijie/
        ├── JPEGImages/
        ├── SegmentationClass/
        ├── ImageSets/Segmentation/
        │   ├── train.txt / val.txt / test.txt
        │   └── tasks/
        └── metadata/
```

## 数据组织

### 训练使用的数据集

当前训练基线仍然使用：

```text
data/jijie/
```

需要至少包含：

```text
data/jijie/
├── JPEGImages/
├── SegmentationClass/
└── ImageSets/Segmentation/
```

注意：

- `JPEGImages` 和 `SegmentationClass` 必须同时存在。
- 本地如果只有 mask、没有 `JPEGImages`，训练和完整评估都无法跑通。
- 服务器上请确认 `data/jijie/JPEGImages` 已存在。

### 原始导出库的角色

原始导出库：

```text
fenge_datasets/
```

主要用于：

- 数据审计
- 医生补标
- 后续增强版 manifest 重建

v0.2 训练本身不依赖服务器上必须保留这份原始库，只要 prepared dataset 完整即可。

## 三条任务线

### 1. mito

标签：

- `1,2,3,4`

目标：

- 线粒体四类状态分割
- 数量、面积、周长、圆度、Feret 等图像级统计

特点：

- 高分辨率小 patch
- 主指标以正类表现为主

### 2. mito_sr

标签：

- `1,2,3,4,5`

目标：

- 线粒体与肌浆网联合分割
- 肌浆网面积
- 肌浆网到最近线粒体边界距离

### 3. sarcomere

标签：

- 训练：`10,11,12,13`
- 量化主目标：`10,12`

目标：

- Z 线 / M 线相关结构分割
- 基础组件级统计

说明：

- v0.2 已打通任务分割与基础组件统计。
- 更高级的肌节几何参数仍需结合后续规则继续校准。

## 任务 manifest

任务 manifest 已生成在：

```text
data/jijie/ImageSets/Segmentation/tasks/
```

结构示例：

```text
data/jijie/ImageSets/Segmentation/tasks/mito/
├── default/
│   ├── train.txt
│   ├── val.txt
│   ├── test.txt
│   ├── train_positive_only.txt
│   ├── train_partial_labeled.txt
│   └── manifest_summary.json
└── folds/
    ├── fold_00/
    ├── fold_01/
    ├── fold_02/
    ├── fold_03/
    └── fold_04/
```

如果后面你更新了审计结果或补标结果，可以重新生成：

```bash
python tools/build_jijie_task_manifests.py \
  --prepared-root ./data/jijie \
  --task-registry ./data/jijie_audit/task_registry.csv \
  --output-root ./data/jijie/ImageSets/Segmentation/tasks \
  --num-folds 5
```

如果服务器上没有 `data/jijie_audit`，也没关系，当前仓库已经带了生成好的任务 manifest，可直接用。

## 训练命令

### 线粒体状态分割

```bash
python train.py \
  --config configs/mito/mito_v0_2.json \
  --gpu-ids 0 \
  --workers 4
```

### 线粒体-肌浆网关系

```bash
python train.py \
  --config configs/mito_sr/mito_sr_v0_2.json \
  --gpu-ids 0 \
  --workers 4
```

### 肌节结构

```bash
python train.py \
  --config configs/sarcomere/sarcomere_v0_2.json \
  --gpu-ids 0 \
  --workers 2
```

### 覆盖配置项

配置文件里的值都可以被命令行覆盖，例如：

```bash
python train.py \
  --config configs/mito/mito_v0_2.json \
  --gpu-ids 0,1 \
  --batch-size 8 \
  --epochs 180 \
  --checkname jijie_mito_v0_2_gpu2
```

## 评估命令

### mito

```bash
python evaluate.py \
  --config configs/mito/mito_v0_2.json \
  --split test \
  --resume run/jijie/jijie_mito_v0_2/model_best.pth.tar \
  --save-dir outputs/eval/jijie_mito_v0_2
```

### mito_sr

```bash
python evaluate.py \
  --config configs/mito_sr/mito_sr_v0_2.json \
  --split test \
  --resume run/jijie/jijie_mito_sr_v0_2/model_best.pth.tar \
  --save-dir outputs/eval/jijie_mito_sr_v0_2
```

### sarcomere

```bash
python evaluate.py \
  --config configs/sarcomere/sarcomere_v0_2.json \
  --split test \
  --resume run/jijie/jijie_sarcomere_v0_2/model_best.pth.tar \
  --save-dir outputs/eval/jijie_sarcomere_v0_2
```

### 保存可视化

```bash
python evaluate.py \
  --config configs/mito/mito_v0_2.json \
  --split test \
  --resume run/jijie/jijie_mito_v0_2/model_best.pth.tar \
  --save-dir outputs/eval/jijie_mito_v0_2_vis \
  --visualize \
  --num-vis-samples 20
```

## 评估输出

评估输出目录下会保存：

- `metrics_summary.json`
- `per_class_metrics.csv`
- `confusion_matrix.csv`
- `confusion_matrix.png`
- `image_task_metrics.csv`
- `object_metrics.csv`
- `detection_metrics.csv`
- `visualizations/*.png`，若启用 `--visualize`

其中：

- `image_task_metrics.csv` 是图像级统计
- `object_metrics.csv` 是逐对象统计
- `detection_metrics.csv` 是对象匹配层面的精确率 / 召回率 / F1 / count error

## v0.2 当前指标覆盖范围

### 已实现

- 像素级：
  - IoU
  - Dice
  - Precision
  - Recall
  - Pixel Accuracy
- 线粒体对象级：
  - 计数
  - 面积
  - 周长
  - 圆度
  - Feret 直径
  - 长宽比
  - form factor
- 线粒体检测级：
  - 对象级 Precision / Recall / F1
  - count absolute error
- 肌浆网关系：
  - 肌浆网面积统计
  - 最近线粒体边界距离

### 尚需后续规则校准

- 高级肌节几何参数：
  - AsR
  - Cur
  - `Cur/AsR`
  - alpha
  - H 参数

当前 `sarcomere` 已经具备独立训练和基础组件级统计能力，但最终论文级几何量化还需要结合更严格的结构规则继续完善。

## 服务器推荐工作流

### 1. 拉取新分支

假设你后面把这个版本推到了：

```text
codex/jijie-v0.2
```

服务器上可以直接：

```bash
git clone https://github.com/DZW131/3thdianjing.git
cd 3thdianjing
git checkout codex/jijie-v0.2
```

### 2. 准备数据

保证服务器上存在完整的：

```text
data/jijie/JPEGImages
data/jijie/SegmentationClass
data/jijie/ImageSets/Segmentation/tasks
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 直接按任务运行

例如先跑 `mito`：

```bash
python train.py --config configs/mito/mito_v0_2.json --gpu-ids 0 --workers 4
```

训练完成后：

```bash
python evaluate.py \
  --config configs/mito/mito_v0_2.json \
  --split test \
  --resume run/jijie/jijie_mito_v0_2/model_best.pth.tar \
  --save-dir outputs/eval/jijie_mito_v0_2
```

## 已知限制

- 本地当前工作区缺少 `data/jijie/JPEGImages`，因此无法在这台机器上完成完整训练 smoke test。
- v0.2 已经把任务化训练、滑窗推理和对象级统计接通，但最终科研版肌节高级几何量化仍需后续校准。
- 物理尺度 `um_per_pixel` 目前尚未完整接回 prepared dataset，因此当前导出的量化结果默认以像素单位为主。

## 推荐的实验记录表

| Task | Config | GPUs | Batch | Epochs | Val Metric | Test Metric | Output Dir | Notes |
| --- | --- | --- | ---: | ---: | --- | --- | --- | --- |
| mito | `configs/mito/mito_v0_2.json` | 1 | 4 | 150 | mean positive Dice | mean positive Dice / object F1 | `outputs/eval/jijie_mito_v0_2` | baseline |
| mito_sr | `configs/mito_sr/mito_sr_v0_2.json` | 1 | 2 | 160 | mean positive Dice | mean positive Dice / SR distance error | `outputs/eval/jijie_mito_sr_v0_2` | relation task |
| sarcomere | `configs/sarcomere/sarcomere_v0_2.json` | 1 | 1 | 180 | mean target IoU | target IoU / basic component stats | `outputs/eval/jijie_sarcomere_v0_2` | geometry baseline |
