# 心肌病超微结构电镜分割与量化

研究代码交接版，DeepLabV3+（默认ResNet101、output stride 16），三个任务：线粒体损伤分类分割、线粒体-肌浆网关系、肌节结构分割。包含整图滑窗预测、量化、医生复核轮廓导出及修订数据微调。**不是临床诊断软件，结果需人工复核。**

交接分支：`codex/reproducible-handoff-20261020`。未替换主模型。原始数据、患者标识、数据库、样本划分需由项目负责人授权提供，不随公开代码发布。

2026-10-22新增肌节结构研究：[实验记录与8例统计](docs/SARCOMERE_PROGRESS_20261022.md)、[SarcGraph接入与复现](docs/SARCGRAPH_INTEGRATION.md)。已实现候选Z-Z像素距离、方向/重叠检查和原图纹理证据；可靠的微米肌节长度及alpha角仍待结构审核与尺度校准。

## 安装与快速检查

2026-11-16同步嵴状态表征及损伤分类探索：
[复现命令、依赖、汇总结果与限制](docs/CRISTAE_REPRESENTATION.md)。
新增公开数据下载、局部表征探针、原模型整图推理及病例分组融合对照。
101个固定GT区域上中度F1由原模型0.5079至融合0.6087，但不是端到端分割提升；
原模型训练重叠未排除，需要新病例独立确认。未替换生产模型，不随仓库发布患者数据或拟合权重。

以下在仓库根目录Linux Bash运行。已工作的服务器环境为Python3.10、RTX5090、torch2.11.0+cu128、torchvision0.26.0+cu128，不是旧4090的cu121环境。

```bash
git clone --depth 1 --branch codex/reproducible-handoff-20261020 https://github.com/DZW131/3thdianjing.git
cd 3thdianjing
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-5090.txt
python -m unittest discover -s tests -v
python train.py --help
python evaluate.py --help
```

版本来自服务器实际环境，未在空白机器全量重装；[验证记录](docs/HANDOFF.md)。`requirements.txt`为较宽范围，`requirements-5090.txt`为相关包的精确版本。GPU驱动必须支持所选wheel。自动标尺OCR使用可选 RapidOCR/ONNX Runtime，安装 `requirements-scale-ocr.txt`；不需要 Tesseract。依赖版本须兼容目标 Python/系统，详见下节。

### 自动标尺读数与微米校准（2026-11-18 补齐交接）

旧 `tools/detect_scale_bars.py` 只测横线，不读数字；现在要求显式 `--scale-bar-um`，仅适合已知所有输入标尺长度相同的情况。**不要把 1 μm 默认值用于不同刻度的图片。**

自动流程入口为 `tools/detect_scale_ocr.py`，核心为 `utils/jijie/scale_ocr.py`：定位标尺，读取邻近数字/单位，在原图及增强图上取得一致读数后计算 `um_per_pixel=scale_bar_um/scale_bar_px`。无法识别或多候选歧义则 `needs_review`，不猜为 1 μm。

```bash
python -m pip install -r requirements-scale-ocr.txt
python tools/detect_scale_ocr.py --image-dir /path/to/JPEGImages --output-dir outputs/scale_ocr --numeric-only --limit 5
python -m unittest discover -s tools -p test_scale_ocr_pipeline.py -v
```

`--numeric-only` 表示负责人明确保证单位全部为微米，OCR 只读数字，不是自动确认单位；混合 nm/μm 时不可使用。默认模式需识别到微米单位。
批量入口非递归扫描 JPG/PNG/TIF/JPEG/TIFF，输出 `calibration.csv`、`summary.json` 和 `review/` 检查图。相同样本名冲突会拒绝；多页 TIF 仅处理首帧；输出目录需不存在。

`evaluate.py` 已接入：在原有评估命令后追加 `--auto-scale-ocr --scale-ocr-numeric-only`；或追加 `--scale-calibration-csv outputs/scale_ocr/calibration.csv` 复用同原图尺度的逐图校准表。不要同时用旧 `--auto-scale-bar` 冒充数字识别。显式 `--um-per-pixel` 与校准 CSV 优先于 OCR。
OCR 路径会检查原图和预测 mask 尺寸，尺寸不匹配则待复核。面积乘比例平方，长度乘比例；失败图继续保留像素单位。逐图导出 OCR 文本、状态、置信度和复核原因。

本次通过 4 项合成/注入OCR输出的测试及语法/CLI检查；本机未安装真实 OCR 推理依赖，没有将本次同步说成真实图片全量复测。历史识别覆盖率不是读数准确率。医院部署仍需真实图人工核对、目标环境依赖验证及误读防护。

## 目录结构

| 路径 | 用途 |
|---|---|
| train.py / evaluate.py | 训练、评估、量化、可视化 |
| configs/{mito,mito_sr,sarcomere} | 三任务与实验配置 |
| modeling / dataloaders | 网络、标签映射、裁剪增强 |
| utils/jijie | 滑窗、量化、后处理、标尺、部分标注指标 |
| tools | 数据准备、批量导出、DB转换、CV及汇总 |
| tests / docs | 无患者数据测试、交接与权重清单 |
| data / run / outputs | 私有数据、权重、结果，Git忽略 |

## 数据组织及标签

医生原始导出为病例/时间目录中的TIF与对应`slice.db`。不要把同名DB扁平化复制。处理后的单通道PNG存原始类别ID，与图像尺寸一致：

```text
data/jijie/
  JPEGImages/<sample_id>.jpg
  SegmentationClass/<sample_id>.png
  ImageSets/Segmentation/tasks/<task>/final_200/{train,val,test}.txt
  metadata/...
```

清单每行一个不带扩展名的ID。`--data-root`与`--manifest-dir`独立设置，外部数据应同时指定。

| 原始ID | 定义 |
|---|---|
| 0 | 背景 |
| 1/2/3/4 | 严重损伤/中度损伤/健康/自噬线粒体 |
| 5 | 肌浆网 |
| 6/7/8/9 | 闰盘/Z线样物质堆积/脂滴/糖原颗粒 |
| 10/11/12/13 | Z线/T管/M线/心肌侧管 |
| 255 | 部分标注实验中的忽略区域 |

mito选择1–4，mito_sr选择1–5，sarcomere选择10–13。内部按顺序重映射为1起始连续ID，肌节的1/2/3/4对应原10/11/12/13。**T管不等于侧管**；历史`t_prior`配置不是推荐方案。

```bash
python tools/prepare_jijie_dataset.py --input-root /path/to/raw_export --output-root data/jijie
python tools/build_jijie_task_manifests.py --help
```

原始准备器的分组划分目标为70%/15%/15%，不等于每次历史实验实际比例。任务清单构建依赖审核过的task registry；复现实验请向负责人获取原始清单。52张修订图按6病例分五折，留出17/10/10/9/6张。不要随机切patch替代病例分组。`--force`可能删除输出目录，默认不用。

## 权重

[权重清单](docs/weights_inventory.json)包含40个文件的相对路径、大小、SHA-256及URL。URL为空表示未发布。下载工具校验大小及哈希，不覆盖不一致文件。

**当前仅纹理分类器已发布到[Release](https://github.com/DZW131/3thdianjing/releases/tag/handoff-weights-20261021)，三个基线及其他实验checkpoint尚未公开上传。** 大文件传输缓慢，本次中止上传；以下基线命令须另加授权的`--ssh USER@HOST --remote-root /path/to/project`才能下载，否则会明确报错。

```bash
python tools/fetch_weights.py --list
python tools/fetch_weights.py --path run/jijie/jijie_mito_v0_5_final200_5090_retrain/model_best.pth.tar
python tools/fetch_weights.py --path run/jijie/mito_texture_classifier.pkl
```

未公开权重须授权SCP获取（EXPERIMENT替换为清单中的名称）：

```bash
python tools/fetch_weights.py --path run/jijie/EXPERIMENT/model_best.pth.tar --ssh USER@HOST --remote-root /path/to/project
```

仅加载可信checkpoint及joblib/pickle，哈希不是安全沙箱。ResNet默认下载ImageNet初始化；加载完整权重推理时`export EM_DISABLE_PRETRAINED=1`可避免该联网请求。这个变量仅对ResNet有效，不要用于需要ImageNet初始化的训练。

## 训练

```bash
python train.py --config configs/mito/mito_v0_5_final200.json --data-root /path/to/prepared --manifest-dir /path/to/prepared/ImageSets/Segmentation/tasks/mito/final_200 --checkname mito_new_run
```

其他任务改用`configs/mito_sr/mito_sr_v0_5_final200.json`或`configs/sarcomere/sarcomere_v0_5_final200.json`，同时替换manifest任务名。CE+Dice、SGD、poly调度；crop为640/768/1024，batch为4/2/1，epoch为150/160/180。基线保存的参数记录也使用这些轮数，**不能笼统称全部200轮**；要200轮显式加`--epochs 200`并作为新实验。

肌节v0_5含Z/M标签膨胀radius=1，并非完全无标签处理；未启用T管/侧管先验。`no_val=true`时末轮被命名model_best，不代表验证最优。权重写入`run/jijie/<checkname>/`。历史JSON合并逻辑仅在CLI值不同于parser默认值时覆盖；要把某项改回默认（如膨胀设0），请复制修改JSON。

修订52张五折微调，独立保存、不替换主模型：

```bash
python tools/run_mito_doctor_revision_cv.py --config configs/mito/mito_v0_9_doctor_revision_cv5.json --baseline-checkpoint run/jijie/jijie_mito_v0_5_final200_5090_retrain/model_best.pth.tar --data-root /path/to/revised52 --folds-root /path/to/revised52/ImageSets/Segmentation/tasks/mito/folds --run-prefix revised52_new_cv
```

原实验val与test相同且参与选模，不是独立测试。三来源融合、MetaSSL及结构先验保留为实验功能，未证明优于主方案；不能称为论文官方复现。

## 评估与可视化

```bash
export EM_DISABLE_PRETRAINED=1
python evaluate.py --config configs/mito/mito_v0_5_final200.json --data-root /path/to/prepared --manifest-dir /path/to/prepared/ImageSets/Segmentation/tasks/mito/final_200 --split test --resume run/jijie/jijie_mito_v0_5_final200_5090_retrain/model_best.pth.tar --save-dir outputs/eval/mito --visualize --num-vis-samples 5 --postprocess-mito --postprocess-classifier run/jijie/mito_texture_classifier.pkl
```

输出包括metrics_summary.json、per_class_metrics.csv、混淆矩阵、量化表及可视化。`--partial-label-eval`另外计算只保留与同类GT重叠预测组件的指标，有选择偏倚，不能代替整图指标。255像素不计分；未标区域设0还是255会影响误检统计。

量化包括组件数量、面积、周长、Feret、形态特征及肌浆网-线粒体距离；组件不必然等于真实独立细胞器。已知尺度用`--um-per-pixel 0.005`（示例，必须换真实值）；`--auto-scale-bar`启用自动识别，仍需人工检查。尺度失败不能把像素值称为微米。α角脚本仅为探索，结构归属问题未解决。

## 整图推理与标注平台导出

先用fetch_weights获取另外两个基线，然后：

```bash
export EM_DISABLE_PRETRAINED=1
python tools/export_doctor_review_predictions.py --data-root /path/to/prepared --all-images --mito-resume run/jijie/jijie_mito_v0_5_final200_5090_retrain/model_best.pth.tar --mito-sr-resume run/jijie/jijie_mito_sr_v0_5_final200_5090_retrain/model_best.pth.tar --postprocess-mito --postprocess-classifier run/jijie/mito_texture_classifier.pkl --output-root outputs/review
python tools/export_sarcomere_predictions.py --data-root /path/to/prepared --all-images --resume run/jijie/jijie_sarcomere_v0_5_final200_5090_retrain/model_best.pth.tar --output-root outputs/review_sarcomere
```

首次加`--max-samples 1`。输出原图、原始类别mask、彩色mask、叠加图和manifest。原始TIF与可编辑DB配对：

```bash
python tools/export_platform_tif_ai_db.py --raw-slice-root /path/to/raw_slices --mito-mask-dir /path/to/mito_masks --sr-mask-dir /path/to/sr_masks --output-root outputs/platform
python tools/validate_platform_db_export.py --help
```

mask路径从导出manifest取得。DB结构需与平台兼容，先导入1张验证坐标和轮廓可编辑。默认仅导出AI对象，`--keep-doctor-marks`可保留源医生对象。不要覆盖源目录；`--overwrite`会替换输出。watershed分离粘连组件不是实例真值。SVS图像不能代替可编辑标注。

## 当前结果

2026-10-14同一52张修订切片，五折平均Positive Dice/IoU：

| 方案 | Dice | IoU |
|---|---:|---:|
| 原模型 | 58.71% | 47.60% |
| 原模型＋后处理 | 75.70% | 67.00% |
| 直接微调 | 61.06% | 49.60% |
| 微调＋原后处理 | 46.00% | 38.52% |

来源：[聚合结果](docs/results/four_way_overall.csv)、[分类别](docs/results/four_way_classes.csv)。仅发布聚合统计，私有原始日志由负责人保管，汇总脚本为`tools/report_mito_four_way.py`。五折均值与合并混淆矩阵宏平均不能混用。当前仍保留原模型＋后处理。

## 肌原纤维区域与 Z 线图融合（2026-11-12 实验，11-16 同步）

本批独立实验不修改原有分割模型。当前推荐方向是**保留原始 SarcGraph 节点和连接，在图的边上加入肌原纤维区域证据**，而不是以绿色四边形替代肌节骨架。

- [区域学习实验](docs/MYOFIBRIL_REGION_LEARNING_20261112.md)：小型三类 U-Net，CellSMB 公开算法输出作为训练目标；肌原纤维空间留出一致性 Dice=0.4703、IoU=0.3074，不是人工金标准准确率。
- [SarcGraph 区域证据融合](docs/SARCGRAPH_REGION_GRAPH_20261112.md)：dev00/dev05 分别保留全部 35/47 条原始边，附加区域支持信息；8 条有区域支持的额外建议仅供核查，未加入原图。没有验证准确率改善。
- [局部 Z 路径实验](docs/Z_LOCAL_PATHS_20261112.md)：局部追踪辅助配对。
- [四边形探索记录](docs/FUSED_SARCOMERE_REGIONS_20261112.md)：保留历史过程，不作为当前完整肌节建模成果。

依赖沿用项目的 PyTorch、numpy、Pillow、pandas、scipy、scikit-image；独立区域网络不需要 TensorFlow。完整实测版本及训练参数见实验说明，不应直接安装误匹配的 Myofibril_paper 权重（其公开模型目标是荧光线粒体）。

```bash
python -m unittest discover -s tools -p test_myofibril_region_learning.py -v
python -m unittest discover -s tools -p test_fused_sarcomere_regions.py -v
python -m unittest discover -s tools -p test_sarcgraph_region_evidence.py -v
python -m unittest discover -s tools -p test_z_local_streamlines.py -v
python tools/pilot_myofibril_region_learning.py --help
python tools/annotate_sarcgraph_with_regions.py --help
```

实跑命令见各说明。运行依赖此前核对的公共图像/标签、授权私有图像和已导出的 Z 节点/边；这些输入不随代码提交，克隆后不能仅凭源码重现实验。独立区域试验权重和患者图像、概率图、打包产物未上传。旧分割权重获取方式不变。原始 SarcGraph 图来自 `tools/analyze_sarcomere_sarcgraph.py`，并非新增的替代配对图。

## 已知问题与交接边界

- 未标区域、保留AI对象、选模偏倚、历史checkpoint/纹理分类器潜在训练重叠限制结果解释，不能称临床准确率或严格外部泛化。
- 精确复现实验需授权数据、原始划分及对应折checkpoint；代码可用不意味着公开数据齐备。
- 未重新长训，检查范围见[HANDOFF](docs/HANDOFF.md)；历史报告脚本依赖历史产物。
- 不纳入无关旧WSI工程、厂商二进制及患者目录；非jijie数据集的旧路径不在本次支持范围。
- 新分支目录移除旧数据，但**Git历史仍含旧数据**，未擅自改写历史或强推。浅克隆不等于消除历史隐私风险。
- 代码及模型再分发/商业使用权限需负责人确认，本次不凭空授予许可。
