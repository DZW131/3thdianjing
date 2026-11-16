# 肌原纤维区域学习实验（2026-11-12）

## 结论

已完成一轮独立区域网络训练和两张本项目 TEM 图像的试推理，不再只依靠手工纹理阈值。
网络能够覆盖部分条带状肌原纤维区域，但碎裂、背景误判和 Z 线位置的缺口仍明显。
目前不能据此宣布完成肌原纤维实例重建，也不能计算可靠的同束肌节长度或 α 角。
未修改现有 DeepLab、Z 线模型、训练配置或生产推理链路。

## 公开模型审计纠正

- `fabda/Myofibril_paper`，审计提交 `11f512a8181fb98540b6e5a89466ed85b46b4d5f`。
  仓库名容易误导：论文方法明确说明公开的 seResNet18 U-Net 用于 MOM-GFP 荧光线粒体分割。
  下载的四组 T0 图像/mask 也支持这一判断；因此停止权重下载和 TensorFlow 安装，未用于本项目推理。
  来源：https://pmc.ncbi.nlm.nih.gov/articles/PMC8027795/ ，Deep Learning segmentation 节。
- `CellSMB/cardiovinci`，已克隆审计提交 `9427f44994c6f539a6d9ecd6790acf348f7dce94`。
  论文包含心肌肌原纤维分割，但此次审计到的仓库主要是 StyleGAN 代码，未发现可直接加载的分割权重。
  这不是宣称作者在所有渠道都未发布权重。
  来源：https://github.com/CellSMB/cardiovinci ，https://pmc.ncbi.nlm.nih.gov/articles/PMC9527637/ 。

## 实际训练内容

- 数据沿用此前核对的 CellSMB v1：11 层同一体数据，raw 为 `11x466x2156`。
- 公开 cell1/cell2 的肌原纤维和线粒体分割合并，训练三类：other=0、myofibril=1、mitochondria=2。
- 标签是公开分割输出，未证实为人工金标准；未标区域归入 other 也是潜在噪声来源。
- mask 的 XY 尺寸是原图两倍，最近邻缩至原图尺寸；仍需解剖配准复核。两类重叠的 6 个像素忽略。
- 独立小型 U-Net，通道 8/16/32/64，GroupNorm，从随机权重开始；不是作者模型复现或预训练权重迁移。
- 训练为所有层的 x=[0,1280)，留出 x=[1536,2156)，中间 256 像素不用。
  相邻层没有随机分配到训练/验证，但依旧只有一个体数据，不能当跨病例验证。
- 800 optimizer steps，每步 4 个随机 128x128 patch，共 3200 次采样，不是 800 epochs。
- Adam，lr=0.001，CE + 两个前景类别的 soft Dice；旋转、翻转、亮度/对比度增强，seed=42。
- 训练归一化阈值只由训练区域计算；留出区域沿用该阈值。
- 128 窗口、96 步长，重叠区域平均概率，argmax 输出类别，无形态学后处理。
- 本地 CPU / Python 3.13 / PyTorch 2.12.1+cpu；完整成功运行约 132 秒。

## 实测一致性（不是临床准确率）

| 类别 | Dice | IoU | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| 肌原纤维 | 0.4703 | 0.3074 | 0.4053 | 0.5601 |
| 线粒体辅助类 | 0.4853 | 0.3204 | 0.4277 | 0.5608 |
| other | 0.8343 | 0.7157 | 0.9258 | 0.7592 |

以上是在所有留出层累计像素混淆矩阵后计算，与公开算法输出比较。
它不能与本项目医生标注的线粒体损伤分类分割指标横向比较。
本项目 dev00/dev05 没有肌原纤维区域 GT，因此没有报告私有数据 Dice/IoU。
私有图缩至 0.25 和 0.125 两个尺度，仅用于观察尺度敏感性，并非已标尺校准的尺度对齐。

## 图像观察与下一步

dev05 的部分肌束内部被连片识别，但 Z 线常被留成缺口；dev00 更碎裂。
线粒体辅助类也有明显误判，不可替代已有的线粒体模型。
公共样本有拼接/平坦背景，部分真实结构可能未包含在公开 mask 内，这些因素均限制训练与评估。

下一步优先：建立少量本项目肌原纤维区域校正样本，明确标签是否包含 Z/I 带；
利用标尺对齐公共数据和 TEM 的物理尺度，再进行目标域微调。
区域 mask 可靠后，还需独立的束边界/实例分离与局部 Z 线归属验证，不能把连通域直接当肌原纤维实例。

## 文件与复现

脚本：`tools/pilot_myofibril_region_learning.py`；测试：`tools/test_myofibril_region_learning.py`。
成功结果目录：`outputs/research/myofibril_region_20261112/region_learning_v2/`。
`region_learning_v1/` 是私有推理阶段数值类型错误中断的首轮，不作为完整交付。
`manifest.json` 保存输入 SHA-256、划分、指标、版本和权重 SHA-256。
权重 `region_pilot.pt` 为 509621 字节，SHA-256：
`c578a60c1d0135363f11f7b815b3f1d46dee3ebc4c13ba500db8dc78064e8375`。

```powershell
python -m unittest discover -s tools -p test_myofibril_region_learning.py -v
python tools/pilot_myofibril_region_learning.py --data outputs/research/cellsmb_audit_20261103/paired_samples_v3 --private-inputs outputs/research/sarcgraph_development_inputs --output outputs/research/myofibril_region_20261112/region_learning_rerun --steps 800
```

输出目录必须不存在。运行依赖 numpy、Pillow、torch；不依赖 TensorFlow。
成功通过 6 项单元测试和语法检查。私有图和数据不应公开上传 GitHub。
