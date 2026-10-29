# 线粒体全图量化报告入口

```bash
python tools/quantify_mito_doctor_report.py --project /path/to/project --prediction-root /path/to/postprocessed/mito --output outputs/mito_quant_new
```

输入目录包含summary.json、review_dataset/SegmentationClass/*.png和对应JPEGImages/*.jpg。
summary须明确postprocess_mito=true，mask为原始ID 0/1/2/3/4。脚本复用已有后处理结果，未重新推理。

采用既有split_mito_instances方法，min_area=30px，min_distance=20px。输出实例形态、每图、每病例及全量分类汇总和来源校验；objects.csv.gz为逐对象明细。

面积为像素面积；周长为Crofton四方向估计；Feret为凸包最大直径；主次轴为二阶中心矩等效椭圆轴。它们与历史边界像素周长、PCA跨度口径不同，不能直接纵向比较。分类数量统计包含边缘实例，border_truncated用于后续排除截断对象的形态统计。占整图面积的分母包括标尺/文字区域，不等同组织面积。

2026-10-29服务器实际完成652张、51个病例前缀、66,020个算法实例的统计，其中6,343个接触图像边缘。严重损伤4,464，中度损伤3,563，健康57,542，自噬451。数据来源为已有baseline加后处理的整图预测。实例数量不是医生确认真值，可能受过分离/欠分离影响。未应用未经确认的物理尺度，无μm或μm²报告。

图像级病例ID、对象明细和医生用Excel保留在私有输出目录，不上传公开仓库。原始出处为预测根目录summary.json、mask哈希及本轮provenance.json。实际统计不包含尚未推理的新增切片，也不合入肌浆网重复对象。
