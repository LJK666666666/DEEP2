# 实验二：目标检测网络应用

## 2.1 实验要求

1. 开发环境为 Ubuntu 18.04 和 PyTorch 1.8.2。  
2. 按照实验具体要求对网络进行修改，以提高网络的性能。  
3. 使用指定的数据集进行网络的训练。  
4. 保存实验结果，包括关键实验参数、最终模型、预测结果、运行截图。  
5. 提交一份针对该实验的报告，格式参照 markdown 模板。

## 2.2 实验任务

1. 修改 SSD 目标检测网络，使用 VOC2007 数据集对网络进行训练。  
2. 设计 QT 界面，完成可视化检测。

## 2.3 实验步骤

### 1. 环境安装

连接网络（可通过手机热点），终端中运行：

```bash
pip install -r requirements.txt
```

### 2. 修改基础 SSD 的 VGG16 骨干网络为 ResNet50

SSD 原论文使用的是 VGG16 作为 backbone，本实验对此进行修改，使用更优秀的 ResNet50 作为 backbone 提取原图的特征信息。

- 修改并使用 `resnet50_ssd300_voc0712_feature_fusion.yaml` 文件。
- 删去 yaml 文件中的 `voc_2012_trainval`。
- 将 `batch_size` 从 32 改为 16 或 8，以避免训练时显存溢出。
- 修改实验程序 `SSD-Resnet-Attention-FeatureFusion-master/ssd/config/defaults.py`，将 `model.device` 参数修改为 `cuda`。

### 3. 增加 FPN 模块

基础的 SSD 网络并没有充分利用多层特征，将低分辨率和高语义特征融合在一起，以在目标检测上获得良好结果。因此需要在基础 SSD 网络中添加 FPN 结构，增强网络的特征融合能力。

### 4. 训练模型

运行以下命令进行训练：

```bash
python3 train.py
```

- 每 6000 次迭代保存一次模型，并保存一次验证结果。
- 结果保存路径：  
  `SSD-Resnet-Attention-FeatureFusion-master/outputs/resnet50_ssd300_voc0712_feature_fusion`

### 5. 使用 QT 对算法模型进行集成

**具体要求：**

- 设计 QT 界面，完成对算法模型的调用及结果读取。
- 实现选择图片并显示检测结果的功能。
- 测试图片不仅来自数据集，还可来自现场拍摄的图片，类别按 VOC2007 数据集定义。
- 可通过调用 `SSD-Resnet-Attention-FeatureFusion-master/demo1.py` 文件来绘制图片（部分程序需按提示补充）。
- 每个实验台配有 USB 摄像头，建议编写程序实时采集现场视频，并将检测框叠加到视频图像上，以提升目标检测的展示效果。

## 2.4 思考题

**使用 K-means++ 算法调整优化 SSD 的 Anchor 框：**

SSD 算法对每一张特征图，按照不同大小和长宽比生成多个默认框，是基于多尺度的方法得到目标检测结果。对于不同尺度的特征图，会设置不同大小和宽高比的先验框。

为了更符合 VOC2007 数据集的特点，使用 K-means++ 方法对目标框统计数据进行聚类，代替原 SSD 中生成的默认框基准大小。

**要求：** 对比和分析修改前后的实验效果。
