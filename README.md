# 深度学习实验二：SSD 目标检测

## 快速开始

```bash
# ============================================
# 1. 环境安装
# ============================================
cd SSD-Resnet-Attention-FeatureFusion-master
pip install -r requirements.txt

# ============================================
# 2. 数据集准备 (VOC2007)
# ============================================
mkdir -p datasets && cd datasets
wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
tar -xf VOCtrainval_06-Nov-2007.tar
tar -xf VOCtest_06-Nov-2007.tar
mv VOCdevkit/VOC2007 ./
rm -rf VOCdevkit *.tar
cd ..

# ============================================
# 3. 训练模型 (原始 Anchor)
# ============================================
python3 train.py --config-file configs/resnet50_ssd300_voc0712_feature_fusion.yaml

# ============================================
# 4. K-means++ 优化 Anchor 对比实验 (思考题)
# ============================================
# 4.1 运行聚类分析
python kmeans_anchor.py --data_dir datasets/VOC2007 --clusters 6

# 4.2 创建优化配置文件
cp configs/resnet50_ssd300_voc0712_feature_fusion.yaml \
   configs/resnet50_ssd300_voc0712_feature_fusion_kmeans.yaml
# 根据聚类结果修改 resnet50_ssd300_voc0712_feature_fusion_kmeans.yaml 中的 MIN_SIZES 和 MAX_SIZES

# 4.3 训练优化 Anchor 模型
python3 train.py --config-file configs/resnet50_ssd300_voc0712_feature_fusion_kmeans.yaml

# 4.4 对比评估
python3 test.py --config-file configs/resnet50_ssd300_voc0712_feature_fusion.yaml
python3 test.py --config-file configs/resnet50_ssd300_voc0712_feature_fusion_kmeans.yaml

# ============================================
# 5. 增强模型训练 (A100 优化, 追求最高性能)
# ============================================
# 使用 Enhanced ResNet101 + CBAM + FPN + DropBlock
# 支持 best 模型自动保存和早停机制
python3 train.py --config-file configs/enhanced_a100_ultimate.yaml \
                 --save_step 20000 --eval_step 2000 \
                 --early_stop_patience 15 --early_stop_min_delta 0.001

# 备用配置 (使用内置模块，兼容性更好)
python3 train.py --config-file configs/enhanced_a100_builtin.yaml \
                 --save_step 20000 --eval_step 2000

# ============================================
# 5.1 学习率调度器选择
# ============================================
# 支持三种学习率调度策略，通过 SOLVER.LR_SCHEDULER 配置

# (1) 固定步长衰减 (默认)
python3 train.py --config-file configs/xxx.yaml SOLVER.LR_SCHEDULER WarmupMultiStepLR

# (2) 余弦退火
python3 train.py --config-file configs/xxx.yaml SOLVER.LR_SCHEDULER WarmupCosineAnnealingLR

# (3) 基于验证集性能自适应调整 (必须配合 --eval_step 使用)
python3 train.py --config-file configs/xxx.yaml --eval_step 2500 \
                 SOLVER.LR_SCHEDULER ReduceLROnPlateau \
                 SOLVER.PLATEAU_PATIENCE 5 SOLVER.PLATEAU_FACTOR 0.1

# ============================================
# 6. Co-DETR + Swin-L 训练 (SOTA 模型, 需安装 MMDetection)
# ============================================
# 安装 MMDetection
pip install -U openmim
mim install mmengine "mmcv>=2.0.0" mmdet

# 训练 Co-DETR + Swin-L
cd mmdet_codetr
python train_codetr.py --config configs/co_detr_swin_l_voc.py

# 测试
python test_codetr.py --config configs/co_detr_swin_l_voc.py \
                      --checkpoint work_dirs/co_detr_swin_l_voc/best_mAP.pth
cd ..

# ============================================
# 7. Qt 界面编译运行
# ============================================
cd ../qt_deep2
qmake external_program.pro
make
./external_program
```

---

## 详细说明

  cd /content/DEEP2/SSD-Resnet-Attention-FeatureFusion-master
  mkdir -p datasets
  cd datasets

  # 下载 VOC2007 训练/验证集
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar

  # 下载 VOC2007 测试集
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar

  # 解压
  tar -xf VOCtrainval_06-Nov-2007.tar
  tar -xf VOCtest_06-Nov-2007.tar

  # 重命名目录（解压后是 VOCdevkit/VOC2007，需要移动）
  mv VOCdevkit/VOC2007 ./

  # 1. 安装依赖
  cd SSD-Resnet-Attention-FeatureFusion-master
  pip install -r requirements.txt

  # 2. 训练模型
  python3 train.py --config-file configs/resnet50_ssd300_voc0712_feature_fusion.yaml

  # 3. 编译 Qt 界面
  cd ../qt_deep2
  qmake external_program.pro
  make

  # 4. 运行检测界面
  ./external_program

  一、训练流程

  步骤 1: 启动训练
  ────────────────
  $ python3 train.py --config-file configs/resnet50_ssd300_voc0712_feature_fusion.yaml

  步骤 2: 配置加载
  ────────────────
  train.py:91  → cfg.merge_from_file(args.config_file)  # 加载 yaml 配置
  train.py:92  → cfg.merge_from_list(args.opts)         # 合并命令行参数
               → 最终配置: DEVICE=cuda, BACKBONE=R50_300, FUSION=True, BATCH_SIZE=16

  步骤 3: 构建模型
  ────────────────
  train.py:23  → model = build_detection_model(cfg)
                 │
                 ├─→ ResNet50 骨干网络 (resnet_input_300.py:411)
                 │     ├─ conv1 + bn1 + relu + maxpool
                 │     ├─ layer1: 3 个 Bottleneck (64→256 通道)
                 │     ├─ layer2: 4 个 Bottleneck (256→512 通道)  → 特征图 38x38
                 │     ├─ layer3: 6 个 Bottleneck (512→1024 通道) → 特征图 19x19
                 │     └─ layer4: 3 个 Bottleneck (1024→2048 通道) → 特征图 10x10
                 │
                 ├─→ FPN 特征融合 (当 FUSION=True)
                 │     ├─ conv2: 512→256, conv3: 1024→256, conv4: 2048→256
                 │     ├─ 上采样对齐到 38x38
                 │     ├─ concat → 768 通道
                 │     └─ conv5 → 512 通道
                 │
                 ├─→ Extra Layers (生成多尺度特征图)
                 │     └─ 输出 6 个特征图: [38x38, 19x19, 10x10, 5x5, 3x3, 1x1]
                 │
                 └─→ SSD BoxHead (检测头)
                       ├─ 分类分支: 预测 21 个类别
                       └─ 回归分支: 预测边界框坐标

  步骤 4: 加载数据
  ────────────────
  train.py:45  → train_loader = make_data_loader(cfg, is_train=True)
                 │
                 └─→ VOCDataset("voc_2007_trainval")
                       ├─ 加载图像和标注
                       ├─ 数据增强: 随机裁剪、翻转、颜色抖动
                       └─ 归一化: 减去 [123, 117, 104] 均值

  步骤 5: 训练循环
  ────────────────
  train.py:47  → do_train(cfg, model, train_loader, optimizer, scheduler, ...)
                 │
                 └─→ for iteration in range(120000):
                       ├─ images, targets = next(train_loader)
                       ├─ loss_dict = model(images, targets)      # 前向传播
                       │     ├─ cls_loss: 分类损失 (交叉熵)
                       │     └─ reg_loss: 回归损失 (Smooth L1)
                       ├─ loss.backward()                         # 反向传播
                       ├─ optimizer.step()                        # 更新参数
                       ├─ if iteration % 6000 == 0: 保存模型
                       └─ if iteration % 1000 == 0: 评估 mAP

  二、推理流程 (Qt + Python)

  步骤 1: Qt 界面启动
  ──────────────────
  $ ./external_program
  MainWindow 构造函数初始化界面

  步骤 2: 选择图片
  ──────────────
  用户点击 "选择文件" 按钮
      │
      ├─→ on_file_name_button_clicked() (mainwindow.cpp:104)
      │     ├─ QFileDialog::getOpenFileName() 弹出文件选择对话框
      │     ├─ 显示路径到 file_name_Edit
      │     └─ WriteCommunication() 将路径写入 /home/b401-25/shiyan2/file_name.txt
      │
      └─→ 通信文件内容: "/path/to/test_image.jpg"

  步骤 3: 开始检测
  ──────────────
  用户点击 "开始检测" 按钮
      │
      ├─→ on_startorend_clicked() (mainwindow.cpp:58)
      │     ├─ ui->state_Edit->setText("开始检测！")
      │     │
      │     └─→ system("cd ... && python3 demo1.py")
      │           │
      │           │  ┌─────────── demo1.py 执行流程 ───────────┐
      │           │  │                                          │
      │           │  │  1. 加载配置和模型                        │
      │           │  │     cfg.merge_from_file(yaml)            │
      │           │  │     model = build_detection_model(cfg)   │
      │           │  │     checkpointer.load(model_006000.pth)  │
      │           │  │                                          │
      │           │  │  2. 读取图片路径                          │
      │           │  │     f = open("file_name.txt")            │
      │           │  │     image_path = f.read()                │
      │           │  │                                          │
      │           │  │  3. 图像预处理                            │
      │           │  │     image = Image.open(path)             │
      │           │  │     images = transforms(image)           │
      │           │  │       ├─ Resize to 300x300               │
      │           │  │       ├─ ToTensor                        │
      │           │  │       └─ Normalize (减均值)               │
      │           │  │                                          │
      │           │  │  4. 模型推理                              │
      │           │  │     result = model(images.to(device))    │
      │           │  │       ├─ 骨干网络提取特征                  │
      │           │  │       ├─ FPN 特征融合                     │
      │           │  │       ├─ 检测头预测                       │
      │           │  │       └─ NMS 非极大值抑制                  │
      │           │  │                                          │
      │           │  │  5. 过滤低置信度结果                       │
      │           │  │     indices = scores > 0.7               │
      │           │  │     boxes = boxes[indices]               │
      │           │  │                                          │
      │           │  │  6. 绘制检测框并保存                       │
      │           │  │     drawn_image = draw_boxes(...)        │
      │           │  │     Image.save("demo/result/result.jpg") │
      │           │  │                                          │
      │           │  └──────────────────────────────────────────┘
      │           │
      ├─→ showImageSignal()  → show_image()  显示原图到左侧 Label
      │
      └─→ showImageSignal2() → show_image2() 显示检测结果到右侧 Label
            └─ 读取 demo/result/result.jpg 并显示

  三、文件修改汇总

  | 文件                                                     | 修改内容                             |
  |--------------------------------------------------------|----------------------------------|
  | ssd/config/defaults.py:7                               | DEVICE = "cpu" → DEVICE = "cuda" |
  | configs/resnet50_ssd300_voc0712_feature_fusion.yaml:22 | 删除 voc_2012_trainval             |
  | configs/resnet50_ssd300_voc0712_feature_fusion.yaml:28 | BATCH_SIZE: 32 → BATCH_SIZE: 16  |
  | demo1.py:40                                            | 补充通信文件路径                         |
  | demo1.py:81                                            | 补充结果保存路径                         |

  四、K-means++ 优化 Anchor 框对比实验

  步骤 1: 运行 K-means++ 聚类分析
  ────────────────────────────────
  # 分析 VOC2007 数据集目标框尺寸，生成优化后的 Anchor 配置
  python kmeans_anchor.py --data_dir datasets/VOC2007 --clusters 6

  # 输出结果保存在 outputs/kmeans_analysis/ 目录下：
  #   - anchor_clusters.png              聚类散点图
  #   - aspect_ratio_distribution.png    宽高比分布图
  #   - anchor_boxes_visualization.png   Anchor 框可视化
  #   - optimized_anchor_config.yaml     优化后的配置

  步骤 2: 创建优化后的配置文件
  ────────────────────────────────
  # 复制原配置文件
  cp configs/resnet50_ssd300_voc0712_feature_fusion.yaml configs/resnet50_ssd300_voc0712_feature_fusion_kmeans.yaml

  # 根据 kmeans_anchor.py 输出的优化配置，修改新配置文件中的 PRIORS 部分
  # 将 MIN_SIZES 和 MAX_SIZES 替换为聚类得到的新值

  步骤 3: 训练原始 Anchor 模型（对照组）
  ────────────────────────────────
  python3 train.py --config-file configs/resnet50_ssd300_voc0712_feature_fusion.yaml

  # 模型保存路径: outputs/resnet50_ssd300_voc0712_feature_fusion/

  步骤 4: 训练优化 Anchor 模型（实验组）
  ────────────────────────────────
  python3 train.py --config-file configs/resnet50_ssd300_voc0712_feature_fusion_kmeans.yaml

  # 模型保存路径: outputs/resnet50_ssd300_voc0712_feature_fusion_kmeans/

  步骤 5: 对比评估两个模型
  ────────────────────────────────
  # 评估原始 Anchor 模型
  python3 test.py --config-file configs/resnet50_ssd300_voc0712_feature_fusion.yaml

  # 评估优化 Anchor 模型
  python3 test.py --config-file configs/resnet50_ssd300_voc0712_feature_fusion_kmeans.yaml

  # 对比指标：mAP@0.5, mAP@0.75, 各类别 AP

  对比实验结果记录表
  ────────────────────────────────
  | 模型配置        | 平均 IoU | mAP@0.5 | mAP@0.75 | 备注           |
  |----------------|---------|---------|----------|---------------|
  | 原始 Anchor     |         |         |          | 默认 SSD 配置   |
  | K-means++ Anchor|         |         |          | 数据驱动优化    |
