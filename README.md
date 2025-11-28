  代码运行流程

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

  四、运行命令总结

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

  