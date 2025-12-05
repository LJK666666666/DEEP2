# Co-DETR + Swin-L 目标检测

基于 MMDetection 的 Co-DETR (Collaborative DETR) 配置，使用 Swin Transformer Large 作为骨干网络。

## 环境安装

```bash
# 1. 安装 PyTorch (根据 CUDA 版本选择)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# 2. 安装 MMDetection 及依赖
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet

# 3. 安装 Co-DETR 额外依赖
pip install fairscale
pip install timm
```

## 数据集准备

VOC2007 数据集需要转换为 COCO 格式，或直接使用 MMDetection 的 VOC 支持：

```bash
# 数据集目录结构
datasets/
└── VOC2007/
    ├── Annotations/
    ├── ImageSets/
    ├── JPEGImages/
    └── ...
```

## 训练命令

```bash
# 单 GPU 训练
python train_codetr.py --config configs/co_detr_swin_l_voc.py

# 多 GPU 训练 (推荐 A100)
torchrun --nproc_per_node=4 train_codetr.py --config configs/co_detr_swin_l_voc.py

# 使用 MMDetection 原生训练脚本
mim train mmdet configs/co_detr_swin_l_voc.py --gpus 1
```

## 测试命令

```bash
python test_codetr.py --config configs/co_detr_swin_l_voc.py \
                      --checkpoint work_dirs/co_detr_swin_l_voc/best_mAP.pth
```

## 模型说明

### Co-DETR (Collaborative DETR)
- 论文: "DETRs with Collaborative Hybrid Assignments Training" (ICCV 2023)
- 特点: 结合了 one-to-one 和 one-to-many 标签分配策略
- 在 DETR 基础上引入辅助检测头加速收敛

### Swin Transformer Large
- 论文: "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" (ICCV 2021)
- 参数量: ~197M
- 特点: 层次化设计，shifted window 机制，线性计算复杂度

### 预期性能
- VOC2007 mAP: 85%+ (使用 COCO 预训练权重)
- 训练时间: ~12h (A100 80GB, batch_size=2)
