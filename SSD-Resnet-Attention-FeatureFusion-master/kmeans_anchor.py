"""
K-means++ 算法优化 SSD Anchor 框

本脚本用于分析 VOC2007 数据集中目标框的尺寸分布，
使用 K-means++ 聚类算法生成更适合该数据集的 Anchor 框尺寸。

使用方法:
    python kmeans_anchor.py --data_dir datasets/VOC2007 --clusters 6

作者: 深度学习实验二
"""

import os
import argparse
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def parse_voc_annotation(ann_dir, image_set_file):
    """
    解析 VOC 数据集的标注文件，提取所有目标框的宽高信息

    Args:
        ann_dir: Annotations 目录路径
        image_set_file: ImageSets/Main/trainval.txt 文件路径

    Returns:
        boxes: numpy array, shape (N, 2), 每行是 [width, height]
        image_sizes: 对应的图像尺寸列表
    """
    boxes = []
    image_sizes = []

    # 读取图像列表
    with open(image_set_file, 'r') as f:
        image_ids = [line.strip() for line in f.readlines()]

    print(f"正在解析 {len(image_ids)} 张图像的标注...")

    for image_id in image_ids:
        ann_file = os.path.join(ann_dir, f"{image_id}.xml")
        if not os.path.exists(ann_file):
            continue

        tree = ET.parse(ann_file)
        root = tree.getroot()

        # 获取图像尺寸
        size = root.find('size')
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)

        # 解析所有目标框
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)

            # 计算框的宽高（归一化到 0-1）
            box_width = (xmax - xmin) / img_width
            box_height = (ymax - ymin) / img_height

            boxes.append([box_width, box_height])
            image_sizes.append([img_width, img_height])

    return np.array(boxes), image_sizes


def iou(box, clusters):
    """
    计算一个框与所有聚类中心的 IoU

    Args:
        box: [width, height]
        clusters: (k, 2) array of cluster centers

    Returns:
        IoU values for each cluster
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_value = intersection / (box_area + cluster_area - intersection + 1e-10)
    return iou_value


def avg_iou(boxes, clusters):
    """
    计算所有框与其最近聚类中心的平均 IoU
    """
    return np.mean([np.max(iou(box, clusters)) for box in boxes])


def kmeans_plusplus(boxes, k, max_iter=300):
    """
    K-means++ 聚类算法

    Args:
        boxes: (N, 2) array, 每行是 [width, height]
        k: 聚类数量
        max_iter: 最大迭代次数

    Returns:
        clusters: (k, 2) array, 聚类中心
    """
    print(f"\n使用 K-means++ 算法进行聚类 (k={k})...")

    # 使用 sklearn 的 KMeans（内置 k-means++ 初始化）
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=max_iter,
                    n_init=10, random_state=42)
    kmeans.fit(boxes)

    clusters = kmeans.cluster_centers_

    # 按面积排序
    areas = clusters[:, 0] * clusters[:, 1]
    sorted_indices = np.argsort(areas)
    clusters = clusters[sorted_indices]

    return clusters


def analyze_aspect_ratios(boxes):
    """
    分析目标框的宽高比分布
    """
    aspect_ratios = boxes[:, 0] / (boxes[:, 1] + 1e-10)

    print("\n=== 宽高比分布分析 ===")
    print(f"最小宽高比: {aspect_ratios.min():.3f}")
    print(f"最大宽高比: {aspect_ratios.max():.3f}")
    print(f"平均宽高比: {aspect_ratios.mean():.3f}")
    print(f"中位数宽高比: {np.median(aspect_ratios):.3f}")

    # 统计常见宽高比区间
    ranges = [(0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, float('inf'))]
    print("\n宽高比区间分布:")
    for low, high in ranges:
        count = np.sum((aspect_ratios >= low) & (aspect_ratios < high))
        percentage = count / len(aspect_ratios) * 100
        print(f"  [{low:.1f}, {high:.1f}): {count} ({percentage:.1f}%)")

    return aspect_ratios


def visualize_results(boxes, clusters, output_dir):
    """
    可视化聚类结果
    """
    os.makedirs(output_dir, exist_ok=True)

    # 图1: 目标框尺寸散点图 + 聚类中心
    plt.figure(figsize=(10, 8))
    plt.scatter(boxes[:, 0], boxes[:, 1], s=1, alpha=0.3, label='Ground Truth Boxes')
    plt.scatter(clusters[:, 0], clusters[:, 1], s=200, c='red', marker='*',
                edgecolors='black', linewidths=2, label='Cluster Centers (Anchors)')
    plt.xlabel('Width (normalized)')
    plt.ylabel('Height (normalized)')
    plt.title('K-means++ Clustering of Bounding Boxes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'anchor_clusters.png'), dpi=150)
    plt.close()
    print(f"\n聚类散点图已保存到: {output_dir}/anchor_clusters.png")

    # 图2: 宽高比直方图
    aspect_ratios = boxes[:, 0] / (boxes[:, 1] + 1e-10)
    plt.figure(figsize=(10, 6))
    plt.hist(aspect_ratios, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Aspect Ratio (width/height)')
    plt.ylabel('Count')
    plt.title('Distribution of Aspect Ratios in VOC2007')
    plt.axvline(x=1.0, color='red', linestyle='--', label='Square (1:1)')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'aspect_ratio_distribution.png'), dpi=150)
    plt.close()
    print(f"宽高比分布图已保存到: {output_dir}/aspect_ratio_distribution.png")

    # 图3: Anchor 框可视化
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')

    colors = plt.cm.rainbow(np.linspace(0, 1, len(clusters)))

    for i, (cluster, color) in enumerate(zip(clusters, colors)):
        w, h = cluster
        # 以中心为原点绘制
        rect = plt.Rectangle((0.5 - w/2, 0.5 - h/2), w, h,
                             fill=False, edgecolor=color, linewidth=2,
                             label=f'Anchor {i+1}: {w:.3f}x{h:.3f}')
        ax.add_patch(rect)

    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.title('Optimized Anchor Boxes (K-means++)')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'anchor_boxes_visualization.png'), dpi=150)
    plt.close()
    print(f"Anchor 框可视化已保存到: {output_dir}/anchor_boxes_visualization.png")


def convert_to_ssd_format(clusters, input_size=300):
    """
    将聚类结果转换为 SSD 配置格式

    Args:
        clusters: (k, 2) array, 归一化的聚类中心
        input_size: SSD 输入图像尺寸

    Returns:
        min_sizes, max_sizes: SSD 格式的先验框尺寸
    """
    # 转换为像素尺寸
    pixel_sizes = clusters * input_size

    # 计算每个 anchor 的等效尺寸 (sqrt(w*h))
    equiv_sizes = np.sqrt(pixel_sizes[:, 0] * pixel_sizes[:, 1])

    # 排序
    equiv_sizes = np.sort(equiv_sizes)

    # 生成 min_sizes 和 max_sizes
    min_sizes = equiv_sizes.tolist()
    max_sizes = []
    for i in range(len(min_sizes)):
        if i < len(min_sizes) - 1:
            max_sizes.append(np.sqrt(min_sizes[i] * min_sizes[i+1]))
        else:
            max_sizes.append(min_sizes[i] * 1.2)  # 最后一个稍大

    return min_sizes, max_sizes


def compare_with_default(clusters, input_size=300):
    """
    与 SSD 默认 Anchor 进行对比
    """
    # SSD300 默认配置
    default_min_sizes = [21, 45, 99, 153, 207, 261]
    default_max_sizes = [45, 99, 153, 207, 261, 315]

    print("\n" + "="*60)
    print("=== SSD 默认 Anchor vs K-means++ 优化 Anchor 对比 ===")
    print("="*60)

    print("\n【SSD 默认配置】")
    print(f"  MIN_SIZES: {default_min_sizes}")
    print(f"  MAX_SIZES: {default_max_sizes}")

    # 计算优化后的配置
    new_min_sizes, new_max_sizes = convert_to_ssd_format(clusters, input_size)

    print("\n【K-means++ 优化配置】")
    print(f"  MIN_SIZES: {[round(x, 1) for x in new_min_sizes]}")
    print(f"  MAX_SIZES: {[round(x, 1) for x in new_max_sizes]}")

    print("\n【聚类中心详情】(归一化坐标)")
    for i, (w, h) in enumerate(clusters):
        ar = w / h
        print(f"  Anchor {i+1}: width={w:.4f}, height={h:.4f}, aspect_ratio={ar:.2f}")

    return new_min_sizes, new_max_sizes


def generate_yaml_config(clusters, input_size=300, output_file=None):
    """
    生成优化后的 YAML 配置
    """
    min_sizes, max_sizes = convert_to_ssd_format(clusters, input_size)

    # 计算宽高比
    aspect_ratios = clusters[:, 0] / clusters[:, 1]
    unique_ratios = set()
    for ar in aspect_ratios:
        if ar > 1:
            unique_ratios.add(round(ar, 1))
        else:
            unique_ratios.add(round(1/ar, 1))

    config = f"""
# ============================================
# K-means++ 优化后的 Anchor 配置
# ============================================
# 使用以下配置替换 YAML 文件中的 PRIORS 部分

MODEL:
  PRIORS:
    MIN_SIZES: {[int(round(x)) for x in min_sizes]}
    MAX_SIZES: {[int(round(x)) for x in max_sizes]}
    # 建议的宽高比 (基于聚类分析): {sorted(unique_ratios)}
    ASPECT_RATIOS: [[2], [2, 3], [2, 3], [2, 3], [2], [2]]

# 聚类中心 (归一化坐标):
"""
    for i, (w, h) in enumerate(clusters):
        config += f"#   Anchor {i+1}: [{w:.4f}, {h:.4f}], aspect_ratio={w/h:.2f}\n"

    if output_file:
        with open(output_file, 'w') as f:
            f.write(config)
        print(f"\n配置已保存到: {output_file}")

    print(config)
    return config


def main():
    parser = argparse.ArgumentParser(description='K-means++ Anchor Box Optimization for SSD')
    parser.add_argument('--data_dir', type=str, default='datasets/VOC2007',
                        help='VOC2007 数据集路径')
    parser.add_argument('--clusters', type=int, default=6,
                        help='聚类数量 (对应 SSD 的 6 个特征图)')
    parser.add_argument('--input_size', type=int, default=300,
                        help='SSD 输入图像尺寸')
    parser.add_argument('--output_dir', type=str, default='outputs/kmeans_analysis',
                        help='输出目录')
    args = parser.parse_args()

    print("="*60)
    print("   K-means++ Anchor Box 优化分析")
    print("   用于 SSD 目标检测网络")
    print("="*60)

    # 路径设置
    ann_dir = os.path.join(args.data_dir, 'Annotations')
    image_set_file = os.path.join(args.data_dir, 'ImageSets', 'Main', 'trainval.txt')

    # 检查路径
    if not os.path.exists(ann_dir):
        print(f"错误: 找不到 Annotations 目录: {ann_dir}")
        return
    if not os.path.exists(image_set_file):
        print(f"错误: 找不到图像列表文件: {image_set_file}")
        return

    # 1. 解析标注文件
    print("\n[Step 1] 解析 VOC2007 标注文件...")
    boxes, image_sizes = parse_voc_annotation(ann_dir, image_set_file)
    print(f"共解析到 {len(boxes)} 个目标框")

    # 2. 分析宽高比分布
    print("\n[Step 2] 分析目标框宽高比分布...")
    analyze_aspect_ratios(boxes)

    # 3. K-means++ 聚类
    print(f"\n[Step 3] 执行 K-means++ 聚类 (k={args.clusters})...")
    clusters = kmeans_plusplus(boxes, args.clusters)

    # 4. 计算平均 IoU
    avg_iou_value = avg_iou(boxes, clusters)
    print(f"\n聚类结果平均 IoU: {avg_iou_value:.4f}")
    print("(IoU 越高表示聚类的 Anchor 框与真实目标框匹配越好)")

    # 5. 与默认配置对比
    print("\n[Step 4] 与 SSD 默认 Anchor 配置对比...")
    new_min_sizes, new_max_sizes = compare_with_default(clusters, args.input_size)

    # 6. 可视化结果
    print("\n[Step 5] 生成可视化结果...")
    visualize_results(boxes, clusters, args.output_dir)

    # 7. 生成 YAML 配置
    print("\n[Step 6] 生成优化后的配置...")
    config_file = os.path.join(args.output_dir, 'optimized_anchor_config.yaml')
    generate_yaml_config(clusters, args.input_size, config_file)

    print("\n" + "="*60)
    print("   分析完成!")
    print("="*60)
    print(f"\n所有结果已保存到: {args.output_dir}/")
    print("\n使用建议:")
    print("1. 查看 anchor_clusters.png 了解聚类效果")
    print("2. 查看 optimized_anchor_config.yaml 获取优化配置")
    print("3. 将优化后的 MIN_SIZES 和 MAX_SIZES 替换到 YAML 配置文件中")
    print("4. 重新训练模型并对比效果")


if __name__ == '__main__':
    main()
