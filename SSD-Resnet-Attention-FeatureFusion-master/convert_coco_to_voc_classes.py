"""
从 COCO 数据集中提取 VOC 的 20 个类别，并进行类别 ID 映射。

使用方法:
    python convert_coco_to_voc_classes.py \
        --coco_ann datasets/COCO2017/Annotations/instances_val2017.json \
        --output datasets/COCO2017/Annotations/instances_val2017_voc.json

    # 同时删除不包含 VOC 类别的图片:
    python convert_coco_to_voc_classes.py \
        --coco_ann datasets/COCO2017/Annotations/instances_val2017.json \
        --output datasets/COCO2017/Annotations/instances_val2017_voc.json \
        --images_dir datasets/COCO2017/JPEGImages \
        --delete_unused

生成的标注文件可以直接用于 VOC 训练的模型测试。
"""

import argparse
import json
import os
from collections import defaultdict


# COCO 类别名 -> COCO 类别 ID 的映射
# 注意：COCO 的类别 ID 不是连续的
COCO_CATEGORIES = {
    'person': 1, 'bicycle': 2, 'car': 3, 'motorcycle': 4, 'airplane': 5,
    'bus': 6, 'train': 7, 'truck': 8, 'boat': 9, 'traffic light': 10,
    'fire hydrant': 11, 'stop sign': 13, 'parking meter': 14, 'bench': 15,
    'bird': 16, 'cat': 17, 'dog': 18, 'horse': 19, 'sheep': 20, 'cow': 21,
    'elephant': 22, 'bear': 23, 'zebra': 24, 'giraffe': 25, 'backpack': 27,
    'umbrella': 28, 'handbag': 31, 'tie': 32, 'suitcase': 33, 'frisbee': 34,
    'skis': 35, 'snowboard': 36, 'sports ball': 37, 'kite': 38, 'baseball bat': 39,
    'baseball glove': 40, 'skateboard': 41, 'surfboard': 42, 'tennis racket': 43,
    'bottle': 44, 'wine glass': 46, 'cup': 47, 'fork': 48, 'knife': 49,
    'spoon': 50, 'bowl': 51, 'banana': 52, 'apple': 53, 'sandwich': 54,
    'orange': 55, 'broccoli': 56, 'carrot': 57, 'hot dog': 58, 'pizza': 59,
    'donut': 60, 'cake': 61, 'chair': 62, 'couch': 63, 'potted plant': 64,
    'bed': 65, 'dining table': 67, 'toilet': 70, 'tv': 72, 'laptop': 73,
    'mouse': 74, 'remote': 75, 'keyboard': 76, 'cell phone': 77, 'microwave': 78,
    'oven': 79, 'toaster': 80, 'sink': 81, 'refrigerator': 82, 'book': 84,
    'clock': 85, 'vase': 86, 'scissors': 87, 'teddy bear': 88, 'hair drier': 89,
    'toothbrush': 90
}

# VOC 类别名 -> VOC 类别 ID (1-20)
VOC_CATEGORIES = {
    'aeroplane': 1, 'bicycle': 2, 'bird': 3, 'boat': 4, 'bottle': 5,
    'bus': 6, 'car': 7, 'cat': 8, 'chair': 9, 'cow': 10,
    'diningtable': 11, 'dog': 12, 'horse': 13, 'motorbike': 14, 'person': 15,
    'pottedplant': 16, 'sheep': 17, 'sofa': 18, 'train': 19, 'tvmonitor': 20
}

# VOC 类别名 -> COCO 类别名 的映射
VOC_TO_COCO_NAME = {
    'aeroplane': 'airplane',
    'bicycle': 'bicycle',
    'bird': 'bird',
    'boat': 'boat',
    'bottle': 'bottle',
    'bus': 'bus',
    'car': 'car',
    'cat': 'cat',
    'chair': 'chair',
    'cow': 'cow',
    'diningtable': 'dining table',
    'dog': 'dog',
    'horse': 'horse',
    'motorbike': 'motorcycle',
    'person': 'person',
    'pottedplant': 'potted plant',
    'sheep': 'sheep',
    'sofa': 'couch',
    'train': 'train',
    'tvmonitor': 'tv'
}

# 构建 COCO 类别 ID -> VOC 类别 ID 的映射
COCO_ID_TO_VOC_ID = {}
for voc_name, voc_id in VOC_CATEGORIES.items():
    coco_name = VOC_TO_COCO_NAME[voc_name]
    coco_id = COCO_CATEGORIES[coco_name]
    COCO_ID_TO_VOC_ID[coco_id] = voc_id

# 需要保留的 COCO 类别 ID 集合
VALID_COCO_IDS = set(COCO_ID_TO_VOC_ID.keys())


def convert_coco_to_voc_classes(coco_ann_path, output_path, images_dir=None, delete_unused=False):
    """
    从 COCO 标注中提取 VOC 类别，并重新映射类别 ID。

    Args:
        coco_ann_path: COCO 标注文件路径 (instances_val2017.json)
        output_path: 输出文件路径
        images_dir: 图片目录路径（用于删除未使用的图片）
        delete_unused: 是否删除不包含 VOC 类别的图片
    """
    print(f"Loading COCO annotations from {coco_ann_path}...")
    with open(coco_ann_path, 'r') as f:
        coco_data = json.load(f)

    print(f"Original annotations: {len(coco_data['annotations'])}")
    print(f"Original images: {len(coco_data['images'])}")
    print(f"Original categories: {len(coco_data['categories'])}")

    # 过滤标注，只保留 VOC 类别
    new_annotations = []
    image_ids_with_annotations = set()

    for ann in coco_data['annotations']:
        coco_cat_id = ann['category_id']
        if coco_cat_id in VALID_COCO_IDS:
            # 映射到 VOC 类别 ID
            new_ann = ann.copy()
            new_ann['category_id'] = COCO_ID_TO_VOC_ID[coco_cat_id]
            new_annotations.append(new_ann)
            image_ids_with_annotations.add(ann['image_id'])

    # 过滤图片，只保留有标注的图片
    new_images = [img for img in coco_data['images'] if img['id'] in image_ids_with_annotations]

    # 创建新的类别列表 (VOC 格式)
    new_categories = []
    for voc_name, voc_id in sorted(VOC_CATEGORIES.items(), key=lambda x: x[1]):
        new_categories.append({
            'id': voc_id,
            'name': voc_name,
            'supercategory': voc_name
        })

    # 构建新的 COCO 格式数据
    new_coco_data = {
        'info': coco_data.get('info', {}),
        'licenses': coco_data.get('licenses', []),
        'images': new_images,
        'annotations': new_annotations,
        'categories': new_categories
    }

    print(f"\nFiltered annotations: {len(new_annotations)}")
    print(f"Filtered images: {len(new_images)}")
    print(f"VOC categories: {len(new_categories)}")

    # 统计每个类别的标注数量
    cat_counts = defaultdict(int)
    for ann in new_annotations:
        cat_counts[ann['category_id']] += 1

    print("\nAnnotations per category:")
    for voc_name, voc_id in sorted(VOC_CATEGORIES.items(), key=lambda x: x[1]):
        print(f"  {voc_id:2d}. {voc_name:12s}: {cat_counts[voc_id]:5d}")

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(new_coco_data, f)

    print(f"\nSaved to {output_path}")

    # 删除不包含 VOC 类别的图片
    if delete_unused and images_dir:
        if not os.path.isdir(images_dir):
            print(f"\nWarning: Images directory not found: {images_dir}")
            return

        # 获取保留的图片文件名
        kept_filenames = set(img['file_name'] for img in new_images)

        # 获取原始所有图片文件名
        all_filenames = set(img['file_name'] for img in coco_data['images'])

        # 计算需要删除的图片
        to_delete = all_filenames - kept_filenames

        if not to_delete:
            print("\nNo unused images to delete.")
            return

        print(f"\nDeleting {len(to_delete)} unused images...")
        deleted_count = 0
        not_found_count = 0

        for filename in to_delete:
            filepath = os.path.join(images_dir, filename)
            if os.path.exists(filepath):
                os.remove(filepath)
                deleted_count += 1
            else:
                not_found_count += 1

        print(f"Deleted: {deleted_count} images")
        if not_found_count > 0:
            print(f"Not found (already deleted?): {not_found_count} images")


def main():
    parser = argparse.ArgumentParser(description='Convert COCO annotations to VOC classes')
    parser.add_argument('--coco_ann', type=str, required=True,
                        help='Path to COCO annotation file (e.g., instances_val2017.json)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for filtered annotations')
    parser.add_argument('--images_dir', type=str, default=None,
                        help='Path to images directory (required if --delete_unused is set)')
    parser.add_argument('--delete_unused', action='store_true',
                        help='Delete images that do not contain any VOC classes')
    args = parser.parse_args()

    if args.delete_unused and not args.images_dir:
        parser.error("--images_dir is required when --delete_unused is set")

    convert_coco_to_voc_classes(args.coco_ann, args.output, args.images_dir, args.delete_unused)


if __name__ == '__main__':
    main()
