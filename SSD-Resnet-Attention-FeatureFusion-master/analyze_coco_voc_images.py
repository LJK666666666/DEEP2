"""
统计 COCO 数据集中包含/不包含 VOC 20 类别的图片数量。

使用方法:
    python analyze_coco_voc_images.py --coco_ann path/to/instances_train2017.json
"""

import argparse
import json
from collections import Counter, defaultdict

# VOC 对应的 COCO 类别 ID
VOC_COCO_IDS = {
    5: 'aeroplane', 2: 'bicycle', 16: 'bird', 9: 'boat', 44: 'bottle',
    6: 'bus', 3: 'car', 17: 'cat', 62: 'chair', 21: 'cow',
    67: 'diningtable', 18: 'dog', 19: 'horse', 4: 'motorbike', 1: 'person',
    64: 'pottedplant', 20: 'sheep', 63: 'sofa', 7: 'train', 72: 'tvmonitor'
}

VALID_COCO_IDS = set(VOC_COCO_IDS.keys())


def analyze_coco_voc_images(coco_ann_path):
    print(f"Loading COCO annotations from {coco_ann_path}...")
    with open(coco_ann_path, 'r') as f:
        data = json.load(f)

    total_images = len(data['images'])
    all_image_ids = set(img['id'] for img in data['images'])

    # 统计每张图片包含的 VOC 类别
    image_voc_classes = defaultdict(set)  # image_id -> set of VOC class names

    for ann in data['annotations']:
        cat_id = ann['category_id']
        if cat_id in VALID_COCO_IDS:
            image_voc_classes[ann['image_id']].add(VOC_COCO_IDS[cat_id])

    # 包含 VOC 类别的图片
    images_with_voc = set(image_voc_classes.keys())
    # 不包含 VOC 类别的图片
    images_without_voc = all_image_ids - images_with_voc

    print("\n" + "=" * 60)
    print("COCO Dataset - VOC Classes Image Statistics")
    print("=" * 60)
    print(f"Total images:                    {total_images:>8}")
    print(f"Images WITH VOC classes:         {len(images_with_voc):>8} ({len(images_with_voc)/total_images*100:.2f}%)")
    print(f"Images WITHOUT VOC classes:      {len(images_without_voc):>8} ({len(images_without_voc)/total_images*100:.2f}%)")
    print("=" * 60)

    # 统计每个 VOC 类别出现在多少张图片中
    class_image_counts = Counter()
    for img_id, classes in image_voc_classes.items():
        for cls in classes:
            class_image_counts[cls] += 1

    print("\nImages per VOC class (how many images contain each class):")
    print("-" * 60)
    max_count = max(class_image_counts.values())
    for name, count in sorted(class_image_counts.items(), key=lambda x: -x[1]):
        pct = count / len(images_with_voc) * 100
        bar_len = int(count / max_count * 25)
        bar = '█' * bar_len
        print(f"  {name:12s}: {count:7d} images ({pct:5.2f}%) {bar}")

    # 统计每张图片包含多少个 VOC 类别
    num_classes_per_image = Counter()
    for img_id, classes in image_voc_classes.items():
        num_classes_per_image[len(classes)] += 1

    print("\n" + "-" * 60)
    print("Number of VOC classes per image:")
    print("-" * 60)
    for num_classes in sorted(num_classes_per_image.keys()):
        count = num_classes_per_image[num_classes]
        pct = count / len(images_with_voc) * 100
        bar_len = int(pct / 2)
        bar = '█' * bar_len
        print(f"  {num_classes:2d} classes: {count:7d} images ({pct:5.2f}%) {bar}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Analyze COCO images for VOC classes')
    parser.add_argument('--coco_ann', type=str, required=True,
                        help='Path to COCO annotation file (e.g., instances_train2017.json)')
    args = parser.parse_args()

    analyze_coco_voc_images(args.coco_ann)


if __name__ == '__main__':
    main()
