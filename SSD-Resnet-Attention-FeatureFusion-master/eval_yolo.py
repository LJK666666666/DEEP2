#!/usr/bin/env python
"""
YOLO11x Evaluation Script

Evaluates YOLO11x model on a dataset and outputs mAP metrics.
Supports VOC format datasets and COCO format datasets.

Usage:
    # Evaluate on VOC2007 test set
    python eval_yolo.py --data_path datasets/VOC2007 --split test

    # Evaluate on COCO2017 val set
    python eval_yolo.py --data_path datasets/COCO2017 --format coco

    # Evaluate on custom images folder
    python eval_yolo.py --images_path path/to/images --annotations_path path/to/annotations

    # Use custom weights
    python eval_yolo.py --weights path/to/yolo.pt --data_path datasets/VOC2007

Requirements:
    - ultralytics
    - torch
    - numpy
    - pycocotools (for mAP calculation)
"""

import argparse
import os
import sys
import json
import numpy as np
from collections import defaultdict
import xml.etree.ElementTree as ET
from tqdm import tqdm


# COCO 80 classes
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# VOC 20 classes
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# COCO to VOC name mapping (for both predictions and ground truth)
COCO_TO_VOC_NAME = {
    'person': 'person', 'bicycle': 'bicycle', 'car': 'car',
    'motorcycle': 'motorbike', 'airplane': 'aeroplane', 'bus': 'bus',
    'train': 'train', 'boat': 'boat', 'bird': 'bird', 'cat': 'cat',
    'dog': 'dog', 'horse': 'horse', 'sheep': 'sheep', 'cow': 'cow',
    'bottle': 'bottle', 'chair': 'chair', 'couch': 'sofa',
    'potted plant': 'pottedplant', 'dining table': 'diningtable', 'tv': 'tvmonitor'
}

# Build mapping for model prediction (COCO index -> VOC name)
COCO_IDX_TO_VOC_NAME = {}
for coco_idx, coco_name in enumerate(COCO_CLASSES):
    if coco_name in COCO_TO_VOC_NAME:
        COCO_IDX_TO_VOC_NAME[coco_idx] = COCO_TO_VOC_NAME[coco_name]


def parse_voc_annotation(xml_path):
    """Parse VOC XML annotation file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        difficult = obj.find('difficult')
        difficult = int(difficult.text) if difficult is not None else 0

        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)

        objects.append({
            'name': name,
            'bbox': [xmin, ymin, xmax, ymax],
            'difficult': difficult
        })

    return objects


def load_coco_annotations(json_path):
    """
    Load COCO format annotations and convert to evaluation format.
    Maps COCO category names to VOC category names.

    Args:
        json_path: Path to COCO JSON annotation file

    Returns:
        image_ids: List of image IDs (as strings)
        image_id_to_filename: Dict mapping image_id to filename
        ground_truths: Dict of class_name -> {img_id: [(bbox, difficult), ...]}
    """
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    # Build category_id -> category_name mapping
    cat_id_to_name = {}
    for cat in coco_data['categories']:
        cat_id_to_name[cat['id']] = cat['name']

    # Build image_id -> filename mapping
    image_id_to_filename = {}
    image_ids = []
    for img in coco_data['images']:
        img_id = str(img['id'])  # Convert to string for consistency
        image_ids.append(img_id)
        image_id_to_filename[img_id] = img['file_name']

    # Parse annotations and map to VOC classes
    ground_truths = defaultdict(dict)  # class_name -> {img_id: [(bbox, difficult), ...]}

    for ann in coco_data['annotations']:
        img_id = str(ann['image_id'])
        cat_id = ann['category_id']
        coco_name = cat_id_to_name.get(cat_id, '')

        # Map COCO name to VOC name
        if coco_name not in COCO_TO_VOC_NAME:
            continue  # Skip non-VOC classes

        voc_name = COCO_TO_VOC_NAME[coco_name]

        # Convert COCO bbox [x, y, width, height] to [xmin, ymin, xmax, ymax]
        x, y, w, h = ann['bbox']
        bbox = [x, y, x + w, y + h]

        # iscrowd is similar to difficult
        difficult = ann.get('iscrowd', 0)

        if img_id not in ground_truths[voc_name]:
            ground_truths[voc_name][img_id] = []
        ground_truths[voc_name][img_id].append((bbox, difficult))

    return image_ids, image_id_to_filename, ground_truths


def compute_iou(box1, box2):
    """Compute IoU between two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def compute_ap(recalls, precisions):
    """Compute Average Precision using 11-point interpolation (VOC style)."""
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        precisions_above = [p for r, p in zip(recalls, precisions) if r >= t]
        if len(precisions_above) > 0:
            ap += max(precisions_above)
    return ap / 11.0


def evaluate_class(detections, ground_truths, iou_threshold=0.5):
    """
    Evaluate detections for a single class.

    Args:
        detections: List of (image_id, confidence, bbox)
        ground_truths: Dict of image_id -> list of (bbox, difficult)
        iou_threshold: IoU threshold for matching

    Returns:
        AP value, or 0.0 if no ground truths exist
    """
    # Count total ground truths (excluding difficult)
    n_gt = sum(
        len([g for g in gts if not g[1]])
        for gts in ground_truths.values()
    )

    # If no ground truths, return 0
    if n_gt == 0:
        return 0.0

    # If no detections, return 0
    if len(detections) == 0:
        return 0.0

    # Sort by confidence
    detections = sorted(detections, key=lambda x: x[1], reverse=True)

    # Track which ground truths have been matched
    gt_matched = {img_id: [False] * len(gts) for img_id, gts in ground_truths.items()}

    tp = np.zeros(len(detections))
    fp = np.zeros(len(detections))

    for i, (img_id, conf, det_bbox) in enumerate(detections):
        if img_id not in ground_truths:
            fp[i] = 1
            continue

        gts = ground_truths[img_id]
        max_iou = 0
        max_idx = -1

        for j, (gt_bbox, difficult) in enumerate(gts):
            iou = compute_iou(det_bbox, gt_bbox)
            if iou > max_iou:
                max_iou = iou
                max_idx = j

        if max_iou >= iou_threshold and max_idx >= 0:
            if not gts[max_idx][1]:  # Not difficult
                if not gt_matched[img_id][max_idx]:
                    tp[i] = 1
                    gt_matched[img_id][max_idx] = True
                else:
                    fp[i] = 1
            # Ignore difficult objects
        else:
            fp[i] = 1

    # Compute precision and recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    # Handle case where all detections matched difficult objects (tp + fp = 0)
    denominator = tp_cumsum + fp_cumsum
    if denominator[-1] == 0:  # No valid detections (all matched difficult GTs)
        return 0.0

    recalls = tp_cumsum / n_gt
    # Avoid division by zero - set precision to 0 where denominator is 0
    precisions = np.divide(tp_cumsum, denominator, out=np.zeros_like(tp_cumsum, dtype=float), where=denominator != 0)

    # Compute AP
    ap = compute_ap(recalls, precisions)
    return ap


def evaluate_yolo(model, images_dir, annotations_dir, image_ids, conf_thres=0.001, batch_size=16):
    """
    Evaluate YOLO model on dataset.

    Args:
        model: YOLO model
        images_dir: Directory containing images
        annotations_dir: Directory containing VOC XML annotations
        image_ids: List of image IDs to evaluate
        conf_thres: Confidence threshold
        batch_size: Batch size for inference (default: 16)

    Returns:
        mAP and per-class AP
    """
    # Collect all detections and ground truths per class
    all_detections = defaultdict(list)  # class_name -> [(img_id, conf, bbox), ...]
    all_ground_truths = defaultdict(dict)  # class_name -> {img_id: [(bbox, difficult), ...]}

    # Prepare image paths and load annotations
    valid_image_ids = []
    valid_image_paths = []

    print(f"\nPreparing {len(image_ids)} images...")
    for img_id in image_ids:
        # Find image path
        img_path = os.path.join(images_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(images_dir, f"{img_id}.png")
        if not os.path.exists(img_path):
            continue

        valid_image_ids.append(img_id)
        valid_image_paths.append(img_path)

        # Load annotations
        ann_path = os.path.join(annotations_dir, f"{img_id}.xml")
        if os.path.exists(ann_path):
            annotations = parse_voc_annotation(ann_path)
            for ann in annotations:
                if ann['name'] in VOC_CLASSES:
                    if img_id not in all_ground_truths[ann['name']]:
                        all_ground_truths[ann['name']][img_id] = []
                    all_ground_truths[ann['name']][img_id].append(
                        (ann['bbox'], ann['difficult'])
                    )

    print(f"Found {len(valid_image_ids)} valid images")
    print(f"Running batch inference (batch_size={batch_size})...")

    # Process images in batches
    num_batches = (len(valid_image_paths) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Inference"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(valid_image_paths))

        batch_paths = valid_image_paths[start_idx:end_idx]
        batch_ids = valid_image_ids[start_idx:end_idx]

        # Run batch inference
        results = model(batch_paths, conf=conf_thres, verbose=False)

        # Process results for each image in batch
        for img_id, result in zip(batch_ids, results):
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                labels = result.boxes.cls.cpu().numpy().astype(int)

                for box, score, label in zip(boxes, scores, labels):
                    if label in COCO_IDX_TO_VOC_NAME:
                        voc_name = COCO_IDX_TO_VOC_NAME[label]
                        all_detections[voc_name].append((img_id, score, box.tolist()))

    # Compute AP for each class
    print("\nComputing mAP...")

    # Debug: Print ground truth and detection counts per class
    print("\nGround truth and detection counts:")
    for class_name in VOC_CLASSES:
        gt_count = sum(len(gts) for gts in all_ground_truths.get(class_name, {}).values())
        det_count = len(all_detections.get(class_name, []))
        print(f"  {class_name}: GT={gt_count}, Det={det_count}")

    aps = {}
    for class_name in VOC_CLASSES:
        detections = all_detections.get(class_name, [])
        ground_truths = all_ground_truths.get(class_name, {})

        ap = evaluate_class(detections, ground_truths, iou_threshold=0.5)
        aps[class_name] = ap

    # Compute mAP (use nanmean to handle any remaining NaN values safely)
    mAP = np.nanmean(list(aps.values()))

    return mAP, aps


def evaluate_yolo_coco(model, images_dir, coco_json_path, conf_thres=0.001, batch_size=16):
    """
    Evaluate YOLO model on COCO format dataset.
    Maps COCO ground truth labels to VOC names for evaluation.

    Args:
        model: YOLO model
        images_dir: Directory containing images
        coco_json_path: Path to COCO JSON annotation file
        conf_thres: Confidence threshold
        batch_size: Batch size for inference (default: 16)

    Returns:
        mAP and per-class AP
    """
    # Load COCO annotations (already mapped to VOC names)
    print(f"\nLoading COCO annotations from {coco_json_path}...")
    image_ids, image_id_to_filename, all_ground_truths = load_coco_annotations(coco_json_path)
    print(f"Loaded {len(image_ids)} images, {sum(sum(len(gts) for gts in class_gts.values()) for class_gts in all_ground_truths.values())} annotations")

    # Collect all detections per class
    all_detections = defaultdict(list)  # class_name -> [(img_id, conf, bbox), ...]

    # Prepare image paths
    valid_image_ids = []
    valid_image_paths = []

    print(f"\nPreparing images...")
    for img_id in image_ids:
        filename = image_id_to_filename.get(img_id, '')
        img_path = os.path.join(images_dir, filename)
        if not os.path.exists(img_path):
            continue

        valid_image_ids.append(img_id)
        valid_image_paths.append(img_path)

    print(f"Found {len(valid_image_ids)} valid images")
    print(f"Running batch inference (batch_size={batch_size})...")

    # Process images in batches
    num_batches = (len(valid_image_paths) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Inference"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(valid_image_paths))

        batch_paths = valid_image_paths[start_idx:end_idx]
        batch_ids = valid_image_ids[start_idx:end_idx]

        # Run batch inference
        results = model(batch_paths, conf=conf_thres, verbose=False)

        # Process results for each image in batch
        for img_id, result in zip(batch_ids, results):
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                labels = result.boxes.cls.cpu().numpy().astype(int)

                for box, score, label in zip(boxes, scores, labels):
                    if label in COCO_IDX_TO_VOC_NAME:
                        voc_name = COCO_IDX_TO_VOC_NAME[label]
                        all_detections[voc_name].append((img_id, score, box.tolist()))

    # Compute AP for each class
    print("\nComputing mAP...")

    # Debug: Print ground truth and detection counts per class
    print("\nGround truth and detection counts:")
    for class_name in VOC_CLASSES:
        gt_count = sum(len(gts) for gts in all_ground_truths.get(class_name, {}).values())
        det_count = len(all_detections.get(class_name, []))
        print(f"  {class_name}: GT={gt_count}, Det={det_count}")

    aps = {}
    for class_name in VOC_CLASSES:
        detections = all_detections.get(class_name, [])
        ground_truths = all_ground_truths.get(class_name, {})

        ap = evaluate_class(detections, ground_truths, iou_threshold=0.5)
        aps[class_name] = ap

    # Compute mAP (use nanmean to handle any remaining NaN values safely)
    mAP = np.nanmean(list(aps.values()))

    return mAP, aps


def main():
    parser = argparse.ArgumentParser(description='Evaluate YOLO11x on VOC/COCO dataset')
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to YOLO weights')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to dataset (e.g., datasets/VOC2007 or datasets/COCO2017)')
    parser.add_argument('--images_path', type=str, default=None,
                        help='Path to images directory')
    parser.add_argument('--annotations_path', type=str, default=None,
                        help='Path to annotations directory or JSON file')
    parser.add_argument('--format', type=str, default='auto', choices=['auto', 'voc', 'coco'],
                        help='Annotation format: auto (detect), voc (XML), coco (JSON)')
    parser.add_argument('--split', type=str, default='test',
                        help='Dataset split (train, val, test, trainval) - for VOC format')
    parser.add_argument('--coco_json', type=str, default=None,
                        help='Path to COCO JSON annotation file (for COCO format)')
    parser.add_argument('--conf_thres', type=float, default=0.001,
                        help='Confidence threshold for evaluation')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for inference (default: 16)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results')
    args = parser.parse_args()

    # Determine format and paths
    ann_format = args.format
    coco_json_path = None

    if args.data_path:
        images_dir = os.path.join(args.data_path, 'JPEGImages')
        annotations_dir = os.path.join(args.data_path, 'Annotations')

        # Auto-detect format
        if ann_format == 'auto':
            # Check if there are JSON files in Annotations directory (COCO format)
            if os.path.isdir(annotations_dir):
                json_files = [f for f in os.listdir(annotations_dir) if f.endswith('.json')]
                if json_files:
                    ann_format = 'coco'
                    # Try to find a suitable JSON file
                    for candidate in ['instances_val2017.json', 'instances_val2017_voc.json',
                                      'instances_train2017.json', 'instances_train2017_voc.json']:
                        if candidate in json_files:
                            coco_json_path = os.path.join(annotations_dir, candidate)
                            break
                    if coco_json_path is None:
                        coco_json_path = os.path.join(annotations_dir, json_files[0])
                else:
                    ann_format = 'voc'
            else:
                ann_format = 'voc'

        if ann_format == 'coco':
            if args.coco_json:
                coco_json_path = args.coco_json
            elif coco_json_path is None:
                # Look for JSON file in Annotations
                json_files = [f for f in os.listdir(annotations_dir) if f.endswith('.json')]
                if json_files:
                    coco_json_path = os.path.join(annotations_dir, json_files[0])
                else:
                    print("Error: No COCO JSON file found. Please specify --coco_json")
                    sys.exit(1)

        split_file = os.path.join(args.data_path, 'ImageSets', 'Main', f'{args.split}.txt')

    elif args.images_path and args.annotations_path:
        images_dir = args.images_path
        annotations_dir = args.annotations_path

        # Check if annotations_path is a JSON file
        if args.annotations_path.endswith('.json'):
            ann_format = 'coco'
            coco_json_path = args.annotations_path
        elif ann_format == 'auto':
            ann_format = 'voc'

        split_file = None
    else:
        print("Error: Please provide either --data_path or both --images_path and --annotations_path")
        sys.exit(1)

    # Load model
    if args.weights is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.weights = os.path.join(base_dir, 'yolo11x.pt')

    if not os.path.exists(args.weights):
        print(f"Error: Weights file not found: {args.weights}")
        sys.exit(1)

    print("=" * 60)
    print("YOLO11x Evaluation")
    print("=" * 60)
    print(f"Weights: {args.weights}")
    print(f"Images: {images_dir}")
    print(f"Format: {ann_format.upper()}")
    if ann_format == 'coco':
        print(f"COCO JSON: {coco_json_path}")
    else:
        print(f"Annotations: {annotations_dir}")
    print(f"Confidence threshold: {args.conf_thres}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")

    try:
        from ultralytics import YOLO
        model = YOLO(args.weights)
        model.to(args.device)
    except ImportError:
        print("Error: ultralytics is not installed. Please run: pip install ultralytics")
        sys.exit(1)

    # Evaluate based on format
    if ann_format == 'coco':
        mAP, aps = evaluate_yolo_coco(model, images_dir, coco_json_path, args.conf_thres, args.batch_size)
    else:
        # VOC format
        # Get image IDs
        if split_file and os.path.exists(split_file):
            with open(split_file, 'r') as f:
                image_ids = [line.strip() for line in f.readlines()]
            print(f"Loaded {len(image_ids)} images from {split_file}")
        else:
            # Get all images from directory
            image_ids = []
            for f in os.listdir(images_dir):
                if f.endswith(('.jpg', '.jpeg', '.png')):
                    image_ids.append(os.path.splitext(f)[0])
            print(f"Found {len(image_ids)} images in {images_dir}")

        if len(image_ids) == 0:
            print("Error: No images found")
            sys.exit(1)

        mAP, aps = evaluate_yolo(model, images_dir, annotations_dir, image_ids, args.conf_thres, args.batch_size)

    # Print results
    print("\n" + "=" * 60)
    print("Results (VOC mAP @ IoU=0.5)")
    print("=" * 60)
    print(f"\n{'Class':<15} {'AP':>10}")
    print("-" * 25)
    for class_name in VOC_CLASSES:
        ap = aps[class_name]
        print(f"{class_name:<15} {ap*100:>9.2f}%")
    print("-" * 25)
    print(f"{'mAP':<15} {mAP*100:>9.2f}%")
    print("=" * 60)

    # Save results
    if args.output:
        results = {
            'mAP': mAP,
            'per_class_AP': aps,
            'config': {
                'weights': args.weights,
                'data_path': args.data_path,
                'format': ann_format,
                'conf_thres': args.conf_thres,
            }
        }
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
