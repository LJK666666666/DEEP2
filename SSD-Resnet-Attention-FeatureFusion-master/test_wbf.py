#!/usr/bin/env python
"""Test script to diagnose WBF issues."""

import os
import sys
import numpy as np

# Add paths
sys.path.insert(0, os.path.dirname(__file__))

from run_yolo_gui import (
    YOLODetector, filter_to_voc, apply_wbf, builtin_wbf,
    COCO_CLASSES, VOC_CLASSES, COCO_IDX_TO_VOC_IDX, COCO_IDX_TO_VOC_NAME
)
from PIL import Image

def test_models():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Test image
    test_image = os.path.join(
        os.path.dirname(__file__),
        'datasets', 'VOC2007', 'JPEGImages', '000001.jpg'
    )

    if not os.path.exists(test_image):
        print(f"Test image not found: {test_image}")
        # Try to find any jpg file
        import glob
        jpgs = glob.glob(os.path.join(os.path.dirname(__file__), 'datasets', '**', '*.jpg'), recursive=True)
        if jpgs:
            test_image = jpgs[0]
            print(f"Using: {test_image}")
        else:
            print("No test images found!")
            return

    print("=" * 60)
    print("Testing YOLO Models and WBF")
    print("=" * 60)

    # Load models
    yolo11_path = os.path.join(base_dir, 'yolo11x.pt')
    yolo8_path = os.path.join(base_dir, 'yolov8x.pt')

    print(f"\n1. Loading yolo11x.pt from: {yolo11_path}")
    detector1 = YOLODetector(yolo11_path, conf_thres=0.25)

    print(f"\n2. Loading yolov8x.pt from: {yolo8_path}")
    detector2 = YOLODetector(yolo8_path, conf_thres=0.25)

    # Test image size
    img = Image.open(test_image)
    image_size = img.size
    print(f"\n3. Test image: {test_image}")
    print(f"   Image size: {image_size}")

    # Run detection with filter_voc=True
    print("\n4. Running detection with filter_voc=True:")

    print("\n   --- yolo11x.pt results ---")
    boxes1, labels1, scores1, names1 = detector1.detect(test_image, filter_voc=True)
    print(f"   Detections: {len(boxes1)}")
    if len(boxes1) > 0:
        print(f"   Labels (VOC indices): {labels1[:10]}...")
        print(f"   Names: {names1[:10]}...")
        print(f"   Scores: {scores1[:10]}...")

    print("\n   --- yolov8x.pt results ---")
    boxes2, labels2, scores2, names2 = detector2.detect(test_image, filter_voc=True)
    print(f"   Detections: {len(boxes2)}")
    if len(boxes2) > 0:
        print(f"   Labels (VOC indices): {labels2[:10]}...")
        print(f"   Names: {names2[:10]}...")
        print(f"   Scores: {scores2[:10]}...")

    # Check label consistency
    print("\n5. Checking label consistency:")
    print(f"   VOC_CLASSES: {VOC_CLASSES}")
    print(f"   Example: VOC_CLASSES[15] = '{VOC_CLASSES[15] if 15 < len(VOC_CLASSES) else 'N/A'}'")

    # Test WBF
    print("\n6. Testing WBF fusion:")

    all_boxes = [
        boxes1 if len(boxes1) > 0 else np.array([]).reshape(0, 4),
        boxes2 if len(boxes2) > 0 else np.array([]).reshape(0, 4)
    ]
    all_scores = [
        scores1 if len(scores1) > 0 else np.array([]),
        scores2 if len(scores2) > 0 else np.array([])
    ]
    all_labels = [
        labels1 if len(labels1) > 0 else np.array([]),
        labels2 if len(labels2) > 0 else np.array([])
    ]

    print(f"   Input boxes shapes: {[b.shape for b in all_boxes]}")
    print(f"   Input labels: {[l[:5].tolist() if len(l) > 0 else [] for l in all_labels]}")

    fused_boxes, fused_scores, fused_labels = apply_wbf(
        all_boxes, all_scores, all_labels,
        image_size=image_size,
        iou_thr=0.55,
        skip_box_thr=0.01,
        weights=[1.0, 1.0]
    )

    print(f"\n   WBF Results:")
    print(f"   Fused detections: {len(fused_boxes)}")
    if len(fused_labels) > 0:
        print(f"   Fused labels: {fused_labels[:10]}...")
        print(f"   Fused scores: {fused_scores[:10]}...")

        # Generate names
        fused_names = []
        for label in fused_labels:
            label = int(label)
            if label < len(VOC_CLASSES):
                fused_names.append(VOC_CLASSES[label])
            else:
                fused_names.append(f"class_{label}")
        print(f"   Fused names: {fused_names[:10]}...")

        # Compare boxes
        print("\n   Box comparison:")
        for i in range(min(3, len(fused_boxes))):
            print(f"\n   Detection {i+1}: {fused_names[i]}")
            print(f"     yolo11x box: {boxes1[i] if i < len(boxes1) else 'N/A'}")
            print(f"     yolo11x score: {scores1[i] if i < len(scores1) else 'N/A'}")
            print(f"     yolov8x box: {boxes2[i] if i < len(boxes2) else 'N/A'}")
            print(f"     yolov8x score: {scores2[i] if i < len(scores2) else 'N/A'}")
            print(f"     WBF fused box: {fused_boxes[i]}")
            print(f"     WBF fused score: {fused_scores[i]}")

    print("\n7. Checking name generation logic:")
    print(f"   VOC_CLASSES[1:] type: {type(VOC_CLASSES[1:])}")
    print(f"   15 in VOC_CLASSES[1:]: {15 in VOC_CLASSES[1:]}")  # This should be False!
    print(f"   'person' in VOC_CLASSES[1:]: {'person' in VOC_CLASSES[1:]}")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)

if __name__ == '__main__':
    test_models()
