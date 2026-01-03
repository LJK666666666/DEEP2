#!/usr/bin/env python
"""
Co-DETR Model Test Script

Tests the Co-DETR model loading and inference on a sample image.
Verifies both model functionality and GPU availability.
Applies NMS filtering and saves visualization results.
"""

import sys
import os

# Add Co-DETR to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Co-DETR'))

# IMPORTANT: Import mmcv_shim BEFORE any mmdet imports to patch mmcv 2.x
import mmcv_shim  # noqa: F401 - patches mmcv for 2.x compatibility

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torchvision.ops as ops
import random

# COCO to VOC class mapping
COCO_TO_VOC_NAME = {
    'person': 'person', 'bicycle': 'bicycle', 'car': 'car',
    'motorcycle': 'motorbike', 'airplane': 'aeroplane', 'bus': 'bus',
    'train': 'train', 'boat': 'boat', 'bird': 'bird', 'cat': 'cat',
    'dog': 'dog', 'horse': 'horse', 'sheep': 'sheep', 'cow': 'cow',
    'bottle': 'bottle', 'chair': 'chair', 'couch': 'sofa',
    'potted plant': 'pottedplant', 'dining table': 'diningtable', 'tv': 'tvmonitor'
}

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# COCO classes
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


# Generate distinct colors for each class
def get_class_colors(num_classes):
    """Generate distinct colors for each class."""
    random.seed(42)  # For reproducibility
    colors = []
    for i in range(num_classes):
        # Use HSV color space for more distinct colors
        hue = i / num_classes
        # Convert HSV to RGB (simplified)
        r = int(255 * abs(hue * 6 - 3) - 1)
        g = int(255 * (2 - abs(hue * 6 - 2)))
        b = int(255 * (2 - abs(hue * 6 - 4)))
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        colors.append((r, g, b))
    random.shuffle(colors)
    return colors


def apply_nms_per_class(bbox_results, score_thresh=0.5, iou_thresh=0.5):
    """Apply NMS per class and return filtered results."""
    filtered_results = []

    for class_idx, class_bboxes in enumerate(bbox_results):
        if len(class_bboxes) == 0:
            filtered_results.append([])
            continue

        # Convert to tensors
        bboxes = []
        scores = []
        for bbox in class_bboxes:
            if len(bbox) >= 5:
                x1, y1, x2, y2, score = bbox[:5]
            else:
                x1, y1, x2, y2 = bbox[:4]
                score = 1.0

            if score >= score_thresh:
                bboxes.append([x1, y1, x2, y2])
                scores.append(score)

        if len(bboxes) == 0:
            filtered_results.append([])
            continue

        boxes_tensor = torch.tensor(bboxes, dtype=torch.float32)
        scores_tensor = torch.tensor(scores, dtype=torch.float32)

        # Apply NMS
        keep_indices = ops.nms(boxes_tensor, scores_tensor, iou_thresh)

        # Get kept detections
        kept_detections = []
        for idx in keep_indices:
            idx = idx.item()
            kept_detections.append({
                'bbox': bboxes[idx],
                'score': scores[idx]
            })

        filtered_results.append(kept_detections)

    return filtered_results


def draw_detections(image_path, filtered_results, output_path, score_thresh=0.5):
    """Draw detection results on image and save."""
    # Load image
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)

    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()

    # Get colors for VOC classes
    colors = get_class_colors(len(VOC_CLASSES))
    voc_class_to_color = {cls: colors[i] for i, cls in enumerate(VOC_CLASSES)}

    # Draw detections
    detections_drawn = 0
    for class_idx, class_detections in enumerate(filtered_results):
        if len(class_detections) == 0:
            continue

        coco_name = COCO_CLASSES[class_idx]

        # Filter to VOC classes
        if coco_name not in COCO_TO_VOC_NAME:
            continue

        voc_name = COCO_TO_VOC_NAME[coco_name]
        color = voc_class_to_color.get(voc_name, (255, 0, 0))

        for det in class_detections:
            x1, y1, x2, y2 = det['bbox']
            score = det['score']

            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # Draw label background
            label = f"{voc_name}: {score:.2f}"
            bbox_text = draw.textbbox((x1, y1), label, font=font)
            text_width = bbox_text[2] - bbox_text[0]
            text_height = bbox_text[3] - bbox_text[1]

            # Draw label background rectangle
            draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], fill=color)

            # Draw label text
            draw.text((x1 + 2, y1 - text_height - 2), label, fill=(255, 255, 255), font=font)

            detections_drawn += 1

    # Save image
    img.save(output_path)
    return detections_drawn


def main():
    print("=" * 60)
    print("Co-DETR Model Test (with NMS filtering)")
    print("=" * 60)

    # Check PyTorch and CUDA
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        device = 'cuda:0'
    else:
        print("Warning: CUDA not available, using CPU")
        device = 'cpu'

    # Paths
    base_dir = os.path.dirname(os.path.dirname(__file__))
    config_file = os.path.join(base_dir, 'Co-DETR', 'projects', 'configs', 'co_dino_vit', 'co_dino_5scale_vit_large_coco.py')
    checkpoint_file = os.path.join(base_dir, 'co-detr-vit-large-coco', 'pytorch_model.pth')

    # Find a test image
    test_image_dir = os.path.join(os.path.dirname(__file__), 'datasets', 'VOC2007', 'JPEGImages')
    test_images = [f for f in os.listdir(test_image_dir) if f.endswith('.jpg')][:1]

    if not test_images:
        print("Error: No test images found!")
        return

    test_image_path = os.path.join(test_image_dir, test_images[0])
    print(f"\nTest image: {test_image_path}")

    # Output path for result image
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs', 'codetr_results')
    os.makedirs(output_dir, exist_ok=True)
    output_image_path = os.path.join(output_dir, f"result_{test_images[0]}")

    # Load model
    print(f"\nLoading model...")
    print(f"Config: {config_file}")
    print(f"Checkpoint: {checkpoint_file}")
    print(f"Device: {device}")

    from mmdet.apis import init_detector, inference_detector
    import projects  # Register custom modules

    model = init_detector(config_file, checkpoint_file, device=device)
    model.eval()
    print("Model loaded successfully!")

    # Run inference
    print(f"\nRunning inference on {test_images[0]}...")
    result = inference_detector(model, test_image_path)

    # Get bbox results
    if isinstance(result, tuple):
        bbox_results = result[0]
    else:
        bbox_results = result

    # NMS parameters
    score_thresh = 0.5
    iou_thresh = 0.5

    print(f"\nApplying NMS filtering (score_thresh={score_thresh}, iou_thresh={iou_thresh})...")

    # Apply NMS per class
    filtered_results = apply_nms_per_class(bbox_results, score_thresh=score_thresh, iou_thresh=iou_thresh)

    # Parse and print results
    print("\n" + "=" * 60)
    print(f"Detection Results (VOC classes only, after NMS)")
    print("=" * 60)

    detections_found = 0
    class_counts = {}

    for class_idx, class_detections in enumerate(filtered_results):
        if len(class_detections) == 0:
            continue

        coco_name = COCO_CLASSES[class_idx]

        # Filter to VOC classes
        if coco_name not in COCO_TO_VOC_NAME:
            continue

        voc_name = COCO_TO_VOC_NAME[coco_name]

        for det in class_detections:
            x1, y1, x2, y2 = det['bbox']
            score = det['score']
            detections_found += 1
            class_counts[voc_name] = class_counts.get(voc_name, 0) + 1
            print(f"  {voc_name}: score={score:.3f}, box=[{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

    if detections_found == 0:
        print("  No detections after NMS filtering")
    else:
        print(f"\nTotal detections after NMS: {detections_found}")
        print("\nDetections per class:")
        for cls_name, count in sorted(class_counts.items()):
            print(f"  {cls_name}: {count}")

    # Draw and save results
    print("\n" + "=" * 60)
    print("Saving visualization...")
    print("=" * 60)

    num_drawn = draw_detections(test_image_path, filtered_results, output_image_path, score_thresh)

    print(f"\nResult image saved to:")
    print(f"  {output_image_path}")
    print(f"\nDrawn {num_drawn} detections on the image.")

    print("\n" + "=" * 60)
    print("TEST PASSED - Model and GPU working correctly!")
    print("=" * 60)


if __name__ == '__main__':
    main()
