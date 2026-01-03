"""
COCO to VOC class mapping module.

Maps COCO 80 classes to VOC 20 classes, filtering out non-VOC classes.
"""

# COCO 80 classes (0-indexed)
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

# VOC 20 classes (with background as index 0)
VOC_CLASSES = [
    '__background__',
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# COCO class name -> VOC class name mapping
# Some COCO classes have different names than VOC
COCO_TO_VOC_NAME = {
    'person': 'person',
    'bicycle': 'bicycle',
    'car': 'car',
    'motorcycle': 'motorbike',  # COCO: motorcycle -> VOC: motorbike
    'airplane': 'aeroplane',    # COCO: airplane -> VOC: aeroplane
    'bus': 'bus',
    'train': 'train',
    'boat': 'boat',
    'bird': 'bird',
    'cat': 'cat',
    'dog': 'dog',
    'horse': 'horse',
    'sheep': 'sheep',
    'cow': 'cow',
    'bottle': 'bottle',
    'chair': 'chair',
    'couch': 'sofa',           # COCO: couch -> VOC: sofa
    'potted plant': 'pottedplant',  # COCO: potted plant -> VOC: pottedplant
    'dining table': 'diningtable',  # COCO: dining table -> VOC: diningtable
    'tv': 'tvmonitor',         # COCO: tv -> VOC: tvmonitor
}

# Build COCO index -> VOC index mapping
# COCO uses 0-indexed, VOC uses 1-indexed (0 is background)
COCO_IDX_TO_VOC_IDX = {}
COCO_IDX_TO_VOC_NAME = {}

for coco_idx, coco_name in enumerate(COCO_CLASSES):
    if coco_name in COCO_TO_VOC_NAME:
        voc_name = COCO_TO_VOC_NAME[coco_name]
        voc_idx = VOC_CLASSES.index(voc_name)
        COCO_IDX_TO_VOC_IDX[coco_idx] = voc_idx
        COCO_IDX_TO_VOC_NAME[coco_idx] = voc_name

# Set of valid COCO indices (those that map to VOC)
VALID_COCO_INDICES = set(COCO_IDX_TO_VOC_IDX.keys())


def filter_coco_to_voc(boxes, labels, scores):
    """
    Filter detection results to only include VOC classes.

    Args:
        boxes: Detection boxes, shape (N, 4)
        labels: COCO class indices, shape (N,)
        scores: Confidence scores, shape (N,)

    Returns:
        filtered_boxes: Filtered boxes
        filtered_labels: VOC class indices (1-indexed)
        filtered_scores: Filtered scores
        filtered_names: VOC class names
    """
    import numpy as np

    if len(labels) == 0:
        return np.array([]), np.array([]), np.array([]), []

    # Create mask for VOC classes
    mask = np.array([label in VALID_COCO_INDICES for label in labels])

    if not mask.any():
        return np.array([]), np.array([]), np.array([]), []

    filtered_boxes = boxes[mask]
    filtered_scores = scores[mask]
    filtered_labels_coco = labels[mask]

    # Convert COCO indices to VOC indices
    filtered_labels_voc = np.array([COCO_IDX_TO_VOC_IDX[label] for label in filtered_labels_coco])
    filtered_names = [COCO_IDX_TO_VOC_NAME[label] for label in filtered_labels_coco]

    return filtered_boxes, filtered_labels_voc, filtered_scores, filtered_names


def get_voc_class_name(voc_idx):
    """Get VOC class name by index (1-indexed)."""
    if 0 < voc_idx < len(VOC_CLASSES):
        return VOC_CLASSES[voc_idx]
    return 'unknown'


# Colors for visualization (one per VOC class)
VOC_COLORS = [
    (0, 0, 0),        # background
    (128, 0, 0),      # aeroplane
    (0, 128, 0),      # bicycle
    (128, 128, 0),    # bird
    (0, 0, 128),      # boat
    (128, 0, 128),    # bottle
    (0, 128, 128),    # bus
    (128, 128, 128),  # car
    (64, 0, 0),       # cat
    (192, 0, 0),      # chair
    (64, 128, 0),     # cow
    (192, 128, 0),    # diningtable
    (64, 0, 128),     # dog
    (192, 0, 128),    # horse
    (64, 128, 128),   # motorbike
    (192, 128, 128),  # person
    (0, 64, 0),       # pottedplant
    (128, 64, 0),     # sheep
    (0, 192, 0),      # sofa
    (128, 192, 0),    # train
    (0, 64, 128),     # tvmonitor
]


def get_voc_color(voc_idx):
    """Get color for VOC class index (1-indexed)."""
    if 0 <= voc_idx < len(VOC_COLORS):
        return VOC_COLORS[voc_idx]
    return (255, 255, 255)
