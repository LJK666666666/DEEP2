#!/usr/bin/env python
"""
YOLO11x Object Detection GUI

A PyQt5 GUI application for running YOLO11x object detection
with automatic COCO to VOC class mapping.
Supports WBF (Weighted Box Fusion) ensemble with multiple models.

Usage:
    python run_yolo_gui.py [--weights path/to/yolo11x.pt]

Requirements:
    - PyQt5
    - ultralytics
    - torch
    - ensemble_boxes (optional, for WBF)
"""

import sys
import os
import argparse
import numpy as np
from PIL import Image
import cv2  # For camera capture

# Try to import ensemble_boxes for WBF
try:
    from ensemble_boxes import weighted_boxes_fusion
    HAS_ENSEMBLE_BOXES = True
except ImportError:
    HAS_ENSEMBLE_BOXES = False
    print("Warning: ensemble_boxes not installed. Using built-in WBF implementation.")
    print("For better performance, install: pip install ensemble-boxes")

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QSlider, QGroupBox, QTextEdit,
    QStatusBar, QMessageBox, QSplitter, QComboBox, QProgressBar,
    QCheckBox
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer


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
    '__background__',
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# COCO to VOC name mapping
COCO_TO_VOC_NAME = {
    'person': 'person', 'bicycle': 'bicycle', 'car': 'car',
    'motorcycle': 'motorbike', 'airplane': 'aeroplane', 'bus': 'bus',
    'train': 'train', 'boat': 'boat', 'bird': 'bird', 'cat': 'cat',
    'dog': 'dog', 'horse': 'horse', 'sheep': 'sheep', 'cow': 'cow',
    'bottle': 'bottle', 'chair': 'chair', 'couch': 'sofa',
    'potted plant': 'pottedplant', 'dining table': 'diningtable', 'tv': 'tvmonitor'
}

# Build COCO index to VOC mapping
COCO_IDX_TO_VOC_IDX = {}
COCO_IDX_TO_VOC_NAME = {}
for coco_idx, coco_name in enumerate(COCO_CLASSES):
    if coco_name in COCO_TO_VOC_NAME:
        voc_name = COCO_TO_VOC_NAME[coco_name]
        voc_idx = VOC_CLASSES.index(voc_name)
        COCO_IDX_TO_VOC_IDX[coco_idx] = voc_idx
        COCO_IDX_TO_VOC_NAME[coco_idx] = voc_name

VALID_COCO_INDICES = set(COCO_IDX_TO_VOC_IDX.keys())


def builtin_wbf(boxes_list, scores_list, labels_list, iou_thr=0.55, skip_box_thr=0.0, weights=None):
    """
    Built-in implementation of Weighted Boxes Fusion (WBF).

    Args:
        boxes_list: List of arrays with boxes for each model (normalized 0-1)
        scores_list: List of arrays with scores for each model
        labels_list: List of arrays with labels for each model
        iou_thr: IoU threshold for box fusion
        skip_box_thr: Skip boxes with score below this threshold
        weights: Weights for each model (default: equal weights)

    Returns:
        fused_boxes, fused_scores, fused_labels
    """
    if weights is None:
        weights = [1.0] * len(boxes_list)
    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    # Collect all boxes with their metadata
    all_boxes = []
    for model_idx, (boxes, scores, labels) in enumerate(zip(boxes_list, scores_list, labels_list)):
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            if score < skip_box_thr:
                continue
            all_boxes.append({
                'box': box,
                'score': score,  # Keep original score
                'weight': weights[model_idx],  # Store weight separately
                'label': int(label),
                'model_idx': model_idx
            })

    if len(all_boxes) == 0:
        return np.array([]), np.array([]), np.array([])

    # Group boxes by label
    label_groups = {}
    for det in all_boxes:
        label = det['label']
        if label not in label_groups:
            label_groups[label] = []
        label_groups[label].append(det)

    # Process each label group
    fused_boxes = []
    fused_scores = []
    fused_labels = []

    def iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0

    for label, detections in label_groups.items():
        # Sort by score descending
        detections = sorted(detections, key=lambda x: -x['score'])

        clusters = []
        for det in detections:
            matched = False
            for cluster in clusters:
                # Check IoU with cluster representative (first box)
                if iou(det['box'], cluster[0]['box']) > iou_thr:
                    cluster.append(det)
                    matched = True
                    break
            if not matched:
                clusters.append([det])

        # Fuse each cluster
        for cluster in clusters:
            # Weighted average of boxes and scores
            total_weight = sum(d['weight'] for d in cluster)
            if total_weight <= 0:
                continue

            # Weighted average of boxes
            fused_box = np.zeros(4)
            weighted_score_sum = 0.0
            for det in cluster:
                w = det['weight'] / total_weight
                fused_box += np.array(det['box']) * w
                weighted_score_sum += det['score'] * det['weight']

            # Final score: weighted average of scores
            fused_score = weighted_score_sum / total_weight

            fused_boxes.append(fused_box)
            fused_scores.append(fused_score)
            fused_labels.append(label)

    if len(fused_boxes) == 0:
        return np.array([]), np.array([]), np.array([])

    return np.array(fused_boxes), np.array(fused_scores), np.array(fused_labels)


def apply_wbf(boxes_list, scores_list, labels_list, image_size, iou_thr=0.55, skip_box_thr=0.01, weights=None):
    """
    Apply Weighted Boxes Fusion to multiple detection results.

    Args:
        boxes_list: List of arrays with boxes for each model (in pixel coordinates)
        scores_list: List of arrays with scores for each model
        labels_list: List of arrays with labels for each model
        image_size: (width, height) of the image
        iou_thr: IoU threshold for box fusion
        skip_box_thr: Skip boxes with score below this threshold
        weights: Weights for each model (default: equal weights)

    Returns:
        fused_boxes (pixel coords), fused_scores, fused_labels
    """
    if weights is None:
        weights = [1.0] * len(boxes_list)

    width, height = image_size

    # Normalize boxes to [0, 1]
    normalized_boxes_list = []
    for boxes in boxes_list:
        if len(boxes) == 0:
            normalized_boxes_list.append(np.array([]))
            continue
        normalized = boxes.copy().astype(np.float32)
        normalized[:, [0, 2]] /= width
        normalized[:, [1, 3]] /= height
        # Clip to [0, 1]
        normalized = np.clip(normalized, 0, 1)
        normalized_boxes_list.append(normalized)

    # Apply WBF
    if HAS_ENSEMBLE_BOXES:
        # Convert to list format required by ensemble_boxes
        boxes_for_wbf = [b.tolist() if len(b) > 0 else [] for b in normalized_boxes_list]
        scores_for_wbf = [s.tolist() if len(s) > 0 else [] for s in scores_list]
        labels_for_wbf = [l.tolist() if len(l) > 0 else [] for l in labels_list]

        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_for_wbf, scores_for_wbf, labels_for_wbf,
            weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr
        )
    else:
        fused_boxes, fused_scores, fused_labels = builtin_wbf(
            normalized_boxes_list, scores_list, labels_list,
            iou_thr=iou_thr, skip_box_thr=skip_box_thr, weights=weights
        )

    # Convert back to pixel coordinates
    if len(fused_boxes) > 0:
        fused_boxes = np.array(fused_boxes)
        fused_boxes[:, [0, 2]] *= width
        fused_boxes[:, [1, 3]] *= height

    return fused_boxes, np.array(fused_scores), np.array(fused_labels).astype(int)


# VOC class colors
VOC_COLORS = [
    (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128),
    (128, 0, 128), (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0),
    (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128),
    (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
    (0, 64, 128)
]


def filter_to_voc(boxes, labels, scores, class_names):
    """Filter detections to VOC classes only."""
    if len(labels) == 0:
        return np.array([]), np.array([]), np.array([]), []

    mask = np.array([label in VALID_COCO_INDICES for label in labels])
    if not mask.any():
        return np.array([]), np.array([]), np.array([]), []

    filtered_boxes = boxes[mask]
    filtered_scores = scores[mask]
    filtered_labels_coco = labels[mask]
    filtered_labels_voc = np.array([COCO_IDX_TO_VOC_IDX[int(l)] for l in filtered_labels_coco])
    filtered_names = [COCO_IDX_TO_VOC_NAME[int(l)] for l in filtered_labels_coco]

    return filtered_boxes, filtered_labels_voc, filtered_scores, filtered_names


class YOLODetector:
    """YOLO11x detector wrapper."""

    def __init__(self, weights_path, device='cuda:0', conf_thres=0.25):
        self.device = device
        self.conf_thres = conf_thres

        try:
            from ultralytics import YOLO
            print(f"Loading YOLO model from {weights_path}...")
            self.model = YOLO(weights_path)
            self.model.to(device)
            print("YOLO model loaded successfully!")
        except ImportError:
            raise ImportError(
                "ultralytics is not installed. Please install it:\n"
                "  pip install ultralytics"
            )

    def detect(self, image_path, filter_voc=True):
        """
        Run detection on an image.

        Args:
            image_path: Path to image file
            filter_voc: If True, filter to VOC classes only

        Returns:
            boxes, labels, scores, names
        """
        results = self.model(image_path, conf=self.conf_thres, verbose=False)

        if len(results) == 0:
            return np.array([]), np.array([]), np.array([]), []

        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        labels = result.boxes.cls.cpu().numpy().astype(int)
        names = [COCO_CLASSES[int(l)] for l in labels]

        if filter_voc:
            boxes, labels, scores, names = filter_to_voc(boxes, labels, scores, names)

        return boxes, labels, scores, names


class DetectionThread(QThread):
    """Background thread for detection."""
    finished = pyqtSignal(object, object, object, object)
    error = pyqtSignal(str)

    def __init__(self, detector, image_path, filter_voc):
        super().__init__()
        self.detector = detector
        self.image_path = image_path
        self.filter_voc = filter_voc

    def run(self):
        try:
            boxes, labels, scores, names = self.detector.detect(
                self.image_path, filter_voc=self.filter_voc
            )
            self.finished.emit(boxes, labels, scores, names)
        except Exception as e:
            self.error.emit(str(e))


class WBFDetectionThread(QThread):
    """Background thread for WBF ensemble detection."""
    finished = pyqtSignal(object, object, object, object, str)  # boxes, labels, scores, names, info
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, detectors, image_path, filter_voc, iou_thr=0.55, weights=None):
        super().__init__()
        self.detectors = detectors  # List of (name, detector) tuples
        self.image_path = image_path
        self.filter_voc = filter_voc
        self.iou_thr = iou_thr
        self.weights = weights

    def run(self):
        try:
            # Get image size
            from PIL import Image
            img = Image.open(self.image_path)
            image_size = img.size  # (width, height)

            # Run detection on all models
            all_boxes = []
            all_scores = []
            all_labels = []
            model_results = []

            for name, detector in self.detectors:
                self.progress.emit(f"Running {name}...")
                boxes, labels, scores, names = detector.detect(
                    self.image_path, filter_voc=self.filter_voc
                )
                all_boxes.append(boxes if len(boxes) > 0 else np.array([]).reshape(0, 4))
                all_scores.append(scores if len(scores) > 0 else np.array([]))
                all_labels.append(labels if len(labels) > 0 else np.array([]))
                model_results.append((name, len(boxes)))

            self.progress.emit("Applying WBF fusion...")

            # Apply WBF
            fused_boxes, fused_scores, fused_labels = apply_wbf(
                all_boxes, all_scores, all_labels,
                image_size=image_size,
                iou_thr=self.iou_thr,
                skip_box_thr=0.01,
                weights=self.weights
            )

            # Generate names from fused labels (labels are VOC indices)
            fused_names = []
            for label in fused_labels:
                label = int(label)
                if 0 < label < len(VOC_CLASSES):
                    fused_names.append(VOC_CLASSES[label])
                else:
                    fused_names.append(f"class_{label}")

            # Build info string
            info_parts = [f"{name}: {count}" for name, count in model_results]
            info_str = " | ".join(info_parts) + f" → WBF: {len(fused_boxes)}"

            self.finished.emit(fused_boxes, fused_labels, fused_scores, fused_names, info_str)

        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")


class CameraThread(QThread):
    """Background thread for camera capture and detection."""
    frame_ready = pyqtSignal(np.ndarray, object, object, object, object)  # frame, boxes, labels, scores, names
    error = pyqtSignal(str)

    def __init__(self, detector, camera_id=0, filter_voc=True):
        super().__init__()
        self.detector = detector
        self.camera_id = camera_id
        self.filter_voc = filter_voc
        self.running = False

    def run(self):
        self.running = True
        cap = cv2.VideoCapture(self.camera_id)

        if not cap.isOpened():
            self.error.emit(f"Cannot open camera {self.camera_id}")
            return

        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            try:
                # Convert BGR to RGB for detection
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Run detection
                results = self.detector.model(frame_rgb, conf=self.detector.conf_thres, verbose=False)

                if len(results) > 0:
                    result = results[0]
                    boxes = result.boxes.xyxy.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy()
                    labels = result.boxes.cls.cpu().numpy().astype(int)
                    names = [COCO_CLASSES[int(l)] for l in labels]

                    if self.filter_voc:
                        boxes, labels, scores, names = filter_to_voc(boxes, labels, scores, names)
                else:
                    boxes, labels, scores, names = np.array([]), np.array([]), np.array([]), []

                self.frame_ready.emit(frame_rgb, boxes, labels, scores, names)

            except Exception as e:
                # Continue even if detection fails on a frame
                self.frame_ready.emit(frame_rgb, np.array([]), np.array([]), np.array([]), [])

        cap.release()

    def stop(self):
        self.running = False
        self.wait()


class ImageLabel(QLabel):
    """Custom label for displaying images with detection boxes."""

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(640, 480)
        self.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        self.setText("Load an image to start detection")

        self.original_image = None
        self.boxes = None
        self.labels = None
        self.scores = None
        self.names = None
        self.score_threshold = 0.25

    def set_image(self, image_path):
        self.original_image = QPixmap(image_path)
        self.boxes = None
        self._update_display()

    def set_detections(self, boxes, labels, scores, names):
        self.boxes = boxes
        self.labels = labels
        self.scores = scores
        self.names = names
        self._update_display()

    def set_score_threshold(self, threshold):
        self.score_threshold = threshold
        self._update_display()

    def set_frame(self, frame, boxes, labels, scores, names):
        """Set a camera frame (numpy array RGB) with detections."""
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        qimage = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.original_image = QPixmap.fromImage(qimage)
        self.boxes = boxes
        self.labels = labels
        self.scores = scores
        self.names = names
        self._update_display()

    def _update_display(self):
        if self.original_image is None:
            return

        display_image = self.original_image.copy()

        if self.boxes is not None and len(self.boxes) > 0:
            painter = QPainter(display_image)
            painter.setRenderHint(QPainter.Antialiasing)
            font = QFont("Arial", 12, QFont.Bold)
            painter.setFont(font)

            for i, (box, label, score) in enumerate(zip(self.boxes, self.labels, self.scores)):
                if score < self.score_threshold:
                    continue

                x1, y1, x2, y2 = map(int, box)
                color_idx = int(label) if int(label) < len(VOC_COLORS) else 0
                color = VOC_COLORS[color_idx]
                qcolor = QColor(*color)

                class_name = self.names[i] if self.names and i < len(self.names) else 'unknown'

                pen = QPen(qcolor, 3)
                painter.setPen(pen)
                painter.drawRect(x1, y1, x2 - x1, y2 - y1)

                label_text = f"{class_name}: {score:.2f}"
                metrics = painter.fontMetrics()
                text_width = metrics.horizontalAdvance(label_text) + 6
                text_height = metrics.height() + 4

                painter.fillRect(x1, y1 - text_height, text_width, text_height, qcolor)
                painter.setPen(QPen(Qt.white))
                painter.drawText(x1 + 3, y1 - 4, label_text)

            painter.end()

        scaled = display_image.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()


class YOLOGui(QMainWindow):
    """Main GUI window for YOLO detection."""

    def __init__(self, weights_path, wbf_weights_path=None):
        super().__init__()
        self.setWindowTitle("Object Detector (with WBF Fusion)")
        self.setGeometry(100, 100, 1200, 800)

        self.weights_path = weights_path
        self.wbf_weights_path = wbf_weights_path
        self.detector = None
        self.wbf_detector = None  # Second detector for WBF
        self.current_image_path = None
        self.camera_thread = None  # Camera thread for real-time detection

        self._init_ui()
        self._init_detector()

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left panel
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        self.image_label = ImageLabel()
        left_layout.addWidget(self.image_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)

        # Right panel
        right_panel = QWidget()
        right_panel.setMaximumWidth(350)
        right_layout = QVBoxLayout(right_panel)

        # File controls
        file_group = QGroupBox("Image")
        file_layout = QVBoxLayout(file_group)

        self.load_btn = QPushButton("Load Image")
        self.load_btn.clicked.connect(self._load_image)
        file_layout.addWidget(self.load_btn)

        self.load_voc_btn = QPushButton("Load from VOC Dataset")
        self.load_voc_btn.clicked.connect(self._load_from_voc)
        file_layout.addWidget(self.load_voc_btn)

        self.image_path_label = QLabel("No image loaded")
        self.image_path_label.setWordWrap(True)
        file_layout.addWidget(self.image_path_label)

        right_layout.addWidget(file_group)

        # Detection controls
        detect_group = QGroupBox("Detection")
        detect_layout = QVBoxLayout(detect_group)

        self.detect_btn = QPushButton("Run Detection")
        self.detect_btn.clicked.connect(self._run_detection)
        self.detect_btn.setEnabled(False)
        detect_layout.addWidget(self.detect_btn)

        # WBF Fusion button
        self.wbf_btn = QPushButton("WBF Fusion")
        self.wbf_btn.setToolTip("Run detection with both models and fuse results using WBF")
        self.wbf_btn.clicked.connect(self._run_wbf_detection)
        self.wbf_btn.setEnabled(False)
        self.wbf_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        detect_layout.addWidget(self.wbf_btn)

        # Camera button
        self.camera_btn = QPushButton("Start Camera")
        self.camera_btn.setToolTip("Start real-time camera detection")
        self.camera_btn.clicked.connect(self._toggle_camera)
        self.camera_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; }")
        detect_layout.addWidget(self.camera_btn)

        self.filter_voc_checkbox = QCheckBox("Filter to VOC 20 classes")
        self.filter_voc_checkbox.setChecked(True)
        detect_layout.addWidget(self.filter_voc_checkbox)

        # Threshold slider
        threshold_layout = QVBoxLayout()
        threshold_layout.addWidget(QLabel("Confidence Threshold:"))

        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(1)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(25)
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)
        threshold_layout.addWidget(self.threshold_slider)

        self.threshold_value_label = QLabel("0.25")
        threshold_layout.addWidget(self.threshold_value_label)

        detect_layout.addLayout(threshold_layout)
        right_layout.addWidget(detect_group)

        # Results
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(200)
        results_layout.addWidget(self.results_text)

        right_layout.addWidget(results_group)

        # VOC Classes
        classes_group = QGroupBox("VOC Classes")
        classes_layout = QVBoxLayout(classes_group)

        classes_text = QTextEdit()
        classes_text.setReadOnly(True)
        classes_text.setMaximumHeight(120)
        classes_text.setText("\n".join([f"{i}: {n}" for i, n in enumerate(VOC_CLASSES) if i > 0]))
        classes_layout.addWidget(classes_text)

        right_layout.addWidget(classes_group)
        right_layout.addStretch()

        # Layout
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def _init_detector(self):
        self.status_bar.showMessage("Loading YOLO models...")
        try:
            # Load primary model
            self.detector = YOLODetector(
                self.weights_path,
                device='cuda:0',
                conf_thres=self.threshold_slider.value() / 100.0
            )

            # Load WBF model if path is provided
            if self.wbf_weights_path and os.path.exists(self.wbf_weights_path):
                self.status_bar.showMessage("Loading YOLOv8x for WBF...")
                self.wbf_detector = YOLODetector(
                    self.wbf_weights_path,
                    device='cuda:0',
                    conf_thres=self.threshold_slider.value() / 100.0
                )
                self.status_bar.showMessage("Both YOLO models loaded successfully")
            else:
                self.wbf_detector = None
                if self.wbf_weights_path:
                    self.status_bar.showMessage(f"Primary model loaded. WBF model not found: {self.wbf_weights_path}")
                else:
                    self.status_bar.showMessage("YOLO model loaded successfully")

        except Exception as e:
            self.status_bar.showMessage(f"Failed to load model: {str(e)[:50]}...")
            QMessageBox.warning(self, "Error", f"Failed to load YOLO model:\n\n{str(e)}")

    def _load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "",
            "Images (*.jpg *.jpeg *.png *.bmp);;All Files (*)"
        )
        if file_path:
            self._set_image(file_path)

    def _load_from_voc(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        voc_dir = os.path.join(base_dir, 'datasets', 'VOC2007', 'JPEGImages')
        if not os.path.exists(voc_dir):
            voc_dir = ""

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select VOC Image", voc_dir,
            "Images (*.jpg *.jpeg *.png *.bmp);;All Files (*)"
        )
        if file_path:
            self._set_image(file_path)

    def _set_image(self, file_path):
        self.current_image_path = file_path
        self.image_label.set_image(file_path)
        self.image_path_label.setText(os.path.basename(file_path))
        self.detect_btn.setEnabled(True)
        # Enable WBF button only if WBF detector is loaded
        self.wbf_btn.setEnabled(self.wbf_detector is not None)
        self.results_text.clear()
        self.status_bar.showMessage(f"Loaded: {file_path}")

    def _run_detection(self):
        if self.current_image_path is None or self.detector is None:
            return

        self.detector.conf_thres = self.threshold_slider.value() / 100.0
        self.detect_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)

        self.detection_thread = DetectionThread(
            self.detector,
            self.current_image_path,
            self.filter_voc_checkbox.isChecked()
        )
        self.detection_thread.finished.connect(self._on_detection_finished)
        self.detection_thread.error.connect(self._on_detection_error)
        self.detection_thread.start()

    def _on_detection_finished(self, boxes, labels, scores, names):
        self.detect_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        self.image_label.set_detections(boxes, labels, scores, names)

        if len(boxes) > 0:
            results_text = f"Detected {len(boxes)} objects:\n\n"
            for i, (box, label, score, name) in enumerate(zip(boxes, labels, scores, names)):
                results_text += f"{i+1}. {name}: {score:.3f}\n"
            self.results_text.setText(results_text)
            self.status_bar.showMessage(f"Detection complete: {len(boxes)} objects")
        else:
            self.results_text.setText("No objects detected")
            self.status_bar.showMessage("No objects detected")

    def _on_detection_error(self, error_msg):
        self.detect_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage(f"Error: {error_msg[:50]}...")
        QMessageBox.warning(self, "Error", f"Detection failed:\n\n{error_msg}")

    def _on_threshold_changed(self, value):
        threshold = value / 100.0
        self.threshold_value_label.setText(f"{threshold:.2f}")
        self.image_label.set_score_threshold(threshold)

    def _run_wbf_detection(self):
        """Run WBF fusion detection with both models."""
        if self.current_image_path is None or self.detector is None or self.wbf_detector is None:
            QMessageBox.warning(self, "Error", "Both models must be loaded to use WBF fusion.")
            return

        # Update confidence threshold for both detectors
        conf_thres = self.threshold_slider.value() / 100.0
        self.detector.conf_thres = conf_thres
        self.wbf_detector.conf_thres = conf_thres

        self.detect_btn.setEnabled(False)
        self.wbf_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)

        # Create detector list with names
        detectors = [
            (os.path.basename(self.weights_path), self.detector),
            (os.path.basename(self.wbf_weights_path), self.wbf_detector)
        ]

        self.wbf_thread = WBFDetectionThread(
            detectors,
            self.current_image_path,
            self.filter_voc_checkbox.isChecked(),
            iou_thr=0.55,
            weights=[1.0, 1.0]  # Equal weights for both models
        )
        self.wbf_thread.finished.connect(self._on_wbf_detection_finished)
        self.wbf_thread.error.connect(self._on_wbf_detection_error)
        self.wbf_thread.progress.connect(self._on_wbf_progress)
        self.wbf_thread.start()

    def _on_wbf_progress(self, message):
        """Handle WBF progress updates."""
        self.status_bar.showMessage(message)

    def _on_wbf_detection_finished(self, boxes, labels, scores, names, info_str):
        """Handle WBF detection completion."""
        self.detect_btn.setEnabled(True)
        self.wbf_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        self.image_label.set_detections(boxes, labels, scores, names)

        if len(boxes) > 0:
            results_text = f"WBF Fusion Results:\n{info_str}\n\n"
            results_text += f"Detected {len(boxes)} objects:\n\n"
            for i, (box, label, score, name) in enumerate(zip(boxes, labels, scores, names)):
                results_text += f"{i+1}. {name}: {score:.3f}\n"
            self.results_text.setText(results_text)
            self.status_bar.showMessage(f"WBF complete: {info_str}")
        else:
            self.results_text.setText(f"WBF Fusion: No objects detected\n{info_str}")
            self.status_bar.showMessage("WBF: No objects detected")

    def _on_wbf_detection_error(self, error_msg):
        """Handle WBF detection error."""
        self.detect_btn.setEnabled(True)
        self.wbf_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage(f"WBF Error: {error_msg[:50]}...")
        QMessageBox.warning(self, "WBF Error", f"WBF detection failed:\n\n{error_msg}")

    def _toggle_camera(self):
        """Toggle camera on/off."""
        if self.camera_thread is not None and self.camera_thread.running:
            self._stop_camera()
        else:
            self._start_camera()

    def _start_camera(self):
        """Start camera detection."""
        if self.detector is None:
            QMessageBox.warning(self, "Error", "Model not loaded yet.")
            return

        self.detector.conf_thres = self.threshold_slider.value() / 100.0

        self.camera_thread = CameraThread(
            self.detector,
            camera_id=0,
            filter_voc=self.filter_voc_checkbox.isChecked()
        )
        self.camera_thread.frame_ready.connect(self._on_camera_frame)
        self.camera_thread.error.connect(self._on_camera_error)
        self.camera_thread.start()

        self.camera_btn.setText("Stop Camera")
        self.camera_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; }")
        self.detect_btn.setEnabled(False)
        self.wbf_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        self.load_voc_btn.setEnabled(False)
        self.status_bar.showMessage("Camera started - real-time detection active")

    def _stop_camera(self):
        """Stop camera detection."""
        if self.camera_thread is not None:
            self.camera_thread.stop()
            self.camera_thread = None

        self.camera_btn.setText("Start Camera")
        self.camera_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; }")
        self.detect_btn.setEnabled(self.current_image_path is not None)
        self.wbf_btn.setEnabled(self.wbf_detector is not None and self.current_image_path is not None)
        self.load_btn.setEnabled(True)
        self.load_voc_btn.setEnabled(True)
        self.status_bar.showMessage("Camera stopped")
        self.results_text.clear()

    def _on_camera_frame(self, frame, boxes, labels, scores, names):
        """Handle camera frame with detections."""
        self.image_label.set_frame(frame, boxes, labels, scores, names)

        # Update results text with detection count
        if len(boxes) > 0:
            self.results_text.setText(f"Real-time Detection:\n\nDetected {len(boxes)} objects")
        else:
            self.results_text.setText("Real-time Detection:\n\nNo objects detected")

    def _on_camera_error(self, error_msg):
        """Handle camera error."""
        self._stop_camera()
        QMessageBox.warning(self, "Camera Error", f"Camera error:\n\n{error_msg}")

    def closeEvent(self, event):
        """Handle window close - stop camera if running."""
        if self.camera_thread is not None and self.camera_thread.running:
            self.camera_thread.stop()
        event.accept()


def main():
    parser = argparse.ArgumentParser(description='Object Detection GUI with WBF Fusion')
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to primary YOLO weights file')
    parser.add_argument('--wbf-weights', type=str, default=None,
                        help='Path to YOLOv8x weights for WBF fusion')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Default primary weights path
    if args.weights is None:
        args.weights = os.path.join(base_dir, 'yolo11x.pt')

    # Default WBF weights path (yolov8x.pt)
    if args.wbf_weights is None:
        args.wbf_weights = os.path.join(base_dir, 'yolov8x.pt')

    if not os.path.exists(args.weights):
        print(f"Error: Primary weights file not found: {args.weights}")
        print("Please provide the correct path using --weights")
        sys.exit(1)

    wbf_available = os.path.exists(args.wbf_weights)

    print("=" * 60)
    print("YOLO11x Object Detection GUI (with WBF Fusion)")
    print("=" * 60)
    print(f"Primary weights: {args.weights}")
    print(f"WBF weights: {args.wbf_weights} {'(found)' if wbf_available else '(NOT FOUND - WBF disabled)'}")

    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = YOLOGui(args.weights, args.wbf_weights if wbf_available else None)
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
