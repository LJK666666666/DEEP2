"""
PyQt5 GUI for Co-DETR Object Detection with VOC class filtering.

Features:
- Load images from file
- Run Co-DETR detection
- Filter to VOC 20 classes
- Display detection results with bounding boxes
- Adjust confidence threshold
"""

import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QSlider, QGroupBox, QTextEdit,
    QStatusBar, QMessageBox, QSplitter, QComboBox, QProgressBar
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import numpy as np
from PIL import Image

from .class_mapping import VOC_CLASSES, get_voc_color


class DetectionThread(QThread):
    """Background thread for running detection."""
    finished = pyqtSignal(object, object, object, object)  # boxes, labels, scores, names
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, detector, image_path):
        super().__init__()
        self.detector = detector
        self.image_path = image_path

    def run(self):
        try:
            self.progress.emit("Running detection...")
            boxes, labels, scores, names = self.detector.detect(self.image_path)
            self.finished.emit(boxes, labels, scores, names)
        except Exception as e:
            self.error.emit(str(e))


class ImageLabel(QLabel):
    """Custom QLabel for displaying images with detection boxes."""

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
        self.score_threshold = 0.3

    def set_image(self, image_path):
        """Load and display an image."""
        self.original_image = QPixmap(image_path)
        self.boxes = None
        self.labels = None
        self.scores = None
        self.names = None
        self._update_display()

    def set_detections(self, boxes, labels, scores, names):
        """Set detection results."""
        self.boxes = boxes
        self.labels = labels
        self.scores = scores
        self.names = names
        self._update_display()

    def set_score_threshold(self, threshold):
        """Update score threshold and refresh display."""
        self.score_threshold = threshold
        self._update_display()

    def _update_display(self):
        """Update the displayed image with detection boxes."""
        if self.original_image is None:
            return

        # Create a copy to draw on
        display_image = self.original_image.copy()

        # Draw detection boxes
        if self.boxes is not None and len(self.boxes) > 0:
            painter = QPainter(display_image)
            painter.setRenderHint(QPainter.Antialiasing)

            font = QFont("Arial", 12, QFont.Bold)
            painter.setFont(font)

            for i, (box, label, score) in enumerate(zip(self.boxes, self.labels, self.scores)):
                if score < self.score_threshold:
                    continue

                x1, y1, x2, y2 = map(int, box)

                # Get color and class name
                color = get_voc_color(int(label))
                qcolor = QColor(*color)

                if self.names is not None and i < len(self.names):
                    class_name = self.names[i]
                else:
                    class_name = VOC_CLASSES[int(label)] if 0 < int(label) < len(VOC_CLASSES) else 'unknown'

                # Draw box
                pen = QPen(qcolor, 2)
                painter.setPen(pen)
                painter.drawRect(x1, y1, x2 - x1, y2 - y1)

                # Draw label background
                label_text = f"{class_name}: {score:.2f}"
                metrics = painter.fontMetrics()
                text_width = metrics.horizontalAdvance(label_text) + 6
                text_height = metrics.height() + 4

                painter.fillRect(x1, y1 - text_height, text_width, text_height, qcolor)

                # Draw label text
                painter.setPen(QPen(Qt.white))
                painter.drawText(x1 + 3, y1 - 4, label_text)

            painter.end()

        # Scale to fit label while maintaining aspect ratio
        scaled = display_image.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.setPixmap(scaled)

    def resizeEvent(self, event):
        """Handle resize events."""
        super().resizeEvent(event)
        self._update_display()


class CoDETRGui(QMainWindow):
    """Main window for Co-DETR object detection GUI."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Co-DETR Object Detector (VOC Classes)")
        self.setGeometry(100, 100, 1200, 800)

        self.detector = None
        self.current_image_path = None
        self.detection_thread = None

        self._init_ui()
        self._init_detector()

    def _init_ui(self):
        """Initialize the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        # Left panel: Image display
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        self.image_label = ImageLabel()
        left_layout.addWidget(self.image_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)

        # Right panel: Controls
        right_panel = QWidget()
        right_panel.setMaximumWidth(350)
        right_layout = QVBoxLayout(right_panel)

        # File controls
        file_group = QGroupBox("Image")
        file_layout = QVBoxLayout(file_group)

        self.load_btn = QPushButton("Load Image")
        self.load_btn.clicked.connect(self._load_image)
        file_layout.addWidget(self.load_btn)

        self.load_folder_btn = QPushButton("Load from VOC Dataset")
        self.load_folder_btn.clicked.connect(self._load_from_voc)
        file_layout.addWidget(self.load_folder_btn)

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

        # Device selection
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("Device:"))
        self.device_combo = QComboBox()

        # Check CUDA availability and set default device
        import torch
        if torch.cuda.is_available():
            self.device_combo.addItems(["cuda:0", "cpu"])
        else:
            self.device_combo.addItems(["cpu", "cuda:0"])
            print("Note: CUDA not available, using CPU by default")

        device_layout.addWidget(self.device_combo)
        detect_layout.addLayout(device_layout)

        # Score threshold slider
        threshold_layout = QVBoxLayout()
        threshold_label = QLabel("Score Threshold:")
        threshold_layout.addWidget(threshold_label)

        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(30)
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)
        threshold_layout.addWidget(self.threshold_slider)

        self.threshold_value_label = QLabel("0.30")
        threshold_layout.addWidget(self.threshold_value_label)

        detect_layout.addLayout(threshold_layout)

        right_layout.addWidget(detect_group)

        # Results display
        results_group = QGroupBox("Detection Results")
        results_layout = QVBoxLayout(results_group)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(200)
        results_layout.addWidget(self.results_text)

        right_layout.addWidget(results_group)

        # VOC Classes reference
        classes_group = QGroupBox("VOC Classes (20)")
        classes_layout = QVBoxLayout(classes_group)

        classes_text = QTextEdit()
        classes_text.setReadOnly(True)
        classes_text.setMaximumHeight(150)
        classes_text.setText("\n".join([f"{i}: {name}" for i, name in enumerate(VOC_CLASSES) if i > 0]))
        classes_layout.addWidget(classes_text)

        right_layout.addWidget(classes_group)

        right_layout.addStretch()

        # Add panels to main layout
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def _init_detector(self):
        """Initialize the Co-DETR detector."""
        self.status_bar.showMessage("Initializing detector...")

        try:
            from .detector import create_detector

            # Get paths relative to this file
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            config_file = os.path.join(
                base_dir,
                'Co-DETR/projects/configs/co_dino_vit/co_dino_5scale_vit_large_coco.py'
            )
            checkpoint_file = os.path.join(
                base_dir,
                'co-detr-vit-large-coco/pytorch_model.pth'
            )

            self.detector = create_detector(
                config_file=config_file,
                checkpoint_file=checkpoint_file,
                device=self.device_combo.currentText(),
                score_thr=self.threshold_slider.value() / 100.0
            )

            self.status_bar.showMessage("Detector initialized successfully")

        except Exception as e:
            self.status_bar.showMessage(f"Failed to initialize detector: {str(e)[:50]}...")
            QMessageBox.warning(
                self,
                "Detector Initialization Failed",
                f"Failed to initialize Co-DETR detector:\n\n{str(e)}\n\n"
                "Please ensure mmdetection is properly installed."
            )

    def _load_image(self):
        """Open file dialog to load an image."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp);;All Files (*)"
        )

        if file_path:
            self._set_image(file_path)

    def _load_from_voc(self):
        """Load image from VOC dataset directory."""
        # Try to find VOC dataset directory
        base_dir = os.path.dirname(os.path.dirname(__file__))
        voc_dir = os.path.join(base_dir, 'datasets', 'VOC2007', 'JPEGImages')

        if not os.path.exists(voc_dir):
            voc_dir = ""

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select VOC Image",
            voc_dir,
            "Images (*.jpg *.jpeg *.png *.bmp);;All Files (*)"
        )

        if file_path:
            self._set_image(file_path)

    def _set_image(self, file_path):
        """Set the current image."""
        self.current_image_path = file_path
        self.image_label.set_image(file_path)
        self.image_path_label.setText(os.path.basename(file_path))
        self.detect_btn.setEnabled(True)
        self.results_text.clear()
        self.status_bar.showMessage(f"Loaded: {file_path}")

    def _run_detection(self):
        """Run detection on the current image."""
        if self.current_image_path is None:
            return

        if self.detector is None:
            QMessageBox.warning(self, "Error", "Detector not initialized")
            return

        # Update detector threshold
        self.detector.score_thr = self.threshold_slider.value() / 100.0

        # Disable controls during detection
        self.detect_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate

        # Run detection in background thread
        self.detection_thread = DetectionThread(self.detector, self.current_image_path)
        self.detection_thread.finished.connect(self._on_detection_finished)
        self.detection_thread.error.connect(self._on_detection_error)
        self.detection_thread.progress.connect(lambda msg: self.status_bar.showMessage(msg))
        self.detection_thread.start()

    def _on_detection_finished(self, boxes, labels, scores, names):
        """Handle detection completion."""
        # Re-enable controls
        self.detect_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        # Update display
        self.image_label.set_detections(boxes, labels, scores, names)

        # Update results text
        if len(boxes) > 0:
            results_text = f"Detected {len(boxes)} objects:\n\n"
            for i, (box, label, score, name) in enumerate(zip(boxes, labels, scores, names)):
                results_text += f"{i+1}. {name}: {score:.3f}\n"
                results_text += f"   Box: [{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]\n"
            self.results_text.setText(results_text)
            self.status_bar.showMessage(f"Detection complete: {len(boxes)} objects found")
        else:
            self.results_text.setText("No objects detected")
            self.status_bar.showMessage("Detection complete: No objects found")

    def _on_detection_error(self, error_msg):
        """Handle detection error."""
        self.detect_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        self.status_bar.showMessage(f"Detection failed: {error_msg[:50]}...")
        QMessageBox.warning(self, "Detection Error", f"Detection failed:\n\n{error_msg}")

    def _on_threshold_changed(self, value):
        """Handle threshold slider change."""
        threshold = value / 100.0
        self.threshold_value_label.setText(f"{threshold:.2f}")
        self.image_label.set_score_threshold(threshold)


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = CoDETRGui()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
