"""
Co-DETR Detector Wrapper

Wraps Co-DETR model for inference with VOC class filtering.
Requires mmdetection and mmcv to be installed.
"""

import os
import sys
import numpy as np
from PIL import Image

# Add Co-DETR to path
CODETR_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'Co-DETR')
if os.path.exists(CODETR_ROOT):
    sys.path.insert(0, CODETR_ROOT)

from .class_mapping import filter_coco_to_voc, VOC_CLASSES, get_voc_color


class CoDETRDetector:
    """
    Co-DETR object detector with VOC class filtering.

    Usage:
        detector = CoDETRDetector(
            config_file='path/to/config.py',
            checkpoint_file='path/to/checkpoint.pth',
            device='cuda:0'
        )
        boxes, labels, scores, names = detector.detect(image_path)
    """

    def __init__(self, config_file=None, checkpoint_file=None, device='cuda:0', score_thr=0.3):
        """
        Initialize Co-DETR detector.

        Args:
            config_file: Path to mmdetection config file
            checkpoint_file: Path to model checkpoint
            device: Device to run inference on ('cuda:0', 'cpu', etc.)
            score_thr: Score threshold for filtering detections
        """
        import torch

        # Auto-fallback to CPU if CUDA is not available
        if device.startswith('cuda') and not torch.cuda.is_available():
            print(f"Warning: CUDA not available, falling back to CPU")
            device = 'cpu'

        self.device = device
        self.score_thr = score_thr
        self.model = None

        # Default paths
        if config_file is None:
            config_file = os.path.join(
                CODETR_ROOT,
                'projects/configs/co_dino_vit/co_dino_5scale_vit_large_coco.py'
            )
        if checkpoint_file is None:
            checkpoint_file = os.path.join(
                os.path.dirname(CODETR_ROOT),
                'co-detr-vit-large-coco/pytorch_model.pth'
            )

        self.config_file = config_file
        self.checkpoint_file = checkpoint_file

        # Load model
        self._load_model()

    def _load_model(self):
        """Load Co-DETR model using mmdetection API."""
        try:
            from mmdet.apis import init_detector
            # Import projects to register custom modules
            import projects  # noqa

            print(f"Loading model from {self.checkpoint_file}...")
            print(f"Using device: {self.device}")
            self.model = init_detector(
                self.config_file,
                self.checkpoint_file,
                device=self.device
            )
            self.model.eval()
            print("Model loaded successfully!")

        except ImportError as e:
            raise ImportError(
                f"Failed to import mmdetection: {e}\n"
                "Please install mmdetection and mmcv:\n"
                "  pip install openmim\n"
                "  mim install mmengine mmcv mmdet"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def detect(self, image):
        """
        Run detection on an image.

        Args:
            image: Image path (str) or numpy array (H, W, C) in RGB format

        Returns:
            boxes: Detection boxes in (x1, y1, x2, y2) format, shape (N, 4)
            labels: VOC class indices (1-indexed), shape (N,)
            scores: Confidence scores, shape (N,)
            names: List of VOC class names
        """
        from mmdet.apis import inference_detector

        # Run inference
        result = inference_detector(self.model, image)

        # Parse result
        boxes, labels, scores = self._parse_result(result)

        # Filter by score threshold
        if len(scores) > 0:
            mask = scores >= self.score_thr
            boxes = boxes[mask]
            labels = labels[mask]
            scores = scores[mask]

        # Filter to VOC classes and map labels
        boxes, labels, scores, names = filter_coco_to_voc(boxes, labels, scores)

        return boxes, labels, scores, names

    def _parse_result(self, result):
        """Parse mmdetection result format."""
        # Result format depends on mmdet version
        # For newer versions, result is DetDataSample
        # For older versions, result is list of arrays

        if hasattr(result, 'pred_instances'):
            # New mmdet format (mmdet >= 3.0)
            pred = result.pred_instances
            boxes = pred.bboxes.cpu().numpy()
            scores = pred.scores.cpu().numpy()
            labels = pred.labels.cpu().numpy()
        elif isinstance(result, tuple):
            # Mask R-CNN style result: (bbox_results, mask_results)
            bbox_results = result[0]
            boxes, scores, labels = self._parse_bbox_results(bbox_results)
        elif isinstance(result, list):
            # Standard bbox result: list of arrays per class
            boxes, scores, labels = self._parse_bbox_results(result)
        else:
            raise ValueError(f"Unknown result format: {type(result)}")

        return boxes, labels, scores

    def _parse_bbox_results(self, bbox_results):
        """Parse bbox results (list of arrays per class)."""
        boxes_list = []
        scores_list = []
        labels_list = []

        for class_idx, class_bboxes in enumerate(bbox_results):
            if len(class_bboxes) > 0:
                # Each bbox is [x1, y1, x2, y2, score]
                boxes_list.append(class_bboxes[:, :4])
                scores_list.append(class_bboxes[:, 4])
                labels_list.append(np.full(len(class_bboxes), class_idx, dtype=np.int32))

        if len(boxes_list) > 0:
            boxes = np.vstack(boxes_list)
            scores = np.concatenate(scores_list)
            labels = np.concatenate(labels_list)
        else:
            boxes = np.array([]).reshape(0, 4)
            scores = np.array([])
            labels = np.array([], dtype=np.int32)

        return boxes, labels, scores

    def visualize(self, image, boxes, labels, scores, names=None):
        """
        Draw detection results on image.

        Args:
            image: PIL Image or numpy array (H, W, C) in RGB
            boxes: Detection boxes (N, 4)
            labels: VOC class indices (N,)
            scores: Confidence scores (N,)
            names: Optional list of class names

        Returns:
            PIL Image with drawn detections
        """
        from PIL import ImageDraw, ImageFont

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        else:
            image = image.copy()

        draw = ImageDraw.Draw(image)

        # Try to load a font
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()

        for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            x1, y1, x2, y2 = box.astype(int)

            # Get class name and color
            if names is not None and i < len(names):
                class_name = names[i]
            else:
                class_name = VOC_CLASSES[label] if 0 < label < len(VOC_CLASSES) else 'unknown'

            color = get_voc_color(label)

            # Draw box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

            # Draw label
            label_text = f"{class_name}: {score:.2f}"
            text_bbox = draw.textbbox((x1, y1), label_text, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]

            # Draw label background
            draw.rectangle([x1, y1 - text_h - 4, x1 + text_w + 4, y1], fill=color)
            draw.text((x1 + 2, y1 - text_h - 2), label_text, fill=(255, 255, 255), font=font)

        return image


class SimpleCoDETRDetector:
    """
    Simplified Co-DETR detector that loads weights directly without full mmdetection.

    This is a fallback when mmdetection is not properly installed.
    It uses a simpler inference approach.
    """

    def __init__(self, checkpoint_file=None, device='cuda:0', score_thr=0.3):
        """
        Initialize simplified detector.

        Note: This detector has limited functionality compared to CoDETRDetector.
        """
        import torch

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.score_thr = score_thr

        if checkpoint_file is None:
            checkpoint_file = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'co-detr-vit-large-coco/pytorch_model.pth'
            )

        self.checkpoint_file = checkpoint_file

        print(f"SimpleCoDETRDetector initialized.")
        print(f"Checkpoint: {checkpoint_file}")
        print(f"Device: {self.device}")
        print("\nNOTE: This simplified detector requires mmdetection for full functionality.")
        print("Please install: pip install openmim && mim install mmengine mmcv mmdet")

    def detect(self, image):
        """
        Placeholder detection method.

        Returns empty results - requires mmdetection for actual inference.
        """
        print("Warning: SimpleCoDETRDetector cannot perform inference without mmdetection.")
        return np.array([]), np.array([]), np.array([]), []


def create_detector(config_file=None, checkpoint_file=None, device='cuda:0', score_thr=0.3):
    """
    Factory function to create the appropriate detector.

    Tries to create CoDETRDetector first, falls back to SimpleCoDETRDetector
    if mmdetection is not available.
    """
    try:
        return CoDETRDetector(config_file, checkpoint_file, device, score_thr)
    except ImportError as e:
        print(f"Warning: Import error - {e}")
        print("Falling back to SimpleCoDETRDetector (limited functionality)")
        return SimpleCoDETRDetector(checkpoint_file, device, score_thr)
    except Exception as e:
        print(f"Warning: Failed to create CoDETRDetector - {e}")
        print("Falling back to SimpleCoDETRDetector (limited functionality)")
        return SimpleCoDETRDetector(checkpoint_file, device, score_thr)
