# Co-DETR GUI Module
# Provides PyQt5 interface for Co-DETR object detection with VOC class mapping

from .class_mapping import (
    COCO_CLASSES,
    VOC_CLASSES,
    COCO_TO_VOC_NAME,
    COCO_IDX_TO_VOC_IDX,
    COCO_IDX_TO_VOC_NAME,
    filter_coco_to_voc,
    get_voc_class_name,
    get_voc_color
)

from .detector import (
    CoDETRDetector,
    SimpleCoDETRDetector,
    create_detector
)

__all__ = [
    'COCO_CLASSES',
    'VOC_CLASSES',
    'COCO_TO_VOC_NAME',
    'COCO_IDX_TO_VOC_IDX',
    'COCO_IDX_TO_VOC_NAME',
    'filter_coco_to_voc',
    'get_voc_class_name',
    'get_voc_color',
    'CoDETRDetector',
    'SimpleCoDETRDetector',
    'create_detector',
]
