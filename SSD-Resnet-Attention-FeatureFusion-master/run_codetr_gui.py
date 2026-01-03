#!/usr/bin/env python
"""
Co-DETR Object Detection GUI

A PyQt5 GUI application for running Co-DETR object detection
with automatic COCO to VOC class mapping.

Usage:
    python run_codetr_gui.py

Requirements:
    - PyQt5
    - torch
    - mmdetection (mmdet)
    - mmcv-full (1.5.0-1.7.0)
    - fvcore, fairscale, timm, einops

Install dependencies:
    pip install PyQt5 torch fvcore fairscale timm einops
    pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
    pip install mmdet==2.25.3
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Add Co-DETR to path BEFORE importing mmdet
CODETR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Co-DETR')
if os.path.exists(CODETR_ROOT):
    sys.path.insert(0, os.path.abspath(CODETR_ROOT))


def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []

    try:
        import PyQt5
    except ImportError:
        missing.append("PyQt5")

    try:
        import torch
    except ImportError:
        missing.append("torch")

    try:
        import numpy
    except ImportError:
        missing.append("numpy")

    try:
        from PIL import Image
    except ImportError:
        missing.append("Pillow")

    if missing:
        print("Missing required dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nPlease install them using:")
        print(f"  pip install {' '.join(missing)}")
        return False

    # Check mmcv version (must be 1.x, not 2.x)
    try:
        import mmcv
        mmcv_version = mmcv.__version__
        if mmcv_version.startswith('2.'):
            print(f"\nWarning: mmcv {mmcv_version} detected, but Co-DETR requires mmcv 1.5.0-1.7.0")
            print("Please downgrade: pip install mmcv-full==1.7.0")
            return False
        print(f"mmcv version: {mmcv_version}")
    except ImportError:
        print("\nWarning: mmcv not installed")
        print("Please install: pip install mmcv-full==1.7.0")
        return False

    # Check mmdet
    try:
        # Import from Co-DETR's mmdet (which has the projects)
        from mmdet.apis import init_detector
        print("mmdet available")
    except ImportError as e:
        print(f"\nWarning: mmdet import failed: {e}")
        print("Please install: pip install mmdet==2.25.3")
        return False

    # Check Co-DETR specific dependencies
    optional_missing = []
    for dep_name in ['fvcore', 'fairscale', 'timm', 'einops']:
        try:
            __import__(dep_name)
        except ImportError:
            optional_missing.append(dep_name)

    if optional_missing:
        print(f"\nMissing Co-DETR dependencies: {', '.join(optional_missing)}")
        print(f"Please install: pip install {' '.join(optional_missing)}")
        return False

    return True


def main():
    """Main entry point."""
    print("=" * 60)
    print("Co-DETR Object Detection GUI")
    print("With automatic COCO to VOC class mapping")
    print("=" * 60)
    print()

    if not check_dependencies():
        sys.exit(1)

    print("\nStarting GUI...")

    from codetr_gui.gui import main as gui_main
    gui_main()


if __name__ == '__main__':
    main()
