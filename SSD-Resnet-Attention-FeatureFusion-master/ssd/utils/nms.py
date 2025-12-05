import sys
import warnings
import torchvision

try:
    import torch
    import torch_extension

    _nms = torch_extension.nms
except ImportError:
    # 使用 torchvision 内置的 NMS（适用于 torchvision >= 0.3.0）
    if hasattr(torchvision.ops, 'nms'):
        _nms = torchvision.ops.nms
    else:
        warnings.warn('No NMS is available. Please upgrade torchvision to 0.3.0+ or compile c++ NMS '
                      'using `cd ext & python build.py build_ext develop`')
        sys.exit(-1)


def nms(boxes, scores, nms_thresh):
    """ Performs non-maximum suppression, run on GPU or CPU according to
    boxes's device.
    Args:
        boxes(Tensor[N, 4]): boxes in (x1, y1, x2, y2) format, use absolute coordinates(or relative coordinates)
        scores(Tensor[N]): scores
        nms_thresh(float): thresh
    Returns:
        indices kept.
    """
    keep = _nms(boxes, scores, nms_thresh)
    return keep


def batched_nms(boxes, scores, idxs, iou_threshold):
    """
    Performs non-maximum suppression in a batched fashion.

    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.

    Parameters
    ----------
    boxes : Tensor[N, 4]
        boxes where NMS will be performed. They
        are expected to be in (x1, y1, x2, y2) format
    scores : Tensor[N]
        scores for each one of the boxes
    idxs : Tensor[N]
        indices of the categories for each one of the boxes.
    iou_threshold : float
        discards all overlapping boxes
        with IoU < iou_threshold

    Returns
    -------
    keep : Tensor
        int64 tensor with the indices of
        the elements that have been kept by NMS, sorted
        in decreasing order of scores
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    # 使用 torchvision 内置的 batched_nms (更稳定，避免溢出问题)
    if hasattr(torchvision.ops, 'batched_nms'):
        return torchvision.ops.batched_nms(boxes, scores, idxs, iou_threshold)

    # fallback: 手动实现 (旧版 torchvision)
    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1.0, device=boxes.device))
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep
