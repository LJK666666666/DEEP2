import torch

from .lr_scheduler import WarmupMultiStepLR, WarmupCosineAnnealingLR, WarmupReduceLROnPlateau


def make_optimizer(cfg, model, lr=None):
    lr = cfg.SOLVER.LR if lr is None else lr
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)


def make_lr_scheduler(cfg, optimizer, milestones=None):
    """
    根据配置创建学习率调度器。

    支持的调度器类型:
    - WarmupMultiStepLR: 固定步长衰减（默认）
    - WarmupCosineAnnealingLR: 余弦退火
    - ReduceLROnPlateau: 基于验证集性能自适应调整

    Args:
        cfg: 配置对象
        optimizer: 优化器
        milestones: MultiStepLR 的里程碑（可选，仅用于 WarmupMultiStepLR）

    Returns:
        学习率调度器
    """
    scheduler_name = cfg.SOLVER.LR_SCHEDULER

    if scheduler_name == "WarmupMultiStepLR":
        return WarmupMultiStepLR(
            optimizer=optimizer,
            milestones=cfg.SOLVER.LR_STEPS if milestones is None else milestones,
            gamma=cfg.SOLVER.GAMMA,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS
        )
    elif scheduler_name == "WarmupCosineAnnealingLR":
        T_max = cfg.SOLVER.COSINE_T_MAX if cfg.SOLVER.COSINE_T_MAX > 0 else cfg.SOLVER.MAX_ITER
        return WarmupCosineAnnealingLR(
            optimizer=optimizer,
            T_max=T_max,
            eta_min=cfg.SOLVER.COSINE_ETA_MIN,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS
        )
    elif scheduler_name == "ReduceLROnPlateau":
        return WarmupReduceLROnPlateau(
            optimizer=optimizer,
            mode='max',  # mAP 越大越好
            factor=cfg.SOLVER.PLATEAU_FACTOR,
            patience=cfg.SOLVER.PLATEAU_PATIENCE,
            min_lr=cfg.SOLVER.PLATEAU_MIN_LR,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS
        )
    else:
        raise ValueError(f"Unknown LR scheduler: {scheduler_name}. "
                         f"Supported: WarmupMultiStepLR, WarmupCosineAnnealingLR, ReduceLROnPlateau")
