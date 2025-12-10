import math
from bisect import bisect_right

from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau


class WarmupMultiStepLR(_LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, warmup_factor=1.0 / 3,
                 warmup_iters=500, last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            alpha = float(self.last_epoch) / self.warmup_iters
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


class WarmupCosineAnnealingLR(_LRScheduler):
    """
    余弦退火学习率调度器，支持预热阶段。

    学习率变化：
    - 预热阶段 (0 ~ warmup_iters): 线性从 warmup_factor * base_lr 增加到 base_lr
    - 余弦退火阶段 (warmup_iters ~ T_max): 从 base_lr 余弦衰减到 eta_min

    Args:
        optimizer: 优化器
        T_max: 余弦退火总迭代数（包括预热）
        eta_min: 最小学习率
        warmup_factor: 预热起始学习率因子
        warmup_iters: 预热迭代数
        last_epoch: 上次迭代数，用于恢复训练
    """

    def __init__(self, optimizer, T_max, eta_min=1e-6, warmup_factor=1.0 / 3,
                 warmup_iters=500, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            # 预热阶段：线性增加
            alpha = float(self.last_epoch) / self.warmup_iters
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # 余弦退火阶段
            # 调整迭代数，从预热结束后开始计算
            progress = (self.last_epoch - self.warmup_iters) / max(1, self.T_max - self.warmup_iters)
            # 限制 progress 在 [0, 1] 范围内
            progress = min(1.0, progress)
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [
                self.eta_min + (base_lr - self.eta_min) * cosine_factor
                for base_lr in self.base_lrs
            ]


class WarmupReduceLROnPlateau:
    """
    基于验证集性能的学习率调度器，支持预热阶段。

    包装 PyTorch 的 ReduceLROnPlateau，在预热阶段使用线性增加的学习率，
    预热结束后根据验证集性能动态调整。

    Args:
        optimizer: 优化器
        mode: 'min' 或 'max'，监控指标的优化方向
        factor: 学习率衰减因子
        patience: 连续多少次评估没有改善时降低学习率
        min_lr: 最小学习率
        warmup_factor: 预热起始学习率因子
        warmup_iters: 预热迭代数
    """

    def __init__(self, optimizer, mode='max', factor=0.1, patience=5,
                 min_lr=1e-6, warmup_factor=1.0 / 3, warmup_iters=500):
        self.optimizer = optimizer
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.current_iter = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

        # 内部使用 ReduceLROnPlateau
        self.plateau_scheduler = ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            verbose=True
        )

        # 标记是否在预热阶段
        self._in_warmup = True

    def step(self, metrics=None):
        """
        更新学习率。

        在预热阶段，每次迭代调用时更新学习率（metrics 应为 None）。
        在正常阶段，每次评估后调用时更新学习率（metrics 应为评估指标）。
        """
        if self._in_warmup and self.current_iter < self.warmup_iters:
            # 预热阶段：线性增加学习率
            alpha = float(self.current_iter) / self.warmup_iters
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = base_lr * warmup_factor
            self.current_iter += 1
        else:
            # 预热结束
            if self._in_warmup:
                self._in_warmup = False
                # 恢复到基础学习率
                for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                    param_group['lr'] = base_lr

            # 正常阶段：基于验证指标调整
            if metrics is not None:
                self.plateau_scheduler.step(metrics)

    def step_iteration(self):
        """每次迭代调用，用于预热阶段的学习率更新"""
        if self._in_warmup and self.current_iter < self.warmup_iters:
            alpha = float(self.current_iter) / self.warmup_iters
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = base_lr * warmup_factor
            self.current_iter += 1
        elif self._in_warmup:
            # 预热刚结束，恢复基础学习率
            self._in_warmup = False
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = base_lr

    def step_metric(self, metrics):
        """每次评估后调用，基于验证指标调整学习率"""
        if not self._in_warmup:
            self.plateau_scheduler.step(metrics)

    def state_dict(self):
        """保存调度器状态"""
        return {
            'current_iter': self.current_iter,
            'in_warmup': self._in_warmup,
            'base_lrs': self.base_lrs,
            'plateau_scheduler': self.plateau_scheduler.state_dict()
        }

    def load_state_dict(self, state_dict):
        """加载调度器状态"""
        self.current_iter = state_dict['current_iter']
        self._in_warmup = state_dict['in_warmup']
        self.base_lrs = state_dict['base_lrs']
        self.plateau_scheduler.load_state_dict(state_dict['plateau_scheduler'])
