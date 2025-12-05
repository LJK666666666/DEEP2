"""
Co-DETR + Swin-L 训练脚本
基于 MMDetection 框架

使用方法:
    python train_codetr.py --config configs/co_detr_swin_l_voc.py
    python train_codetr.py --config configs/co_detr_swin_l_voc.py --resume work_dirs/co_detr_swin_l_voc/latest.pth
"""

import argparse
import os
import os.path as osp

from mmengine.config import Config
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector with MMDetection')
    parser.add_argument('--config', required=True, help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume', type=str, default=None, help='resume from checkpoint')
    parser.add_argument('--amp', action='store_true', help='enable automatic-mixed-precision training')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher'
    )
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # 加载配置
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher

    # 设置工作目录
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    # 恢复训练
    if args.resume:
        cfg.resume = True
        cfg.load_from = args.resume

    # 启用 AMP
    if args.amp:
        cfg.optim_wrapper = dict(
            type='AmpOptimWrapper',
            optimizer=cfg.optim_wrapper.optimizer,
            loss_scale='dynamic'
        )

    # 设置随机种子
    cfg.randomness = dict(seed=args.seed, deterministic=False)

    # 构建 Runner 并开始训练
    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == '__main__':
    main()
