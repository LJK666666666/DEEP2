"""
Co-DETR + Swin-L 测试脚本
基于 MMDetection 框架

使用方法:
    python test_codetr.py --config configs/co_detr_swin_l_voc.py --checkpoint work_dirs/co_detr_swin_l_voc/best_mAP.pth
"""

import argparse
import os
import os.path as osp

from mmengine.config import Config
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description='Test a detector with MMDetection')
    parser.add_argument('--config', required=True, help='test config file path')
    parser.add_argument('--checkpoint', required=True, help='checkpoint file')
    parser.add_argument('--work-dir', help='the dir to save evaluation results')
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

    # 加载模型权重
    cfg.load_from = args.checkpoint

    # 构建 Runner 并开始测试
    runner = Runner.from_cfg(cfg)
    runner.test()


if __name__ == '__main__':
    main()
