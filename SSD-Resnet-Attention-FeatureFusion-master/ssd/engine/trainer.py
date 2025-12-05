import collections
import datetime
import logging
import os
import time
import torch
import torch.distributed as dist

from ssd.engine.inference import do_evaluation
from ssd.utils import dist_util
from ssd.utils.metric_logger import MetricLogger


def write_metric(eval_result, prefix, summary_writer, global_step):
    for key in eval_result:
        value = eval_result[key]
        tag = '{}/{}'.format(prefix, key)
        if isinstance(value, collections.Mapping):
            write_metric(value, tag, summary_writer, global_step)
        else:
            summary_writer.add_scalar(tag, value, global_step=global_step)


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = dist_util.get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(cfg, model,
             data_loader,
             optimizer,
             scheduler,
             checkpointer,
             device,
             arguments,
             args):
    logger = logging.getLogger("SSD.trainer")
    logger.info("Start training ...")
    meters = MetricLogger()

    model.train()
    save_to_disk = dist_util.get_rank() == 0
    if args.use_tensorboard and save_to_disk:
        import tensorboardX

        summary_writer = tensorboardX.SummaryWriter(log_dir=os.path.join(cfg.OUTPUT_DIR, 'tf_logs'))
    else:
        summary_writer = None

    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    start_training_time = time.time()
    end = time.time()

    # 记录最佳 mAP 和早停计数器
    best_mAP = arguments.get("best_mAP", 0.0)
    early_stop_counter = arguments.get("early_stop_counter", 0)
    early_stop_patience = getattr(args, 'early_stop_patience', -1)
    early_stop_min_delta = getattr(args, 'early_stop_min_delta', 0.001)
    early_stopped = False

    logger.info(f"Starting with best_mAP: {best_mAP:.4f}")
    if early_stop_patience > 0:
        logger.info(f"Early stopping enabled: patience={early_stop_patience}, min_delta={early_stop_min_delta}")

    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        iteration = iteration + 1
        arguments["iteration"] = iteration

        images = images.to(device)
        targets = targets.to(device)
        loss_dict = model(images, targets=targets)
        loss = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(total_loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time)
        if iteration % args.log_step == 0:
            eta_seconds = meters.time.global_avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            logger.info(
                meters.delimiter.join([
                    "iter: {iter:06d}",
                    "lr: {lr:.5f}",
                    '{meters}',
                    "eta: {eta}",
                    'mem: {mem}M',
                ]).format(
                    iter=iteration,
                    lr=optimizer.param_groups[0]['lr'],
                    meters=str(meters),
                    eta=eta_string,
                    mem=round(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0),
                )
            )
            if summary_writer:
                global_step = iteration
                summary_writer.add_scalar('losses/total_loss', losses_reduced, global_step=global_step)
                for loss_name, loss_item in loss_dict_reduced.items():
                    summary_writer.add_scalar('losses/{}'.format(loss_name), loss_item, global_step=global_step)
                summary_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)

        # 定期保存 checkpoint (间隔可设置较大)
        if iteration % args.save_step == 0:
            checkpointer.save("model_{:06d}".format(iteration), **arguments)

        # 评估并保存 best 模型
        if args.eval_step > 0 and iteration % args.eval_step == 0 and not iteration == max_iter:
            eval_results = do_evaluation(cfg, model, distributed=args.distributed, iteration=iteration)

            # 提取 mAP 并检查是否是 best
            if dist_util.get_rank() == 0:
                if eval_results and len(eval_results) > 0:
                    current_mAP = eval_results[0]['metrics'].get('mAP', 0.0)
                    logger.info(f"Iteration {iteration}: mAP = {current_mAP:.4f}, best_mAP = {best_mAP:.4f}")

                    # 检查是否有显著提升
                    if current_mAP > best_mAP + early_stop_min_delta:
                        # 有显著提升，重置计数器
                        best_mAP = current_mAP
                        early_stop_counter = 0
                        arguments["best_mAP"] = best_mAP
                        arguments["early_stop_counter"] = early_stop_counter
                        checkpointer.save("model_best", **arguments)
                        logger.info(f"*** New best model saved! mAP: {best_mAP:.4f} ***")
                    else:
                        # 无显著提升，增加计数器
                        early_stop_counter += 1
                        arguments["early_stop_counter"] = early_stop_counter
                        if early_stop_patience > 0:
                            logger.info(f"No improvement. Early stop counter: {early_stop_counter}/{early_stop_patience}")

                    # 检查是否触发早停
                    if early_stop_patience > 0 and early_stop_counter >= early_stop_patience:
                        logger.info(f"*** Early stopping triggered! No improvement for {early_stop_patience} evaluations. ***")
                        logger.info(f"*** Best mAP: {best_mAP:.4f} ***")
                        early_stopped = True

                    # 写入 tensorboard
                    if summary_writer:
                        for eval_result, dataset in zip(eval_results, cfg.DATASETS.TEST):
                            write_metric(eval_result['metrics'], 'metrics/' + dataset, summary_writer, iteration)
                        summary_writer.add_scalar('metrics/best_mAP', best_mAP, global_step=iteration)
                        if early_stop_patience > 0:
                            summary_writer.add_scalar('metrics/early_stop_counter', early_stop_counter, global_step=iteration)

            model.train()  # *IMPORTANT*: change to train mode after eval.

            # 早停退出
            if early_stopped:
                break

    # 保存最终模型
    checkpointer.save("model_final", **arguments)

    # 打印最终结果
    if early_stopped:
        logger.info(f"Training early stopped at iteration {iteration}. Best mAP: {best_mAP:.4f}")
    else:
        logger.info(f"Training finished. Best mAP: {best_mAP:.4f}")

    # compute training time
    total_training_time = int(time.time() - start_training_time)
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info("Total training time: {} ({:.4f} s / it)".format(total_time_str, total_training_time / iteration))
    return model
