import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from utils.loss_func import (
    wbce_with_wiou_loss,
    fg_feat_similarity_loss,
    bg_feat_similarity_loss,
)
from utils.utils import clip_gradient, AverageMeter
from datetime import datetime, timedelta
import time
from accelerate import DistributedType

"""
This file calculates distributed global metrics
Loss calculation uses is_empty_prob
"""


def train_stage(train_loader, model, optimizer, epoch, opt, logger, writer=None, accelerator=None):
    """
    Train model for one epoch, record global loss and save checkpoints, display estimated completion time for each epoch.
    Compatible with MULTI_GPU (DDP) and DEEPSPEED, supports no/fp16/bf16 mixed precision.
    """
    model.train()
    if accelerator.is_main_process:
        logger.info("=" * 35 + f" Training Epoch: {epoch} " + "=" * 35)
        print("=" * 35, f"Training Epoch: {epoch}", "=" * 35)

    model_save_path = opt.train_model_save_path
    if accelerator.is_main_process:
        os.makedirs(model_save_path, exist_ok=True)

    loss_meter = AverageMeter(window_size=opt.batch_record_interval)
    batch_time_meter = AverageMeter()
    epoch_start_time = time.time()
    total_batches = len(train_loader)

    try:
        for batch_idx, batch_data in enumerate(train_loader, start=1):
            batch_start_time = time.time()

            query_img = batch_data["query_img"]
            support_img = batch_data["support_img"]
            support_mask = batch_data["support_mask"]
            text = batch_data["text"]
            query_mask = batch_data["query_mask"]

            optimizer.zero_grad()
            with accelerator.autocast():
                # pred_mask, _ = model(
                #     query_image_inputs=query_img,
                #     support_image_inputs=support_img,
                #     change_text_inputs=text,
                #     support_mask_inputs=support_mask,
                #     multimask_output=opt.multimask_output,
                # )
                pred_mask, query_image_embeddings, comb_support_feat = model(
                    query_image_inputs=query_img,
                    support_image_inputs=support_img,
                    change_text_inputs=text,
                    support_mask_inputs=support_mask,
                    multimask_output=opt.multimask_output,
                )

                target_mask = F.interpolate(query_mask, size=pred_mask.size()[2:], mode="bilinear", align_corners=False)
                segmentation_loss = wbce_with_wiou_loss(pred_mask, target_mask)
                feat_loss = 5 * fg_feat_similarity_loss(
                    query_image_embeddings, comb_support_feat, query_mask
                ) + 5 * bg_feat_similarity_loss(query_image_embeddings, comb_support_feat, query_mask)

                batch_loss = segmentation_loss + feat_loss
                # batch_loss = segmentation_loss

            accelerator.backward(batch_loss)

            if accelerator.distributed_type == DistributedType.MULTI_GPU:
                clip_gradient(optimizer=optimizer, grad_clip=opt.gradient_clip)
            optimizer.step()

            loss_meter.update(batch_loss.item())
            batch_time_meter.update(time.time() - batch_start_time)

            avg_batch_time = batch_time_meter.average
            remaining_batches = total_batches - batch_idx
            eta_str = str(timedelta(seconds=int(avg_batch_time * remaining_batches)))

            if accelerator.is_main_process and (batch_idx == 1 or batch_idx % opt.batch_record_interval == 0 or batch_idx == total_batches):
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [Epo: {epoch:03d}/{opt.epoch:03d}] => "
                    f"[Batch: {batch_idx:04d}/{total_batches:04d}] => "
                    f"[BLoss: {batch_loss.item():.4f}] => "
                    f"[LAvgLoss: {loss_meter.average:.4f}] => "
                    f"[Lr: {current_lr}] => [ETA: {eta_str}]"
                )

        # Calculate global average loss
        local_avg_loss = loss_meter.average

        accelerator.wait_for_everyone()
        global_avg_loss = accelerator.gather(torch.tensor(local_avg_loss, device=accelerator.device)).mean().item()

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_duration_str = str(timedelta(seconds=int(time.time() - epoch_start_time)))

        if accelerator.is_main_process:
            logger.info(
                f"[Train Info]: [Epoch {epoch:03d}/{opt.epoch:03d}], "
                f"[LocalAvgLoss: {local_avg_loss:.4f}], [GlobalAvgLoss: {global_avg_loss:.4f}], "
                f"[Lr: {current_lr}], [Duration: {epoch_duration_str}]"
            )
            print(
                f"[Train Info]: [Epoch {epoch:03d}/{opt.epoch:03d}], "
                f"[LocalAvgLoss: {local_avg_loss:.4f}], [GlobalAvgLoss: {global_avg_loss:.4f}], "
                f"[Lr: {current_lr}], [Duration: {epoch_duration_str}]"
            )

            if writer is not None:
                writer.add_scalar("Train/LearningRate", current_lr, epoch)
                writer.add_scalar("Train/LocalTotalLoss", local_avg_loss, epoch)
                writer.add_scalar("Train/GlobalTotalLoss", global_avg_loss, epoch)
                writer.add_scalar("Train/EpochDuration", time.time() - epoch_start_time, epoch)

            if epoch % opt.train_model_save_epoch == 0:
                checkpoint_path = os.path.join(model_save_path, f"checkpoint_epoch_{epoch}")
                if accelerator.distributed_type == DistributedType.DEEPSPEED:
                    accelerator.save_state(checkpoint_path)
                    logger.info(f"[Train Info]: Saved DeepSpeed checkpoint at epoch {epoch} to {checkpoint_path}")
                    print(f"[Train Info]: Saved DeepSpeed checkpoint at epoch {epoch}")
                else:
                    checkpoint_path += ".pth"
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": accelerator.unwrap_model(model).state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": global_avg_loss,
                        },
                        checkpoint_path,
                    )
                    logger.info(f"[Train Info]: Saved DDP checkpoint at epoch {epoch} to {checkpoint_path}")
                    print(f"[Train Info]: Saved DDP checkpoint at epoch {epoch}")

    except KeyboardInterrupt:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            print(">>> Keyboard Interrupt: Saving model and exiting!")
            logger.info("[Train Info]: Keyboard Interrupt: Saving model and exiting!")
            interrupt_path = os.path.join(model_save_path, f"interrupted_checkpoint_epoch_{epoch}")
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                accelerator.save_state(interrupt_path)
                logger.info(f"[Train Info]: Saved interrupted DeepSpeed checkpoint at epoch {epoch} to {interrupt_path}")
                print(f"[Train Info]: Saved interrupted DeepSpeed checkpoint at epoch {epoch}")
            else:
                interrupt_path += ".pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": accelerator.unwrap_model(model).state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    interrupt_path,
                )
                logger.info(f"[Train Info]: Saved interrupted DDP checkpoint at epoch {epoch} to {interrupt_path}")
                print(f"[Train Info]: Saved interrupted DDP checkpoint at epoch {epoch}")
        raise

    return global_avg_loss


def val_stage(test_loader, model, optimizer, epoch, opt, logger, writer=None, accelerator=None):
    model.eval()
    if accelerator.is_main_process:
        logger.info("=" * 35 + f" Val Epoch: {epoch} " + "=" * 35)
        print("=" * 35, f"Val Epoch: {epoch}", "=" * 35)

    model_save_path = opt.train_model_save_path
    if accelerator.is_main_process:
        os.makedirs(model_save_path, exist_ok=True)

    global best_metric_dict, best_score, best_epoch
    if "best_score" not in globals():
        best_score = float("-inf")  # Initialize to negative infinity since higher (mDice + mIoU - MAE) is better
        best_epoch = -1
        best_metric_dict = {}

    dice_meter = AverageMeter()
    mae_meter = AverageMeter()
    iou_meter = AverageMeter()
    mdice_meter = AverageMeter()
    miou_meter = AverageMeter()
    total_samples = 0

    batch_time_meter = AverageMeter()
    epoch_start_time = time.time()
    total_batches = len(test_loader)

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader, start=1):
            batch_start_time = time.time()

            query_img = batch_data["query_img"]
            support_img = batch_data["support_img"]
            support_mask = batch_data["support_mask"]
            text = batch_data["text"]
            query_mask = batch_data["query_mask"]

            with accelerator.autocast():
                # pred_mask, _ = model(
                #     query_image_inputs=query_img,
                #     support_image_inputs=support_img,
                #     change_text_inputs=text,
                #     support_mask_inputs=support_mask,
                #     multimask_output=opt.multimask_output,
                # )
                pred_mask, _, _ = model(
                    query_image_inputs=query_img,
                    support_image_inputs=support_img,
                    change_text_inputs=text,
                    support_mask_inputs=support_mask,
                    multimask_output=opt.multimask_output,
                )

                pred_mask = F.interpolate(pred_mask, size=query_mask.size()[2:], mode="bilinear", align_corners=False)

                pred = torch.sigmoid(pred_mask)
                pred_min = torch.amin(pred, dim=(1, 2, 3), keepdim=True)
                pred_max = torch.amax(pred, dim=(1, 2, 3), keepdim=True)
                pred = (pred - pred_min) / (pred_max - pred_min + 1e-8)

            dice_batch = compute_dice(pred, query_mask)
            mae_batch = compute_mae(pred, query_mask)
            iou_batch = compute_iou(pred, query_mask)
            mdice_batch = compute_mdice(pred, query_mask)
            miou_batch = compute_miou(pred, query_mask)

            dice_meter.update(dice_batch.mean().item(), n=dice_batch.size(0))
            mae_meter.update(mae_batch.mean().item(), n=mae_batch.size(0))
            iou_meter.update(iou_batch.mean().item(), n=iou_batch.size(0))
            mdice_meter.update(mdice_batch.mean().item(), n=mdice_batch.size(0))
            miou_meter.update(miou_batch.mean().item(), n=miou_batch.size(0))
            total_samples += pred.size(0)

            batch_time_meter.update(time.time() - batch_start_time)

            avg_batch_time = batch_time_meter.average
            remaining_batches = total_batches - batch_idx
            eta_str = str(timedelta(seconds=int(avg_batch_time * remaining_batches)))

            if accelerator.is_main_process and (batch_idx == 1 or batch_idx % opt.batch_record_interval == 0 or batch_idx == total_batches):
                print(
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [Epoch: {epoch:03d}/{opt.epoch:03d}] => "
                    f"[Batch: {batch_idx:04d}/{total_batches:04d}] => "
                    f"[LocalDice: {dice_batch.mean().item():.4f}] => "
                    f"[LocalMAE: {mae_batch.mean().item():.4f}] => "
                    f"[LocalIoU: {iou_batch.mean().item():.4f}] => "
                    f"[LocalmDice: {mdice_batch.mean().item():.4f}] => "
                    f"[LocalmIoU: {miou_batch.mean().item():.4f}] => [ETA: {eta_str}]"
                )

        # Calculate global weighted average metrics
        # local_avg_dice = dice_meter.average
        # local_avg_mae = mae_meter.average
        # local_avg_iou = iou_meter.average
        # local_avg_mdice = mdice_meter.average
        # local_avg_miou = miou_meter.average
        local_total_samples = total_samples

        accelerator.wait_for_everyone()
        dice_per_process = torch.tensor(dice_meter.total_sum, device=accelerator.device)
        mae_per_process = torch.tensor(mae_meter.total_sum, device=accelerator.device)
        iou_per_process = torch.tensor(iou_meter.total_sum, device=accelerator.device)
        mdice_per_process = torch.tensor(mdice_meter.total_sum, device=accelerator.device)
        miou_per_process = torch.tensor(miou_meter.total_sum, device=accelerator.device)
        samples_per_process = torch.tensor(local_total_samples, device=accelerator.device)

        global_dice_sum = accelerator.gather(dice_per_process).sum().item()
        global_mae_sum = accelerator.gather(mae_per_process).sum().item()
        global_iou_sum = accelerator.gather(iou_per_process).sum().item()
        global_mdice_sum = accelerator.gather(mdice_per_process).sum().item()
        global_miou_sum = accelerator.gather(miou_per_process).sum().item()
        global_samples_sum = accelerator.gather(samples_per_process).sum().item()

        global_avg_dice_weighted = global_dice_sum / global_samples_sum
        global_avg_mae_weighted = global_mae_sum / global_samples_sum
        global_avg_iou_weighted = global_iou_sum / global_samples_sum
        global_avg_mdice_weighted = global_mdice_sum / global_samples_sum
        global_avg_miou_weighted = global_miou_sum / global_samples_sum

        epoch_duration_str = str(timedelta(seconds=int(time.time() - epoch_start_time)))

        if accelerator.is_main_process:
            best_dice = best_metric_dict.get("dice", 0.0)
            best_mae = best_metric_dict.get("mae", 0.0)
            best_iou = best_metric_dict.get("iou", 0.0)
            best_mdice = best_metric_dict.get("mdice", 0.0)
            best_miou = best_metric_dict.get("miou", 0.0)

            logger.info(
                f"[Val Info]: Epoch: {epoch}, "
                f"Global Dice: {global_avg_dice_weighted:.4f}, Global MAE: {global_avg_mae_weighted:.4f}, Global IoU: {global_avg_iou_weighted:.4f}, "
                f"Global mDice: {global_avg_mdice_weighted:.4f}, Global mIoU: {global_avg_miou_weighted:.4f}, "
                f"[Duration: {epoch_duration_str}]"
            )
            logger.info(
                f"[Best Info]: BestEpoch: {best_epoch}, Best Dice: {best_dice:.4f}, Best MAE: {best_mae:.4f}, "
                f"Best IoU: {best_iou:.4f}, Best mDice: {best_mdice:.4f}, Best mIoU: {best_miou:.4f}"
            )
            print(
                f"[Val Info]: Epoch {epoch} - "
                f"Global Dice: {global_avg_dice_weighted:.4f}, Global MAE: {global_avg_mae_weighted:.4f}, Global IoU: {global_avg_iou_weighted:.4f}, "
                f"Global mDice: {global_avg_mdice_weighted:.4f}, Global mIoU: {global_avg_miou_weighted:.4f}, "
                f"[Duration: {epoch_duration_str}]"
            )
            print(
                f"[Best Info]: BestEpoch: {best_epoch}, "
                f"Best Dice: {best_dice:.4f}, Best MAE: {best_mae:.4f}, Best IoU: {best_iou:.4f}, "
                f"Best mDice: {best_mdice:.4f}, Best mIoU: {best_miou:.4f}"
            )

            # # Use global weighted average mDice + mIoU - MAE to determine best model (higher is better)
            # current_score = global_avg_mdice_weighted + global_avg_miou_weighted - global_avg_mae_weighted
            # # Use global weighted average mDice + mIoU to determine best model (higher is better)
            # current_score = global_avg_mdice_weighted + global_avg_miou_weighted

            # Use global weighted average Dice + IoU to determine best model (higher is better)
            current_score = global_avg_dice_weighted + global_avg_iou_weighted
            if current_score > best_score:
                best_score = current_score
                best_epoch = epoch
                best_metric_dict = {
                    "dice": global_avg_dice_weighted,
                    "mae": global_avg_mae_weighted,
                    "iou": global_avg_iou_weighted,
                    "mdice": global_avg_mdice_weighted,
                    "miou": global_avg_miou_weighted,
                }
                best_model_path = os.path.join(model_save_path, "best_model")
                best_model_full_path = os.path.join(model_save_path, "best_model_full.pth")

                if accelerator.distributed_type == DistributedType.DEEPSPEED:
                    accelerator.save_state(best_model_path)
                    logger.info(f"[Val Info]: New best DeepSpeed model saved at epoch {epoch} to {best_model_path}")
                    print(f"[Val Info]: New best DeepSpeed model saved at epoch {epoch}")
                else:
                    torch.save(accelerator.unwrap_model(model).state_dict(), best_model_path + ".pth")
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": accelerator.unwrap_model(model).state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                        },
                        best_model_full_path,
                    )
                    logger.info(
                        f"[Val Info]: New best DDP model saved at epoch {epoch} with Dice: {global_avg_dice_weighted:.4f}, "
                        f"MAE: {global_avg_mae_weighted:.4f}, IoU: {global_avg_iou_weighted:.4f}, "
                        f"mDice: {global_avg_mdice_weighted:.4f}, mIoU: {global_avg_miou_weighted:.4f}"
                    )
                    print(
                        f"[Val Info]: New best DDP model saved at epoch {epoch} with mDice: {global_avg_mdice_weighted:.4f}, "
                        f"mIoU: {global_avg_miou_weighted:.4f}, MAE: {global_avg_mae_weighted:.4f}, "
                        # f"Score (mDice+mIoU-MAE): {current_score:.4f}"
                        f"Score (Dice+IoU): {current_score:.4f}"
                    )

            if writer is not None:
                writer.add_scalar("Val/GlobalDice", global_avg_dice_weighted, epoch)
                writer.add_scalar("Val/GlobalMAE", global_avg_mae_weighted, epoch)
                writer.add_scalar("Val/GlobalIoU", global_avg_iou_weighted, epoch)
                writer.add_scalar("Val/GlobalmDice", global_avg_mdice_weighted, epoch)
                writer.add_scalar("Val/GlobalmIoU", global_avg_miou_weighted, epoch)
                writer.add_scalar("Val/EpochDuration", time.time() - epoch_start_time, epoch)

    return global_avg_dice_weighted, global_avg_mae_weighted, global_avg_iou_weighted, global_avg_mdice_weighted, global_avg_miou_weighted


# ---- Metric Calculation Functions ----
def compute_dice(pred: torch.Tensor, gt: torch.Tensor, smooth=1e-5):
    assert pred.shape == gt.shape, f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}"
    pred = pred.view(pred.size(0), -1)
    gt = gt.view(gt.size(0), -1)
    intersection = (pred * gt).sum(dim=1)
    dice = (2.0 * intersection + smooth) / (pred.sum(dim=1) + gt.sum(dim=1) + smooth)
    return dice


def compute_mae(pred: torch.Tensor, gt: torch.Tensor):
    assert pred.shape == gt.shape, f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}"
    pred = pred.view(pred.size(0), -1)
    gt = gt.view(gt.size(0), -1)
    mae = torch.abs(pred - gt).mean(dim=1)
    return mae


def compute_iou(pred: torch.Tensor, gt: torch.Tensor, smooth=1e-5):
    assert pred.shape == gt.shape, f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}"
    pred = pred.view(pred.size(0), -1)
    gt = gt.view(gt.size(0), -1)
    intersection = (pred * gt).sum(dim=1)
    union = pred.sum(dim=1) + gt.sum(dim=1) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou


def compute_mdice(pred: torch.Tensor, gt: torch.Tensor, smooth=1e-5):
    """
    Calculate mDice, considering both foreground and background Dice coefficients.
    """
    assert pred.shape == gt.shape, f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}"

    # Foreground Dice
    dice_foreground = compute_dice(pred, gt, smooth)

    # Background Dice: invert foreground and background
    pred_background = 1 - pred  # Background prediction (1 becomes 0, 0 becomes 1)
    gt_background = 1 - gt  # Background GT
    dice_background = compute_dice(pred_background, gt_background, smooth)

    # mDice: average of foreground and background Dice
    mdice = (dice_foreground + dice_background) / 2.0
    return mdice


def compute_miou(pred: torch.Tensor, gt: torch.Tensor, smooth=1e-5):
    """
    Calculate mIoU, considering both foreground and background IoU.
    """
    assert pred.shape == gt.shape, f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}"

    # Foreground IoU
    iou_foreground = compute_iou(pred, gt, smooth)

    # Background IoU: invert foreground and background
    pred_background = 1 - pred  # Background prediction
    gt_background = 1 - gt  # Background GT
    iou_background = compute_iou(pred_background, gt_background, smooth)

    # mIoU: average of foreground and background IoU
    miou = (iou_foreground + iou_background) / 2.0
    return miou
