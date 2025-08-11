import torch
import torch.nn.functional as F
import os
from utils.utils import AverageMeter
from datetime import datetime, timedelta
import time
import csv
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2


# def val_metric(test_loader, model, opt, logger, accelerator=None, output_csv_name="per_sample_metrics.csv"):
#     model.eval()
#     if accelerator.is_main_process:
#         print("=" * 35, "Valid model", "=" * 35)

#     # Save test results and logs
#     model_save_path = opt.vaild_model_save_path
#     if accelerator.is_main_process:
#         os.makedirs(model_save_path, exist_ok=True)
#         output_csv_path = os.path.join(model_save_path, output_csv_name)
#         with open(output_csv_path, "w", newline="") as csvfile:
#             fieldnames = [
#                 "Id",
#                 "Query_img",
#                 "Query_mask",
#                 "Support_img",
#                 "Support_mask",
#                 "Text",
#                 "Compose",
#                 "Dataset",
#                 "Target",
#                 # "cat_id",  # Add cat_id
#                 "query_cat",  # Add query_cat
#                 "Dice",
#                 "IoU",
#                 "MAE",
#                 "mDice",
#                 "mIoU",
#             ]
#             writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#             writer.writeheader()

#     dice_meter = AverageMeter()
#     mae_meter = AverageMeter()
#     iou_meter = AverageMeter()
#     mdice_meter = AverageMeter()
#     miou_meter = AverageMeter()
#     total_samples = 0

#     batch_time_meter = AverageMeter()
#     epoch_start_time = time.time()
#     total_batches = len(test_loader)

#     with torch.no_grad():
#         for batch_idx, batch_data in enumerate(test_loader, start=1):
#             batch_start_time = time.time()

#             # Model input data
#             query_img = batch_data["query_img"]
#             support_img = batch_data["support_img"]
#             support_mask = batch_data["support_mask"]
#             text_tokens = batch_data["text"]
#             query_mask = batch_data["query_mask"]

#             # Get metadata
#             pair_ids = batch_data["pair_id"]
#             query_img_names = batch_data.get("query_img_name", None)
#             query_mask_names = batch_data.get("query_mask_name", None)
#             support_img_names = batch_data.get("support_img_name", None)
#             support_mask_names = batch_data.get("support_mask_name", None)
#             text_strings = batch_data.get("text_string", None)
#             composes = batch_data["compose"]
#             dataset_names = batch_data["dataset"]
#             target_classes = batch_data["target"]
#             # cat_id = batch_data["cat_id"]  # Get cat_id
#             query_cat = batch_data["query_cat"]  # Get query_cat

#             with accelerator.autocast():
#                 pred_mask, _, _ = model(
#                     query_image_inputs=query_img,
#                     support_image_inputs=support_img,
#                     change_text_inputs=text_tokens,
#                     support_mask_inputs=support_mask,
#                     multimask_output=opt.multimask_output,
#                 )

#                 pred_mask = F.interpolate(pred_mask, size=query_mask.size()[2:], mode="bilinear", align_corners=False)
#                 # is_empty_prob = torch.sigmoid(is_empty_logit)
#                 # is_empty_mask = (is_empty_prob > 0.5).float()
#                 # pred_mask = pred_mask * (1 - is_empty_mask).unsqueeze(-1).unsqueeze(-1)

#                 pred = torch.sigmoid(pred_mask)
#                 pred_min = torch.amin(pred, dim=(1, 2, 3), keepdim=True)
#                 pred_max = torch.amax(pred, dim=(1, 2, 3), keepdim=True)
#                 pred = (pred - pred_min) / (pred_max - pred_min + 1e-8)

#             # Calculate metrics for each sample
#             dice_batch = compute_dice(pred, query_mask)
#             mae_batch = compute_mae(pred, query_mask)
#             iou_batch = compute_iou(pred, query_mask)
#             mdice_batch = compute_mdice(pred, query_mask)
#             miou_batch = compute_miou(pred, query_mask)

#             # Update averages (only for calculating global metrics)
#             dice_meter.update(dice_batch.mean().item(), n=dice_batch.size(0))
#             mae_meter.update(mae_batch.mean().item(), n=mae_batch.size(0))
#             iou_meter.update(iou_batch.mean().item(), n=iou_batch.size(0))
#             mdice_meter.update(mdice_batch.mean().item(), n=mdice_batch.size(0))
#             miou_meter.update(miou_batch.mean().item(), n=miou_batch.size(0))
#             total_samples += pred.size(0)

#             # Stream write to CSV file
#             if accelerator.is_main_process:
#                 with open(output_csv_path, "a", newline="") as csvfile:
#                     writer = csv.DictWriter(
#                         csvfile,
#                         fieldnames=[
#                             "Id",
#                             "Query_img",
#                             "Query_mask",
#                             "Support_img",
#                             "Support_mask",
#                             "Text",
#                             "Compose",
#                             "Dataset",
#                             "Target",
#                             # "cat_id",
#                             "query_cat",
#                             "Dice",
#                             "MAE",
#                             "IoU",
#                             "mDice",
#                             "mIoU",
#                         ],
#                     )
#                     for i in range(pred.size(0)):
#                         sample_info = {
#                             "Id": pair_ids[i].item() if torch.is_tensor(pair_ids[i]) else pair_ids[i],
#                             "Query_img": (query_img_names[i] if query_img_names else f"sample_{pair_ids[i]}_query_img"),
#                             "Query_mask": (query_mask_names[i] if query_mask_names else f"sample_{pair_ids[i]}_query_mask"),
#                             "Support_img": (support_img_names[i] if support_img_names else f"sample_{pair_ids[i]}_support_img"),
#                             "Support_mask": (support_mask_names[i] if support_mask_names else f"sample_{pair_ids[i]}_support_mask"),
#                             "Text": text_strings[i] if text_strings else "",
#                             "Compose": composes[i].item() if torch.is_tensor(composes[i]) else composes[i],
#                             "Dataset": dataset_names[i],
#                             "Target": target_classes[i],
#                             # "cat_id": cat_id[i].item() if torch.is_tensor(cat_id[i]) else cat_id[i],  # Write cat_id
#                             "query_cat": query_cat[i].item() if torch.is_tensor(query_cat[i]) else query_cat[i],  # Write query_cat
#                             "Dice": "{:.4f}".format(dice_batch[i].item()),  # Keep 4 decimal places
#                             "IoU": "{:.4f}".format(iou_batch[i].item()),  # Keep 4 decimal places
#                             "MAE": "{:.4f}".format(mae_batch[i].item()),  # Keep 4 decimal places
#                             "mDice": "{:.4f}".format(mdice_batch[i].item()),  # Keep 4 decimal places
#                             "mIoU": "{:.4f}".format(miou_batch[i].item()),  # Keep 4 decimal places
#                         }
#                         writer.writerow(sample_info)

#             batch_time_meter.update(time.time() - batch_start_time)

#             avg_batch_time = batch_time_meter.average
#             remaining_batches = total_batches - batch_idx
#             eta_str = str(timedelta(seconds=int(avg_batch_time * remaining_batches)))

#             if accelerator.is_main_process and (batch_idx % 10 == 0 or batch_idx == total_batches):
#                 print(
#                     f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
#                     f"[Batch: {batch_idx:04d}/{total_batches:04d}] => "
#                     f"[ETA: {eta_str}]"
#                 )

#         # Calculate global weighted average metrics
#         local_total_samples = total_samples

#         accelerator.wait_for_everyone()
#         dice_per_process = torch.tensor(dice_meter.total_sum, device=accelerator.device)
#         mae_per_process = torch.tensor(mae_meter.total_sum, device=accelerator.device)
#         iou_per_process = torch.tensor(iou_meter.total_sum, device=accelerator.device)
#         mdice_per_process = torch.tensor(mdice_meter.total_sum, device=accelerator.device)
#         miou_per_process = torch.tensor(miou_meter.total_sum, device=accelerator.device)
#         samples_per_process = torch.tensor(local_total_samples, device=accelerator.device)

#         global_dice_sum = accelerator.gather(dice_per_process).sum().item()
#         global_mae_sum = accelerator.gather(mae_per_process).sum().item()
#         global_iou_sum = accelerator.gather(iou_per_process).sum().item()
#         global_mdice_sum = accelerator.gather(mdice_per_process).sum().item()
#         global_miou_sum = accelerator.gather(miou_per_process).sum().item()
#         global_samples_sum = accelerator.gather(samples_per_process).sum().item()

#         global_avg_dice_weighted = global_dice_sum / global_samples_sum
#         global_avg_mae_weighted = global_mae_sum / global_samples_sum
#         global_avg_iou_weighted = global_iou_sum / global_samples_sum
#         global_avg_mdice_weighted = global_mdice_sum / global_samples_sum
#         global_avg_miou_weighted = global_miou_sum / global_samples_sum

#         epoch_duration_str = str(timedelta(seconds=int(time.time() - epoch_start_time)))

#         if accelerator.is_main_process:
#             logger.info(
#                 f"Global Dice: {global_avg_dice_weighted:.4f}, Global MAE: {global_avg_mae_weighted:.4f}, Global IoU: {global_avg_iou_weighted:.4f}, "
#                 f"Global mDice: {global_avg_mdice_weighted:.4f}, Global mIoU: {global_avg_miou_weighted:.4f}, [Duration: {epoch_duration_str}]"
#             )

#             print(
#                 f"Global Dice: {global_avg_dice_weighted:.4f}, Global MAE: {global_avg_mae_weighted:.4f}, Global IoU: {global_avg_iou_weighted:.4f}, "
#                 f"Global mDice: {global_avg_mdice_weighted:.4f}, Global mIoU: {global_avg_miou_weighted:.4f}, [Duration: {epoch_duration_str}]"
#             )

#             logger.info(f"Per-sample metrics saved to {output_csv_path}")

#     # Return global average metrics
#     return {
#         "global_metrics": {
#             "dice": global_avg_dice_weighted,
#             "mae": global_avg_mae_weighted,
#             "iou": global_avg_iou_weighted,
#             "mdice": global_avg_mdice_weighted,
#             "miou": global_avg_miou_weighted,
#         },
#         "per_sample_metrics": None,
#     }


# # ---- Metric calculation functions, different from train, with binarization ----
# def compute_dice(pred: torch.Tensor, gt: torch.Tensor, smooth=1e-5, threshold=0.5):
#     """
#     Calculate Dice coefficient, supporting special cases of all-zero masks.

#     Args:
#         pred: Predicted mask, probability values after sigmoid, shape [B, H, W] or [B, C, H, W]
#         gt: Ground truth mask, binarized (0 or 1), same shape as pred
#         smooth: Smoothing factor to avoid division by zero
#         threshold: Binarization threshold, default 0.5

#     Returns:
#         dice: Dice coefficient for each sample, shape [B]
#     """
#     assert pred.shape == gt.shape, f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}"

#     # Flatten to [B, N], N=H*W or C*H*W
#     pred = pred.view(pred.size(0), -1)
#     gt = gt.view(gt.size(0), -1)

#     # Binarize pred (convert probability values to 0 or 1)
#     pred = (pred > threshold).float()

#     # Calculate intersection and union
#     intersection = (pred * gt).sum(dim=1)
#     pred_sum = pred.sum(dim=1)
#     gt_sum = gt.sum(dim=1)

#     # Handle all-zero special case
#     all_zero_mask = (gt_sum == 0) & (pred_sum == 0)
#     dice = torch.zeros_like(intersection)

#     # For non-all-zero samples, calculate Dice normally
#     normal_mask = ~all_zero_mask
#     dice[normal_mask] = (2.0 * intersection[normal_mask] + smooth) / (pred_sum[normal_mask] + gt_sum[normal_mask] + smooth)

#     # For all-zero samples (both gt and pred are all zeros), return 1
#     dice[all_zero_mask] = 1.0

#     return dice


# def compute_mae(pred: torch.Tensor, gt: torch.Tensor):
#     """
#     Calculate MAE (Mean Absolute Error), supporting all-zero masks.

#     Args:
#         pred: Predicted mask, probability values after sigmoid, shape [B, H, W] or [B, C, H, W]
#         gt: Ground truth mask, binarized (0 or 1), same shape as pred

#     Returns:
#         mae: MAE for each sample, shape [B]
#     """
#     assert pred.shape == gt.shape, f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}"

#     # Flatten to [B, N], N=H*W or C*H*W
#     pred = pred.view(pred.size(0), -1)
#     gt = gt.view(gt.size(0), -1)

#     # Calculate MAE directly
#     mae = torch.abs(pred - gt).mean(dim=1)
#     return mae


# def compute_iou(pred: torch.Tensor, gt: torch.Tensor, smooth=1e-5, threshold=0.5):
#     """
#     Calculate IoU (Intersection over Union), supporting special cases of all-zero masks.

#     Args:
#         pred: Predicted mask, probability values after sigmoid, shape [B, H, W] or [B, C, H, W]
#         gt: Ground truth mask, binarized (0 or 1), same shape as pred
#         smooth: Smoothing factor to avoid division by zero
#         threshold: Binarization threshold, default 0.5

#     Returns:
#         iou: IoU for each sample, shape [B]
#     """
#     assert pred.shape == gt.shape, f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}"

#     # Flatten to [B, N], N=H*W or C*H*W
#     pred = pred.view(pred.size(0), -1)
#     gt = gt.view(gt.size(0), -1)

#     # Binarize pred (convert probability values to 0 or 1)
#     pred = (pred > threshold).float()

#     # Calculate intersection and union
#     intersection = (pred * gt).sum(dim=1)
#     pred_sum = pred.sum(dim=1)
#     gt_sum = gt.sum(dim=1)
#     union = pred_sum + gt_sum - intersection

#     # Handle all-zero special case
#     all_zero_mask = (gt_sum == 0) & (pred_sum == 0)
#     iou = torch.zeros_like(intersection)

#     # For non-all-zero samples, calculate IoU normally
#     normal_mask = ~all_zero_mask
#     iou[normal_mask] = (intersection[normal_mask] + smooth) / (union[normal_mask] + smooth)

#     # For all-zero samples (both gt and pred are all zeros), return 1
#     iou[all_zero_mask] = 1.0

#     return iou


# def compute_mdice(pred: torch.Tensor, gt: torch.Tensor, smooth=1e-5):
#     """
#     Calculate mDice, considering both foreground and background Dice coefficients.
#     """
#     assert pred.shape == gt.shape, f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}"

#     # Foreground Dice
#     dice_foreground = compute_dice(pred, gt, smooth)

#     # Background Dice: invert foreground and background
#     pred_background = 1 - pred  # Background prediction (1 becomes 0, 0 becomes 1)
#     gt_background = 1 - gt  # Background GT
#     dice_background = compute_dice(pred_background, gt_background, smooth)

#     # mDice: average of foreground and background Dice
#     mdice = (dice_foreground + dice_background) / 2.0
#     return mdice


# def compute_miou(pred: torch.Tensor, gt: torch.Tensor, smooth=1e-5):
#     """
#     Calculate mIoU, considering both foreground and background IoU.
#     """
#     assert pred.shape == gt.shape, f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}"

#     # Foreground IoU
#     iou_foreground = compute_iou(pred, gt, smooth)

#     # Background IoU: invert foreground and background
#     pred_background = 1 - pred  # Background prediction
#     gt_background = 1 - gt  # Background GT
#     iou_background = compute_iou(pred_background, gt_background, smooth)

#     # mIoU: average of foreground and background IoU
#     miou = (iou_foreground + iou_background) / 2.0
#     return miou


def save_hard_pred_masks(
    test_loader,
    model,
    opt,
    logger,
    accelerator=None,
    dataset_path="/data/dataset",  # Dataset root path
    pred_save_dir="predictions",  # Prediction mask save directory
):
    """
    Only save prediction masks, read GT size for resize, name and save according to specified format.
    1. Generate pred_mask and adjust to GT size
    2. Get GT filename and path from batch_data
    3. Save pred_mask as png format, named as pair_id+"_"+query_mask_name
    4. Automatically create save folder
    """
    model.eval()
    if accelerator.is_main_process:
        print("=" * 35, "Save model predictions", "=" * 35)

    # Set prediction save path
    model_save_path = opt.vaild_model_save_path
    pred_save_path = os.path.join(model_save_path, pred_save_dir)

    if accelerator.is_main_process:
        os.makedirs(pred_save_path, exist_ok=True)
        print(f"[INFO] Prediction masks will be saved to: {pred_save_path}")

    batch_time_meter = AverageMeter()
    epoch_start_time = time.time()
    total_batches = len(test_loader)

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader, start=1):
            batch_start_time = time.time()

            # Model input data
            query_img = batch_data["query_img"]  # [B, C, H, W]
            support_img = batch_data["support_img"]  # [B, C, H, W]
            support_mask = batch_data["support_mask"]  # [B, 1, H, W]
            text_tokens = batch_data["text"]

            # Get metadata
            pair_ids = batch_data["pair_id"]
            query_mask_names = batch_data["query_mask_name"]
            datasets = batch_data["dataset"]
            targets = batch_data["target"]

            with accelerator.autocast():
                # Model inference to get prediction mask
                pred_mask, _, _ = model(
                    query_image_inputs=query_img,
                    support_image_inputs=support_img,
                    change_text_inputs=text_tokens,
                    support_mask_inputs=support_mask,
                    multimask_output=opt.multimask_output,
                )

                # Convert pred_mask to probability values and normalize
                pred = torch.sigmoid(pred_mask)
                pred_min = torch.amin(pred, dim=(1, 2, 3), keepdim=True)
                pred_max = torch.amax(pred, dim=(1, 2, 3), keepdim=True)
                pred = (pred - pred_min) / (pred_max - pred_min + 1e-8)

            # Process and save prediction masks in main process
            if accelerator.is_main_process:
                batch_size = query_img.size(0)
                for i in range(batch_size):
                    query_mask_name = query_mask_names[i]
                    dataset = datasets[i]
                    target = targets[i]
                    gt_mask_path = os.path.join(dataset_path, dataset, "mask", str(target), query_mask_name)

                    # Read GT size
                    if os.path.exists(gt_mask_path):
                        try:
                            gt_mask = Image.open(gt_mask_path)
                            gt_size = gt_mask.size
                            if gt_size[0] <= 1 or gt_size[1] <= 1:
                                logger.error(f"Invalid GT size {gt_size} for {gt_mask_path}, skipping sample")
                                continue
                            gt_mask.close()
                            # print(f"[DEBUG] GT size: {gt_size}")
                        except Exception as e:
                            logger.error(f"Failed to read GT mask size for {gt_mask_path}: {str(e)}")
                            continue
                    else:
                        logger.warning(f"GT mask not found: {gt_mask_path}, skipping sample")
                        continue

                    # Resize pred_mask to GT size
                    pred_i = pred[i].cpu().numpy()
                    pred_i = np.squeeze(pred_i)
                    if len(pred_i.shape) != 2:
                        logger.error(f"pred_i shape error before resize: {pred_i.shape}, expected (H, W)")
                        continue
                    # print(f"[DEBUG] pred_i shape before resize: {pred_i.shape}")

                    pred_i = cv2.resize(pred_i, gt_size, interpolation=cv2.INTER_LINEAR)
                    # print(f"[DEBUG] pred_i shape after resize: {pred_i.shape}")
                    if len(pred_i.shape) != 2:
                        logger.error(f"pred_i shape error after resize: {pred_i.shape}, expected (H, W)")
                        continue

                    # Binarization
                    pred_i = (pred_i > 0.5).astype(np.uint8) * 255
                    # print(f"[DEBUG] pred_i shape after binarization: {pred_i.shape}, dtype: {pred_i.dtype}")

                    # Save prediction mask
                    pred_filename = f"{pair_ids[i]}_{query_mask_name}"
                    pred_filepath = os.path.join(pred_save_path, pred_filename)
                    try:
                        if pred_i.shape[0] <= 1 or pred_i.shape[1] <= 1:
                            logger.error(f"Invalid pred_i shape {pred_i.shape} for {pred_filepath}, skipping save")
                            continue
                        pred_img = Image.fromarray(pred_i)
                        pred_img.save(pred_filepath)
                        # print(f"[INFO] Successfully saved: {pred_filepath}")
                    except Exception as e:
                        logger.error(f"Failed to save prediction mask {pred_filepath}: {str(e)}")
                        print(
                            f"[DEBUG] pred_i info: shape={pred_i.shape}, dtype={pred_i.dtype}, "
                            f"min={pred_i.min()}, max={pred_i.max()}, unique_values={np.unique(pred_i)}"
                        )
                        continue

            batch_time_meter.update(time.time() - batch_start_time)

            if accelerator.is_main_process and (batch_idx % 10 == 0 or batch_idx == total_batches):
                avg_batch_time = batch_time_meter.average
                remaining_batches = total_batches - batch_idx
                eta_str = str(timedelta(seconds=int(avg_batch_time * remaining_batches)))
                print(
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
                    f"[Batch: {batch_idx:04d}/{total_batches:04d}] => "
                    f"[ETA: {eta_str}]"
                )

    epoch_duration_str = str(timedelta(seconds=int(time.time() - epoch_start_time)))

    if accelerator.is_main_process:
        logger.info(f"Predictions saved to {pred_save_path}, [Duration: {epoch_duration_str}]")
        print(f"Predictions saved to {pred_save_path}, [Duration: {epoch_duration_str}]")


def save_soft_pred_masks(
    test_loader,
    model,
    opt,
    logger,
    accelerator=None,
    dataset_path="/data/dataset",  # Dataset root path
    pred_save_dir="predictions",  # Prediction mask save directory
):
    """
    Only save prediction masks, read GT size for resize, name and save according to specified format.
    1. Generate pred_mask and adjust to GT size
    2. Get GT filename and path from batch_data
    3. Save pred_mask as png format, named as pair_id+"_"+query_mask_name
    4. Automatically create save folder
    """
    model.eval()
    if accelerator.is_main_process:
        print("=" * 35, "Save model predictions", "=" * 35)

    # Set prediction save path
    model_save_path = opt.vaild_model_save_path
    pred_save_path = os.path.join(model_save_path, pred_save_dir)

    if accelerator.is_main_process:
        os.makedirs(pred_save_path, exist_ok=True)
        print(f"[INFO] Prediction masks will be saved to: {pred_save_path}")

    batch_time_meter = AverageMeter()
    epoch_start_time = time.time()
    total_batches = len(test_loader)

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader, start=1):
            batch_start_time = time.time()

            # Model input data
            query_img = batch_data["query_img"]  # [B, C, H, W]
            support_img = batch_data["support_img"]  # [B, C, H, W]
            support_mask = batch_data["support_mask"]  # [B, 1, H, W]
            text_tokens = batch_data["text"]

            # Get metadata
            pair_ids = batch_data["pair_id"]  # Sample ID
            query_mask_names = batch_data["query_mask_name"]  # GT mask filename
            datasets = batch_data["dataset"]  # Dataset name
            targets = batch_data["target"]  # Target class

            with accelerator.autocast():
                # Model inference to get prediction mask
                pred_mask, _, _ = model(
                    query_image_inputs=query_img,
                    support_image_inputs=support_img,
                    change_text_inputs=text_tokens,
                    support_mask_inputs=support_mask,
                    multimask_output=opt.multimask_output,
                )

                # Convert pred_mask to probability values and normalize
                pred = torch.sigmoid(pred_mask)
                pred_min = torch.amin(pred, dim=(1, 2, 3), keepdim=True)
                pred_max = torch.amax(pred, dim=(1, 2, 3), keepdim=True)
                pred = (pred - pred_min) / (pred_max - pred_min + 1e-8)

            # Process and save prediction masks in main process
            if accelerator.is_main_process:
                batch_size = query_img.size(0)
                for i in range(batch_size):
                    query_mask_name = query_mask_names[i]
                    dataset = datasets[i]
                    target = targets[i]
                    gt_mask_path = os.path.join(dataset_path, dataset, "mask", str(target), query_mask_name)

                    # Read GT size
                    if os.path.exists(gt_mask_path):
                        try:
                            gt_mask = Image.open(gt_mask_path)
                            gt_size = gt_mask.size
                            if gt_size[0] <= 1 or gt_size[1] <= 1:
                                logger.error(f"Invalid GT size {gt_size} for {gt_mask_path}, skipping sample")
                                continue
                            gt_mask.close()
                            # print(f"[DEBUG] GT size: {gt_size}")
                        except Exception as e:
                            logger.error(f"Failed to read GT mask size for {gt_mask_path}: {str(e)}")
                            continue
                    else:
                        logger.warning(f"GT mask not found: {gt_mask_path}, skipping sample")
                        continue

                    # Resize pred_mask to GT size
                    pred_i = pred[i].cpu().numpy()
                    pred_i = np.squeeze(pred_i)
                    if len(pred_i.shape) != 2:
                        logger.error(f"pred_i shape error before resize: {pred_i.shape}, expected (H, W)")
                        continue
                    # print(f"[DEBUG] pred_i shape before resize: {pred_i.shape}")

                    pred_i = cv2.resize(pred_i, gt_size, interpolation=cv2.INTER_LINEAR)
                    # print(f"[DEBUG] pred_i shape after resize: {pred_i.shape}")
                    if len(pred_i.shape) != 2:
                        logger.error(f"pred_i shape error after resize: {pred_i.shape}, expected (H, W)")
                        continue

                    # Convert to grayscale (not binarized)
                    # pred_i = (pred_i > 0.5).astype(np.uint8) * 255
                    pred_i = (pred_i * 255).astype(np.uint8)
                    # print(f"[DEBUG] pred_i shape after conversion: {pred_i.shape}, dtype: {pred_i.dtype}")

                    # Save prediction mask
                    pred_filename = f"{pair_ids[i]}_{query_mask_name}"
                    pred_filepath = os.path.join(pred_save_path, pred_filename)
                    try:
                        if pred_i.shape[0] <= 1 or pred_i.shape[1] <= 1:
                            logger.error(f"Invalid pred_i shape {pred_i.shape} for {pred_filepath}, skipping save")
                            continue
                        pred_img = Image.fromarray(pred_i)
                        pred_img.save(pred_filepath)
                        # print(f"[INFO] Successfully saved: {pred_filepath}")
                    except Exception as e:
                        logger.error(f"Failed to save prediction mask {pred_filepath}: {str(e)}")
                        print(
                            f"[DEBUG] pred_i info: shape={pred_i.shape}, dtype={pred_i.dtype}, "
                            f"min={pred_i.min()}, max={pred_i.max()}, unique_values={np.unique(pred_i)}"
                        )
                        continue

            batch_time_meter.update(time.time() - batch_start_time)

            if accelerator.is_main_process and (batch_idx % 10 == 0 or batch_idx == total_batches):
                avg_batch_time = batch_time_meter.average
                remaining_batches = total_batches - batch_idx
                eta_str = str(timedelta(seconds=int(avg_batch_time * remaining_batches)))
                print(
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
                    f"[Batch: {batch_idx:04d}/{total_batches:04d}] => "
                    f"[ETA: {eta_str}]"
                )

    epoch_duration_str = str(timedelta(seconds=int(time.time() - epoch_start_time)))

    if accelerator.is_main_process:
        logger.info(f"Predictions saved to {pred_save_path}, [Duration: {epoch_duration_str}]")
        print(f"Predictions saved to {pred_save_path}, [Duration: {epoch_duration_str}]")
