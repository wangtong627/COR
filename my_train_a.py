import os
import yaml
import torch

# import torch.nn as nn
import torch.optim as optim
from timm.scheduler import CosineLRScheduler
import torch.backends.cudnn as cudnn
from utils.dataloader import get_train_loader, get_vaild_loader
from utils.trainer_v3_g import train_stage, val_stage
from utils.utils import init_logger, adjust_lr

# from lib.build_model import build_model
from lib.build_model import build_model_with_query_support_feat
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import argparse
from argparse import Namespace
from accelerate import Accelerator, DistributedType
import random
import numpy as np


def load_config(config_file):
    """Load configuration from YAML file and return Namespace object"""
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)
    return Namespace(**config_dict)


def parse_args():
    """Parse command line arguments, only used to get YAML config file path"""
    parser = argparse.ArgumentParser(description="Train a SAM-based segmentation model")
    parser.add_argument(
        "--config",
        type=str,
        default="/l/users/tong.wang/my_models/my_model_3_a/config/train_config_m3.yaml",
        help="Path to the YAML configuration file",
    )
    return parser.parse_args()


def set_random_seed(seed, deterministic=True):
    """Set random seed to ensure reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    """Set random seed for DataLoader workers"""
    worker_seed = torch.initial_seed() % 2**32 + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def main():
    """Main training loop"""

    # Parse command line arguments to get YAML file path
    args = parse_args()
    opt = load_config(args.config)  # Use YAML file path specified from command line

    # Initialize Accelerator, depends on --config_file parameter from accelerate launch
    accelerator = Accelerator()

    # Set random seed (before any random operations)
    seed = 42  # Customizable seed value
    set_random_seed(seed, deterministic=True)
    if accelerator.is_main_process:
        print(f">>> Set random seed: {seed}")

    # Initialize logging
    os.makedirs(opt.train_model_save_path, exist_ok=True)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = init_logger(save_path=opt.train_model_save_path, current_time=current_time)
    if accelerator.is_main_process:
        logger.info(f">>> Training Config: {vars(opt)}")
        logger.info(f">>> Distributed Type: {accelerator.distributed_type}, Mixed Precision: {accelerator.mixed_precision}")
        print(f">>> Distributed Type: {accelerator.distributed_type}, Mixed Precision: {accelerator.mixed_precision}")

    # Initialize TensorBoard
    tensorboard_log_dir = os.path.join(opt.train_model_save_path, f"summary_{current_time}")
    writer = SummaryWriter(log_dir=tensorboard_log_dir)

    # Check GPU availability
    device = accelerator.device

    # Verify file paths
    for path in [
        opt.load_checkpoint_path,
        opt.load_sam_pretrained_checkpoint,
        opt.load_siglip_pretrained_checkpoint,
    ]:
        if path and not os.path.exists(path):
            if accelerator.is_main_process:
                logger.error(f"File not found: {path}")
                print(f"ERROR: File not found: {path}")
            raise FileNotFoundError(f"File not found: {path}")

    # Initialize model
    my_model = build_model_with_query_support_feat(
        sam_model=opt.sam_model_name,
        siglip_model=opt.siglip_model_name,
        sam_checkpoint_path=opt.load_sam_pretrained_checkpoint,
        siglip_checkpoint_path=opt.load_siglip_pretrained_checkpoint,
        mask_pooling=opt.mask_pooling,
    )

    # Only optimize non-frozen parameters
    trainable_params = [p for p in my_model.parameters() if p.requires_grad]

    # # Print trainable_params information (only in main process)
    # if accelerator.is_main_process:
    #     logger.info(f">>> Trainable parameters count: {len(trainable_params)}")
    #     logger.info(f">>> Trainable parameters numel: {sum(p.numel() for p in trainable_params)}")
    #     if trainable_params:
    #         logger.info(f">>> Sample trainable param shape: {trainable_params[0].shape}")

    # Initialize optimizer
    if opt.optimizer == "Adam":
        my_optimizer = optim.Adam(trainable_params, lr=opt.lr)  # Use trainable_params
        if accelerator.is_main_process:
            logger.info(">>> Using optimizer: Adam")
            print(">>> Using optimizer: Adam")
    elif opt.optimizer == "AdamW":
        my_optimizer = optim.AdamW(trainable_params, lr=opt.lr)  # Use trainable_params
        if accelerator.is_main_process:
            logger.info(">>> Using optimizer: AdamW")
            print(">>> Using optimizer: AdamW")
    else:
        my_optimizer = optim.SGD(trainable_params, lr=opt.lr, momentum=0.9)  # Use trainable_params
        if accelerator.is_main_process:
            logger.info(">>> Using optimizer: SGD")
            print(">>> Using optimizer: SGD")

    # Initialize learning rate scheduler
    if opt.lr_scheduler == "CosineAnnealingLR":
        if accelerator.is_main_process:
            logger.info(">>> Using LR Scheduler: CosineAnnealingLR")
            print(">>> Using LR Scheduler: CosineAnnealingLR")
        scheduler = optim.lr_scheduler.CosineAnnealingLR(my_optimizer, T_max=opt.epoch, eta_min=0.1 * opt.lr)
    elif opt.lr_scheduler == "CosineAnnealingWarmRestarts":
        if accelerator.is_main_process:
            logger.info(">>> Using LR Scheduler: CosineAnnealingWarmRestarts")
            print(">>> Using LR Scheduler: CosineAnnealingWarmRestarts")
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(my_optimizer, T_0=10, T_mult=2, eta_min=0.1 * opt.lr)
    elif opt.lr_scheduler == "TimmCosineLRScheduler":
        if accelerator.is_main_process:
            logger.info(">>> Using LR Scheduler: TimmCosineLRScheduler")
            print(">>> Using LR Scheduler: TimmCosineLRScheduler")
        scheduler = CosineLRScheduler(
            my_optimizer,
            t_initial=opt.epoch - 5,  # Length of cosine annealing phase = total epochs - warmup epochs
            lr_min=0.1 * opt.lr,
            warmup_t=5,  # Warmup for 5 epochs
            warmup_lr_init=0.1 * opt.lr,  # Initial warmup learning rate
            warmup_prefix=True,
        )
    elif opt.lr_scheduler == "ExponentialLR":
        if accelerator.is_main_process:
            logger.info(">>> Using LR Scheduler: ExponentialLR")
            print(">>> Using LR Scheduler: ExponentialLR")
        scheduler = optim.lr_scheduler.ExponentialLR(my_optimizer, gamma=0.95)
    else:
        if accelerator.is_main_process:
            logger.info(">>> Using LR Scheduler: None")
            print(">>> Using LR Scheduler: None")
        scheduler = None

    # Initialize data loaders
    train_loader = get_train_loader(
        opt.train_csv,
        opt.dataset_path,
        support_img_size=384,
        text_tokenizer=opt.siglip_model_name,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=8,
        worker_init_fn=worker_init_fn,  # Add worker_init_fn
    )
    val_loader = get_vaild_loader(
        opt.val_csv,
        opt.dataset_path,
        support_img_size=384,
        text_tokenizer=opt.siglip_model_name,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=8,
        worker_init_fn=worker_init_fn,  # Add worker_init_fn
    )

    # Use accelerator.prepare to wrap
    my_model, my_optimizer, train_loader, val_loader = accelerator.prepare(my_model, my_optimizer, train_loader, val_loader)

    if accelerator.is_main_process:
        print(f">>> Training with {train_loader.dataset.dataset_size} samples")
        logger.info(f">>> Training with {train_loader.dataset.dataset_size} samples")
        print(f">>> Validating with {val_loader.dataset.dataset_size} samples")
        logger.info(f">>> Validating with {val_loader.dataset.dataset_size} samples")

    # Load checkpoint: dynamically select based on distributed type
    if opt.load_checkpoint_path is not None:
        if accelerator.distributed_type == DistributedType.DEEPSPEED:
            accelerator.load_state(opt.load_checkpoint_path)
            if accelerator.is_main_process:
                logger.info(f">>> Loaded DeepSpeed checkpoint from {opt.load_checkpoint_path}")
                print(f">>> Loaded DeepSpeed checkpoint from {opt.load_checkpoint_path}")
            start_epoch = 1  # DeepSpeed doesn't explicitly save epoch, need manual management
        else:  # MULTI_GPU (DDP)
            checkpoint = torch.load(opt.load_checkpoint_path, map_location=device)
            # Add strict=False
            my_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            my_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"]  # Don't add 1 here since loading from checkpoint
            if accelerator.is_main_process:
                logger.info(f">>> Loaded DDP checkpoint from {opt.load_checkpoint_path} at epoch {start_epoch}")
                print(f">>> Loaded DDP checkpoint from {opt.load_checkpoint_path} at epoch {start_epoch}")

        # Check frozen and trainable parameters
        if accelerator.is_main_process:
            print("\n>>> Checking parameters after loading checkpoint:")
            frozen_params = 0
            trainable_params = 0

            for name, param in my_model.named_parameters():
                num_params = param.numel()  # Calculate number of elements in current parameter
                if not param.requires_grad:
                    frozen_params += num_params
                    print(f" - Frozen: {name}, Shape: {param.shape}, Params: {num_params}")
                else:
                    trainable_params += num_params
                    print(f" - Trainable: {name}, Shape: {param.shape}, Params: {num_params}")

            print(f"\n>>> Summary:")
            print(f" - Total Frozen Parameters: {frozen_params:,}")
            print(f" - Total Trainable Parameters: {trainable_params:,}")
            print(f" - Total Parameters: {frozen_params + trainable_params:,}")
            print(">>> Finished checking parameters.\n")
    else:
        start_epoch = 1
        if accelerator.is_main_process:
            logger.info(">>> Starting training from scratch")
            print(">>> Starting training from scratch")

    # Training loop
    try:
        for epoch in range(start_epoch, opt.epoch + 1):

            # Because of warmup, need to schedule learning rate before training
            if scheduler is not None:
                scheduler.step(epoch) if isinstance(scheduler, CosineLRScheduler) else scheduler.step()

            train_stage(
                train_loader=train_loader,
                model=my_model,
                optimizer=my_optimizer,
                epoch=epoch,
                opt=opt,
                logger=logger,
                writer=writer,
                accelerator=accelerator,
            )

            val_stage(
                test_loader=val_loader,
                model=my_model,
                optimizer=my_optimizer,
                epoch=epoch,
                opt=opt,
                logger=logger,
                writer=writer,
                accelerator=accelerator,
            )
    except Exception as e:
        if accelerator.is_main_process:
            logger.error(f"Training interrupted with error: {str(e)}")
            print(f"Training interrupted with error: {str(e)}")
        raise
    finally:
        writer.close()
        if accelerator.is_main_process:
            logger.info(">>> Training finished!")
            print(">>> Training finished!")


if __name__ == "__main__":
    main()
