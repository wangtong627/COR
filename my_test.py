"""
Validation Step 1: Load trained model parameters and test model performance
"""

import os
import yaml
import torch


import torch.backends.cudnn as cudnn
from utils.dataloader import get_vaild_loader
from utils.vailder import val_metric, val_visual_for_compare_v2, save_hard_pred_masks, save_soft_pred_masks
from utils.utils import init_val_logger
from lib.build_model import build_model_with_query_support_feat
from datetime import datetime
import argparse
from argparse import Namespace
from accelerate import Accelerator, DistributedType
import torch.distributed as dist


# Manual seed setting function
def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def main():
    """Main training loop"""
    # Parse command line arguments to get YAML file path
    args = parse_args()
    opt = load_config(args.config)  # Use YAML file path specified from command line

    # Initialize Accelerator, depends on --config_file parameter from accelerate launch
    accelerator = Accelerator()
    # Set global seed
    set_seed(0)

    # Initialize logging
    os.makedirs(opt.vaild_model_save_path, exist_ok=True)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"val_log_{current_time}.log"
    logger = init_val_logger(save_path=opt.vaild_model_save_path, file_name=log_filename)
    if accelerator.is_main_process:
        logger.info(f">>> Training Config: {vars(opt)}")
        logger.info(f">>> Distributed Type: {accelerator.distributed_type}, Mixed Precision: {accelerator.mixed_precision}")
        print(f">>> Distributed Type: {accelerator.distributed_type}, Mixed Precision: {accelerator.mixed_precision}")

    # Check GPU availability
    device = accelerator.device
    cudnn.benchmark = True

    # Initialize model
    my_model = build_model_with_query_support_feat(
        sam_model=opt.sam_model_name,
        siglip_model=opt.siglip_model_name,
        sam_checkpoint_path=None,
        siglip_checkpoint_path=None,
        mask_pooling=opt.mask_pooling,
    )

    # Initialize data loaders
    num_workers = min(torch.multiprocessing.cpu_count() // accelerator.num_processes, 8)
    # Test-Base
    val_loader_A = get_vaild_loader(
        opt.val_csv_A,
        opt.dataset_path,
        support_img_size=384,
        text_tokenizer=opt.siglip_model_name,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # Test-Novel
    val_loader_B = get_vaild_loader(
        opt.val_csv_B,
        opt.dataset_path,
        support_img_size=384,
        text_tokenizer=opt.siglip_model_name,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # Use accelerator.prepare to wrap
    (my_model, val_loader_A, val_loader_B) = accelerator.prepare(my_model, val_loader_A, val_loader_B)
    # Note: Currently only testing phase, no optimizer needed. If training is added, please add my_optimizer

    if accelerator.is_main_process:
        print(f">>> val_loader_A with {val_loader_A.dataset.dataset_size} samples")
        logger.info(f">>> val_loader_A with {val_loader_A.dataset.dataset_size} samples")
        print(f">>> val_loader_B with {val_loader_B.dataset.dataset_size} samples")
        logger.info(f">>> val_loader_B with {val_loader_B.dataset.dataset_size} samples")

    # Load checkpoint
    if opt.load_checkpoint_path is not None:
        try:
            checkpoint = torch.load(opt.load_checkpoint_path, map_location=device)
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint

            # Check if using DDP, if so, adjust state_dict key names
            if accelerator.distributed_type == DistributedType.MULTI_GPU:
                new_state_dict = {}
                for key, value in state_dict.items():
                    if not key.startswith("module."):
                        new_key = f"module.{key}"
                    else:
                        new_key = key
                    new_state_dict[new_key] = value
                state_dict = new_state_dict

            # Get current state_dict of the model
            model_state_dict = my_model.state_dict()

            # Check if keys in state_dict match the model's state_dict
            missing_keys = [key for key in model_state_dict.keys() if key not in state_dict]
            unexpected_keys = [key for key in state_dict.keys() if key not in model_state_dict]

            # Load parameters
            my_model.load_state_dict(state_dict, strict=True)

            # Verify parameter values are correctly updated after loading
            loaded_state_dict = my_model.state_dict()
            mismatches = []
            for key in model_state_dict.keys():
                if not torch.equal(model_state_dict[key], loaded_state_dict[key]):
                    mismatches.append(key)

            # Log loading results
            if accelerator.is_main_process:
                logger.info(f">>> Loaded checkpoint from {opt.load_checkpoint_path}.")
                print(f">>> Loaded checkpoint from {opt.load_checkpoint_path}.")

                if missing_keys:
                    logger.warning(f"Missing keys in checkpoint: {missing_keys}")
                    print(f"Missing keys in checkpoint: {missing_keys}")
                if unexpected_keys:
                    logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")
                    print(f"Unexpected keys in checkpoint: {unexpected_keys}")
                if mismatches:
                    logger.warning(f"Parameters not correctly updated: {mismatches}")
                    print(f"Parameters not correctly updated: {mismatches}")
                if not missing_keys and not unexpected_keys and not mismatches:
                    logger.info("All parameters were correctly loaded and updated.")
                    print("All parameters were correctly loaded and updated.")
        except RuntimeError as e:
            if accelerator.is_main_process:
                logger.error(f"Failed to load checkpoint: {str(e)}")
                print(f"Failed to load checkpoint: {str(e)}")
            raise
        except Exception as e:
            if accelerator.is_main_process:
                logger.error(f"Unexpected error while loading checkpoint: {str(e)}")
                print(f"Unexpected error while loading checkpoint: {str(e)}")
            raise
    else:
        if accelerator.is_main_process:
            logger.warning("No checkpoint to load.")
            print("No checkpoint to load.")
        raise ValueError("No checkpoint to load.")

    # Start testing
    try:
        ##### Test-Base #####
        if accelerator.is_main_process:
            print("Start Valid Test_A...")
            logger.info("Start Valid Test_A...")

        # Prediction result visualization
        save_hard_pred_masks(
            val_loader_A,
            my_model,
            opt,
            logger,
            accelerator=accelerator,
            dataset_path=opt.dataset_path,  # Dataset root path
            pred_save_dir="hard_pred_Test_1",  # Prediction mask save directory
        )

        ##### Test-Novel #####
        if accelerator.is_main_process:
            print("Start Valid Test_B...")
            logger.info("Start Valid Test_B...")

        # Prediction result visualization
        save_hard_pred_masks(
            val_loader_B,
            my_model,
            opt,
            logger,
            accelerator=accelerator,
            dataset_path=opt.dataset_path,  # Dataset root path
            pred_save_dir="hard_pred_Test_2",  # Prediction mask save directory
        )

    except Exception as e:
        if accelerator.is_main_process:
            logger.error(f"Error during validation: {str(e)}")
            print(f"Error during validation: {str(e)}")
        accelerator.wait_for_everyone()
        raise
    finally:
        if accelerator.is_main_process:
            logger.info(">>> Validation finished!")
            print(">>> Validation finished!")

        # Resolve NCCL resource cleanup issue
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
