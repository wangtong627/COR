import torch
import numpy as np
import logging

import os
import torch
import numpy as np


class AverageMeter:
    """
    Track and compute the average of a sequence of values with sliding window support.

    This class records the current value, sum, count, and average of a series of values,
    and can calculate the average of the most recent updates within a window.
    Suitable for monitoring losses or other metrics during training or validation.
    """

    def __init__(self, window_size=40):
        """
        Initialize AverageMeter object.

        Args:
            window_size (int): Size of sliding window for computing recent average, default is 40.
        """
        self.window_size = window_size
        self.reset()

    def reset(self):
        """Reset all statistics and clear records."""
        self.current_value = 0.0  # Current value
        self.total_sum = 0.0  # Cumulative sum
        self.count = 0  # Update count
        self.average = 0.0  # Current average
        self.history = []  # History value list

    def update(self, value, n=1):
        """
        Update statistics, record new value and recalculate average.

        Args:
            value (float): New value to add
            n (int): Number of samples for this value, default is 1
        """
        self.current_value = value
        self.total_sum += value * n
        self.count += n
        self.average = self.total_sum / self.count if self.count > 0 else 0.0
        self.history.append(value)

    def get_window_average(self):
        """
        Calculate average of the most recent window_size updates.

        Returns:
            torch.Tensor: Average within sliding window. If history has fewer than window_size records,
                         returns average of all available records.
        """
        if not self.history:
            return torch.tensor(0.0)
        start_idx = max(len(self.history) - self.window_size, 0)
        window_values = self.history[start_idx:]
        return torch.mean(torch.tensor(window_values, dtype=torch.float32))


def get_logger(filename, verbosity=1, name=None):
    """
    logger = get_logger('/path/to/exp/exp.log')

    logger.info('start training!')
    for epoch in range(MAX_EPOCH):
        ...
        loss = ...
        acc = ...
        logger.info('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}'.format(epoch , MAX_EPOCH, loss, acc ))
        ...

    logger.info('finish training!')
    """

    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def init_logger(save_path=None, current_time=None):
    logging.basicConfig(
        filename=os.path.join(save_path, f"log_{current_time}.log"),
        format="[%(asctime)s - %(filename)s - %(levelname)s : %(message)s]",
        level=logging.INFO,
        filemode="a",
        datefmt="%Y-%m-%d %I:%M:%S %p",
    )
    logger = logging.getLogger()
    return logger


def init_val_logger(save_path=None, file_name=None):
    logging.basicConfig(
        filename=os.path.join(save_path, file_name),
        format="[%(asctime)s - %(filename)s - %(levelname)s : %(message)s]",
        level=logging.INFO,
        filemode="a",
        datefmt="%Y-%m-%d %I:%M:%S %p",
    )
    logger = logging.getLogger()
    return logger


def clip_gradient(optimizer, grad_clip=1.0):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    The clamp_ operation constrains gradients within the range [-grad_clip, grad_clip].

    Args:
        optimizer: Optimizer object containing parameters whose gradients need to be clipped
        grad_clip: Upper and lower bounds for gradient clipping (scalar, float type)
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for para_group in optimizer.param_groups:
        para_group["lr"] = init_lr * decay


# Example usage
if __name__ == "__main__":
    meter = AverageMeter(window_size=3)
    meter.update(1.0)
    meter.update(2.0)
    meter.update(3.0)
    print(f"Current Value: {meter.current_value:.4f}")  # 3.0000
    print(f"Overall Average: {meter.average:.4f}")  # 2.0000
    print(f"Window Average: {meter.get_window_average():.4f}")  # 2.0000
    meter.update(4.0)
    print(f"Window Average after update: {meter.get_window_average():.4f}")  # 3.0000
