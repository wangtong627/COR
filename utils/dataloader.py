import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance
from torchvision import transforms
import open_clip
import random
import numpy as np

# ---- data augmentation ----


def randomCrop(img, gt):
    border = 30
    img_width = img.size[0]
    img_height = img.size[1]
    cropped_width = np.random.randint(img_width - border, img_width)
    cropped_height = np.random.randint(img_height - border, img_height)
    cropped_region = (
        (img_width - cropped_width) >> 1,
        (img_height - cropped_height) >> 1,
        (img_width + cropped_width) >> 1,
        (img_height + cropped_height) >> 1,
    )
    return img.crop(cropped_region), gt.crop(cropped_region)


def randomRotation(img, gt):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        img = img.rotate(random_angle, mode)
        gt = gt.rotate(random_angle, mode)
    return img, gt


def colorEnhance(img):
    bright_intensity = random.randint(5, 15) / 10
    img = ImageEnhance.Brightness(img).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10
    img = ImageEnhance.Contrast(img).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10
    img = ImageEnhance.Color(img).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10
    img = ImageEnhance.Sharpness(img).enhance(sharp_intensity)
    return img


def randomGaussian(img, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(img)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):
        randX = random.randint(0, img.shape[0] - 1)
        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:
            img[randX, randY] = 0
        else:
            img[randX, randY] = 255
    return Image.fromarray(img)


class TrainingDataset(Dataset):

    def __init__(self, csv_path, dataset_path, support_img_size=384, text_tokenizer=None):
        """
        Args:
            csv_file (str): CSV file path
            root_dir (str): Root directory path
            transform (callable, optional): Image preprocessing transforms
            mask_transform (callable, optional): Mask preprocessing transforms
            text_tokenizer (callable, optional): Text processing function
        """
        self.dataset_csv = pd.read_csv(csv_path)
        # self.dataset_csv = self.dataset_csv.iloc[:20]  # TODO: Take 20 for testing
        # Only keep samples with Compose == 0
        self.dataset_csv = self.dataset_csv[self.dataset_csv["Compose"] == 0]

        self.dataset_path = dataset_path
        self.query_img_size = 1024  # use sam pretrain can't change
        self.support_img_size = support_img_size  # Siglip 384
        self.support_mask_size = support_img_size
        # transforms
        self.query_image_transforms = transforms.Compose(
            [
                transforms.Resize((self.query_img_size, self.query_img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.query_gt_transforms = transforms.Compose(
            [
                transforms.Resize((self.query_img_size, self.query_img_size)),
                transforms.ToTensor(),  # Compress values to [0, 1]
            ]
        )
        self.support_image_transforms = transforms.Compose(
            [
                transforms.Resize((self.support_img_size, self.support_img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.support_gt_transforms = transforms.Compose(
            [
                transforms.Resize((self.support_mask_size, self.support_mask_size)),
                transforms.ToTensor(),  # Compress values to [0, 1]
            ]
        )

        # self.text_tokenizer = text_tokenizer
        self.siglip_text_tokenizer = open_clip.get_tokenizer(text_tokenizer)
        self.dataset_size = len(self.dataset_csv)
        # print(">>> Training with {} samples".format(self.dataset_size))

    def rgb_loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def binary_loader(self, path):
        with open(path, "rb") as f:
            gt = Image.open(f)
            return gt.convert("L")

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        row = self.dataset_csv.iloc[idx]
        pair_id = row["Id"]
        compose = row["Compose"]
        dataset_name = row["Dataset"]
        target_class = row["Target"]

        # # Build Query paths
        # query_img_path = os.path.join(self.dataset_path, dataset_name, "Image", row["Query_img"])
        # query_mask_path = os.path.join(self.dataset_path, dataset_name, "Mask", target_class, row["Query_mask"])

        # # Build Support paths
        # support_img_path = os.path.join(self.dataset_path, dataset_name, "Image", row["Support_img"])
        # support_mask_path = os.path.join(self.dataset_path, dataset_name, "Mask", "sup", row["Support_mask"])

        # Build Query paths
        query_img_path = os.path.join(self.dataset_path, dataset_name, "image", row["Query_img"])
        query_mask_path = os.path.join(self.dataset_path, dataset_name, "mask", target_class, row["Query_mask"])

        # Build Support paths
        support_img_path = os.path.join(self.dataset_path, dataset_name, "image", row["Support_img"])
        support_mask_path = os.path.join(self.dataset_path, dataset_name, "mask", "sup", row["Support_mask"])

        # Read images and masks, perform data augmentation
        query_img = self.rgb_loader(query_img_path)
        query_mask = self.binary_loader(query_mask_path)
        query_img, query_mask = randomCrop(query_img, query_mask)
        query_img, query_mask = randomRotation(query_img, query_mask)
        query_img = colorEnhance(query_img)
        query_mask = randomPeper(query_mask)
        query_img = self.query_image_transforms(query_img)
        query_mask = self.query_gt_transforms(query_mask)

        support_img = self.rgb_loader(support_img_path)
        support_mask = self.binary_loader(support_mask_path)
        support_img = self.support_image_transforms(support_img)
        support_mask = self.support_gt_transforms(support_mask)

        # Process text
        text = row["Text"]
        text = self.siglip_text_tokenizer(text).squeeze(0)  # [1, 64] -> [64]

        pair = {
            "pair_id": pair_id,  # 1
            "query_img": query_img,  # size: [3, 1024, 1024]
            "query_mask": query_mask,  # size: [1, 1024, 1024]
            "support_img": support_img,  # size: [3, 384, 384]
            "support_mask": support_mask,  # size: [1, 27, 27]
            "text": text,  # size: [64]
            "compose": compose,  # 0
            "dataset": dataset_name,  # Test_1
            "target": target_class,  # 0q1n
        }
        return pair


def get_train_loader(
    csv_path,
    dataset_path,
    support_img_size=384,
    text_tokenizer=None,
    batch_size=8,
    shuffle=True,
    num_workers=12,
    pin_memory=True,
    prefetch_factor=4,
    worker_init_fn=None,
):
    """
    Create training data loader, ensure reproducibility by controlling worker thread random seeds through worker_init_fn.

    Args:
        csv_path (str): Training CSV file path
        dataset_path (str): Dataset path
        support_img_size (int): Support image size, default is 384
        text_tokenizer (str, optional): Text tokenizer name, default is None
        batch_size (int): Batch size, default is 8
        shuffle (bool): Whether to shuffle data, default is True
        num_workers (int): Number of worker threads, default is 12
        pin_memory (bool): Whether to use pinned memory, default is True
        prefetch_factor (int): Prefetch factor, default is 4
        worker_init_fn (callable, optional): DataLoader worker thread initialization function, used to fix random seeds

    Returns:
        torch.utils.data.DataLoader: Training data loader
    """
    dataset = TrainingDataset(csv_path, dataset_path, support_img_size=support_img_size, text_tokenizer=text_tokenizer)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        worker_init_fn=worker_init_fn,
    )
    return data_loader


class VaildingDataset(Dataset):

    def __init__(self, csv_path, dataset_path, support_img_size=384, text_tokenizer=None):
        """
        Args:
            csv_file (str): CSV file path
            root_dir (str): Root directory path
            transform (callable, optional): Image preprocessing transforms
            mask_transform (callable, optional): Mask preprocessing transforms
            text_tokenizer (callable, optional): Text processing function
            ** No data augmentation needed during testing
        """
        self.dataset_csv = pd.read_csv(csv_path)
        # self.dataset_csv = self.dataset_csv.iloc[:20]  # TODO: Take 20 for testing
        # Only keep samples with Compose == 0
        self.dataset_csv = self.dataset_csv[self.dataset_csv["Compose"] == 0]

        self.dataset_path = dataset_path
        self.query_img_size = 1024  # use sam pretrain can't change
        self.support_img_size = support_img_size  # Siglip 384
        self.support_mask_size = support_img_size
        # transforms
        self.query_image_transforms = transforms.Compose(
            [
                transforms.Resize((self.query_img_size, self.query_img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.query_gt_transforms = transforms.Compose(
            [
                transforms.Resize((self.query_img_size, self.query_img_size)),
                transforms.ToTensor(),  # Compress values to [0, 1]
            ]
        )
        self.support_image_transforms = transforms.Compose(
            [
                transforms.Resize((self.support_img_size, self.support_img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.support_gt_transforms = transforms.Compose(
            [
                transforms.Resize((self.support_mask_size, self.support_mask_size)),
                transforms.ToTensor(),  # Compress values to [0, 1]
            ]
        )

        # self.text_tokenizer = text_tokenizer
        self.siglip_text_tokenizer = open_clip.get_tokenizer(text_tokenizer)
        self.dataset_size = len(self.dataset_csv)
        # print(">>> Training with {} samples".format(self.dataset_size))

    def rgb_loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def binary_loader(self, path):
        with open(path, "rb") as f:
            gt = Image.open(f)
            return gt.convert("L")

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        row = self.dataset_csv.iloc[idx]
        pair_id = row["Id"]
        compose = row["Compose"]
        dataset_name = row["Dataset"]
        target_class = row["Target"]
        query_cat = row["query_cat"]  # Add query_cat

        # # Build Query paths
        # query_img_path = os.path.join(self.dataset_path, dataset_name, "Image", row["Query_img"])
        # query_mask_path = os.path.join(self.dataset_path, dataset_name, "Mask", target_class, row["Query_mask"])

        # # Build Support paths
        # support_img_path = os.path.join(self.dataset_path, dataset_name, "Image", row["Support_img"])
        # support_mask_path = os.path.join(self.dataset_path, dataset_name, "Mask", "sup", row["Support_mask"])

        # Build Query paths
        query_img_path = os.path.join(self.dataset_path, dataset_name, "image", row["Query_img"])
        query_mask_path = os.path.join(self.dataset_path, dataset_name, "mask", target_class, row["Query_mask"])

        # Build Support paths
        support_img_path = os.path.join(self.dataset_path, dataset_name, "image", row["Support_img"])
        support_mask_path = os.path.join(self.dataset_path, dataset_name, "mask", "sup", row["Support_mask"])

        # Read images and masks, validation doesn't need data augmentation
        query_img = self.rgb_loader(query_img_path)
        query_mask = self.binary_loader(query_mask_path)
        query_img = self.query_image_transforms(query_img)
        query_mask = self.query_gt_transforms(query_mask)

        support_img = self.rgb_loader(support_img_path)
        support_mask = self.binary_loader(support_mask_path)
        support_img = self.support_image_transforms(support_img)
        support_mask = self.support_gt_transforms(support_mask)

        # Different from Train here
        text_string = row["Text"]  # Save original text
        text_tokens = self.siglip_text_tokenizer(text_string).squeeze(0)  # Tokenized text

        pair = {
            "pair_id": pair_id,
            "query_img": query_img,
            "query_mask": query_mask,
            "support_img": support_img,
            "support_mask": support_mask,
            "text": text_tokens,  # Tokenized text
            "text_string": text_string,  # Original text
            "compose": compose,
            "dataset": dataset_name,
            "target": target_class,
            "query_cat": query_cat,  # Add query_cat
            "query_img_name": row["Query_img"],
            "query_mask_name": row["Query_mask"],
            "support_img_name": row["Support_img"],
            "support_mask_name": row["Support_mask"],
        }
        return pair


def get_vaild_loader(
    csv_path,
    dataset_path,
    support_img_size=384,
    text_tokenizer=None,
    batch_size=8,
    shuffle=False,
    num_workers=12,
    pin_memory=True,
    prefetch_factor=4,
    worker_init_fn=None,
):
    """
    Create validation data loader, ensure reproducibility by controlling worker thread random seeds through worker_init_fn.

    Args:
        csv_path (str): Validation CSV file path
        dataset_path (str): Dataset path
        support_img_size (int): Support image size, default is 384
        text_tokenizer (str, optional): Text tokenizer name, default is None
        batch_size (int): Batch size, default is 8
        shuffle (bool): Whether to shuffle data, default is False
        num_workers (int): Number of worker threads, default is 12
        pin_memory (bool): Whether to use pinned memory, default is True
        prefetch_factor (int): Prefetch factor, default is 4
        worker_init_fn (callable, optional): DataLoader worker thread initialization function, used to fix random seeds

    Returns:
        torch.utils.data.DataLoader: Validation data loader
    """
    dataset = VaildingDataset(csv_path, dataset_path, support_img_size=support_img_size, text_tokenizer=text_tokenizer)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        worker_init_fn=worker_init_fn,
    )
    return data_loader


class TestDataset_Single:
    """
    TestDataset class for loading test datasets, no data augmentation transforms
    Output prediction results, transform to the same size as GT
    """

    def __init__(self, csv_path, dataset_path, support_img_size, text_tokenizer):
        self.dataset_csv = pd.read_csv(csv_path)
        self.dataset_path = dataset_path
        self.query_img_size = 1024
        self.support_img_size = support_img_size
        self.support_mask_size = support_img_size
        self.query_image_transforms = transforms.Compose(
            [
                transforms.Resize((self.query_img_size, self.query_img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        # GT is not input, no transformation needed
        # self.query_gt_transforms = transforms.Compose(
        #     [
        #         # transforms.Resize((self.query_img_size, self.query_img_size)),
        #         transforms.ToTensor(),
        #     ]
        # )
        self.support_image_transforms = transforms.Compose(
            [
                transforms.Resize((self.support_img_size, self.support_img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.support_gt_transforms = transforms.Compose(
            [
                transforms.Resize((self.support_mask_size, self.support_mask_size)),
                transforms.ToTensor(),
            ]
        )

        self.siglip_text_tokenizer = open_clip.get_tokenizer(text_tokenizer)
        self.dataset_size = len(self.dataset_csv)
        print(">>> Testing with {} samples".format(self.dataset_size))

    def rgb_loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def binary_loader(self, path):
        with open(path, "rb") as f:
            gt = Image.open(f)
            return gt.convert("L")

    # def __len__(self):
    #     return self.dataset_size

    def load_data(self, idx):
        row = self.dataset_csv.iloc[idx]
        pair_id = row["Id"]
        compose = row["Compose"]
        dataset_name = row["Dataset"]
        target_class = row["Target"]

        query_img_path = os.path.join(self.dataset_path, dataset_name, "Image", row["Query_img"])
        pred_query_img_name = row["Query_mask"]
        query_mask_path = os.path.join(self.dataset_path, dataset_name, "Mask", target_class, row["Query_mask"])
        support_img_path = os.path.join(self.dataset_path, dataset_name, "Image", row["Support_img"])
        support_mask_path = os.path.join(self.dataset_path, dataset_name, "Mask", "sup", row["Support_mask"])
        query_image = self.rgb_loader(query_img_path)
        query_image = self.query_image_transforms(query_image).unsqueeze(0)  # [1, 3, 1024, 1024]
        query_mask = self.binary_loader(query_mask_path)  # GT only needs reading, no transformation
        # query_mask = self.query_gt_transforms(query_mask)
        support_image = self.rgb_loader(support_img_path)
        support_image = self.support_image_transforms(support_image).unsqueeze(0)  # [1, 3, 384, 384]
        support_mask = self.binary_loader(support_mask_path)
        support_mask = self.support_gt_transforms(support_mask).unsqueeze(0)  # [1, 3, 27, 27]
        # Process text
        text = row["Text"]
        text = self.siglip_text_tokenizer(text)  # [1, 64]
        return {
            "pair_id": pair_id,
            "query_img": query_image,
            "gt": query_mask,
            "support_img": support_image,
            "support_mask": support_mask,
            "compose": compose,
            "dataset": dataset_name,
            "target": target_class,
            "pred_query_img_name": pred_query_img_name,
        }
