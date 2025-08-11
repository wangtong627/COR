<div align="center">
<h1> Composed Object Retrieval (COR) </h1>
<h3>Composed Object Retrieval: Object-level Retrieval via Composed Expressions</h3>

Tong Wang<sup>1,2</sup>, Guanyu Yang<sup>1,\*</sup>, Nian Liu<sup>2,3,\*</sup>, Zongyan Han<sup>2</sup>, Jinxing Zhou<sup>2</sup>, Salman Khan<sup>2</sup>, Fahad Shahbaz Khan<sup>2</sup>

<sup>1</sup> Southeast University, <sup>2</sup> Mohamed Bin Zayed University of Artificial Intelligence, <sup>3</sup> Northwestern Polytechnical University  
<small><span style="color:#E63946; font-weight:bold;">*</span> indicates corresponding authors</small>

[![License](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/BUAADreamer/CCRK/blob/main/licence)
[![arXiv](https://img.shields.io/badge/arXiv-2508.04424-red)](https://arxiv.org/abs/2508.04424)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97-Dataset-blue)](https://huggingface.co/datasets/TongWang-NJ/COR_Bench_V1)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97-Checkpoint-blue)](https://huggingface.co/TongWang-NJ/CORE_COR_Bench_V1)
[![OneDrive Dataset](https://img.shields.io/badge/OneDrive-Dataset-blue)](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/tong_wang_mbzuai_ac_ae/EgPAHh93bBVJq_s34RBmuWIBPU2XmBDdGmIEAAkg2lAo-w?e=stRoK8)
[![OneDrive Checkpoint](https://img.shields.io/badge/OneDrive-Checkpoint-blue)](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/tong_wang_mbzuai_ac_ae/Er1V5c9G9EtAnQERvFQur_4Brn8M81rYtSuVNuerUIaWbw)

</div>

## ❗ Update
**We will release the code repository and dataset within one week.**

## 💡 Introduction
Retrieving fine-grained visual content based on user intent is a persistent challenge in multi-modal systems. Existing Composed Image Retrieval (CIR) methods, which combine reference images with textual descriptions, are limited to image-level matching and cannot localize specific objects. To address this, we introduce **Composed Object Retrieval (COR)**, a novel task that advances beyond image-level retrieval to achieve object-level precision. COR enables the retrieval and segmentation of target objects using composed expressions that integrate reference objects with retrieval texts. This task poses significant challenges in retrieval flexibility, requiring systems to accurately identify objects that satisfy the composed expressions while distinguishing them from semantically similar but irrelevant objects within the same scene. 
To support this task, we present **COR127K**, the first large-scale benchmark for COR, comprising 127,166 retrieval triplets across 408 categories with diverse semantic transformations. We also propose **CORE**, a unified end-to-end model that integrates reference region encoding, adaptive visual-textual interaction, and region-level contrastive learning. Extensive experiments demonstrate that CORE significantly outperforms existing models in both base and novel categories, establishing a robust and effective baseline for this challenging task. This work paves the way for future research in fine-grained multi-modal retrieval.
![](figures/framework.png)

## 🌐 COR127K Dataset
You can download the COR127K (COR_Bench_V1.0) dataset from the following links:
- From [Hugging Face](https://huggingface.co/datasets/TongWang-NJ/COR_Bench_V1)
- From [OneDrive](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/tong_wang_mbzuai_ac_ae/EgPAHh93bBVJq_s34RBmuWIBPU2XmBDdGmIEAAkg2lAo-w?e=stRoK8)

## 📦 Installation

The training and testing experiments are conducted using PyTorch. Below are the steps to set up the environment and install the necessary dependencies.

### Prerequisites

OurModel has been tested on Ubuntu OS with the following environments. It may work on other operating systems, but compatibility is not guaranteed.

1. **Clone the repository**:

   ```bash
   git clone https://github.com/wangtong627/COR.git
   ```

2. **Navigate to the project directory**:

   ```bash
   cd OpenSeg-R
   ```

3. **Create a virtual environment**:

   ```bash
   conda create -n COR_Base python=3.10
   ```

4. **Activate the virtual environment**:

   ```bash
   conda activate COR_Base
   ```

5. **Install dependencies**:
   We provide `environment.yml` and `requirements.txt` for setting up the environment. You can use either of the following commands:

   ```bash
   conda env create -f environment.yml
   ```

   or

   ```bash
   pip install -r requirements.txt
   ```


## 🌐 COR127K Dataset

You can download the COR127K (COR_Bench_V1.0) dataset from the following links:

- From [Hugging Face](https://huggingface.co/datasets/TongWang-NJ/COR_Bench_V1)
- From [OneDrive](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/tong_wang_mbzuai_ac_ae/EgPAHh93bBVJq_s34RBmuWIBPU2XmBDdGmIEAAkg2lAo-w?e=stRoK8)

### Dataset Structure

The dataset is organized as follows:

```
/data2/wang_tong/proj_cirseg/COR_Bench_V1
├── csv_data
└── dataset
    ├── Test_1
    │   ├── image
    │   └── mask
    │       ├── 1q0n
    │       ├── 1q1n
    │       ├── 1q2n
    │       ├── 2q0n
    │       ├── 2q1n
    │       ├── 3q0n
    │       └── sup
    ├── Test_2
    │   ├── image
    │   └── mask
    │       ├── 1q0n
    │       ├── 1q1n
    │       ├── 1q2n
    │       ├── 2q0n
    │       ├── 2q1n
    │       ├── 3q0n
    │       └── sup
    └── Train
        ├── image
        └── mask
            ├── 1q0n
            ├── 1q1n
            ├── 1q2n
            ├── 2q0n
            ├── 2q1n
            ├── 3q0n
            └── sup
```


## 🏫 Baseline Model

The checkpoint for our CORE model is available at:

- From [Hugging Face](https://huggingface.co/TongWang-NJ/CORE_COR_Bench_V1)
- From [OneDrive](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/tong_wang_mbzuai_ac_ae/Er1V5c9G9EtAnQERvFQur_4Brn8M81rYtSuVNuerUIaWbw)


## 🛠 Training

Training is performed using the `accelerate` library. The configuration file is located at `train_config_m3.yaml`. To start training, run the following command:

```bash
accelerate launch \
  --config_file your_model_path/config/train_config/a_cfg.yaml \
  your_model_path/my_train_a.py \
  --config your_model_path/config/train_config/train_config_m3.yaml
```

Ensure that `your_model_path` is replaced with the actual path to your model directory.


## 📊 Citation

If this codebase is useful to you, please consider citing:

```bibtex
@article{wang2025cor,
  title={Composed Object Retrieval: Object-level Retrieval via Composed Expressions},
  author={Tong Wang and Guanyu Yang and Nian Liu and Zongyan Han and Jinxing Zhou and Salman Khan and Fahad Shahbaz Khan},
  journal={arXiv preprint arXiv:2508.04424},
  year={2025},
  url={https://arxiv.org/abs/2508.04424},
}
```


## 📝 Acknowledgements

We would like to thank all contributors, funding sources, and supporters who made this project possible. Specific acknowledgments will be updated soon.
