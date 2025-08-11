<div align="center">
<h1> Composed Object Retrieval (COR) </h1>
<h3>Composed Object Retrieval: Object-level Retrieval via Composed Expressions</h3>

Tong Wang<sup>1,2</sup>, Guanyu Yang<sup>1,\*</sup>, Nian Liu<sup>2,3,\*</sup>, Zongyan Han<sup>2</sup>, Jinxing Zhou<sup>2</sup>, Salman Khan<sup>2</sup>, Fahad Shahbaz Khan<sup>2</sup>

<sup>1</sup> Southeast University, <sup>2</sup> Mohamed Bin Zayed University of Artificial Intelligence, <sup>3</sup> Northwestern Polytechnical University  
<small><span style="color:#E63946; font-weight:bold;">*</span> indicates corresponding authors</small>

[![License](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/BUAADreamer/CCRK/blob/main/licence)
[![arXiv](https://img.shields.io/badge/arXiv-2508.04424-red)](https://arxiv.org/abs/2508.04424)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97 Dataset-blue)](https://huggingface.co/datasets/TongWang-NJ/COR_Bench_V1)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97 Checkpoints-blue)](https://huggingface.co/TongWang-NJ/CORE_COR_Bench_V1)
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
You can download the COR127K dataset from the following links:
- [Hugging Face Dataset 🤗](https://huggingface.co/datasets/TongWang-NJ/COR_Bench_V1)
- [OneDrive Dataset](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/tong_wang_mbzuai_ac_ae/EgPAHh93bBVJq_s34RBmuWIBPU2XmBDdGmIEAAkg2lAo-w?e=stRoK8)

## 🏫 Baseline Model
The pre-trained weights for our CORE model are available at:
- [Hugging Face Model 🤗](https://huggingface.co/TongWang-NJ/CORE_COR_Bench_V1)
- [OneDrive Checkpoint](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/tong_wang_mbzuai_ac_ae/Er1V5c9G9EtAnQERvFQur_4Brn8M81rYtSuVNuerUIaWbw)


## 📊 Citation
If this codebase is useful to you, please consider citing:
```
@article{wang2025cor,
      title={Composed Object Retrieval: Object-level Retrieval via Composed Expressions}, 
      author={Tong Wang and Guanyu Yang and Nian Liu and Zongyan Han and Jinxing Zhou and Salman Khan and Fahad Shahbaz Khan},
      journal={arXiv preprint arXiv:2508.04424},
      year={2025},
      url={https://arxiv.org/abs/2508.04424}, 
}
```

## 📝 Acknowledgements
TODO: Acknowledge contributors, funding sources, and any other relevant support for the project.
