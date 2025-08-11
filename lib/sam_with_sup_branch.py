# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple

from lib.sam_model.image_encoder import ImageEncoderViT
from lib.sam_model.mask_decoder import MaskDecoder
from lib.sam_model.my_prompt_encoder import PromptEncoder  # Modified
from lib.support_branch import SupportBranch  # Modified


class CirSegModelWithQuerySupportFeat(nn.Module):
    mask_threshold: float = 0.0
    # low_res_masks are raw logits without sigmoid or softmax, values can be any real numbers (positive or negative). Threshold 0.0 is the natural split point for logits
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,  # Modified
        support_branch: SupportBranch,  # Modified
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.support_branch = support_branch  # Add a multimodal prompt
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(
        self,
        query_image_inputs,
        support_image_inputs,
        change_text_inputs,
        support_mask_inputs,
        multimask_output=True,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Return values:
          (list(dict)): A list for input images, where each element is a dictionary containing the following keys.
            - `'masks'`: (`torch.Tensor`) Batch of binary mask predictions, shape BxCxHxW, where `B` is the number of input prompts, `C` is determined by `multimask_output`, and `(H, W)` is the original image size.
            - `'iou_predictions'`: (`torch.Tensor`) Model's prediction of mask quality, shape BxC.
            - `'low_res_logits'`: (`torch.Tensor`) Low-resolution logits, shape BxCxHxW, where `H=W=256`. Can be passed as mask input to subsequent prediction iterations.
        """
        # Get batch size
        B = query_image_inputs.shape[0]

        # image_encoder
        query_image_embeddings = self.image_encoder(query_image_inputs)  # [2, 256, 64, 64]

        # support branch
        comb_support_feat = self.support_branch(support_image_inputs, change_text_inputs, support_mask_inputs)  # [2, 1, 256]

        # Prompt encoding (batch generation of dense_embeddings)
        dense_embeddings = self.prompt_encoder(B)  # [B, 256, 64, 64]

        # Mask decoding (batch processing)

        low_res_masks, iou_predictions, _ = self.mask_decoder(
            image_embeddings=query_image_embeddings,  # [B, 256, 64, 64]
            image_pe=self.prompt_encoder.get_dense_pe(),  # [1, 256, 64, 64]
            sparse_prompt_embeddings=comb_support_feat,  # [B, 1, 256]
            dense_prompt_embeddings=dense_embeddings,  # [B, 256, 64, 64]
            multimask_output=multimask_output,
        )  # low_res_masks: [B, C, 256, 256], iou_predictions: [B, C]

        # Handle empty masks

        if multimask_output:
            # Select best mask
            best_mask_idx = iou_predictions.argmax(dim=1)  # [B]
            final_masks = low_res_masks[torch.arange(B), best_mask_idx]  # [B, 256, 256]
            final_masks = final_masks.unsqueeze(1)
        else:
            final_masks = low_res_masks
        # final_masks: [B, 1, 256, 256]; query_image_embeddings: [B, 256, 64, 64]; comb_support_feat: [B, 1, 256]
        return final_masks, query_image_embeddings, comb_support_feat
