import torch
import open_clip
from functools import partial

from lib.sam_model.image_encoder import ImageEncoderViT
from lib.sam_model.mask_decoder import MaskDecoder
from lib.sam_model.my_prompt_encoder import PromptEncoder
from lib.sam_model.transformer import TwoWayTransformer

from lib.sam_with_sup_branch import CirSegModelWithQuerySupportFeat
from lib.support_branch import SupportBranch


def build_model_with_query_support_feat(
    sam_model="sam_base",
    siglip_model="ViT-SO400M-14-SigLIP-384",
    sam_checkpoint_path=None,
    siglip_checkpoint_path=None,
    mask_pooling="MaskedPooling",
):
    """
    Build a model based on SAM (Segment Anything Model) and only load image_encoder and mask_decoder pretrained parameters.
    ** SupportBranch provides sparse prompt embedding
    Arguments:
      sam_checkpoint_path (str, optional): SAM pretrained weights path.
      siglip_checkpoint_path (str, optional): SigLIP pretrained weights path.

    Returns:
      SamWithSupBranch: Configured SAM model instance.
    """
    if sam_model == "sam_large":
        # ViT-L specific parameters
        encoder_embed_dim = 1024
        encoder_depth = 24
        encoder_num_heads = 16
        encoder_global_attn_indexes = [5, 11, 17, 23]
    elif sam_model == "sam_base":
        # ViT-B specific parameters
        encoder_embed_dim = 768
        encoder_depth = 12
        encoder_num_heads = 12
        encoder_global_attn_indexes = [2, 5, 8, 11]
    elif sam_model == "sam_huge":
        encoder_embed_dim = 1280
        encoder_depth = 32
        encoder_num_heads = 16
        encoder_global_attn_indexes = [7, 15, 23, 31]
    else:
        raise ValueError(f"Invalid SAM model: {sam_model}")

    image_size = 1024
    prompt_embed_dim = 256
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size  # 64 * 64

    # Initialize model
    model = CirSegModelWithQuerySupportFeat(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),  # Keep this part
        support_branch=SupportBranch(clip_model=siglip_model, siglip_path=siglip_checkpoint_path, mask_pooling=mask_pooling),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),  # 64 * 64
            # input_image_size=(image_size, image_size),  # 1024 * 1024
            # mask_in_chans=16,
        ),  # Only keep dense embedding for this part
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )

    # Load SAM pretrained parameters
    if sam_checkpoint_path is not None:
        state_dict = torch.load(sam_checkpoint_path)

        # Filter to keep only image_encoder and mask_decoder
        image_encoder_state_dict = {k.replace("image_encoder.", ""): v for k, v in state_dict.items() if k.startswith("image_encoder.")}
        mask_decoder_state_dict = {k.replace("mask_decoder.", ""): v for k, v in state_dict.items() if k.startswith("mask_decoder.")}
        prompt_encoder_state_dict = {
            k.replace("prompt_encoder.", ""): v for k, v in state_dict.items() if k.startswith("prompt_encoder.dense_embedding")
        }

        # Load SAM state_dict
        model.image_encoder.load_state_dict(image_encoder_state_dict, strict=False)
        model.mask_decoder.load_state_dict(mask_decoder_state_dict, strict=False)
        model.prompt_encoder.load_state_dict(prompt_encoder_state_dict, strict=False)
        print(f"Load SAM Checkpoint: {sam_checkpoint_path}.")

        # Freeze SigLIP and SAM encoder parameters
        model.support_branch.siglip.freeze()
        model.image_encoder.freeze()  # Freeze image_encoder parameters
        print("Freeze weight of SAM_Image_Encoder and SigLIP.")

        # Freeze mask_decoder.iou_prediction_head parameters
        for param in model.mask_decoder.iou_prediction_head.parameters():
            param.requires_grad = False
        print("Freeze weight of mask_decoder.iou_prediction_head.")

    return model
