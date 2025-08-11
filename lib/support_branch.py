import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip

from lib.support_model.siglip_openclip import SigLIP
from lib.support_model.cir_feature_fuse import CirFuseModule

# from lib.support_model.cir_feature_fuse import CirFuseModuleV3
from lib.support_model.mask_adapter import MaskAdapterPooling, MaskedPooling, LayerNorm


class SupportBranch(nn.Module):

    def __init__(self, clip_model: str, siglip_path: str, mask_pooling: str = "MaskedPooling"):
        super().__init__()

        self.siglip = SigLIP(clip_model, siglip_path)
        if clip_model == "ViT-SO400M-14-SigLIP-384":
            siglip_dim = 1152
        elif clip_model == "ViT-B-16-SigLIP-384" or clip_model == "ViT-B-16-SigLIP2-384":
            siglip_dim = 768
        elif clip_model == "ViT-L-16-SigLIP-384" or clip_model == "ViT-L-16-SigLIP2-384":
            siglip_dim = 1024
        else:
            raise ValueError(f"Invalid SigLIP model: {clip_model}")

        self.siglip_dim = siglip_dim
        if mask_pooling == "MaskAdapterPooling":
            self.mask_pooling = MaskAdapterPooling(
                x_in_channel=self.siglip_dim,
                mask_adatpet_network_in_channel=512,  # 256 or 512
                mask_downscaling_mid_channel=16,
                mask_adatpet_network_mid_channel=256,  # 256
                num_output_maps=8,
            )
        elif mask_pooling == "MaskedPooling":
            self.mask_pooling = MaskedPooling()
        else:
            raise ValueError(f"Invalid mask pooling method: {mask_pooling}")

        # Define Combiner module
        self.cir_fuse = CirFuseModule(image_embed_dim=self.siglip_dim, text_embed_dim=self.siglip_dim)

        self.ln_channel_first = LayerNorm(normalized_shape=self.siglip_dim, eps=1e-6, data_format="channels_first")
        self.ln_channel_last = LayerNorm(normalized_shape=self.siglip_dim, eps=1e-6, data_format="channels_last")
        self.dim_proj = nn.Sequential(
            nn.Linear(self.siglip_dim, 512),
            nn.GELU(),
            nn.Dropout(0.8),  # Dropout probability set to 0.5, can be adjusted as needed
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.8),
        )

    def forward(self, support_input, change_text, mask_input):
        image_feat, text_feat, _, image_last_hidden_states_nchw = self.siglip(support_input, change_text)
        image_last_hidden_states_nchw = self.ln_channel_first(image_last_hidden_states_nchw)
        support_feat = self.mask_pooling(image_last_hidden_states_nchw, mask_input)  # [N, 1, 1152]
        support_feat = self.ln_channel_last(support_feat)
        support_feat = support_feat.squeeze(1)  # [N, D]
        text_feat = text_feat.squeeze(1)  # [N, D]

        # Method 1: Based on Combiner module
        fused_feat = self.cir_fuse.compose_img_text(support_feat, text_feat)
        fused_feat = fused_feat["repres"]  # Extract fused representation

        # # Method 2: Direct fusion of image and text features using torch.add
        # fused_feat = torch.add(support_feat, text_feat)  # [N, D]

        # # Method 3 (Ablation experiment 1): Remove Text branch (DGX27)
        # fused_feat = support_feat  # [N, D]

        # # Method 4 (Ablation experiment 2): Remove Mask branch (DGX28)
        # fused_feat = self.cir_fuse.compose_img_text(image_feat, text_feat)
        # fused_feat = fused_feat["repres"]  # Extract fused representation

        # # Method 5 (Ablation experiment 3): Only use image features, remove Mask and Text branches (DGX29)
        # fused_feat = image_feat  # [N, D]

        # # Method 6 (Ablation experiment 4): Only use text features, remove Mask and Image branches (DGX30)
        # fused_feat = text_feat  # [N, D]

        # Map to the same dimension as SAM's Decoder prompt features through 2-layer Linear
        comb_sup_feat = F.normalize(self.dim_proj(fused_feat), p=2, dim=-1)
        comb_sup_feat = comb_sup_feat.unsqueeze(1)  # [N, 1, 256]
        return comb_sup_feat
