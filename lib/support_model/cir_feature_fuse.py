# For fusing Text and refer image representations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# Multimodal fusion module
class CirFuseModule(nn.Module):
    """
    For fusing Text and refer image representations
    Args:
        image_embed_dim: Image representation dimension
        text_embed_dim: Text representation dimension
    Returns:
        N * D
    """

    def __init__(self, image_embed_dim, text_embed_dim):
        super().__init__()
        self.atten_Image = nn.Sequential(
            nn.Linear(image_embed_dim + text_embed_dim, image_embed_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(image_embed_dim, image_embed_dim),
            nn.Sigmoid(),
        )
        self.atten_Text = nn.Sequential(
            nn.Linear(image_embed_dim + text_embed_dim, text_embed_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(text_embed_dim, text_embed_dim),
            nn.Sigmoid(),
        )
        self.dynamic_scalar = nn.Sequential(
            nn.Linear(image_embed_dim + text_embed_dim, image_embed_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(image_embed_dim, 1),
            nn.Sigmoid(),
        )

    def compose_img_text(self, image_features, text_features):
        """
        image_features: N x D
        text_features: N x D
        """
        raw_combined_features = torch.cat((image_features, text_features), -1)
        atten_I = self.atten_Image(raw_combined_features)
        atten_T = self.atten_Text(raw_combined_features)
        image_features = atten_I * image_features
        text_features = atten_T * text_features
        new_combined_features = torch.cat((image_features, text_features), -1)
        dynamic = self.dynamic_scalar(new_combined_features)
        com_fearues = dynamic * image_features + (1 - dynamic) * text_features

        representations = {
            "repres": F.normalize(com_fearues),  # self.normalization_layer(com_fearues),
            "fuseimg": image_features,
            "fusetxt": text_features,
            "dynamic_scalar": dynamic,
        }
        return representations
