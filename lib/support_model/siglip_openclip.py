import torch
import torch.nn as nn
import open_clip
import torch.nn.functional as F


class SigLIP(nn.Module):
    def __init__(self, model_name: str = "ViT-SO400M-14-SigLIP-384", pretrained: str = None):
        super(SigLIP, self).__init__()

        # Load open_clip model and preprocessor
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        if pretrained is not None:
            print(f"Load SigLIP Checkpoint: {pretrained}.")
        self.text_tokenizer = open_clip.get_tokenizer(model_name)

    def freeze(self):
        """Freeze all parameters of the SigLIP model"""
        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def get_img_siglip_feature(self, image_inputs) -> tuple:
        with torch.no_grad():
            # Extract global image features
            image_features = self.model.encode_image(image_inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # Extract patch-level features
            vision_model = self.model.visual.trunk
            x = vision_model.patch_embed(image_inputs)
            x = vision_model.pos_drop(x + vision_model.pos_embed)
            for block in vision_model.blocks:
                x = block(x)
            image_last_hidden_states_nqc = vision_model.norm(x)

            # Convert to (N, C, H, W) format
            N, num_patches, hidden_size = image_last_hidden_states_nqc.shape
            H = W = int(num_patches**0.5)
            image_last_hidden_states_nchw = (
                image_last_hidden_states_nqc.view(N, H * W, hidden_size).permute(0, 2, 1).view(N, hidden_size, H, W)
            )

        return image_features, image_last_hidden_states_nqc, image_last_hidden_states_nchw

    @torch.no_grad()
    def get_text_siglip_feature(self, text_tokens, normalize=True) -> torch.Tensor:
        self.eval()
        # Tokenization processing
        # text_tokens = self.text_tokenizer(text_list)
        # print("text_tokens shape:", text_tokens.shape)  # text_tokens shape: torch.Size([2, 64])

        text_features = self.model.encode_text(text_tokens)

        if normalize:
            text_features = F.normalize(text_features, dim=-1)
        # print("text_embs final:", text_features.shape)  # text_embs final: torch.Size([2, 1152])

        return text_features

    def forward(self, image_inputs, text_list) -> tuple:
        image_features, image_last_hidden_states_nqc, image_last_hidden_states_nchw = self.get_img_siglip_feature(image_inputs)
        text_features = self.get_text_siglip_feature(text_list)
        return image_features, text_features, image_last_hidden_states_nqc, image_last_hidden_states_nchw


if __name__ == "__main__":
    # Initialize model
    siglip_model = SigLIP(
        # pretrained="/l/users/tong.wang/wt_segmentation/parameter/open_clip_ViT-SO400M-14-SigLIP-384.bin",
        pretrained=None,
        model_name="ViT-SO400M-14-SigLIP-384",
    )

    # Test with random input
    image_input = torch.randn(2, 3, 384, 384)

    # Prepare text input
    text_list = ["Change color from white to blue", "Another random text"]
    siglip_text_tokenizer = open_clip.get_tokenizer("ViT-SO400M-14-SigLIP-384")
    text_tokens = siglip_text_tokenizer(text_list)
    print("Text tokens dtype:", text_tokens.dtype)
    print("Text tokens shape:", text_tokens.shape)  # [2, 64]

    # Forward pass
    image_features, text_features, image_last_hidden_states_nqc, image_last_hidden_states_nchw = siglip_model(image_input, text_tokens)

    # Print output shapes
    print("Image features shape:", image_features.shape)
    print("Text features shape:", text_features.shape)
    print("Patch-level features shape:", image_last_hidden_states_nqc.shape)
    print("Image-level features shape:", image_last_hidden_states_nchw.shape)
    """
    ViT-SO400M-14-SigLIP-384: 
    Image features shape: torch.Size([2, 1152])
    Text features shape: torch.Size([2, 1152])
    Patch-level features shape: torch.Size([2, 729, 1152])
    Image-level features shape: torch.Size([2, 1152, 27, 27])

    ViT-B-16-SigLIP-384
    Final output shape: torch.Size([1, 576, 768])

    ViT-B-16-SigLIP2-384
    Final output shape: torch.Size([1, 576, 768])
    
    ViT-L-16-SigLIP-384
    Final output shape: torch.Size([1, 576, 1024])

    ViT-L-16-SigLIP2-384
    Final output shape: torch.Size([1, 576, 1024])
    """
