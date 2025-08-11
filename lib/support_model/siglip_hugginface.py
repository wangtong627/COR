import os
import torch
from transformers import AutoModel, AutoImageProcessor, AutoTokenizer


class SigLIP(torch.nn.Module):
    def __init__(self, model_name: str):
        super().__init__()

        # 加载模型
        self.model = AutoModel.from_pretrained(model_name)

    # def __call__(self, image_inputs, text_inputs) -> tuple:

    #     return self.forward(image_inputs, text_inputs)

    def get_img_siglip_feature(self, image_inputs) -> tuple:

        with torch.no_grad():
            image_features = self.model.get_image_features(**image_inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # 提取 patch-level 特征
            vision_outputs = self.model.vision_model(**image_inputs, output_hidden_states=True)
            # shape: [batch_size, num_patches, hidden_size] 第27层（最后一层）的输出
            image_last_hidden_states_nqc = vision_outputs.hidden_states[-1]
            # print("Number of hidden states:", len(vision_outputs.hidden_states))  # 应为 28
            # print("Last hidden state shape:", vision_outputs.hidden_states[-1].shape)  # [2, 729, 1152]
            # 转换为 (N, C, H, W) 格式
            N, num_patches, hidden_size = image_last_hidden_states_nqc.shape
            H = W = int(num_patches**0.5)
            image_last_hidden_states_nchw = (
                image_last_hidden_states_nqc.view(N, H * W, hidden_size)
                .permute(0, 2, 1)
                .view(N, hidden_size, H, W)
            )

        return image_features, image_last_hidden_states_nqc, image_last_hidden_states_nchw

    def get_text_siglip_feature(self, text_inputs) -> torch.Tensor:

        with torch.no_grad():
            text_features = self.model.get_text_features(**text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def forward(self, image_inputs, text_inputs) -> tuple:

        image_features, image_last_hidden_states_nqc, image_last_hidden_states_nchw = (
            self.get_img_siglip_feature(image_inputs)
        )
        text_features = self.get_text_siglip_feature(text_inputs)

        return image_features, text_features, image_last_hidden_states_nqc, image_last_hidden_states_nchw


# 测试代码
if __name__ == "__main__":
    # 初始化模型和工具
    siglip_model_path = "/l/users/tong.wang/wt_segmentation/parameter/siglip-so400m-patch14-384"
    siglip_model = SigLIP(siglip_model_path)

    # 准备图像输入
    # 这里如果给PIL图像，需要使用 AutoImageProcessor 处理
    image_inputs = {"pixel_values": torch.randn(2, 3, 384, 384)}  # 批量大小为 2

    # 准备文本输入
    text_list = ["Change color from white to blue", "Another random text"]
    tokenizer = AutoTokenizer.from_pretrained(siglip_model_path)
    text_inputs = tokenizer(text_list, return_tensors="pt", padding=True, truncation=True)

    # 使用随机张量测试，直接调用实例
    image_features, text_features, image_last_hidden_states_nqc, image_last_hidden_states_nchw = (
        siglip_model(image_inputs, text_inputs)
    )

    print("Image features shape:", image_features.shape)
    print("Text features shape:", text_features.shape)
    print("Patch-level features shape:", image_last_hidden_states_nqc.shape)
    print("image-level features shape:", image_last_hidden_states_nchw.shape)
    """
    Image features shape: torch.Size([2, 1152])
    Text features shape: torch.Size([2, 1152])
    Patch-level features shape: torch.Size([2, 729, 1152])
    image-level features shape: torch.Size([2, 1152, 27, 27])
    """
