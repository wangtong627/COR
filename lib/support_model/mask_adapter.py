import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class MaskedPooling(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, clip_feature: torch.Tensor, mask: torch.Tensor):
        """
        clip_feature: [B, C, H, W]
        mask: [B, 1, H, W]
        pooled_clip_feature: [B, 1, C]
        """
        if mask.shape[2:] != clip_feature.shape[2:]:
            mask = F.interpolate(mask, size=clip_feature.shape[2:], mode="bilinear", align_corners=False)

        pooled_clip_feature = clip_feature * mask  # [B, C, H, W]
        pooled_clip_feature = pooled_clip_feature.sum((2, 3)) / (mask.sum((2, 3)) + 1e-8)
        pooled_clip_feature = pooled_clip_feature.squeeze(1)
        return pooled_clip_feature


class MaskAdapterPooling(nn.Module):

    def __init__(
        self,
        x_in_channel=1152,
        mask_adatpet_network_in_channel=256,
        mask_downscaling_mid_channel=16,
        mask_adatpet_network_mid_channel=256,
        num_output_maps=16,
    ):
        super().__init__()
        self.channel_clip_to_maskadapter = ChannelReduction(
            in_channel=x_in_channel, out_channel=mask_adatpet_network_in_channel
        )

        self.get_mask_map = GenerateMaskAdapterMap(
            clip_in_channel=mask_adatpet_network_in_channel,
            mask_downscaling_mid_channel=mask_downscaling_mid_channel,
            mid_channel=mask_adatpet_network_mid_channel,
            num_output_maps=num_output_maps,
        )

        self.num_output_maps = num_output_maps

    def forward(self, clip_feature, mask):
        """
        x.shape = (B, C, H, W)
        mask.shape = (B, 1, H, W)
        """
        if mask.shape[-2:] != clip_feature.shape[-2:]:
            mask = F.interpolate(mask, size=clip_feature.shape[-2:], mode="bilinear", align_corners=False)
        clip_vis_dense = self.channel_clip_to_maskadapter(clip_feature)
        semantic_activation_maps = self.get_mask_map(clip_vis_dense, mask)  # (n 16 h w)
        # if semantic_activation_maps.shape[-2:] != clip_feature.shape[-2:]:
        maps_for_pooling = F.interpolate(
            semantic_activation_maps,
            size=clip_feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        B, C = clip_feature.size(0), clip_feature.size(1)
        N = maps_for_pooling.size(1)
        num_instances = N // self.num_output_maps  # 16/16 = 1
        maps_for_pooling = F.softmax(F.logsigmoid(maps_for_pooling).view(B, N, -1), dim=-1)  # B 16 P
        pooled_clip_feature = torch.bmm(
            maps_for_pooling,
            clip_feature.view(B, C, -1).permute(0, 2, 1),  # B 16 P @ B P C = B 16 C
        )
        # get average
        pooled_clip_feature = (
            pooled_clip_feature.reshape(B, num_instances, self.num_output_maps, -1).mean(dim=-2).contiguous()
        )
        return pooled_clip_feature


class ChannelReduction(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, 1)
        self.norm = LayerNorm(out_channel, data_format="channels_first")
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class GenerateMaskAdapterMap(nn.Module):

    def __init__(
        self,
        clip_in_channel=768,
        mask_downscaling_mid_channel=16,
        mid_channel=768,
        num_output_maps=16,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            mask_downscaling_mid_channel: Number of channels in the mask downscaling network.
            mid_channel: Number of channels in the ConvNeXt blocks.
            num_output_maps: Number of output channels.
        """
        super().__init__()

        self.clip_in_channel = clip_in_channel

        self.fuse = nn.Conv2d(self.clip_in_channel, mid_channel, 1)

        self.cnext1 = ConvNextBlock(mid_channel)

        self.cnext2 = ConvNextBlock(mid_channel)

        self.cnext3 = ConvNextBlock(mid_channel)

        self.norm = LayerNorm(mid_channel, data_format="channels_last")
        self.final = nn.Conv2d(mid_channel, num_output_maps, 1)

        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_downscaling_mid_channel // 4, kernel_size=3, stride=2, padding=1),
            LayerNorm(mask_downscaling_mid_channel // 4, data_format="channels_first"),
            nn.GELU(),
            nn.Conv2d(
                mask_downscaling_mid_channel // 4,
                mask_downscaling_mid_channel,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            LayerNorm(mask_downscaling_mid_channel, data_format="channels_first"),
            nn.GELU(),
            nn.Conv2d(mask_downscaling_mid_channel, self.clip_in_channel, kernel_size=1),
        )

    def forward(self, clip_feature, masks):
        """
        Args:
            clip_feature (B C H W) C是clip_in_channel
            masks (B Q H W)  这边 Q 取 1
        Returns:
            outputs (B ? H W)  这里 ? 取 num_output_maps 后面会对 ? 取平均
        """
        N = masks.size(1)
        masks = rearrange(masks, "B N H W -> (B N) H W").unsqueeze(dim=1)

        clip_feature = repeat(clip_feature, "B C H W -> (B N) C H W", N=N)

        H, W = clip_feature.shape[-2:]
        masks = F.interpolate(masks.float(), size=(H * 4, W * 4), mode="bilinear", align_corners=False)
        masks = self.mask_downscaling(masks)

        outputs = clip_feature + masks

        outputs = self.fuse(outputs)

        outputs = self.cnext1(outputs)

        outputs = self.cnext2(outputs)

        outputs = self.cnext3(outputs)

        outputs = outputs.permute(0, 2, 3, 1)
        outputs = self.norm(outputs.contiguous())
        outputs = outputs.permute(0, 3, 1, 2)

        outputs = self.final(outputs.contiguous())

        outputs = rearrange(outputs, "(B N) C H W -> B (N C) H W", N=N)

        return outputs


class ConvNextBlock(nn.Module):
    """ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, kernel_size=7, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim
        )  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


if __name__ == "__main__":
    img_tensor = torch.randn(2, 1152, 27, 27)
    mask_tensor = torch.randn(2, 1, 27, 27)
    # model = MaskAdapterPooling(
    #     x_in_channel=1152,
    #     mask_adatpet_network_in_channel=256,
    #     mask_downscaling_mid_channel=16,
    #     mask_adatpet_network_mid_channel=256,
    #     num_output_maps=16,
    # )
    model = MaskAdapterPooling()
    out = model(img_tensor, mask_tensor)
    print(out.shape)
    """
    GenerateMaskAdapterMap: torch.Size([2, 16, 14, 14])表示生成了16个候选map
    Final output: torch.Size([2, 1, 1152])
    """
