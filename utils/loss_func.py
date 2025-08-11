import torch
import torch.nn.functional as F


def wbce_with_wiou_loss(pred, mask, w1=1.0, w2=1.0):
    """
    Args:
        pred (torch.Tensor): Predicted logits, shape [N, C, H, W]
        mask (torch.Tensor): Ground truth mask, shape [N, C, H, W], values in [0, 1]

        w1 (float): Weight for weighted BCE
        w2 (float): Weight for weighted IoU

    Returns:
        torch.Tensor: Total loss
    """
    # Calculate edge weights
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

    # Weighted BCE loss
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    # Weighted IoU loss
    pred_prob = torch.sigmoid(pred)  # Convert logits to probabilities
    inter = ((pred_prob * mask) * weit).sum(dim=(2, 3))
    union = ((pred_prob + mask) * weit).sum(dim=(2, 3)) - inter
    wiou = 1 - (inter + 1e-6) / (union + 1e-6)  # Avoid division by zero

    # Combined loss
    total_loss = w1 * wbce + w2 * wiou
    return total_loss.mean()


def mask_pooling(embeddings, mask):
    """
    Perform mask pooling on embeddings to generate features with dimension [B, 1, C].

    Args:
        embeddings (torch.Tensor): Input features, shape [B, C, H, W].
        mask (torch.Tensor): Input mask, shape [B, 1, H, W], values in [0, 1].

    Returns:
        torch.Tensor: Pooled features, shape [B, 1, C].
    """
    if mask.shape[2:] != embeddings.shape[2:]:
        mask = F.interpolate(mask, size=embeddings.shape[2:], mode="bilinear", align_corners=False)

    mask = mask.clamp(min=0, max=1)
    pooled_features = embeddings * mask
    mask_sum = mask.sum((2, 3)) + 1e-8
    pooled_features = pooled_features.sum((2, 3)) / mask_sum
    pooled_features = F.normalize(pooled_features, p=2, dim=-1)  # L2 normalization
    pooled_features = pooled_features.unsqueeze(1)

    return pooled_features


def fg_feat_similarity_loss(query_image_embeddings, comb_support_feat, query_mask):
    """
    Calculate feature loss, compute foreground feature cosine similarity loss only for non-empty samples (query_mask not all zeros).
    Empty samples (query_mask all zeros) are skipped directly with loss = 0.

    Args:
        query_image_embeddings (torch.Tensor): Query image features, shape [B, C, H, W].
        comb_support_feat (torch.Tensor): Support branch features, shape [B, 1, C], normalized.
        query_mask (torch.Tensor): Query mask, shape [B, 1, H, W], values in [0, 1].

    Returns:
        torch.Tensor: Feature loss (scalar), based only on non-empty samples.
    """
    # Detect non-empty samples
    mask_sum = query_mask.sum(dim=(1, 2, 3))  # [B]
    valid = mask_sum > 0  # [B]

    if not valid.any():
        return torch.tensor(0.0, device=query_image_embeddings.device)

    # Perform masked pooling and loss calculation only for non-empty samples
    query_feat = mask_pooling(query_image_embeddings[valid], query_mask[valid])  # [V, 1, C], V is number of non-empty samples
    support_feat = comb_support_feat[valid]  # [V, 1, C]

    # Calculate cosine similarity loss
    loss = 1 - F.cosine_similarity(query_feat, support_feat, dim=-1).mean()
    return loss


def bg_feat_similarity_loss(query_image_embeddings, comb_support_feat, query_mask):
    """
    Calculate background feature loss, compute background feature cosine similarity loss with foreground features for valid background samples (1 - query_mask not all zeros).
    Goal: Minimize similarity between background features and support branch foreground features (cosine similarity approaching -1).
    All-foreground samples (1 - query_mask all zeros) are skipped with loss = 0.

    Args:
        query_image_embeddings (torch.Tensor): Query image features, shape [B, C, H, W].
        comb_support_feat (torch.Tensor): Support branch features, shape [B, 1, C], normalized.
        query_mask (torch.Tensor): Query mask, shape [B, 1, H, W], values in [0, 1].

    Returns:
        torch.Tensor: Background feature loss (scalar), based on valid background samples.
    """
    # Calculate background mask
    bg_mask = 1 - query_mask  # [B, 1, H, W]

    # Detect valid background samples (background mask not all zeros)
    bg_mask_sum = bg_mask.sum(dim=(1, 2, 3))  # [B]
    valid = bg_mask_sum > 0  # [B]

    if not valid.any():
        return torch.tensor(0.0, device=query_image_embeddings.device)

        # No valid background samples, return loss = 0
    if not valid.any():
        return torch.tensor(0.0, device=query_image_embeddings.device)

    # Calculate background features for valid samples only
    bg_feat = mask_pooling(query_image_embeddings[valid], bg_mask[valid])  # [n_valid, C]

    # Expand support features to match valid sample count
    support_feat = comb_support_feat[valid].squeeze(1)  # [n_valid, C]

    # Calculate cosine similarity and add 1, target is minimization
    similarity = F.cosine_similarity(bg_feat, support_feat, dim=1)  # [n_valid]
    loss = (similarity + 1).mean()

    return loss
