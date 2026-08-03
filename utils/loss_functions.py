"""Loss functions for segmentation.

Provides:
- standard_bce: Plain cross-entropy (baseline)
- weighted_bce_dice: Inverse-frequency weighted CE + Dice loss (proposed)
"""

import torch
import torch.nn.functional as F
from .dice_score import dice_loss


def standard_bce(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Standard Binary Cross-Entropy (CrossEntropyLoss for 2-class).

    Args:
        pred: (B, C, H, W) logits — C=2 for binary
        target: (B, H, W) class indices in {0, 1}
    """
    return F.cross_entropy(pred, target)


def weighted_bce_dice(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight_clamp: tuple = (0.3, 3.0),
    dice_weight: float = 1.0,
) -> torch.Tensor:
    """
    Weighted BCE + Dice Loss with inverse frequency class weighting.

    Args:
        pred: (B, C, H, W) logits — C=2 for binary
        target: (B, H, W) class indices in {0, 1}
        weight_clamp: (min, max) clamp for inverse frequency weights
        dice_weight: multiplier for Dice loss component
    """
    # Inverse frequency class weights
    n_fg = (target == 1).sum().float()
    n_bg = (target == 0).sum().float()
    total = n_fg + n_bg

    w_fg = (total / (2 * n_fg + 1e-6)).clamp(*weight_clamp)
    w_bg = (total / (2 * n_bg + 1e-6)).clamp(*weight_clamp)
    weight = torch.tensor([w_bg, w_fg], device=pred.device, dtype=pred.dtype)

    # Weighted Cross-Entropy
    bce = F.cross_entropy(pred, target, weight=weight)

    # Dice Loss on foreground probability
    prob = F.softmax(pred, dim=1)[:, 1]  # (B, H, W) — foreground prob
    dice = dice_loss(prob, target.float(), multiclass=False)

    return bce + dice_weight * dice


# Registry for easy lookup by name
LOSS_REGISTRY = {
    "standard_bce": standard_bce,
    "weighted_bce_dice": weighted_bce_dice,
}


def get_loss_fn(name: str):
    """Get loss function by name."""
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss: {name}. Available: {list(LOSS_REGISTRY.keys())}")
    return LOSS_REGISTRY[name]