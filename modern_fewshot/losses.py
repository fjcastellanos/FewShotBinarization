from __future__ import annotations

import torch
import torch.nn.functional as F


def partial_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    use_masking: bool,
) -> torch.Tensor:
    """Pixel-wise cross entropy with an explicit supervision mask.

    No magic target value (255, -100, ...) is used.  ``valid_mask`` is the
    single source of truth for whether a pixel is annotated.

    Args:
        logits: [B, C, H, W]
        target: [B, H, W], class indices. Unknown pixels may contain 0 because
            they are ignored when ``use_masking`` is True.
        valid_mask: [B, H, W], 1/True for annotated pixels.
        use_masking: If False, every target pixel contributes to the loss.
            This deliberately reproduces the "unknown becomes background"
            baseline used in the ablation.
    """
    if logits.ndim != 4:
        raise ValueError(f"Expected logits [B,C,H,W], got {tuple(logits.shape)}")
    if target.ndim != 3 or valid_mask.ndim != 3:
        raise ValueError("target and valid_mask must have shape [B,H,W]")

    loss_map = F.cross_entropy(logits, target.long(), reduction="none")

    if not use_masking:
        return loss_map.mean()

    valid = valid_mask.to(dtype=loss_map.dtype)
    denom = valid.sum()
    if denom.item() == 0:
        # Keep a differentiable zero in the graph. The sampler normally avoids
        # this case for masked training, but this makes the loss robust.
        return logits.sum() * 0.0
    return (loss_map * valid).sum() / denom
