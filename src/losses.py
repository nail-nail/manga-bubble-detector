"""BCE + Dice loss.

Why both: bubble pixels are a small minority of a page.

* BCE averages over every pixel, so background dominates it and predicting
  "no bubble" everywhere already gives a fairly low loss.
* Dice measures overlap with the foreground, so that lazy answer scores badly
  regardless of how much background there is.

BCE supplies smooth, well-behaved gradients. Dice supplies class balance.
"""

from __future__ import annotations

import torch
from torch import nn


class DiceLoss(nn.Module):
    """Soft Dice loss, computed per image and then averaged over the batch."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).flatten(1)
        targets = targets.flatten(1)
        intersection = (probs * targets).sum(dim=1)
        total = probs.sum(dim=1) + targets.sum(dim=1)
        # `smooth` avoids 0/0 on pages with no bubbles. Those pages then have a
        # loss near 0 when the prediction is empty and a growing loss otherwise.
        dice = (2.0 * intersection + self.smooth) / (total + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(
        self, bce_weight: float = 1.0, dice_weight: float = 1.0, smooth: float = 1.0
    ) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(logits, targets) + self.dice_weight * self.dice(
            logits, targets
        )
