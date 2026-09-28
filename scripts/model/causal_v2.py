"""Position-local normalization and receptive-field planning for causal TCNs."""
import torch.nn as nn


class PositionwiseLayerNorm(nn.Module):
    """Normalize channels independently at each position of a (B, C, L) tensor."""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


def context_dilations(target_length, initial):
    """Extend the existing dilation schedule until the backbone covers the input."""
    dilations = list(initial)
    while 25 + 2 * sum(dilations) < target_length:
        dilations.append(2 * dilations[-1])
    return dilations
