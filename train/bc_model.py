"""The same small CNN architecture as the original Mikey17 VanillaBC."""

import torch
from torch import nn


class VanillaBC(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(128 * 2 * 2 + 6, 128), nn.ReLU(),
            nn.Linear(128, 6),
        )

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.vision(image), state], dim=1))
