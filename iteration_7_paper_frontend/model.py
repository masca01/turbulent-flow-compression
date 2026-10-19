"""Paper-inspired front end with the iteration-6 residual AE backbone.

Only the first two downsampling operations change. The challenge grid is
divisible by four, so pooling preserves the previous stage/bottleneck shapes.
This hybrid is not a reproduction of the paper's complete architecture.
"""

import math
from torch import nn
from iteration_4_stable_deterministic.model import (
    CHANNELS, Decoder, ResidualBlock, ae_loss,
)

ARCHITECTURE = "challenge11_paper_frontend_residual_v1"


class Encoder(nn.Module):
    def __init__(self, H, W, C_in, latent_dim):
        super().__init__()
        if H < 4 or W < 4 or H % 4 or W % 4:
            raise ValueError("Paper-front-end encoder requires H and W divisible by 4.")
        layers = []
        in_ch = C_in
        for stage, out_ch in enumerate(CHANNELS):
            if stage < 2:
                layers.extend([
                    nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2, stride=2),
                    ResidualBlock(out_ch),
                ])
            else:
                layers.extend([
                    nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
                    nn.ELU(),
                    ResidualBlock(out_ch),
                ])
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        flat = math.ceil(H / 32) * math.ceil(W / 32) * CHANNELS[-1]
        self.fc = nn.Sequential(
            nn.Linear(flat, 256),
            nn.LayerNorm(256, elementwise_affine=False),
            nn.ELU(),
        )
        self.latent = nn.Linear(256, latent_dim)

    def forward(self, x):
        return self.latent(self.fc(self.conv(x).flatten(1)))
