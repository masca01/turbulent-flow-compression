"""CNN-AE described in Challenge 1.1, section II.C.2 of challengePaper.pdf.

Unspecified choices: same padding; decoder channels 32 -> 16 -> 8 -> 3;
ReLU after decoder expansion and both transposed convolutions, linear latent
and final output. These are reconstruction choices, not verified author code.
"""

import torch
from torch import nn
from torch.nn import functional as F

ARCHITECTURE = "challenge11_paper_cnn_v1"
CHANNELS = (16, 32)


def _check_grid(H, W):
    if H < 4 or W < 4 or H % 4 or W % 4:
        raise ValueError("The paper CNN implementation requires H and W divisible by 4.")


class Encoder(nn.Module):
    def __init__(self, H, W, C_in, latent_dim):
        super().__init__()
        _check_grid(H, W)
        self.conv = nn.Sequential(
            nn.Conv2d(C_in, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(32 * (H // 4) * (W // 4), latent_dim)

    def forward(self, x):
        return self.fc(self.conv(x).flatten(1))


class Decoder(nn.Module):
    def __init__(self, H, W, C_out, latent_dim):
        super().__init__()
        _check_grid(H, W)
        self.Hb, self.Wb = H // 4, W // 4
        self.fc = nn.Sequential(nn.Linear(latent_dim, 32 * self.Hb * self.Wb), nn.ReLU())
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(8, C_out, 3, padding=1),
        )

    def forward(self, z):
        return self.deconv(self.fc(z).view(-1, 32, self.Hb, self.Wb))


def ae_loss(enc, dec, x):
    """Deterministic reconstruction MSE; the latent is the encoder output."""
    return F.mse_loss(dec(enc(x)), x)
