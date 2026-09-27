"""Challenge 1.1, iteration 1: residual beta-VAE, with no shared-project imports."""

import math
import torch
from torch import nn
from torch.nn import functional as F

ARCHITECTURE = "challenge11_residual_v1"
CHANNELS = (8, 16, 32, 64, 128)


class ResidualBlock(nn.Module):
    """Two stride-1 convolutions; a shortcut within one resolution only."""

    def __init__(self, channels):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.activation = nn.ELU()

    def forward(self, x):
        return self.activation(x + self.branch(x))


class Encoder(nn.Module):
    def __init__(self, H, W, C_in, latent_dim):
        super().__init__()
        layers = []
        in_ch = C_in
        for out_ch in CHANNELS:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
                nn.ELU(),
                ResidualBlock(out_ch),
            ])
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        flat = math.ceil(H / 32) * math.ceil(W / 32) * CHANNELS[-1]
        self.fc = nn.Sequential(nn.Linear(flat, 256), nn.ELU())
        self.mu = nn.Linear(256, latent_dim)
        self.logv = nn.Linear(256, latent_dim)

    def forward(self, x):
        h = self.fc(self.conv(x).flatten(1))
        return self.mu(h), self.logv(h)


class Decoder(nn.Module):
    def __init__(self, H, W, C_out, latent_dim):
        super().__init__()
        self.H, self.W = H, W
        self.Hb, self.Wb, self.ch_b = math.ceil(H / 32), math.ceil(W / 32), CHANNELS[-1]
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, self.Hb * self.Wb * self.ch_b), nn.ELU())
        layers = []
        # Process each resolution BEFORE upsampling. All information comes from z.
        in_ch = self.ch_b
        for out_ch in (64, 32, 16, 8, 8):
            layers.extend([
                ResidualBlock(in_ch),
                nn.ConvTranspose2d(in_ch, out_ch, 3, stride=2,
                                   padding=1, output_padding=1),
                nn.ELU(),
            ])
            in_ch = out_ch
        layers.append(nn.Conv2d(in_ch, C_out, 3, padding=1))
        self.deconv = nn.Sequential(*layers)

    def forward(self, z):
        x = self.fc(z).view(-1, self.ch_b, self.Hb, self.Wb)
        x = self.deconv(x)
        r0 = (x.shape[2] - self.H) // 2
        c0 = (x.shape[3] - self.W) // 2
        return x[:, :, r0:r0 + self.H, c0:c0 + self.W]


def vae_loss(enc, dec, x, beta):
    """Same stochastic training objective as the baseline, including at beta=0."""
    mu, logv = enc(x)
    logv = torch.clamp(logv, -10, 10)
    z = mu + torch.exp(0.5 * logv) * torch.randn_like(mu)
    recon = F.mse_loss(dec(z), x)
    kl = (-0.5 * (1 + logv - mu.pow(2) - logv.exp())).sum(1).mean()
    return recon + beta * kl, recon, kl
