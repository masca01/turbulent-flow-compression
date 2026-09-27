"""Original beta-VAE, vendored unchanged from the MMAE 597 project.

Architecture and stochastic loss copied from rom/vae.py to make this repository
self-contained. State-dict keys remain compatible with all existing checkpoints.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def _n_conv(H, W):
    """Number of stride-2 conv layers."""
    return min(6, max(2, math.floor(math.log2(min(H, W))) - 2))


_CH = [8, 16, 32, 64, 128, 256]


class Encoder(nn.Module):
    def __init__(self, H, W, C_in, latent_dim):
        super().__init__()
        n = _n_conv(H, W)
        chs = _CH[:n]
        layers, in_ch = [], C_in
        for out_ch in chs:
            layers += [nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1), nn.ELU()]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        Hb = math.ceil(H / 2**n)
        Wb = math.ceil(W / 2**n)
        flat = Hb * Wb * chs[-1]
        self.fc   = nn.Sequential(nn.Linear(flat, 256), nn.ELU())
        self.mu   = nn.Linear(256, latent_dim)
        self.logv = nn.Linear(256, latent_dim)

    def forward(self, x):
        x = self.conv(x).flatten(1)
        h = self.fc(x)
        return self.mu(h), self.logv(h)


class Decoder(nn.Module):
    def __init__(self, H, W, C_out, latent_dim):
        super().__init__()
        n = _n_conv(H, W)
        chs = _CH[:n]
        self.H, self.W = H, W
        Hb = math.ceil(H / 2**n)
        Wb = math.ceil(W / 2**n)
        ch_b = chs[-1]
        self.Hb, self.Wb, self.ch_b = Hb, Wb, ch_b

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, Hb * Wb * ch_b), nn.ELU())

        rev = list(reversed(chs))          # e.g. [256, 128, 64, 32, 16, 8]
        layers, in_ch = [], rev[0]
        for out_ch in rev[1:]:
            layers += [nn.ConvTranspose2d(in_ch, out_ch, 3, stride=2,
                                          padding=1, output_padding=1), nn.ELU()]
            in_ch = out_ch
        layers += [nn.ConvTranspose2d(in_ch, in_ch, 3, stride=2,
                                      padding=1, output_padding=1), nn.ELU(),
                   nn.Conv2d(in_ch, C_out, 3, padding=1)]
        self.deconv = nn.Sequential(*layers)

    def forward(self, z):
        x = self.fc(z).view(-1, self.ch_b, self.Hb, self.Wb)
        x = self.deconv(x)
        # center-crop to target spatial size
        h, w = x.shape[2], x.shape[3]
        r0 = (h - self.H) // 2
        c0 = (w - self.W) // 2
        return x[:, :, r0:r0+self.H, c0:c0+self.W]


def vae_loss(enc, dec, x, beta):
    mu, logv = enc(x)
    logv = torch.clamp(logv, -10, 10)
    z = mu + torch.exp(0.5 * logv) * torch.randn_like(mu)
    xhat = dec(z)
    recon = F.mse_loss(xhat, x)
    kl = (-0.5 * (1 + logv - mu.pow(2) - logv.exp())).sum(1).mean()
    return recon + beta * kl, recon, kl
