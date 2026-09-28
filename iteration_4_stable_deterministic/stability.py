"""Numerical checks for the iteration-4 architecture."""
import torch
from torch.nn import functional as F

SATURATION_LIMIT = 0.95
LATENT_STD_FLOOR = 1e-6


class TrainingInstability(RuntimeError):
    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


def check_bottleneck(pre_activation, latent):
    # Small CPU reductions also make the diagnostic independent of MPS reductions.
    pre = pre_activation.detach().float().cpu()
    z = latent.detach().float().cpu()
    if not torch.isfinite(pre).all() or not torch.isfinite(z).all():
        raise TrainingInstability("Non-finite encoder activations")
    saturation = float((pre <= -20).float().mean(dim=1).max())
    spread = float(z.std(dim=0, unbiased=False).max()) if len(z) > 1 else None
    diagnostics = {"hidden_pre_abs_max": float(pre.abs().max()),
                   "max_hidden_saturation_fraction": saturation,
                   "max_latent_batch_std": spread}
    if saturation >= SATURATION_LIMIT:
        raise TrainingInstability("Dense ELU saturation: at least 95% of a snapshot's hidden units are saturated", diagnostics)
    if spread is not None and spread <= LATENT_STD_FLOOR:
        raise TrainingInstability("Latent collapse: different snapshots have effectively constant latent vectors", diagnostics)
    return diagnostics


def checked_loss(enc, dec, x):
    """Same forward/MSE as ae_loss, with checks before any optimiser update."""
    if not bool(torch.isfinite(x).all()):
        raise TrainingInstability("Non-finite input batch")
    features = enc.conv(x).flatten(1)
    raw = enc.fc[0](features)
    pre = raw
    for layer in enc.fc[1:-1]:
        pre = layer(pre)
    z = enc.latent(enc.fc[-1](pre))
    diagnostics = check_bottleneck(pre, z)
    diagnostics["dense_raw_abs_max"] = float(raw.detach().float().cpu().abs().max())
    loss = F.mse_loss(dec(z), x)
    if not bool(torch.isfinite(loss)):
        raise TrainingInstability("Non-finite reconstruction loss", diagnostics)
    return loss, diagnostics


def clip_gradients(parameters, max_norm):
    if not 0 < max_norm < float("inf"):
        raise ValueError("Gradient norm limit must be positive and finite")
    try:
        norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm,
                                             error_if_nonfinite=True, foreach=False)
    except RuntimeError as exc:
        raise TrainingInstability(f"Gradient clipping failed; optimiser update skipped: {exc}") from exc
    return float(norm)
