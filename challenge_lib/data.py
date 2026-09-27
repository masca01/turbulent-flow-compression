"""
Reader for the Challenge 1.1 training file (BL2 turbulent boundary layer, x-y planes).

Challenge1.1_train.h5
    Uplane, Vplane, Wplane   (5000, 148, 4000) float32   velocity FLUCTUATIONS u', v', w'
    x                        (1, 4000)  uniform streamwise grid   (units of L_o)
    y                        (1, 148)   stretched wall-normal grid (y = 0 is the wall)

The whole file is ~35 GB, too big to hold twice in RAM. The three datasets are stored
contiguously (no chunks, no compression), so they are opened as read-only np.memmap arrays:
nothing is loaded until a snapshot is indexed and the OS page cache keeps what fits.

Layout returned to the network: [N, C=3, H=148 (y), W=4000 (x)], C = (u, v, w).
"""

import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import h5py

from .paths import DATA_DIR
DATA_FILE = str(DATA_DIR / "BL2" / "Challenge1.1_train.h5")
GRID_FILE = str(DATA_DIR / "BL2" / "BLdns2_parameters.h5")
COMPS     = ("Uplane", "Vplane", "Wplane")
COMP_NAMES = ["u", "v", "w"]


class BLPlanes:
    """Memory-mapped view of the three velocity components."""

    def __init__(self, path=DATA_FILE):
        self.path = path
        with h5py.File(path, "r") as h:
            self.x = h["x"][0].astype(np.float64)
            self.y = h["y"][0].astype(np.float64)
            self._mm = []
            for k in COMPS:
                d = h[k]
                off = d.id.get_offset()
                if d.chunks is not None or d.compression is not None or off is None:
                    raise RuntimeError(f"{k}: not a contiguous dataset, cannot memory-map it")
                self._mm.append(np.memmap(path, dtype=d.dtype, mode="r", offset=off, shape=d.shape))
        self.N, self.H, self.W = self._mm[0].shape
        self.C = len(COMPS)

    def get(self, idx):
        """Snapshots idx (sorted internally for sequential reads) → float32 [n, C, H, W]."""
        idx = np.asarray(idx)
        order = np.argsort(idx)
        out = np.empty((len(idx), self.C, self.H, self.W), dtype=np.float32)
        for c, m in enumerate(self._mm):
            out[order, c] = m[idx[order]]
        return out

    def y_weights(self):
        """Trapezoidal quadrature weights in y (the grid is stretched), normalised to mean 1."""
        w = np.zeros_like(self.y)
        dy = np.diff(self.y)
        w[:-1] += 0.5 * dy; w[1:] += 0.5 * dy
        return (w / w.mean()).astype(np.float32)


def split(N, val_frac=0.1, seed=7):
    """Same split as beta_vae.py: random permutation (seed 7), first 10 % → validation."""
    idx = np.random.default_rng(seed).permutation(N)
    n_val = max(1, round(val_frac * N))
    return np.sort(idx[n_val:]), np.sort(idx[:n_val])


def stats(data, tr_idx, cache=None, block=64):
    """Per-channel mean / std and the time-mean field, over the TRAINING snapshots only
    (streamed in blocks; float64 accumulators). Cached to `cache` (.npz) if given."""
    if cache and os.path.exists(cache):
        z = np.load(cache)
        if np.array_equal(z["tr_idx"], tr_idx):
            return z["mu_C"], z["std_C"], z["mean_field"]
    C, H, W = data.C, data.H, data.W
    s1 = np.zeros((C, H, W)); s2 = np.zeros(C)
    for i in range(0, len(tr_idx), block):
        b = data.get(tr_idx[i:i + block]).astype(np.float64)
        s1 += b.sum(0); s2 += (b ** 2).sum(axis=(0, 2, 3))
        print(f"\r  stats: {min(i + block, len(tr_idx))}/{len(tr_idx)} snapshots", end="", flush=True)
    print()
    n = len(tr_idx)
    mean_field = (s1 / n).astype(np.float32)
    mu_C = (s1.sum(axis=(1, 2)) / (n * H * W))
    var_C = s2 / (n * H * W) - mu_C ** 2
    std_C = np.sqrt(np.maximum(var_C, 0)); std_C = np.where(std_C < 1e-12, 1.0, std_C)
    mu_C, std_C = mu_C.astype(np.float32), std_C.astype(np.float32)
    if cache:
        np.savez(cache, mu_C=mu_C, std_C=std_C, mean_field=mean_field, tr_idx=tr_idx)
    return mu_C, std_C, mean_field


class Prefetcher:
    """Bounded parallel reads; propagate read failures instead of hanging."""

    def __init__(self, data, batches, depth=3, workers=3):
        self.data, self.batches = data, list(batches)
        self.depth, self.workers = depth, workers

    def __iter__(self):
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            pending = {}
            next_job = 0
            for j in range(len(self.batches)):
                while next_job < min(len(self.batches), j + self.depth):
                    pending[next_job] = pool.submit(self.data.get, self.batches[next_job])
                    next_job += 1
                yield pending.pop(j).result()

    def __len__(self):
        return len(self.batches)
