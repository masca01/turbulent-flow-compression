"""Challenge 1.1 iteration 3: train a wide deterministic residual autoencoder.

Defaults: latent=32, batch=32, Adam lr=3e-4, 500 epochs; 4500 train / 500 validation.
Five residual stages with channels 16, 32, 64, 128, 256; no latent sampling or KL.
Run --help for smoke-test, evaluation and resume options. Generated files go to the external ARTIFACTS directory.
"""

import os, sys, time, json, math, datetime
import argparse
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from challenge_lib import paths, data as bl_data
from iteration_3_wide_deterministic.model import ARCHITECTURE, CHANNELS, Encoder, Decoder, ae_loss
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
paths.configure_matplotlib()
EXPERIMENT_RESULTS = paths.experiment_results("iteration_3_wide_deterministic")

# ══ CONFIG ══
LATENT_DIM = 32
BATCH_SIZE = 32
N_EPOCHS   = 500
LR         = 3e-4
VAL_FRAC   = 0.1
RNG_SEED   = 7       # train/val split
TORCH_SEED = 7       # network initialisation and batch order
VAL_EVERY  = None    # None = auto (every ~10% of epochs)
SMOKE_TEST = False
MAX_TRAIN  = None    # None = all training snapshots; an int = the first N (smoke tests)
RESULTS    = str(EXPERIMENT_RESULTS)    # one sub-folder per run (run_dir), see the docstring
# ═══════════════════════════════════════════════════════════


def _cli(argv):
    global N_EPOCHS, LATENT_DIM, LR, MAX_TRAIN, BATCH_SIZE, DEVICE, SMOKE_TEST, RESULTS
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=N_EPOCHS)
    p.add_argument("--latent", type=int, default=LATENT_DIM)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--max-train", type=int)
    p.add_argument("--device", choices=("cpu", "mps"), default=str(DEVICE))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--eval-only", nargs="?", const="", metavar="CHECKPOINT")
    p.add_argument("--smoke-test", action="store_true",
                   help="2 epochs, 4 training + 4 validation snapshots, batch 2; separate smoke outputs")
    args = p.parse_args(argv)
    if args.epochs < 1 or args.latent < 1 or args.batch_size < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        p.error("epochs, latent and batch-size must be positive; lr must be finite and positive")
    if args.max_train is not None and args.max_train < 1:
        p.error("max-train must be positive")
    if args.resume and args.eval_only is not None:
        p.error("choose resume or eval-only")
    if args.device == "mps" and not torch.backends.mps.is_available():
        p.error("MPS is unavailable; use --device cpu")
    N_EPOCHS, LATENT_DIM, LR = args.epochs, args.latent, args.lr
    BATCH_SIZE, MAX_TRAIN, DEVICE = args.batch_size, args.max_train, torch.device(args.device)
    SMOKE_TEST = args.smoke_test
    if SMOKE_TEST:
        N_EPOCHS, MAX_TRAIN, BATCH_SIZE = 2, 4, 2
        RESULTS = str(EXPERIMENT_RESULTS / "smoke")
    return vars(args)


def atomic_save(payload, path):
    temp = path + ".tmp"
    torch.save(payload, temp)
    os.replace(temp, path)


def rng_state():
    state = {"cpu": torch.get_rng_state()}
    if DEVICE.type == "mps":
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng(state):
    torch.set_rng_state(state["cpu"])
    if DEVICE.type == "mps" and "mps" in state:
        torch.mps.set_rng_state(state["mps"])


def check_architecture(ck):
    if ck.get("architecture") != ARCHITECTURE:
        raise ValueError("This script only accepts iteration_3_wide_deterministic checkpoints.")


def run_name():
    tag = f"BL2ch11_wide_deterministic_lat{LATENT_DIM}_ep{N_EPOCHS}"
    return tag + (f"_ntr{MAX_TRAIN}" if MAX_TRAIN else "")


def run_dir(n_train):
    """results/lat<L>_ep<E>_ntr<N>: the folder of this run's files."""
    return os.path.join(RESULTS, f"lat{LATENT_DIM}_ep{N_EPOCHS}_ntr{n_train}")


class _Tee:
    """Print to the terminal and append to the run's train.log."""
    def __init__(self, path):
        self.term, self.f = sys.stdout, open(path, "a", buffering=1)
    def write(self, s):
        self.term.write(s); self.f.write(s)
    def flush(self):
        self.term.flush(); self.f.flush()


def write_parameters(out, extra):
    """parameters.json of the run: the settings, merged with what the run adds later."""
    path = os.path.join(out, "parameters.json")
    p = json.load(open(path)) if os.path.exists(path) else {}
    p.update(extra)
    with open(path, "w") as f:
        json.dump(p, f, indent=2)


# ─────────────────────────── Evaluation ───────────────────────────

def evaluate(enc, dec, data, idx, mu_C, std_C, mean_field, batch=None):
    """One pass over snapshots idx: deterministic normalised MSE, NMSE / rootNMSE
    (overall, per component, y-weighted), Ek about the training mean field, det(R)."""
    batch = batch or BATCH_SIZE
    enc.eval(); dec.eval()
    C = data.C
    mu_t  = torch.tensor(mu_C, device=DEVICE)[None, :, None, None]
    std_t = torch.tensor(std_C, device=DEVICE)[None, :, None, None]
    mf = mean_field.astype(np.float64)[None]
    wy = data.y_weights().astype(np.float64)[None, None, :, None]
    SSE = np.zeros(C); EN = np.zeros(C); SSEw = np.zeros(C); ENw = np.zeros(C); ENk = np.zeros(C)
    tot_r = 0.0; n_samples = 0; Z = []
    batches = [idx[i:i + batch] for i in range(0, len(idx), batch)]
    with torch.no_grad():
        for arr in bl_data.Prefetcher(data, batches):
            x = (torch.from_numpy(arr).to(DEVICE) - mu_t) / std_t
            z = enc(x)
            xhat = dec(z)
            n_samples += len(arr)
            Z.append(z.cpu().numpy())
            # the metrics in float64 on the CPU (sporadic wrong sums seen with MPS reductions)
            q = arr.astype(np.float64)
            rec = xhat.cpu().numpy().astype(np.float64) * std_C[None, :, None, None] + mu_C[None, :, None, None]
            e2 = (rec - q) ** 2
            SSE  += e2.sum(axis=(0, 2, 3));        EN  += (q ** 2).sum(axis=(0, 2, 3))
            SSEw += (e2 * wy).sum(axis=(0, 2, 3)); ENw += (q ** 2 * wy).sum(axis=(0, 2, 3))
            ENk  += ((q - mf) ** 2).sum(axis=(0, 2, 3))
            tot_r += len(arr) * (e2 / std_C.astype(np.float64)[None, :, None, None] ** 2).mean()   # = MSE of the normalised fields
    Z = np.concatenate(Z)
    Cov = np.atleast_2d(np.cov(Z.T)); d = np.sqrt(np.diag(Cov)); d[d < 1e-12] = 1.0
    R = Cov / np.outer(d, d); np.fill_diagonal(R, 1.0)
    # Ek of rom.vae.compute_ek: error of the reconstructed fluctuation about the training mean
    # field; (rec - mf) - (q - mf) = rec - q, so the numerator is the same SSE.
    m = {
        "val_loss": tot_r / n_samples, "val_recon": tot_r / n_samples,
        "NMSE": SSE.sum() / EN.sum(), "rootNMSE": math.sqrt(SSE.sum() / EN.sum()),
        "rootNMSE_comp": dict(zip(bl_data.COMP_NAMES, np.sqrt(SSE / EN).tolist())),
        "NMSE_yw": SSEw.sum() / ENw.sum(), "rootNMSE_yw": math.sqrt(SSEw.sum() / ENw.sum()),
        "Ek": (1 - SSE.sum() / ENk.sum()) * 100,
        "Ek_comp": dict(zip(bl_data.COMP_NAMES, ((1 - SSE / ENk) * 100).tolist())),
        "detR": float(np.linalg.det(R)),
        "latent_std": Z.std(0).tolist(),
    }
    return m


def timing(enc, dec, data, idx, mu_C, std_C, n=50):
    """Wall time per snapshot (batch 1, like a streaming use): compression = normalise + encode,
    reconstruction = decode + de-normalise. Device synchronised; data already in memory."""
    enc.eval(); dec.eval()
    mu_t  = torch.tensor(mu_C, device=DEVICE)[None, :, None, None]
    std_t = torch.tensor(std_C, device=DEVICE)[None, :, None, None]
    n = min(n, len(idx))
    arr = torch.from_numpy(data.get(idx[:n])).to(DEVICE)
    sync = torch.mps.synchronize if DEVICE.type == "mps" else (lambda: None)
    tc = tr = 0.0
    with torch.no_grad():
        for i in range(-5, n):                         # 5 warm-up passes
            q = arr[max(i, 0):max(i, 0) + 1]
            sync(); t0 = time.perf_counter()
            z = enc((q - mu_t) / std_t); sync(); t1 = time.perf_counter()
            rec = dec(z) * std_t + mu_t; sync(); t2 = time.perf_counter()
            if i >= 0: tc += t1 - t0; tr += t2 - t1
    return tc / n, tr / n


# ─────────────────────────── Figures ───────────────────────────

def plot_curves(hist, path, title):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = np.arange(1, len(hist["tr_recon"]) + 1)
    va = np.array(hist["va_epoch"])
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].semilogy(ep, hist["tr_recon"], label="train MSE")
    ax[0].semilogy(va, hist["va_recon"], "o-", label="validation MSE")
    ax[0].set(xlabel="Epoch", ylabel="MSE (normalised)", title="Reconstruction loss")
    ax[0].legend()
    ax[1].plot(va, hist["va_rootNMSE"], "o-", color="green")
    ax[1].set(xlabel="Epoch", ylabel="Root-NMSE", title="Validation reconstruction error")
    ax[2].plot(va, hist["va_Ek"], "o-", color="purple")
    ax[2].set(xlabel="Epoch", ylabel="Ek (%)", title="Validation reconstruction score")
    for a in ax:
        a.grid(True, which="both", alpha=0.4)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_snapshot(enc, dec, data, t, mu_C, std_C, path, title):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import h5py
    with h5py.File(bl_data.GRID_FILE, "r") as h:
        d_in = float(h["delta99_inlet"][()].ravel()[0])
    X, Y = data.x / d_in, data.y / d_in
    q = data.get([t])
    enc.eval(); dec.eval()
    with torch.no_grad():
        x = torch.from_numpy((q - mu_C[None, :, None, None]) / std_C[None, :, None, None]).to(DEVICE)
        z = enc(x)
        rec = dec(z).cpu().numpy() * std_C[None, :, None, None] + mu_C[None, :, None, None]
    fig, ax = plt.subplots(9, 1, figsize=(16, 15), sharex=True)
    for c, name in enumerate(bl_data.COMP_NAMES):
        mx = np.abs(q[0, c]).max()
        err = rec[0, c] - q[0, c]
        rn = np.sqrt((err ** 2).sum() / (q[0, c] ** 2).sum())
        for r, (f, lab, lim) in enumerate([(q[0, c], f"{name}' DNS", mx), (rec[0, c], f"{name}' wide residual AE", mx),
                                           (err, f"{name}' error (root-NMSE {rn:.3f})", mx)]):
            a = ax[3 * c + r]
            im = a.pcolormesh(X, Y, f, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto", rasterized=True)
            a.set_ylabel(r"$y/\delta_{in}$"); a.set_title(lab, fontsize=10, loc="left")
            fig.colorbar(im, ax=a, pad=0.01, fraction=0.02)
    ax[-1].set_xlabel(r"$x/\delta_{in}$")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ─────────────────────────── Main ───────────────────────────

def build(H, W, C):
    torch.manual_seed(TORCH_SEED)
    return Encoder(H, W, C, LATENT_DIM).to(DEVICE), Decoder(H, W, C, LATENT_DIM).to(DEVICE)


def finish(enc, dec, data, val_idx, mu_C, std_C, mean_field, hist, name, best_epoch, n_train, out):
    m = evaluate(enc, dec, data, val_idx, mu_C, std_C, mean_field)
    tc, tr = timing(enc, dec, data, val_idx, mu_C, std_C)
    n_dof = data.C * data.H * data.W
    m.update({"architecture": ARCHITECTURE, "latent": LATENT_DIM, "epochs": N_EPOCHS, "best_epoch": best_epoch,
              "n_train": n_train, "n_val": len(val_idx),
              "compression_ratio": n_dof / LATENT_DIM,
              "t_compress_s_per_snapshot": tc, "t_reconstruct_s_per_snapshot": tr,
              "device": str(DEVICE)})
    m.pop("latent_std")
    with open(os.path.join(out, f"metrics_{name}.json"), "w") as f:
        json.dump(m, f, indent=2)
    if hist.get("va_epoch"):
        plot_curves(hist, os.path.join(out, f"curves_{name}.png"),
                    f"Wide deterministic residual AE  |  BL2 Challenge 1.1  |  latent={LATENT_DIM}")
    plot_snapshot(enc, dec, data, int(val_idx[0]), mu_C, std_C,
                  os.path.join(out, f"recon_{name}.png"),
                  f"Validation snapshot {int(val_idx[0])}  |  latent={LATENT_DIM}  |  "
                  f"val root-NMSE={m['rootNMSE']:.3f}")
    print(f"\n  validation: root-NMSE={m['rootNMSE']:.4f} (u {m['rootNMSE_comp']['u']:.3f}, "
          f"v {m['rootNMSE_comp']['v']:.3f}, w {m['rootNMSE_comp']['w']:.3f})  "
          f"y-weighted={m['rootNMSE_yw']:.4f}  Ek={m['Ek']:.2f}%  detR={m['detR']:.3f}")
    print(f"  compression ratio {n_dof / LATENT_DIM:.0f}   encode {tc * 1e3:.2f} ms   decode {tr * 1e3:.2f} ms per snapshot")
    print(f"  wrote {out}/*_{name}.*")
    write_parameters(out, {"finished": datetime.datetime.now().isoformat(timespec="seconds"),
                           "best_epoch": best_epoch, "rootNMSE": m["rootNMSE"], "Ek": m["Ek"]})
    return m


def main():
    global LATENT_DIM, N_EPOCHS
    opts = _cli(sys.argv[1:])
    os.makedirs(RESULTS, exist_ok=True)
    data = bl_data.BLPlanes()
    C, H, W = data.C, data.H, data.W
    tr_idx, val_idx = bl_data.split(data.N, VAL_FRAC, RNG_SEED)
    if MAX_TRAIN: tr_idx = tr_idx[:MAX_TRAIN]
    if SMOKE_TEST: val_idx = val_idx[:4]
    name = run_name()
    out = run_dir(len(tr_idx))
    best_path = os.path.join(out, f"model_residualAE_{name}.pt")
    last_path = os.path.join(out, f"last_residualAE_{name}.pt")

    if opts["eval_only"] is not None:
        ckpt = opts["eval_only"] or best_path
        if not Path(ckpt).resolve().is_relative_to(EXPERIMENT_RESULTS.resolve()):
            raise ValueError("Evaluation checkpoints must be inside the external iteration_3_wide_deterministic results directory.")
        out = os.path.dirname(os.path.abspath(ckpt))                   # results go next to the checkpoint
        sys.stdout = _Tee(os.path.join(out, "train.log"))
        print(f"\n=== eval-only {datetime.datetime.now():%Y-%m-%d %H:%M} ({ckpt}) ===")
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        check_architecture(ck)
        LATENT_DIM, N_EPOCHS = ck["latent_dim"], ck["config"]["epochs"]
        name = os.path.splitext(os.path.basename(ckpt))[0].replace("model_residualAE_", "")
        enc, dec = build(H, W, C)
        enc.load_state_dict(ck["enc_state"]); dec.load_state_dict(ck["dec_state"])
        finish(enc, dec, data, ck["val_idx"], ck["mu_C"], ck["std_C"], ck["fluc_mean"],
               ck["history"], name, ck.get("best_epoch"), len(ck["tr_idx"]), out)
        return

    if opts["resume"] and not os.path.exists(last_path):
        raise FileNotFoundError(f"No resume checkpoint: {last_path}")
    if not opts["resume"] and any(Path(out).glob("*.pt")):
        raise FileExistsError(f"Checkpoints already exist in {out}. Use --resume or move the existing run.")
    os.makedirs(out, exist_ok=True)
    sys.stdout = _Tee(os.path.join(out, "train.log"))
    print(f"\n=== {'resume' if opts['resume'] else 'run'} {datetime.datetime.now():%Y-%m-%d %H:%M}: "
          f"{' '.join(sys.argv)} ===")
    print(f"  results -> {out}")
    print("  normalisation from the training snapshots ...")
    mu_C, std_C, mean_field = bl_data.stats(data, tr_idx, os.path.join(RESULTS, f"norm_stats_ntr{len(tr_idx)}.npz"))
    mu_t  = torch.tensor(mu_C, device=DEVICE)[None, :, None, None]
    std_t = torch.tensor(std_C, device=DEVICE)[None, :, None, None]

    enc, dec = build(H, W, C)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=LR)
    val_every = VAL_EVERY or max(1, round(N_EPOCHS / 10))
    hist = {k: [] for k in ["tr_loss", "tr_recon", "va_loss", "va_recon",
                            "va_rootNMSE", "va_rootNMSE_comp", "va_rootNMSE_yw", "va_Ek", "va_detR", "va_epoch"]}
    best_val, best_state, best_epoch, start = float("inf"), None, None, 1
    if opts["resume"] and os.path.exists(last_path):
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        check_architecture(ck)
        expected = {"latent": LATENT_DIM, "batch": BATCH_SIZE, "lr": LR,
                    "epochs": N_EPOCHS, "device": str(DEVICE)}
        if ck["config"] != expected or not np.array_equal(ck["tr_idx"], tr_idx) or not np.array_equal(ck["val_idx"], val_idx):
            raise ValueError("Resume settings/split/device differ; use the original command with --resume.")
        enc.load_state_dict(ck["enc"]); dec.load_state_dict(ck["dec"]); opt.load_state_dict(ck["opt"])
        hist, best_val, best_state, best_epoch = ck["hist"], ck["best_val"], ck["best_state"], ck["best_epoch"]
        start = ck["epoch"] + 1
        restore_rng(ck["rng"])
        print(f"  resumed from epoch {ck['epoch']}")

    n_params = sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in dec.parameters())
    if not opts["resume"] or not os.path.exists(os.path.join(out, "parameters.json")):
        write_parameters(out, {
            "architecture": ARCHITECTURE, "channels": list(CHANNELS), "smoke_test": SMOKE_TEST,
            "implementation": "wide residual AE; deterministic latent and reconstruction MSE",
            "deterministic_latent": True,
            "run": os.path.basename(out), "latent": LATENT_DIM, "epochs": N_EPOCHS,
            "batch": BATCH_SIZE, "lr": LR, "val_frac": VAL_FRAC, "split_seed": RNG_SEED,
            "torch_seed": TORCH_SEED, "n_train": len(tr_idx), "n_val": len(val_idx),
            "max_train": MAX_TRAIN, "grid": [C, H, W], "n_params": n_params, "device": str(DEVICE),
            "data_file": data.path, "command": " ".join(sys.argv),
            "started": datetime.datetime.now().isoformat(timespec="seconds")})
    print(f"\n{'=' * 64}")
    print(f"  Wide deterministic residual AE (iteration 3)  |  BL2 Challenge 1.1  |  {H}x{W}  C={C}  Nt={data.N}")
    print(f"  latent={LATENT_DIM}  lr={LR}  batch={BATCH_SIZE}  epochs={N_EPOCHS}")
    print(f"  device={DEVICE}  train={len(tr_idx)}  val={len(val_idx)}  params={n_params / 1e6:.1f}M")
    print(f"{'=' * 64}\n", flush=True)

    t_run = time.time()
    for epoch in range(start, N_EPOCHS + 1):
        t_ep = time.time()
        enc.train(); dec.train()
        perm = np.random.default_rng([TORCH_SEED, epoch]).permutation(tr_idx)
        batches = [perm[i:i + BATCH_SIZE] for i in range(0, len(perm), BATCH_SIZE)]
        s_l = 0.0
        n_seen = 0
        for arr in bl_data.Prefetcher(data, batches):
            xb = (torch.from_numpy(arr).to(DEVICE) - mu_t) / std_t
            loss = ae_loss(enc, dec, xb)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
            opt.zero_grad(); loss.backward(); opt.step()
            n = len(arr); n_seen += n
            s_l += n * loss.item()
        nb = n_seen
        hist["tr_loss"].append(s_l / nb); hist["tr_recon"].append(s_l / nb)

        if epoch % val_every == 0 or epoch == N_EPOCHS:
            m = evaluate(enc, dec, data, val_idx, mu_C, std_C, mean_field)
            for k in ["loss", "recon", "rootNMSE", "Ek", "detR"]:
                hist[f"va_{k}"].append(m[k if k in ("rootNMSE", "Ek", "detR") else f"val_{k}"])
            hist["va_epoch"].append(epoch)
            hist["va_rootNMSE_comp"].append(m["rootNMSE_comp"])
            hist["va_rootNMSE_yw"].append(m["rootNMSE_yw"])
            if not np.isfinite(m["val_loss"]):
                raise FloatingPointError(f"Non-finite validation loss at epoch {epoch}")
            marker = ""
            if m["val_loss"] < best_val:
                best_val, best_epoch = m["val_loss"], epoch
                best_state = {"enc": {k: v.cpu().clone() for k, v in enc.state_dict().items()},
                              "dec": {k: v.cpu().clone() for k, v in dec.state_dict().items()}}
                marker = "  * best"
                atomic_save({"architecture": ARCHITECTURE, "enc_state": best_state["enc"], "dec_state": best_state["dec"],
                            "mu_C": mu_C, "std_C": std_C, "fluc_mean": mean_field,
                            "H": H, "W": W, "C": C, "latent_dim": LATENT_DIM,
                            "comp_names": bl_data.COMP_NAMES, "tr_idx": tr_idx, "val_idx": val_idx,
                            "data_file": data.path, "history": hist, "best_epoch": best_epoch,
                            "config": {"batch": BATCH_SIZE, "lr": LR, "epochs": N_EPOCHS,
                                       "torch_seed": TORCH_SEED, "split_seed": RNG_SEED}}, best_path)
            atomic_save({"architecture": ARCHITECTURE, "rng": rng_state(),
                        "tr_idx": tr_idx, "val_idx": val_idx,
                        "config": {"latent": LATENT_DIM, "batch": BATCH_SIZE,
                                   "lr": LR, "epochs": N_EPOCHS, "device": str(DEVICE)}, "enc": enc.state_dict(), "dec": dec.state_dict(), "opt": opt.state_dict(),
                        "epoch": epoch, "hist": hist, "best_val": best_val,
                        "best_state": best_state, "best_epoch": best_epoch}, last_path)
            history_path = Path(out) / f"history_{name}.json"
            history_temp = history_path.with_suffix(".json.tmp")
            history_temp.write_text(json.dumps(hist, indent=2) + "\n")
            os.replace(history_temp, history_path)
            print(f"Ep {epoch:4d}/{N_EPOCHS} | tr {s_l / nb:.5f} | va {m['val_loss']:.5f} | "
                  f"rootNMSE {m['rootNMSE']:.4f} | Ek {m['Ek']:.1f}% | detR {m['detR']:.4f}{marker} "
                  f"| {time.time() - t_ep:.0f}s/ep  [{(time.time() - t_run) / 3600:.2f} h]", flush=True)
        else:
            print(f"Ep {epoch:4d}/{N_EPOCHS} | tr {s_l / nb:.5f} | {time.time() - t_ep:.0f}s/ep", flush=True)

    enc.load_state_dict(best_state["enc"]); dec.load_state_dict(best_state["dec"])
    finish(enc, dec, data, val_idx, mu_C, std_C, mean_field, hist, name, best_epoch, len(tr_idx), out)


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
