"""Evaluate completed checkpoints on native and shared holdouts; never train."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from challenge_lib import paths, data as bl_data
from iteration_1_residual import train_residual_betavae as residual
from iteration_2_paper_cnn import train_paper_cnn as paper

RUNS = [
    ("Residual: 4500 train, N=30", residual, "iteration_1_residual", "lat30_beta0_ep500_ntr4500"),
    ("Residual: 4900 train, N=30", residual, "iteration_1_residual", "lat30_beta0_ep500_ntr4900"),
    ("Paper-style CNN: 4500 train, N=32", paper, "iteration_2_paper_cnn", "lat32_ep500_ntr4500"),
]


def load_checkpoint(folder):
    matches = list(folder.glob("model_*.pt"))
    if len(matches) != 1:
        raise ValueError(f"Expected one best checkpoint in {folder}")
    return matches[0], torch.load(matches[0], map_location="cpu", weights_only=False)


def main():
    torch.set_num_threads(4)
    residual.DEVICE = paper.DEVICE = torch.device("cpu")
    _, reference = load_checkpoint(paths.experiment_results(RUNS[1][2]) / RUNS[1][3])
    common = reference["val_idx"].copy()
    if len(common) != 100 or len(np.unique(common)) != 100:
        raise ValueError("Expected 100 distinct shared holdout snapshots")
    del reference
    data = bl_data.BLPlanes()
    output = paths.ANALYSIS_DIR / "comparison_2026-09-28"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    old_split = None
    for label, module, experiment, run in RUNS:
        folder = paths.experiment_results(experiment) / run
        checkpoint, ck = load_checkpoint(folder)
        module.check_architecture(ck)
        if np.intersect1d(ck["tr_idx"], ck["val_idx"]).size:
            raise ValueError("Training/validation overlap")
        if not np.isin(common, ck["val_idx"]).all():
            raise ValueError("Shared holdout must be unseen by every model")
        if len(ck["tr_idx"]) == 4500:
            values = {k: ck[k] for k in ("tr_idx", "val_idx", "mu_C", "std_C", "fluc_mean")}
            if old_split is None:
                old_split = values
            elif not all(np.array_equal(old_split[k], v) for k, v in values.items()):
                raise ValueError("4500-snapshot runs differ in split or normalisation")
        else:
            if not np.isin(old_split["tr_idx"], ck["tr_idx"]).all():
                raise ValueError("4900-snapshot training set must contain original training set")
        enc = module.Encoder(ck["H"], ck["W"], ck["C"], ck["latent_dim"])
        dec = module.Decoder(ck["H"], ck["W"], ck["C"], ck["latent_dim"])
        enc.load_state_dict(ck["enc_state"])
        dec.load_state_dict(ck["dec_state"])

        def evaluate(indices):
            args = (enc, dec, data, indices, ck["mu_C"], ck["std_C"], ck["fluc_mean"])
            return (module.evaluate(*args, ck["beta"], batch=8) if module is residual
                    else module.evaluate(*args, batch=8))

        print(f"Evaluating {label} on CPU...", flush=True)
        native = evaluate(ck["val_idx"])
        saved = json.loads(next(folder.glob("metrics_*.json")).read_text())
        for key in ("rootNMSE", "rootNMSE_yw", "Ek"):
            if not np.isclose(native[key], saved[key], rtol=1e-6, atol=1e-6):
                raise ValueError(f"Saved {key} not reproduced for {label}")
        shared = native if np.array_equal(common, ck["val_idx"]) else evaluate(common)
        history = ck["history"]
        records.append({"label": label, "checkpoint": str(checkpoint),
                        "parameters": json.loads((folder / "parameters.json").read_text()),
                        "native_cpu_metrics": native, "common_100_metrics": shared,
                        "validation_epochs": history["va_epoch"],
                        "validation_rootNMSE": history["va_rootNMSE"]})
        print(json.dumps({"label": label, "shared": shared}), flush=True)
        del enc, dec, ck
    report = {"common_validation_indices": common.tolist(), "runs": records}
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")

    paths.configure_matplotlib()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["#007e80", "#d78622", "#923b89"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for record, color in zip(records, colors):
        axes[0].plot(record["validation_epochs"], np.array(record["validation_rootNMSE"]) * 100,
                     "o-", color=color, markersize=3, label=record["label"])
    axes[0].set(title="Training histories (native validation sets)", xlabel="Epoch",
                ylabel="Root-NMSE (%)")
    axes[0].legend(fontsize=8)
    for i, (record, color) in enumerate(zip(records, colors)):
        m = record["common_100_metrics"]
        values = [m["rootNMSE"], *[m["rootNMSE_comp"][c] for c in ("u", "v", "w")], m["rootNMSE_yw"]]
        axes[1].bar(np.arange(5) + (i - 1) * .25, np.array(values) * 100, width=.24, color=color)
    axes[1].set(title="Same 100 held-out snapshots, epoch 500", ylabel="Root-NMSE (%)",
                xticks=np.arange(5), xticklabels=["Overall", "u", "v", "w", "y-weighted"])
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Challenge 1.1 — completed 500-epoch runs (lower error is better)")
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)
    print(f"Wrote {output}", flush=True)


if __name__ == "__main__":
    main()
