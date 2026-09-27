# Challenge 1.1 — iteration 1: residual beta-VAE

This architecture experiment uses its own model and the repository's shared
data/path helpers. It does not import `CODING/ROM_TurbulentFlows`. Existing
checkpoints and the original training settings are preserved.

## Run

Use the same Python environment as the original challenge run; it needs PyTorch,
NumPy, h5py and Matplotlib.

```bash
cd iteration_1_residual  # from the repository root
python3 -u train_residual_betavae.py
```

On macOS, optionally keep the computer awake for the run:

```bash
caffeinate -i python3 -u train_residual_betavae.py
```

Use one of these commands to launch, not both. The script selects MPS when
available; `--device cpu` forces CPU execution. The migration does not launch a new full training run. The existing GPU run can finish before you launch this one.

Defaults: **beta 0, latent 30, 500 epochs, batch 32, Adam learning rate 3e-4**.
The original seed-7 split is retained: **4,500 training / 500 validation snapshots**.
Normalisation and the mean field use only the training snapshots. Latent sampling
remains active during training, including at beta zero; evaluation uses the mean.

Change settings explicitly, for example:

```bash
python3 -u train_residual_betavae.py --latent 30 --beta 0 --epochs 500
```

For the 4,900-training / 100-validation experiment, from the repository root:

```bash
caffeinate -i python3 -u iteration_1_residual/train_residual_betavae.py --val-frac 0.02 --latent 30 --beta 0 --epochs 500 --device mps
```

This trains from scratch with the same architecture, seeds, batch size and learning
rate. The 100 validation snapshots are a subset of the original 500-snapshot
holdout; the other 400 join training. Normalisation is recomputed using the 4,900
training snapshots only. Outputs go to the separate external folder
`../../ARTIFACTS/results/iteration_1_residual/lat30_beta0_ep500_ntr4900/`.
Repeat these flags with `--resume` to resume this run. In IDLE, set `VAL_FRAC = 0.02`
in the CONFIG block before F5. The default remains the original 4,500/500 split.
To isolate the data-size effect, compare both trained models on the same 100
held-out snapshots; the previous 500-snapshot validation average is not directly
equivalent.

## Architecture

Each encoder stage contains the original 3x3 stride-2 convolution and ELU,
followed by a residual block. The block computes
`ELU(h + Conv3x3(ELU(Conv3x3(h))))`, with stride 1 and padding 1 in both
convolutions. There is no batch normalisation or dropout.

| Encoder stage | Output per snapshot |
|---|---|
| Input u, v, w | 3 x 148 x 4000 |
| Downsample + residual block | 8 x 74 x 2000 |
| Downsample + residual block | 16 x 37 x 1000 |
| Downsample + residual block | 32 x 19 x 500 |
| Downsample + residual block | 64 x 10 x 250 |
| Downsample + residual block | 128 x 5 x 125 |
| Flatten, dense + ELU | 80,000 -> 256 |
| Mean and log-variance heads | 30 each; one 30-value sampled latent |

The decoder expands the latent to 128 x 5 x 125, applies a residual block
**before each upsampling operation**, and upsamples through channels
128 -> 64 -> 32 -> 16 -> 8 -> 8. Upsampling retains the baseline's 3x3
transposed convolutions, stride 2, padding 1, output padding 1, and ELU.
The final linear 3x3 convolution produces three channels at 160 x 4000;
the original centre crop produces exactly 148 x 4000.

All shortcuts are internal to individual blocks. There are no encoder-to-decoder
connections bypassing the latent. Only 30 numbers per snapshot reach the decoder.

- Encoder: 15 convolutions, 5 residual blocks, **20,987,388 parameters**.
- Decoder: 16 convolution/transpose layers, 5 residual blocks, **2,972,171 parameters**.
- Total: **23,959,559 parameters**, versus the original 23,172,903 at latent 30.
- Nominal snapshot compression ratio: **59,200:1**, excluding model storage.

## Files and outputs

- `model.py`: local encoder, decoder, residual blocks and stochastic VAE loss.
- `../challenge_lib/data.py`: shared streamed reader; data locations come from `challenge_lib/paths.py`.
- `train_residual_betavae.py`: training, evaluation, checkpointing and figures.
- `../../ARTIFACTS/results/iteration_1_residual/lat30_beta0_ep500_ntr4500/`: default full-run outputs.
- `../../ARTIFACTS/results/iteration_1_residual/smoke/`: isolated smoke-test outputs, not performance results.
- `../../ARTIFACTS/migration/iteration_1_residual/verification.json`: historical setup checks.
- `../../ARTIFACTS/migration/iteration_1_residual/original_file_hashes.json`: historical source fingerprints.

The shared reader resolves `challenge_lib.paths.DATA_DIR / "BL2"` independently
of the working directory; the default is the sibling `Challenge/DATA/BL2` folder.
It opens the HDF5 data read-only.
Generated results and normalisation caches are stored outside Git in
`../../ARTIFACTS/results/iteration_1_residual/`. Plotting caches use
`../../ARTIFACTS/cache/`. See the root README for environment overrides.
No copy of the 35.5 GB dataset is made.

Outputs include `train.log`, `parameters.json`, best model and resume checkpoints,
metrics JSON, learning curves, and a DNS/reconstruction/error figure.
Validation and checkpoint saving occur every 10% of the requested epochs
(every 50 epochs by default), and at the final epoch. Best-model selection uses
normalised validation MSE plus beta times KL.

Physical-space error metrics and validation losses accumulate on CPU in float64.
Epoch losses weight batches by their number of snapshots; this corrects the original
script's equal weighting of the smaller final batch. The per-batch training objective,
normalisation, and physical-space metric definitions are otherwise retained.

## Resume and evaluate

From this folder, resume a default run with:

```bash
python3 -u train_residual_betavae.py --resume
```

For a non-default run, repeat its original options and add `--resume`. Resume
restores the optimiser, best model, history and PyTorch random state, and checks
the settings, device and split. It continues from the last saved epoch; it does not
extend the requested epoch count. Missing resume checkpoints cause an error.
Starting a fresh run in a folder with existing checkpoints also causes an error.

Re-evaluate the default best model:

```bash
python3 train_residual_betavae.py --eval-only
```

Or pass an explicit iteration-1 best-model checkpoint to `--eval-only`.
Evaluation checkpoints must be inside this experiment's external result subtree; the architecture
identifier rejects baseline checkpoints. Metrics and plots are saved beside
the checkpoint.

## Verification

The setup smoke test uses actual full-resolution snapshots, two epochs, four
training snapshots, four disjoint validation snapshots, and batch size two:

```bash
python3 train_residual_betavae.py --smoke-test --device cpu
```

Its checkpoints are retained in `../../ARTIFACTS/results/iteration_1_residual/smoke`. To revisit that completed check:

```bash
python3 train_residual_betavae.py --smoke-test --device cpu --resume
python3 train_residual_betavae.py --smoke-test --device cpu --eval-only
```

These checks exercise training, full-size reconstruction, checkpoint loading,
metrics and plotting. They are CPU checks, not a full-batch MPS memory benchmark
or evidence of reconstruction quality. With only four validation snapshots,
the smoke-test latent correlation matrix is rank deficient; det(R) is not useful.
