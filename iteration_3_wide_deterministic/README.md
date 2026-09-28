# Iteration 3: wide deterministic residual autoencoder

Combines latent dimension 32, doubled convolutional widths, and deterministic
latent training. It retains the five downsampling stages and residual blocks of
iteration 1. This tests the changes jointly, without isolating each contribution.
The variance head, latent sampling and KL term are removed: this is a deterministic
autoencoder trained with reconstruction MSE. Earlier training entrypoints and
checkpoints remain separate.

## Configuration

| Setting | Value |
|---|---|
| Latent dimension | 32 |
| Encoder channels | 16, 32, 64, 128, 256 |
| Decoder channels | 256 -> 128 -> 64 -> 32 -> 16 -> 16 -> 3 |
| Training / validation | 4500 / 500 snapshots, seed 7 |
| Epochs | 500 |
| Batch size | 32 |
| Optimiser | Adam, learning rate 3e-4 |
| Activation | ELU; linear latent and final output |
| Normalisation | Training-only per-component z-score |
| Validation/checkpoint interval | Every 50 epochs by default, including epoch 100 |

Each encoder stage has a stride-2 3x3 convolution followed by a two-convolution
residual block. On the challenge grid, the final feature map is 256 x 5 x 125;
the dense path is 160000 -> 256 -> 32. The decoder expands the latent to
256 x 5 x 125, applies a residual block before each transposed convolution, and
centre-crops from 160 x 4000 to 148 x 4000. There are no connections bypassing
the latent bottleneck, no dropout and no batch normalisation.

- Encoder: 42,933,408 parameters; decoder: 7,246,995 parameters.
- Total: **50,180,403**, versus 23,959,559 for the previous latent-30 residual
  model and 76,975,603 for the latent-32 paper-style CNN.
- Nominal snapshot compression: 55,500:1, excluding model storage.

## Launch manually

From the repository root:

```bash
caffeinate -i /Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -u \
  iteration_3_wide_deterministic/train_wide_deterministic.py \
  --latent 32 --epochs 500 --batch-size 32 --lr 0.0003 --device mps
```

This starts a fresh run when you execute it. No training was launched during
implementation. For IDLE, open `train_wide_deterministic.py` and use F5; its
defaults match this command. Do not restart a Shell that is already training.

Append `--resume` to the same command to continue from the last saved checkpoint.
Resume requires matching settings and restores optimiser state, history and RNG;
it does not extend the epoch budget. Existing checkpoints prevent accidental
fresh-run overwrites. There is no `--beta` option because there is no KL loss.

## Results and evaluation

Outputs remain outside Git:

```text
Challenge/ARTIFACTS/results/iteration_3_wide_deterministic/lat32_ep500_ntr4500/
```

Files include parameters, training log, best and last checkpoints, final metrics,
learning curves, a reconstruction plot, and `history_*.json`. Each validation
point preserves overall, u/v/w and wall-normal-weighted root-NMSE so epoch-100
component scores remain available after checkpoints are replaced. Physical
error metrics accumulate in float64 on CPU.

Evaluate a completed default checkpoint without training:

```bash
python3 -B iteration_3_wide_deterministic/train_wide_deterministic.py --eval-only
```

The unique architecture identifier rejects checkpoints from earlier iterations.
Data paths and external artifact overrides use the shared `challenge_lib`
helpers. No data is copied and no code imports the separate ROM_TurbulentFlows
project.

## Validation performed

```bash
python3 -B iteration_3_wide_deterministic/test_model.py
python3 -B -m unittest discover -s tests -v
```

Checks cover deterministic train/eval forwards without RNG consumption, MSE
gradients, full challenge-grid geometry using meta tensors, signed output,
checkpoint round trips and architecture rejection, and imports from another
working directory without creating run folders. These checks do not execute a
training loop or optimiser updates. Full-batch MPS memory use and convergence
have not been tested. The optional `--smoke-test --device cpu` launches a separate
two-epoch run with four training snapshots; it was not executed here.
