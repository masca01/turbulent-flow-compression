# Iteration 4: stable deterministic residual autoencoder

A new experiment trained **from random initialisation at epoch 1**. It does not
load iteration 3's weights or optimiser. Iteration 3's code and outputs remain
unchanged. This iteration addresses its observed dense-ELU saturation and adds
diagnostics for future instability; long-run stability still requires testing.

## Architecture and training changes

The five residual stages and channel widths remain 16, 32, 64, 128, 256. The
decoder remains the wider iteration-3 decoder. The encoder dense path becomes:

```text
160000 -> Linear(256) -> LayerNorm(256, affine=False) -> ELU -> Linear(32)
```

LayerNorm centres and scales each snapshot's 256 hidden values before the ELU.
This targets the failure observed in iteration 3, where large negative dense
activations drove every ELU output to -1. Fixed normalisation has no trainable
scale or bias that could undo it and adds no parameters. The model still has
50,180,403 parameters at latent 32 on the full grid, with nominal 55,500:1
snapshot compression excluding model storage. It remains deterministic, with
pure reconstruction MSE and no variance head, sampling or KL penalty.

| Setting | Default |
|---|---|
| Latent dimension | 32 |
| Training / validation | 4500 / 500, seed 7 |
| Epochs | 500, starting at 1 |
| Batch size | 32 |
| Optimiser | Adam |
| Learning rate | Cosine decay from 3e-5 to 3e-6 |
| Global gradient norm limit | 1.0 |
| Validation and checkpoints | Every 5 epochs and the final epoch |
| Input normalisation | Training-only per-component z-score |

Every batch checks input/loss/activation finiteness, severe hidden saturation and
constant latent vectors before an update. Non-finite gradients are rejected;
finite gradients are clipped. A four-snapshot validation probe checks the model
after every epoch, including its final update, before checkpoints are saved.

Every batch logs loss, learning rate, pre-clipping gradient norm, raw dense
activation magnitude, saturation and latent spread. A failed batch/probe guard
writes `failure.json`, stops execution and leaves existing checkpoints untouched.
There is no automatic restart or fallback. Validation also rejects non-finite
loss and fully collapsed latents. These controls reduce risk and detect the
known failure; they cannot guarantee convergence or prevent every instability.

## Launch manually

From the repository root:

```bash
caffeinate -i /Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -u \
  iteration_4_stable_deterministic/train_stable.py \
  --latent 32 --epochs 500 --batch-size 32 --lr 0.00003 --min-lr 0.000003 \
  --grad-clip 1.0 --val-every 5 --device mps
```

No simulation was launched during implementation. In IDLE, open
`train_stable.py` and press F5 when ready; its CONFIG defaults match the command.
For a longer experiment, choose `--epochs 1000` at the initial launch. That uses
a separate output directory and decays the learning rate over all 1000 epochs.
The smaller learning rate may require more epochs than iteration 3; its early
learning speed should not be assumed to match that run.

Outputs are outside Git:

```text
Challenge/ARTIFACTS/results/iteration_4_stable_deterministic/lat32_ep500_ntr4500/
```

The folder contains parameters, best/last checkpoints, `train.log`, per-batch
diagnostics, per-component validation history and final metrics/figures. Checkpoints
from earlier architectures are rejected even if their tensor shapes match.
Existing checkpoints prevent accidental fresh-run overwrites.

To resume this new experiment after an interruption, repeat its original options
with `--resume`. That restores optimiser, RNG and history and requires the same
learning-rate schedule, clipping, validation cadence, split and epoch target.
It does not extend the epoch target. If a guard stopped the run, inspect
`failure.json` before deciding whether to resume.

## Verification

```bash
python3 -B iteration_4_stable_deterministic/test_model.py
python3 -B iteration_4_stable_deterministic/test_stability.py -v
python3 -B -m unittest discover -s tests -v
```

Checks cover deterministic forwards, finite gradients, full-grid geometry,
signed output, checkpoint compatibility, and directory-independent imports.
An extreme-negative-activation test reproduces the old saturation and confirms
that the new LayerNorm keeps ELU gradients nonzero. Additional tests cover
clipping, failure detection, the cosine schedule, fresh initialisation, history,
resume settings, and protection of saved checkpoints on failure. Integration
tests use tiny synthetic data/models and mocked optimiser steps; no parameter
updates or real-data training are executed. Full-batch MPS memory use and
long-run convergence have not been validated.
