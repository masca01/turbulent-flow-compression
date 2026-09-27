# Iteration 2: paper CNN-AE baseline

An independent PyTorch implementation of the architecture described in
`Challenge/challengePaper.pdf`, section II.C.2, page 6. This is a reconstruction
from the paper, **not a verified copy of the authors' source code**. The official
website's download page returned HTTP 403 during implementation.

Model and training code live here; the shared reader in `challenge_lib` opens data
read-only. Results and caches live outside Git in `../../ARTIFACTS/`. See the root
README for the directory layout and environment overrides.

## Run

Use your existing Python environment (PyTorch, NumPy, h5py, Matplotlib):

```bash
cd iteration_2_paper_cnn  # from the repository root
caffeinate -i python3 -u train_paper_cnn.py
```

Current CONFIG: **latent 30, 100 epochs, batch 32, Adam lr 3e-4**, seed 7.
This preserves the user's IDLE settings. Use `--latent 32` for the paper's N=32 comparison point.
MPS is selected when available; `--device cpu` selects CPU.
The migration does not start, stop or restart any full training run.

For comparison with the residual experiment at exactly the same latent size and
epoch budget, use a separate run:

```bash
python3 -u train_paper_cnn.py --latent 30 --epochs 500
```

Further options: `--lr`, `--batch-size`, `--max-train`, `--resume`, `--eval-only`.
Run `--help` for usage. There is no beta option: this is a deterministic AE.

## Architecture

| Stage | Operation | Output per snapshot |
|---|---|---|
| Input | u, v, w as channels | 3 x 148 x 4000 |
| Encoder block 1 | 3x3 conv, 16 filters, ReLU, 2x2 max-pool | 16 x 74 x 2000 |
| Encoder block 2 | 3x3 conv, 32 filters, ReLU, 2x2 max-pool | 32 x 37 x 1000 |
| Bottleneck | Flatten and linear layer | 1,184,000 -> N |
| Decoder expansion | Linear layer, ReLU, reshape | N -> 32 x 37 x 1000 |
| Upsampling 1 | 4x4 transposed conv, stride 2, ReLU | 16 x 74 x 2000 |
| Upsampling 2 | 4x4 transposed conv, stride 2, ReLU | 8 x 148 x 4000 |
| Output | Linear 3x3 conv | 3 x 148 x 4000 |

No residual blocks, latent sampling, variance head, KL penalty, dropout or batch
normalisation. The final layer supports signed velocities. The loss is ordinary
MSE between the normalised input and reconstruction. No cropping is needed on
the challenge grid; this implementation requires both dimensions divisible by 4.

At N=32: **76,975,603 parameters**, and nominal snapshot compression **55,500:1**,
excluding network storage. Most parameters are in the two dense layers.

## What the paper specifies, and what we chose

The paper specifies the two encoder blocks (16 and 32 filters, 3x3 kernels, ReLU,
2x2 max-pooling), a dense latent bottleneck, dense decoder expansion, two 4x4
stride-2 transposed convolutions, a final 3x3 output convolution, Adam, MSE, and
100 epochs.

The description does **not** specify every implementation or training detail.
Our choices are:

- Padding 1 for 3x3 convolutions and 4x4 transposed convolutions.
- Decoder channels 32 -> 16 -> 8 -> 3; ReLU after expansion and each upsampling.
- Linear latent layer and linear final output.
- Latent size is configurable. The user's current default is 30; `--latent 32`
  selects a point in the paper's latent sweep.
- Adam lr 3e-4, batch 32, and seed 7, following our existing experiments.
- Per-component z-score from the training snapshots only, following our existing
  comparison protocol. The paper does not specify this normalisation recipe.
- The same **4,500 training / 500 validation split** as both previous models,
  rather than using all 5,000 available snapshots for training. This supports an
  internal comparison; it does not reproduce the official withheld test.
- Best-validation checkpoint selection and validation every 10 epochs by default.

Do not interpret the resulting score as an exact replication of the paper until
these choices and the official evaluation code have been checked against the
authors' implementation.

## Checkpoints, metrics and resume

Full-run outputs go to `../../ARTIFACTS/results/iteration_2_paper_cnn/lat30_ep100_ntr4500/`. This includes a log,
parameters JSON, best-model checkpoint, last-training checkpoint, learning curves,
snapshot reconstruction plot and metrics JSON. A run refuses to overwrite existing
checkpoints unless resumed. Checkpoint files are replaced atomically.

Validation/checkpoint saving occurs every 10% of the requested epochs, including
the final epoch. Resume restores weights, Adam state, history and RNG state, and
checks that the settings, device and snapshot indices match. To resume:

```bash
python3 -u train_paper_cnn.py --resume
```

For a non-default run, repeat its original options and add `--resume`.
Resume continues the original epoch budget; it does not extend it.

Re-evaluate the default best model:

```bash
python3 train_paper_cnn.py --eval-only
```

An explicit best-checkpoint path may follow `--eval-only`. Only this iteration's
checkpoints inside this experiment's external result subtree are accepted; evaluation writes beside the checkpoint.

Physical-space metrics are computed in CPU float64: overall/per-component
root-NMSE, a wall-normal quadrature-weighted variant, and Ek. A latent correlation
determinant is retained as a diagnostic, not as a loss or an independence guarantee.
Loss averages weight each batch by its snapshot count. Timings are raw per-snapshot
encode/decode times, not the paper's POD-normalised timings.

## Verification

```bash
python3 -B test_model.py
python3 -B train_paper_cnn.py --smoke-test --device cpu
```

The smoke test uses two epochs and four real full-resolution training snapshots
plus four disjoint validation snapshots, with batch size two. Its files live in
`../../ARTIFACTS/results/iteration_2_paper_cnn/smoke/`; they do not measure useful reconstruction performance. The
completed smoke run can be checked with:

```bash
python3 -B train_paper_cnn.py --latent 32 --smoke-test --device cpu --eval-only
python3 -B train_paper_cnn.py --latent 32 --smoke-test --device cpu --resume
```

`test_model.py` checks deterministic training forwards, MSE gradients, full-grid
output geometry and the ability to reconstruct negative velocities. The smoke
test additionally exercises data streaming, checkpoint saving/loading and figures.
These are CPU checks, not a full-batch MPS memory or runtime benchmark.
