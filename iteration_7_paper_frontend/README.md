# Iteration 7: paper-inspired convolution/pooling front end

Test whether extracting features before early downsampling improves Challenge 1.1
reconstruction relative to iteration 6. This is a hybrid inspired by the paper's
first two encoder blocks, not a reproduction of the full paper network.

## Controlled architecture change

The first two encoder stages now use:

```text
Conv 3x3, stride 1 -> ReLU -> MaxPool 2x2, stride 2 -> existing residual block
```

Their channels remain 16 and 32. Previously these stages used a stride-2
convolution followed by ELU and the same residual block. This experiment changes
both the downsampling operation and the early activation; any improvement cannot
be attributed to pooling alone.

The remaining three stages still use stride-2 convolution, ELU and residual
refinement, with channels 64, 128 and 256. Shapes for the full input are:

```text
3x148x4000 -> 16x74x2000 -> 32x37x1000 -> 64x19x500
           -> 128x10x250 -> 256x5x125
           -> flatten 160000 -> Dense 256 -> LayerNorm -> ELU -> latent 32
```

The decoder and stability helpers are imported unchanged from iteration 4, as in
iteration 6. Reconstruction uses only the latent vector, with a signed linear
output. No latent sampling or KL penalty is used. Pooling requires input height
and width divisible by four; the challenge grid meets that requirement.

Trainable parameters remain **50,180,403**. The parameter initialization sequence
is preserved at seed 7, although the forward activations differ. The encoder has
a new architecture identifier; iteration-6 checkpoints cannot be resumed here.

## Training and comparison

Keep iteration 6's settings: 500 epochs, latent 32, batch 32, seed-7 split of
4500 training / 500 validation snapshots, Adam with cosine LR 3e-4 to 1e-4,
gradient clipping at 1.0, validation every five epochs, and training-only
normalization. Initialize from random weights, not an existing checkpoint.

Compare matched-epoch validation histories and the final selected checkpoint
against iteration 6: overall rootNMSE **39.7866%**, streamwise u **33.6484%**,
best epoch 495. Also compare v, w, y-weighted errors and inference timings.
These are internal validation results, not the organizers' withheld test scores.

Outputs are isolated outside Git in:

```text
Challenge/ARTIFACTS/results/iteration_7_paper_frontend/lat32_ep500_ntr4500/
```

Best/last checkpoints, histories, batch diagnostics and failure reporting follow
the existing trainer. Existing checkpoints prevent accidental fresh-run overwrite.

## Launch manually

From the repository root, in a free Terminal:

```bash
caffeinate -i /Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -u \
  iteration_7_paper_frontend/train_paper_frontend.py \
  --epochs 500 --latent 32 --batch-size 32 --lr 3e-4 --min-lr 1e-4 --device mps
```

This starts immediately; it is not a queue command. In IDLE, open
`train_paper_frontend.py` and press F5 in a free Shell. Do not restart an active
training Shell. The assistant has not launched training or queued this experiment.

## Approximate duration

Budget **12-15 hours** for 500 epochs on the same Mac/MPS setup, running alone.
This is an estimate, not a measured iteration-7 training benchmark. Iteration 6's
logged epochs totaled 38,580 seconds (10.72 hours), with the last 100 averaging
74.23 seconds each. Shape-only counting gives approximately 9.432 versus 10.135
billion convolution/dense forward MACs per snapshot, a 7.5% increase. This count
excludes pooling, activations, normalization, diagnostics, backward work and I/O;
it does not translate directly to wall time.

Larger early activation maps and max-pool backward storage can raise memory
pressure and runtime beyond the arithmetic increase. Full-batch MPS memory and
throughput remain unmeasured. After launching, average epochs 6-25 (including
validation/checkpoint epochs) and multiply seconds per epoch by 500/3600 to
refine the total-hours estimate. Simultaneous workloads can extend it substantially.

## Checks without training

```bash
python3 -B iteration_7_paper_frontend/test_experiment.py -v
python3 -B -m unittest discover -s tests -v
```

Checks cover full-grid geometry on the meta device, exact parameter count,
unchanged seeded parameter values, the intended front-end operations, synthetic
CPU gradients and stability checks, signed outputs, configuration/schedule
matching, checkpoint rejection, output isolation and imports from an unrelated
directory. No dataset, real checkpoint evaluation or optimizer update is needed.
