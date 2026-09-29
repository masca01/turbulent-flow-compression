# Iteration 6: higher learning-rate floor

This experiment tests whether a larger learning rate late in training improves
the stabilized deterministic residual autoencoder. It starts from random weights,
with the same seed as iteration 5. It does not continue iteration 5's checkpoint.

Iteration 5 completed 500 epochs. Its selected epoch-495 checkpoint scored
39.9795% overall validation root-NMSE and 33.8186% for streamwise velocity u.
Overall error improved from 40.2812% at epoch 400 to 39.9795% at epoch 495.
The last epochs had maximum pre-clipping gradient norms around 0.19–0.21,
below the limit of 1.0. A nonzero raw gradient norm does not establish an optimal
learning rate: Adam rescales updates using its moment estimates.

Only the minimum of the cosine schedule changes relative to iteration 5:

| Setting | Iteration 5 | Iteration 6 |
|---|---:|---:|
| Initial learning rate | 3e-4 | 3e-4 |
| Final learning rate | 3e-5 | 1e-4 |
| Epochs | 500 | 500 |
| Latent dimension | 32 | 32 |
| Training / validation | 4500 / 500 | 4500 / 500 |
| Batch size | 32 | 32 |
| Gradient norm limit | 1.0 | 1.0 |
| Validation interval | 5 epochs | 5 epochs |

The schedule starts at the same rate, stays higher after epoch 1, and ends at
3.33 times the previous rate. Raising the floor also raises intermediate rates,
so this tests the complete schedule rather than a switch only at epoch 400.
Compare best validation MSE and overall/component root-NMSE with iteration 5.
An improvement is a hypothesis, not a guaranteed result.

The script imports iteration 4's model and stability helpers, as iteration 5
does. It retains bottleneck LayerNorm, gradient clipping, finite-value and latent
collapse checks, and separate best/last checkpoints. Architecture, initial weights,
split, normalization and deterministic batch order stay the same.

## Launch manually

From the repository root, when ready:

```bash
caffeinate -i /Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -u \
  iteration_6_higher_lr_floor/train_higher_lr_floor.py \
  --epochs 500 --lr 3e-4 --min-lr 1e-4 --device mps
```

This starts immediately, not through a queue. For IDLE, open
`train_higher_lr_floor.py` and press F5 in a free Shell. Default settings match
the command above. The assistant has not launched training.

Outputs remain outside Git in:

```text
Challenge/ARTIFACTS/results/iteration_6_higher_lr_floor/lat32_ep500_ntr4500/
```

Existing iteration-5 outputs are preserved. To resume an interrupted iteration-6
run, append `--resume` to its original command with unchanged settings. Resume
loads only iteration 6's own last checkpoint and checks the saved configuration.

## Verification without training

```bash
python3 -B iteration_6_higher_lr_floor/test_experiment.py -v
python3 -B -m unittest discover -s tests -v
```

These checks verify the controlled configuration, seeded initialization, schedule
bounds and output isolation. They do not read the dataset or perform optimizer
updates. Final iteration-5 metrics and histories are the source of the figures above.
