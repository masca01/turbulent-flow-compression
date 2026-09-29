# Iteration 5: higher-learning-rate comparison

At iteration 4's epoch 470, root-NMSE was 60.83% overall and 52.18% for u.
The model remained input-dependent, with no hidden saturation in the inspected
batches. Around epoch 473, its learning rate was 3.19e-6 and none of the latest
66 inspected batches required gradient clipping. A larger learning rate is a
reasonable experiment, but the effect of LayerNorm versus iteration 3 has not
been isolated.

This experiment starts from random weights and changes **only the learning-rate
schedule** relative to iteration 4:

| Setting | Iteration 4 | Iteration 5 |
|---|---:|---:|
| Initial learning rate | 3e-5 | 3e-4 |
| Final cosine learning rate | 3e-6 | 3e-5 |
| Epochs | 500 | 500 |
| Latent dimension | 32 | 32 |
| Training / validation | 4500 / 500 | 4500 / 500 |
| Batch size | 32 | 32 |
| Gradient norm limit | 1.0 | 1.0 |
| Validation/checkpoint interval | 5 epochs | 5 epochs |

Model classes and stability helpers are imported directly from iteration 4, so
the architecture, fixed bottleneck LayerNorm, parameter count, guards, seeds and
initial weights match. The standalone training entrypoint uses its own external
output subtree. It does not alter or load weights from the current experiment.
Increasing learning rate does not guarantee improved accuracy or stability.

## Queue manually after the current run

The inspected iteration-4 Python process has PID **61417**. From the repository
root, run this once in another Terminal tab while the original run continues:

```bash
caffeinate -i /Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -u \
  iteration_5_higher_lr/queue_after_iteration4.py --after-pid 61417
```

Keep both Terminal tabs open and the Mac connected to power with the lid open.
The queue checks the process identity and waits without interrupting it. Once
it exits, the queue requires completed parameters, history and finite metrics
before starting iteration 5. If iteration 4 has already finished successfully,
iteration 5 starts immediately. If it failed or was interrupted, the queue stops.
PID reuse or an unrelated process is rejected, and a file lock prevents duplicate
queue launchers. No automatic restarts are performed if iteration 5 fails.

`--dry-run` inspects readiness and prints the command without waiting, launching
training or writing output files. Only this dry-run mode was executed during
implementation. The PID above identifies this particular run; inspect a new
process's PID before using the queue for a later run.

The queue writes status to `Challenge/ARTIFACTS/queues/iteration5.json` after you
launch it. Training outputs remain outside Git:

```text
Challenge/ARTIFACTS/results/iteration_5_higher_lr/lat32_ep500_ntr4500/
```

## Direct launch and resume

Use direct launch only when no other simulation is running:

```bash
caffeinate -i python3 -u iteration_5_higher_lr/train_higher_lr.py --epochs 500 --device mps
```

Append `--resume` with the same settings to resume iteration 5's own last
checkpoint. Do not use the queue to resume an existing iteration-5 run: it refuses
an existing run to prevent duplicates. To use IDLE, open `train_higher_lr.py` and
press F5 when ready; that launches immediately rather than queueing.

## Validation

```bash
python3 -B iteration_5_higher_lr/test_experiment.py -v
python3 -B -m unittest discover -s tests -v
```

Tests confirm identical model classes and seeded initial weights, unchanged
non-LR settings, a tenfold schedule increase, output isolation, queue waiting,
successful-completion checks, dry-run behaviour, duplicate rejection and PID
identity checks. All subprocess launches in queue tests are mocked. Neither
training nor the real queue was launched by the assistant.
