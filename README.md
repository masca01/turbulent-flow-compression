# Turbulent flow compression

Convolutional beta-VAE and CNN autoencoder experiments for **Challenge 1.1** of
[Data-Driven Reduced-Complexity Modeling of Fluid Flows: A Community Challenge](https://arxiv.org/abs/2601.06183).
The task compresses three-component boundary-layer velocity fluctuations on a
148 x 4000 grid.

## Directory layout

Keep source code in this Git repository and large files in sibling directories:

```text
Challenge/
├── turbulent-flow-compression/           # Git repository: code and documentation
│   ├── challenge_lib/
│   │   ├── paths.py                      # All data/artifact locations
│   │   ├── data.py                       # Shared streamed HDF5 reader
│   │   └── baseline_model.py             # Original beta-VAE, vendored locally
│   ├── train_betavae_bl.py               # Original model's training entry point
│   ├── iteration_1_residual/
│   │   ├── model.py
│   │   ├── train_residual_betavae.py
│   │   └── analysis/compare_results.py
│   ├── iteration_2_paper_cnn/
│   │   ├── model.py
│   │   ├── train_paper_cnn.py
│   │   └── test_model.py
│   ├── tests/
│   └── requirements.txt
├── DATA/BL2/                             # HDF5 input data, outside Git
│   ├── Challenge1.1_train.h5
│   └── BLdns2_parameters.h5
├── ARTIFACTS/                            # Generated files, outside Git
│   ├── results/
│   │   ├── baseline/
│   │   ├── iteration_1_residual/
│   │   └── iteration_2_paper_cnn/
│   ├── analysis/iteration_1_residual/
│   ├── cache/
│   └── migration/                        # Local migration records/source backup
├── challengePaper.pdf                   # Reference paper, outside Git
└── CODE -> turbulent-flow-compression    # Local compatibility alias
```

The local `results` links within the repository preserve older paths for running
jobs and historical checkpoint metadata. They are ignored by Git. New code uses
`challenge_lib.paths` and does not need these links. Generated logs, checkpoints,
normalisation statistics, figures and comparison reports are kept in ARTIFACTS.
Python bytecode caches are ignored by Git.

## Install

Use Python 3.10 or later in Terminal or IDLE:

```bash
python3 -m pip install -r requirements.txt
```

No editable install or other repository is required. Entry points resolve paths
from their source location, regardless of the current working directory.

Place the challenge data in `../DATA/BL2`. To use another location, set absolute
paths in the environment before launching Python/IDLE:

```bash
export TFC_DATA_DIR="/absolute/path/to/DATA"
export TFC_ARTIFACTS_DIR="/absolute/path/to/ARTIFACTS"
```

`TFC_DATA_DIR` contains the `BL2` subdirectory. `TFC_ARTIFACTS_DIR` contains
`results`, `analysis`, and `cache`. Both locations must be outside this repository.
Importing the code does not start training or require the dataset to be present.

## Run from Terminal

From the repository root:

```bash
# Original beta-VAE: latent 30, beta 0, 500 epochs
python3 -u train_betavae_bl.py

# Residual beta-VAE: latent 30, beta 0, 500 epochs
python3 -u iteration_1_residual/train_residual_betavae.py

# Paper-description CNN-AE: current user configuration is latent 30, 100 epochs
python3 -u iteration_2_paper_cnn/train_paper_cnn.py
```

Choose one run at a time. On macOS, prefix a command with `caffeinate -i` to prevent
idle sleep. These commands start immediately; they do not queue automatically.

CLI flags override the CONFIG block. For example, to run the paper CNN at the
paper's nearby N=32 comparison point:

```bash
python3 -u iteration_2_paper_cnn/train_paper_cnn.py --latent 32 --epochs 100
```

For a matched latent-size/epoch-budget comparison:

```bash
python3 -u iteration_2_paper_cnn/train_paper_cnn.py --latent 30 --epochs 500
```

## Run from IDLE

Open the relevant training script **inside this repository**, edit its CONFIG
block, save, and use Run -> Run Module (F5). The current paper-CNN configuration
(latent 30, 100 epochs) has been preserved. No working-directory change is needed.
Do not press F5 in a Shell that is already training: IDLE restarts that Shell.

During migration, the existing IDLE job kept its loaded code and original paths
through compatibility links. Its outputs now physically reside in
`../ARTIFACTS/results/iteration_2_paper_cnn/`.

## Results, resume and evaluation

By default, all models retain the same seed-7 split: 4,500 training snapshots and 500 validation
snapshots. The residual script also accepts `--val-frac 0.02` for the separate
4,900-training / 100-validation experiment. Normalisation uses training snapshots only. Validation scores are
internal holdout scores, not the organisers' withheld test scores.

Each experiment has its own output subtree and parameter-specific run folders:

- Baseline: `../ARTIFACTS/results/baseline/lat30_beta0_ep500_ntr4500/`
- Residual: `../ARTIFACTS/results/iteration_1_residual/lat30_beta0_ep500_ntr4500/`
- Paper CNN: `../ARTIFACTS/results/iteration_2_paper_cnn/lat30_ep100_ntr4500/`

Runs contain `train.log`, `parameters.json`, best-model and last-training `.pt`
checkpoints, metrics JSON and figures. Append `--resume` to the original command
with the same options to resume. Append `--eval-only` to evaluate its best model;
an explicit best-checkpoint path can follow `--eval-only`. For iterations 1 and 2,
explicit checkpoints must be inside that experiment's external result subtree.
Their existing settings/split/device checks remain in place.

Architecture, learning objectives, stored state-dict keys and numerical training
recipes were not changed by the reorganisation. Existing checkpoints are compatible.
The data reader is shared across experiments; it uses bounded background reads
and propagates read errors instead of hanging.

The paper CNN is reconstructed from the published description, with unspecified
choices documented in [its README](iteration_2_paper_cnn/README.md). It is not a
verified copy of the authors' implementation.

## Verification and comparison

```bash
python3 -B -m unittest discover -s tests -v
python3 -B iteration_2_paper_cnn/test_model.py
python3 -B iteration_1_residual/analysis/compare_results.py
```

The comparison script reads existing checkpoints, evaluates them on the CPU and
writes to `../ARTIFACTS/analysis/iteration_1_residual/`. It does not retrain models.
It includes the completed paper-description CNN run alongside the original and
residual models. See [the completed-run analysis](docs/results_2026-09-27.md) for
the results and the unresolved metric differences with the publication.
Full-resolution smoke runs are available with `--smoke-test --device cpu` on
iterations 1 and 2, and write into their external `results/<iteration>/smoke/` trees.
Use a separate `TFC_ARTIFACTS_DIR` to keep verification runs isolated.

Only source, tests, requirements and documentation should be committed. `.gitignore`
excludes result links, data formats, checkpoints, logs and caches. No input dataset,
checkpoint or reference PDF is required in the Git history.
