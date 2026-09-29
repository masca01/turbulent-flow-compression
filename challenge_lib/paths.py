"""Repository-independent locations for input data and generated artifacts.

Defaults use DATA/ and ARTIFACTS/ beside the repository. Environment overrides
must be absolute paths, so Terminal and IDLE resolve them identically.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _external_path(variable, default):
    value = Path(os.environ.get(variable, str(default))).expanduser()
    if not value.is_absolute():
        raise ValueError(f"{variable} must be an absolute path: {value}")
    value = value.resolve()
    if value.is_relative_to(REPO_ROOT):
        raise ValueError(f"{variable} must be outside the Git repository: {value}")
    return value


DATA_DIR = _external_path("TFC_DATA_DIR", REPO_ROOT.parent / "DATA")
ARTIFACTS_DIR = _external_path("TFC_ARTIFACTS_DIR", REPO_ROOT.parent / "ARTIFACTS")
RESULTS_DIR = ARTIFACTS_DIR / "results"
ANALYSIS_DIR = ARTIFACTS_DIR / "analysis"
CACHE_DIR = ARTIFACTS_DIR / "cache"


def experiment_results(name):
    if name not in ("baseline", "iteration_1_residual", "iteration_2_paper_cnn",
                    "iteration_3_wide_deterministic", "iteration_4_stable_deterministic",
                    "iteration_5_higher_lr"):
        raise ValueError(f"Unknown experiment: {name}")
    return RESULTS_DIR / name


def configure_matplotlib():
    os.environ["MPLCONFIGDIR"] = str(CACHE_DIR / "matplotlib")
