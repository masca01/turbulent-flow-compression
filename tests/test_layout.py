"""Check relocation, external artifacts and imports without training or real data."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class LayoutTests(unittest.TestCase):
    def test_entrypoints_from_an_unrelated_directory_without_data(self):
        # run_path reproduces IDLE's __file__ context without invoking main().
        with tempfile.TemporaryDirectory() as temp:
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
                       TFC_DATA_DIR=str(Path(temp) / "no-data"),
                       TFC_ARTIFACTS_DIR=str(Path(temp) / "outputs"))
            source = '''import json, runpy
from pathlib import Path
root = Path(ROOT_VALUE)
scripts = ["train_betavae_bl.py", "iteration_1_residual/train_residual_betavae.py", "iteration_2_paper_cnn/train_paper_cnn.py", "iteration_3_wide_deterministic/train_wide_deterministic.py", "iteration_4_stable_deterministic/train_stable.py", "iteration_5_higher_lr/train_higher_lr.py"]
reports = []
for script in scripts:
    ns = runpy.run_path(str(root / script), run_name="import_check")
    reports.append({"results": ns["RESULTS"], "data": ns["bl_data"].DATA_FILE, "model": ns["Encoder"].__module__})
print(json.dumps(reports))
'''.replace('ROOT_VALUE', repr(str(ROOT)))
            result = subprocess.run([sys.executable, "-B", "-c", source], cwd=temp,
                                    env=env, capture_output=True, text=True, check=True)
            reports = json.loads(result.stdout)
            self.assertEqual([r["model"] for r in reports], [
                "challenge_lib.baseline_model", "iteration_1_residual.model", "iteration_2_paper_cnn.model",
                "iteration_3_wide_deterministic.model", "iteration_4_stable_deterministic.model",
                "iteration_4_stable_deterministic.model"])
            for report in reports:
                self.assertTrue(Path(report["results"]).is_relative_to((Path(temp) / "outputs").resolve()))
                self.assertEqual(Path(report["data"]), (Path(temp) / "no-data/BL2/Challenge1.1_train.h5").resolve())
            self.assertFalse((Path(temp) / "outputs").exists(), "Imports should not create run folders")

    def test_artifacts_cannot_be_redirected_into_git(self):
        env = dict(os.environ, PYTHONPATH=str(ROOT), TFC_ARTIFACTS_DIR=str(ROOT / "outputs"))
        result = subprocess.run([sys.executable, "-B", "-c", "import challenge_lib.paths"],
                                env=env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be outside the Git repository", result.stderr)

    def test_relative_external_paths_are_rejected(self):
        env = dict(os.environ, PYTHONPATH=str(ROOT), TFC_DATA_DIR="../data")
        result = subprocess.run([sys.executable, "-B", "-c", "import challenge_lib.paths"],
                                env=env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be an absolute path", result.stderr)


if __name__ == "__main__":
    unittest.main()
