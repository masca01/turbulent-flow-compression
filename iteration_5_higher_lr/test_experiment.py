"""Validate the LR-only experiment and queue; never launch real training."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from iteration_4_stable_deterministic import train_stable as old
from iteration_5_higher_lr import train_higher_lr as new
from iteration_5_higher_lr import queue_after_iteration4 as queue

IDENTITY = "Mon Sep 28 11:57:27 2026 /usr/bin/Python -u iteration_4_stable_deterministic/train_stable.py --epochs 500 --device mps"


class ExperimentTests(unittest.TestCase):
    def test_only_learning_rate_recipe_changes(self):
        self.assertIs(new.Encoder, old.Encoder)
        self.assertIs(new.Decoder, old.Decoder)
        self.assertIs(new.checked_loss, old.checked_loss)
        for key in ("LATENT_DIM", "BATCH_SIZE", "N_EPOCHS", "GRAD_CLIP", "VAL_FRAC", "RNG_SEED", "TORCH_SEED", "VAL_EVERY"):
            self.assertEqual(getattr(new, key), getattr(old, key))
        for epoch in (1, 100, 250, 500):
            self.assertAlmostEqual(new.learning_rate(epoch), 10 * old.learning_rate(epoch))
        self.assertNotEqual(new.run_dir(4500), old.run_dir(4500))
        with mock.patch.object(old, "DEVICE", torch.device("cpu")), mock.patch.object(new, "DEVICE", torch.device("cpu")):
            a, b = old.build(4, 4, 3), new.build(4, 4, 3)
            for previous, current in zip(a, b):
                for key, value in previous.state_dict().items():
                    torch.testing.assert_close(value, current.state_dict()[key], rtol=0, atol=0)


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.previous = root / "previous"
        self.previous.mkdir()
        (self.previous / "parameters.json").write_text(json.dumps({"finished": "done", "epochs": 500, "n_train": 4500, "n_val": 500}))
        (self.previous / "history_run.json").write_text(json.dumps({"tr_loss": [0.] * 500, "va_epoch": [500]}))
        (self.previous / "metrics_run.json").write_text(json.dumps({"rootNMSE": .6}))
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.stack.enter_context(mock.patch.object(queue, "PREVIOUS", self.previous))
        self.stack.enter_context(mock.patch.object(queue, "TARGET", root / "target"))
        self.stack.enter_context(mock.patch.object(queue, "QUEUE_DIR", root / "queue"))
        self.sleep = self.stack.enter_context(mock.patch.object(queue.time, "sleep"))
        self.launch = self.stack.enter_context(mock.patch.object(queue.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)))

    def test_waits_for_process_and_launches_once(self):
        with mock.patch.object(queue, "process_info", side_effect=[IDENTITY, IDENTITY, None]):
            self.assertEqual(queue.dispatch(1234), 0)
        self.sleep.assert_called_once()
        self.launch.assert_called_once_with(queue.command(), cwd=queue.ROOT)

    def test_dry_run_never_waits_or_launches(self):
        with mock.patch.object(queue, "process_info", return_value=IDENTITY):
            self.assertEqual(queue.dispatch(1234, dry_run=True), 0)
        self.launch.assert_not_called()
        self.sleep.assert_not_called()
        self.assertFalse(queue.QUEUE_DIR.exists())

    def test_completed_predecessor_can_launch_immediately(self):
        with mock.patch.object(queue, "process_info", return_value=None):
            queue.dispatch(1234)
        self.launch.assert_called_once()

    def test_failed_predecessor_does_not_launch(self):
        (self.previous / "parameters.json").write_text('{}')
        with mock.patch.object(queue, "process_info", return_value=None):
            with self.assertRaises(RuntimeError):
                queue.dispatch(1234)
        self.launch.assert_not_called()

    def test_pid_reuse_does_not_launch(self):
        with mock.patch.object(queue, "process_info", side_effect=[IDENTITY, IDENTITY.replace("Mon", "Tue")]):
            with self.assertRaises(RuntimeError):
                queue.dispatch(1234)
        self.launch.assert_not_called()

    def test_duplicate_run_does_not_launch(self):
        queue.TARGET.mkdir()
        (queue.TARGET / "parameters.json").write_text('{}')
        with self.assertRaises(FileExistsError):
            queue.dispatch(1234)
        self.launch.assert_not_called()

    def test_unrelated_process_is_rejected(self):
        with mock.patch.object(queue, "process_info", return_value=IDENTITY.replace("train_stable.py", "other.py")):
            with self.assertRaises(ValueError):
                queue.dispatch(1234)
        self.launch.assert_not_called()


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
