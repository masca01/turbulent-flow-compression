"""Synthetic stability/runner tests; optimiser steps are mocked, never executed."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch import nn
from iteration_4_stable_deterministic import train_stable as training
from iteration_4_stable_deterministic.model import Encoder, Decoder, ae_loss
from iteration_4_stable_deterministic.stability import (
    TrainingInstability, check_bottleneck, checked_loss, clip_gradients,
)


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Identity()
        self.fc = nn.Sequential(nn.Linear(48, 256), nn.LayerNorm(256, elementwise_affine=False), nn.ELU())
        self.latent = nn.Linear(256, 4)

    def forward(self, x):
        return self.latent(self.fc(x.flatten(1)))


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 48)

    def forward(self, z):
        return self.fc(z).view(-1, 3, 4, 4)


class SyntheticData:
    N, C, H, W, path = 20, 3, 4, 4, "synthetic"

    def __init__(self):
        self.values = np.random.default_rng(7).normal(size=(20, 3, 4, 4)).astype(np.float32)

    def get(self, indices):
        return self.values[indices]


class StabilityTests(unittest.TestCase):
    def test_normalisation_prevents_observed_all_negative_saturation(self):
        enc = Encoder(4, 4, 3, 4)
        raw = torch.linspace(-540427., -69733., 256).requires_grad_()
        self.assertTrue(torch.equal(nn.ELU()(raw), torch.full_like(raw, -1.)))
        normalised = enc.fc[1](raw)
        hidden = enc.fc[2](normalised)
        self.assertFalse(torch.any(hidden == -1.))
        hidden.square().sum().backward()
        self.assertTrue(torch.isfinite(raw.grad).all())
        self.assertGreater(float(raw.grad.abs().max()), 0)

    def test_guards_reject_collapse_and_nonfinite(self):
        pre, z = torch.zeros(4, 256), torch.randn(4, 32)
        check_bottleneck(pre, z)
        for bad_pre, bad_z in ((pre - 70000, z), (pre, torch.ones_like(z)),
                               (pre + float("inf"), z), (pre, z * float("nan"))):
            with self.assertRaises(TrainingInstability):
                check_bottleneck(bad_pre, bad_z)

    def test_guarded_forward_preserves_loss(self):
        enc, dec = Encoder(33, 65, 3, 32), Decoder(33, 65, 3, 32)
        x = torch.randn(2, 3, 33, 65)
        loss, _ = checked_loss(enc, dec, x)
        torch.testing.assert_close(loss, ae_loss(enc, dec, x), rtol=0, atol=0)

    def test_gradient_clipping_and_nonfinite_rejection(self):
        p = nn.Parameter(torch.tensor([1., 2.]))
        p.grad = torch.tensor([3., 4.])
        self.assertEqual(clip_gradients([p], 1.), 5.)
        torch.testing.assert_close(p.grad, torch.tensor([.6, .8]))
        torch.testing.assert_close(p, torch.tensor([1., 2.]))
        p.grad = torch.tensor([float("inf"), 1.])
        with self.assertRaises(TrainingInstability):
            clip_gradients([p], 1.)

    def test_cosine_schedule_endpoints(self):
        with mock.patch.object(training, "N_EPOCHS", 500):
            self.assertAlmostEqual(training.learning_rate(1), training.LR)
            self.assertAlmostEqual(training.learning_rate(500), training.MIN_LR)

    def fixture(self, directory):
        stack = contextlib.ExitStack()
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        stack.enter_context(mock.patch.object(training, "RESULTS", str(directory)))
        stack.enter_context(mock.patch.object(training, "EXPERIMENT_RESULTS", directory))
        stack.enter_context(mock.patch.object(training.bl_data, "BLPlanes", SyntheticData))
        stack.enter_context(mock.patch.object(training.bl_data, "stats", return_value=(
            np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32), np.zeros((3, 4, 4)))))
        stack.enter_context(mock.patch.object(training, "build", lambda *args: (TinyEncoder(), TinyDecoder())))
        metrics = {"val_loss": .25, "val_recon": .25, "rootNMSE": .5,
                   "rootNMSE_comp": {"u": .5, "v": .5, "w": .5}, "rootNMSE_yw": .5,
                   "Ek": 75., "detR": .1, "latent_std": [1.] * 4}
        stack.enter_context(mock.patch.object(training, "evaluate", return_value=metrics))
        stack.enter_context(mock.patch.object(training, "finish"))
        stack.enter_context(mock.patch.object(training, "_Tee", lambda path: sys.stdout))
        step = stack.enter_context(mock.patch.object(torch.optim.Adam, "step"))
        args = ["train_stable.py", "--epochs", "2", "--latent", "4", "--batch-size", "8", "--val-every", "1", "--device", "cpu"]
        return stack, step, args

    def test_fresh_start_checkpoints_history_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            stack, step, args = self.fixture(directory)
            with stack:
                with mock.patch.object(sys, "argv", args):
                    training.main()
                self.assertEqual(step.call_count, 6)  # Mock calls only: zero real updates.
                out = directory / "lat4_ep2_ntr18"
                last = torch.load(next(out.glob("last_*.pt")), weights_only=False)
                self.assertEqual(last["epoch"], 2)
                self.assertEqual(last["hist"]["va_epoch"], [1, 2])
                self.assertEqual(last["opt"]["state"], {})
                self.assertEqual(last["opt"]["param_groups"][0]["lr"], training.MIN_LR)
                self.assertEqual(json.loads((out / "parameters.json").read_text())["initialisation"], "random")
                with mock.patch.object(sys, "argv", args + ["--resume"]):
                    training.main()
                self.assertEqual(step.call_count, 6)
                with mock.patch.object(sys, "argv", args + ["--resume", "--grad-clip", "2"]):
                    with self.assertRaises(ValueError):
                        training.main()

    def test_failed_batch_skips_update_and_keeps_existing_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            stack, step, args = self.fixture(directory)
            with stack:
                with mock.patch.object(sys, "argv", args):
                    training.main()
                out = directory / "lat4_ep2_ntr18"
                last = next(out.glob("last_*.pt"))
                saved = last.read_bytes()
                enc, dec, data = TinyEncoder(), TinyDecoder(), SyntheticData()
                opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()))
                step.reset_mock()
                with mock.patch.object(training, "checked_loss", side_effect=TrainingInstability("Synthetic collapse")):
                    with self.assertRaises(TrainingInstability):
                        training.train_epoch(enc, dec, data, np.arange(18), torch.zeros(1, 3, 1, 1),
                                             torch.ones(1, 3, 1, 1), opt, 3, str(out), torch.zeros(2, 3, 4, 4))
                step.assert_not_called()
                self.assertEqual(last.read_bytes(), saved)
                self.assertTrue((out / "failure.json").exists())


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
