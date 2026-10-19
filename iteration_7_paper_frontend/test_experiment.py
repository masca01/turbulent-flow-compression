"""Synthetic CPU and shape-only checks; no dataset or optimizer updates."""

from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from iteration_6_higher_lr_floor import train_higher_lr_floor as previous
from iteration_7_paper_frontend import train_paper_frontend as current
from iteration_7_paper_frontend.model import Encoder, Decoder, ResidualBlock, ae_loss


class PaperFrontendTests(unittest.TestCase):
    def test_full_grid_geometry_and_parameter_budget(self):
        shapes = []
        with torch.device("meta"):
            enc, dec = Encoder(148, 4000, 3, 32), Decoder(148, 4000, 3, 32)
            for layer in enc.conv:
                if isinstance(layer, ResidualBlock):
                    layer.register_forward_hook(lambda m, i, o: shapes.append(tuple(o.shape[1:])))
            x = torch.empty(32, 3, 148, 4000)
            z = enc(x)
            result = dec(z)
        self.assertEqual(shapes, [(16, 74, 2000), (32, 37, 1000),
                                  (64, 19, 500), (128, 10, 250), (256, 5, 125)])
        self.assertEqual(tuple(z.shape), (32, 32))
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(sum(p.numel() for m in (enc, dec) for p in m.parameters()), 50180403)

    def test_frontend_and_unchanged_initial_weights(self):
        with mock.patch.object(previous, "DEVICE", torch.device("cpu")), \
             mock.patch.object(current, "DEVICE", torch.device("cpu")):
            old = previous.build(36, 68, 3)
            new = current.build(36, 68, 3)
        layers = list(new[0].conv)
        for offset in (0, 4):
            self.assertIsInstance(layers[offset], nn.Conv2d)
            self.assertEqual(layers[offset].stride, (1, 1))
            self.assertIsInstance(layers[offset + 1], nn.ReLU)
            self.assertIsInstance(layers[offset + 2], nn.MaxPool2d)
            self.assertIsInstance(layers[offset + 3], ResidualBlock)
        self.assertEqual([m.stride for m in layers[8:] if isinstance(m, nn.Conv2d)], [(2, 2)] * 3)
        # Parameter-free operations must not change the initialization sequence.
        for a, b in zip(old, new):
            self.assertEqual(len(list(a.parameters())), len(list(b.parameters())))
            for pa, pb in zip(a.parameters(), b.parameters()):
                torch.testing.assert_close(pa, pb, rtol=0, atol=0)
        self.assertIs(current.Decoder, previous.Decoder)

    def test_gradients_determinism_and_stability_checks(self):
        torch.manual_seed(7)
        enc, dec = Encoder(36, 68, 3, 32), Decoder(36, 68, 3, 32)
        x = torch.randn(2, 3, 36, 68)
        y = dec(enc(x))
        self.assertEqual(y.shape, x.shape)
        torch.testing.assert_close(y, dec(enc(x)), rtol=0, atol=0)
        loss, diagnostics = current.checked_loss(enc, dec, x)
        torch.testing.assert_close(loss, ae_loss(enc, dec, x))
        self.assertGreater(diagnostics["max_latent_batch_std"], current.LATENT_STD_FLOOR)
        loss.backward()
        for p in list(enc.parameters()) + list(dec.parameters()):
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
        self.assertGreater(float(enc.conv[0].weight.grad.abs().sum()), 0)
        current.clip_gradients(list(enc.parameters()) + list(dec.parameters()), 1.)
        with torch.no_grad():
            dec.deconv[-1].weight.zero_()
            dec.deconv[-1].bias.copy_(torch.tensor([-1., 0., 1.]))
            out = dec(enc(x))
        torch.testing.assert_close(out[:, 0], torch.full_like(out[:, 0], -1.))
        torch.testing.assert_close(out[:, 2], torch.full_like(out[:, 2], 1.))

    def test_configuration_output_and_checkpoint_isolation(self):
        self.assertEqual(previous.training_config(), current.training_config())
        self.assertEqual(previous.VAL_FRAC, current.VAL_FRAC)
        for e in range(1, 501):
            self.assertEqual(previous.learning_rate(e), current.learning_rate(e))
        out = Path(current.run_dir(4500))
        self.assertEqual(out.parent.name, "iteration_7_paper_frontend")
        self.assertFalse(out.is_relative_to(current.REPO_ROOT))
        self.assertNotEqual(out, Path(previous.run_dir(4500)))
        self.assertNotEqual(current.run_name(), previous.run_name())
        current.check_architecture({"architecture": current.ARCHITECTURE})
        with self.assertRaises(ValueError):
            current.check_architecture({"architecture": previous.ARCHITECTURE})
        with self.assertRaises(ValueError):
            previous.check_architecture({"architecture": current.ARCHITECTURE})

    def test_reject_unsupported_pooling_geometry(self):
        for h, w in [(3, 4), (33, 68), (36, 65)]:
            with self.assertRaisesRegex(ValueError, "divisible by 4"):
                Encoder(h, w, 3, 32)


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
