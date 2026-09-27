"""Functional checks for deterministic compression, gradients and full-grid geometry."""

import unittest
import torch
from torch.nn import functional as F
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from iteration_2_paper_cnn.model import Encoder, Decoder, ae_loss


class PaperCNNTests(unittest.TestCase):
    def test_deterministic_training_and_gradients(self):
        torch.manual_seed(7)
        enc, dec = Encoder(16, 20, 3, 4), Decoder(16, 20, 3, 4)
        enc.train(); dec.train()
        x = torch.randn(2, 3, 16, 20)
        rng = torch.get_rng_state().clone()
        z = enc(x)
        self.assertEqual(tuple(z.shape), (2, 4))
        first = dec(z)
        second = dec(enc(x))
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        loss = ae_loss(enc, dec, x)
        torch.testing.assert_close(loss, F.mse_loss(first, x))
        loss.backward()
        for parameter in list(enc.parameters()) + list(dec.parameters()):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_full_grid_geometry_without_allocating_activations(self):
        with torch.device('meta'):
            enc = Encoder(148, 4000, 3, 32)
            dec = Decoder(148, 4000, 3, 32)
            x = torch.zeros(2, 3, 148, 4000)
            z = enc(x)
            reconstructed = dec(z)
        self.assertEqual(tuple(z.shape), (2, 32))
        self.assertEqual(reconstructed.shape, x.shape)

    def test_signed_velocity_output(self):
        dec = Decoder(16, 20, 3, 4)
        with torch.no_grad():
            dec.deconv[-1].weight.zero_()
            dec.deconv[-1].bias.copy_(torch.tensor([-1., 0., 1.]))
            result = dec(torch.zeros(1, 4))
        torch.testing.assert_close(result[0, 0], torch.full((16, 20), -1.))
        torch.testing.assert_close(result[0, 2], torch.ones(16, 20))


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
