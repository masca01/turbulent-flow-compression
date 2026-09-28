"""CPU model checks only: no training loop or optimiser updates."""
import io
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F
from iteration_3_wide_deterministic.model import ARCHITECTURE, Encoder, Decoder, ae_loss
from iteration_3_wide_deterministic import train_wide_deterministic as training


class WideDeterministicTests(unittest.TestCase):
    def test_deterministic_training_and_mse_gradients(self):
        torch.manual_seed(7)
        enc, dec = Encoder(33, 65, 3, 32), Decoder(33, 65, 3, 32)
        enc.train(); dec.train()
        x = torch.randn(2, 3, 33, 65)
        rng = torch.get_rng_state().clone()
        z = enc(x)
        self.assertEqual(tuple(z.shape), (2, 32))
        first = dec(z)
        self.assertEqual(first.shape, x.shape)
        torch.testing.assert_close(first, dec(enc(x)), rtol=0, atol=0)
        loss = ae_loss(enc, dec, x)
        torch.testing.assert_close(loss, F.mse_loss(first, x))
        loss.backward()
        for parameter in list(enc.parameters()) + list(dec.parameters()):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        enc.eval(); dec.eval()
        with torch.no_grad():
            torch.testing.assert_close(first, dec(enc(x)), rtol=0, atol=0)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_full_challenge_grid_geometry(self):
        with torch.device("meta"):
            enc, dec = Encoder(148, 4000, 3, 32), Decoder(148, 4000, 3, 32)
            x = torch.zeros(32, 3, 148, 4000)
            z = enc(x)
            result = dec(z)
        self.assertEqual(tuple(z.shape), (32, 32))
        self.assertEqual(result.shape, x.shape)

    def test_signed_output_and_checkpoint_roundtrip(self):
        dec = Decoder(33, 65, 3, 32)
        with torch.no_grad():
            dec.deconv[-1].weight.zero_()
            dec.deconv[-1].bias.copy_(torch.tensor([-1., 0., 1.]))
        buffer = io.BytesIO()
        torch.save({"architecture": ARCHITECTURE, "dec_state": dec.state_dict()}, buffer)
        buffer.seek(0)
        ck = torch.load(buffer, map_location="cpu", weights_only=True)
        training.check_architecture(ck)
        restored = Decoder(33, 65, 3, 32)
        restored.load_state_dict(ck["dec_state"], strict=True)
        with torch.no_grad():
            out = restored(torch.zeros(1, 32))
        torch.testing.assert_close(out[0, 0], torch.full((33, 65), -1.))
        torch.testing.assert_close(out[0, 2], torch.full((33, 65), 1.))
        for other in ("challenge11_residual_v1", "challenge11_paper_cnn_v1"):
            with self.assertRaises(ValueError):
                training.check_architecture({"architecture": other})


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
