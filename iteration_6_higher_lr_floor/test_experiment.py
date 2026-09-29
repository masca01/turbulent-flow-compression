"""Check the LR-floor experiment without real data or optimizer updates."""

from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from iteration_5_higher_lr import train_higher_lr as previous
from iteration_6_higher_lr_floor import train_higher_lr_floor as current


class ExperimentTests(unittest.TestCase):
    def test_controlled_configuration_and_initialization(self):
        before, after = previous.training_config(), current.training_config()
        before.pop("min_lr")
        after.pop("min_lr")
        self.assertEqual(before, after)
        self.assertEqual(previous.VAL_FRAC, current.VAL_FRAC)
        for key in ("Encoder", "Decoder", "checked_loss", "clip_gradients"):
            self.assertIs(getattr(previous, key), getattr(current, key))
        with mock.patch.object(previous, "DEVICE", torch.device("cpu")), \
             mock.patch.object(current, "DEVICE", torch.device("cpu")):
            old_models = previous.build(4, 4, 3)
            new_models = current.build(4, 4, 3)
            for old, new in zip(old_models, new_models):
                for key, value in old.state_dict().items():
                    torch.testing.assert_close(value, new.state_dict()[key], rtol=0, atol=0)

    def test_schedule_is_bounded_and_higher_after_first_epoch(self):
        rates = [current.learning_rate(e) for e in range(1, 501)]
        self.assertAlmostEqual(rates[0], 3e-4)
        self.assertAlmostEqual(rates[-1], 1e-4)
        self.assertAlmostEqual(rates[0], previous.learning_rate(1))
        self.assertTrue(all(1e-4 <= lr <= 3e-4 for lr in rates))
        self.assertTrue(all(a >= b for a, b in zip(rates, rates[1:])))
        for epoch in range(2, 501):
            self.assertGreater(rates[epoch - 1], previous.learning_rate(epoch))

    def test_outputs_and_checkpoint_names_are_separate(self):
        output = Path(current.run_dir(4500))
        self.assertEqual(output.parent.name, "iteration_6_higher_lr_floor")
        self.assertNotEqual(output, Path(previous.run_dir(4500)))
        self.assertNotEqual(current.run_name(), previous.run_name())
        self.assertFalse(output.is_relative_to(current.REPO_ROOT))


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
