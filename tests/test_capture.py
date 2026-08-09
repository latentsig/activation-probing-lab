import unittest

from activation_probing_lab.capture import checkpoint_step, layer_indices


class CaptureUtilitiesTest(unittest.TestCase):
    def test_layer_fractions_map_to_hidden_state_indices(self) -> None:
        self.assertEqual(layer_indices(36, [0.25, 0.5, 0.75, 1.0]), [9, 18, 27, 36])

    def test_invalid_layer_fraction_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            layer_indices(36, [0.0])

    def test_checkpoint_steps(self) -> None:
        self.assertEqual(checkpoint_step("base"), 0)
        self.assertEqual(checkpoint_step("checkpoint-40"), 40)
        self.assertEqual(checkpoint_step("final-adapter"), -1)


if __name__ == "__main__":
    unittest.main()
