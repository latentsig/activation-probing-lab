import tempfile
import unittest
from pathlib import Path

from activation_probing_lab.backends import backend_name, get_backend
from activation_probing_lab.backends.mlx import (
    discover_mlx_checkpoints,
    mlx_training_config,
)


class BackendTest(unittest.TestCase):
    def test_backend_defaults_to_cuda(self) -> None:
        self.assertEqual(backend_name({}), "cuda")
        self.assertEqual(get_backend({}).name, "cuda")

    def test_mlx_backend_is_selected(self) -> None:
        self.assertEqual(get_backend({"backend": "mlx"}).name, "mlx")

    def test_unknown_backend_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            backend_name({"backend": "mps"})

    def test_mlx_config_maps_pilot_steps_and_lora_scale(self) -> None:
        config = {
            "seed": 7,
            "model": {"name": "mlx-community/test", "max_length": 192},
            "train": {
                "max_steps": 10,
                "batch_size": 1,
                "gradient_accumulation_steps": 1,
                "learning_rate": 2e-4,
                "logging_steps": 1,
                "save_steps": 2,
                "num_layers": -1,
                "lora": {"rank": 16, "alpha": 32, "dropout": 0.05},
            },
        }
        generated = mlx_training_config(config, Path("data"), Path("checkpoints"))
        self.assertEqual(generated["iters"], 10)
        self.assertEqual(generated["save_every"], 2)
        self.assertEqual(generated["lora_parameters"]["scale"], 2.0)
        self.assertTrue(generated["mask_prompt"])

    def test_mlx_checkpoint_discovery_uses_numbered_adapter_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_dir = root / "checkpoints"
            checkpoint_dir.mkdir()
            for step in (10, 2, 8):
                (checkpoint_dir / f"{step:07d}_adapters.safetensors").touch()
            config = {
                "_root": directory,
                "train": {"output_dir": "checkpoints"},
            }
            names = [name for name, _ in discover_mlx_checkpoints(config)]
            self.assertEqual(names, ["base", "checkpoint-2", "checkpoint-8", "checkpoint-10"])


if __name__ == "__main__":
    unittest.main()
