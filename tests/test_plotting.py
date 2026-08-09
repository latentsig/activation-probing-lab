import json
import tempfile
import unittest
from pathlib import Path

from activation_probing_lab.plotting import _loss_history


class PlottingTest(unittest.TestCase):
    def test_mlx_manifest_supplies_loss_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "checkpoints"
            output.mkdir()
            (output / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "metrics": [
                            {"step": 1, "loss": 2.5},
                            {"step": 2, "loss": 1.25},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            config = {"_root": directory, "train": {"output_dir": "checkpoints"}}
            self.assertEqual(_loss_history(config), [(1, 2.5), (2, 1.25)])


if __name__ == "__main__":
    unittest.main()
