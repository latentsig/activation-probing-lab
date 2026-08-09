import tempfile
import unittest
from pathlib import Path

from activation_probing_lab.io import read_jsonl
from activation_probing_lab.toy_data import generate_toy_data, target_rule


class ToyDataTest(unittest.TestCase):
    def test_target_rule_is_xor(self) -> None:
        self.assertEqual(target_rule("cobalt", "north"), 0)
        self.assertEqual(target_rule("cobalt", "south"), 1)
        self.assertEqual(target_rule("amber", "north"), 1)
        self.assertEqual(target_rule("amber", "south"), 0)

    def test_probe_splits_are_balanced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = {
                "seed": 7,
                "_root": directory,
                "data": {
                    "train_path": "data/train.jsonl",
                    "probe_path": "data/probe.jsonl",
                    "train_examples": 100,
                    "probe_train_examples": 64,
                    "probe_id_examples": 32,
                    "probe_transfer_examples": 32,
                    "shortcut_strength": 0.95,
                },
            }
            paths = generate_toy_data(config)
            rows = read_jsonl(paths["probe"])
            for split in ("probe_train", "probe_id", "probe_transfer"):
                split_rows = [row for row in rows if row["split"] == split]
                target_counts = [
                    sum(row["target"] == value for row in split_rows) for value in (0, 1)
                ]
                shortcut_counts = [
                    sum(row["shortcut"] == value for row in split_rows) for value in (0, 1)
                ]
                self.assertLessEqual(abs(target_counts[0] - target_counts[1]), 1)
                self.assertLessEqual(abs(shortcut_counts[0] - shortcut_counts[1]), 1)
            self.assertTrue(Path(paths["train"]).exists())


if __name__ == "__main__":
    unittest.main()
