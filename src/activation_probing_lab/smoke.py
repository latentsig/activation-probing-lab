from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from .config import resolve_path
from .io import read_jsonl
from .plotting import plot_probe_results
from .probes import run_probes
from .toy_data import generate_toy_data


def smoke_demo(config: dict[str, Any], output_dir: str | Path) -> list[Path]:
    demo_config = deepcopy(config)
    output = Path(output_dir).expanduser().resolve()
    demo_config["capture"]["output_dir"] = str(output / "activations")
    demo_config["probe"]["output_dir"] = str(output / "report")
    demo_config["train"]["output_dir"] = str(output / "checkpoints")
    demo_config["probe"]["bootstrap_samples"] = min(
        100, int(demo_config["probe"].get("bootstrap_samples", 100))
    )

    generate_toy_data(demo_config)
    rows = read_jsonl(resolve_path(demo_config, demo_config["data"]["probe_path"]))
    target = np.asarray([row["target"] for row in rows], dtype=np.int8)
    shortcut = np.asarray([row["shortcut"] for row in rows], dtype=np.int8)
    split = np.asarray([row["split"] for row in rows])
    groups = np.asarray([row["group"] for row in rows])
    ids = np.asarray([row["id"] for row in rows])

    rng = np.random.default_rng(int(demo_config.get("seed", 42)))
    steps = (0, 20, 40, 60, 80, 100)
    fractions = np.asarray(demo_config["capture"]["layer_fractions"], dtype=np.float32)
    dimensions = 64
    base_noise = rng.normal(0, 1, size=(len(fractions), len(rows), dimensions))
    target_directions = rng.normal(size=(len(fractions), dimensions))
    shortcut_directions = rng.normal(size=(len(fractions), dimensions))
    target_directions /= np.linalg.norm(target_directions, axis=1, keepdims=True)
    shortcut_directions /= np.linalg.norm(shortcut_directions, axis=1, keepdims=True)

    activation_dir = Path(demo_config["capture"]["output_dir"])
    activation_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(demo_config["train"]["output_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    synthetic_losses = [2.4, 1.72, 1.31, 1.08, 0.94, 0.86]
    (checkpoint_dir / "trainer_state.json").write_text(
        json.dumps(
            {
                "log_history": [
                    {"step": step, "loss": loss}
                    for step, loss in zip(steps, synthetic_losses, strict=True)
                ]
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    target_sign = (target * 2 - 1).astype(np.float32)
    shortcut_sign = (shortcut * 2 - 1).astype(np.float32)
    depth_gain = np.asarray([0.35, 0.8, 1.0, 0.65], dtype=np.float32)
    for step in steps:
        progress = step / max(steps)
        activations = base_noise.copy()
        for layer_index in range(len(fractions)):
            target_amplitude = (0.05 + 1.8 * progress) * depth_gain[layer_index]
            shortcut_amplitude = (0.9 + 0.7 * progress) * (0.6 + depth_gain[layer_index])
            activations[layer_index] += (
                target_amplitude
                * target_sign[:, None]
                * target_directions[layer_index][None, :]
            )
            activations[layer_index] += (
                shortcut_amplitude
                * shortcut_sign[:, None]
                * shortcut_directions[layer_index][None, :]
            )
            activations[layer_index] += rng.normal(
                0, 0.08, size=activations[layer_index].shape
            )
        name = "base" if step == 0 else f"checkpoint-{step}"
        np.savez_compressed(
            activation_dir / f"{name}.npz",
            activations=activations.astype(np.float32),
            ids=ids,
            target=target,
            shortcut=shortcut,
            split=split,
            group=groups,
            layer_indices=np.asarray([9, 18, 27, 36], dtype=np.int16),
            layer_fractions=fractions,
            checkpoint=np.asarray(name),
            step=np.asarray(step, dtype=np.int32),
        )

    results_path = run_probes(demo_config)
    plots = plot_probe_results(demo_config, results_path)
    print(f"CPU smoke demo complete. Open {plots[0]}")
    return [results_path, *plots]
