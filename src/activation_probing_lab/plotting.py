from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import resolve_path


def _loss_history(config: dict[str, Any]) -> list[tuple[int, float]]:
    train_dir = resolve_path(config, config["train"]["output_dir"])
    state_paths = list(train_dir.glob("checkpoint-*/trainer_state.json"))
    if (train_dir / "trainer_state.json").exists():
        state_paths.append(train_dir / "trainer_state.json")
    if not state_paths:
        return []

    def state_step(path: Path) -> int:
        if path.parent.name.startswith("checkpoint-"):
            return int(path.parent.name.split("-")[-1])
        return 10**9

    state_path = max(state_paths, key=state_step)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    history = []
    for entry in state.get("log_history", []):
        if "loss" in entry and "step" in entry:
            history.append((int(entry["step"]), float(entry["loss"])))
    return history


def plot_probe_results(config: dict[str, Any], results_path: str | Path) -> list[Path]:
    with Path(results_path).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Probe result file is empty")

    for row in rows:
        row["step"] = int(row["step"])
        row["layer_fraction"] = float(row["layer_fraction"])
        for key in ("transfer_auc", "anchor_transfer_auc", "permuted_transfer_auc"):
            row[key] = float(row[key])

    positive_steps = [row["step"] for row in rows if row["step"] >= 0]
    final_step = max(positive_steps, default=0)
    for row in rows:
        if row["step"] < 0:
            row["step"] = final_step

    colors = {0.25: "#6f7bf7", 0.5: "#26a6a1", 0.75: "#e58b42", 1.0: "#d4587a"}
    losses = _loss_history(config)
    panel_count = 3 if losses else 2
    fig, axes = plt.subplots(1, panel_count, figsize=(6 * panel_count, 4.8))
    probe_axes = axes[1:] if losses else axes
    if losses:
        loss_axis = axes[0]
        loss_axis.plot(
            [step for step, _ in losses],
            [loss for _, loss in losses],
            color="#39445f",
            linewidth=2.4,
        )
        loss_axis.set_title("Training objective")
        loss_axis.set_xlabel("Training step")
        loss_axis.set_ylabel("Completion loss")
        loss_axis.grid(alpha=0.18)

    for axis, label in zip(probe_axes, ("target", "shortcut"), strict=True):
        label_rows = [row for row in rows if row["label"] == label]
        fractions = sorted({row["layer_fraction"] for row in label_rows})
        for fraction in fractions:
            series = sorted(
                (row for row in label_rows if row["layer_fraction"] == fraction),
                key=lambda row: row["step"],
            )
            axis.plot(
                [row["step"] for row in series],
                [row["transfer_auc"] for row in series],
                marker="o",
                linewidth=2,
                color=colors.get(fraction),
                label=f"{fraction:.0%} depth",
            )
            axis.plot(
                [row["step"] for row in series],
                [row["anchor_transfer_auc"] for row in series],
                linestyle=":",
                linewidth=1.4,
                alpha=0.75,
                color=colors.get(fraction),
            )
        axis.axhline(0.5, color="#8a90a0", linestyle="--", linewidth=1)
        axis.set_title(f"{label.title()} decodability")
        axis.set_xlabel("Training step")
        axis.grid(alpha=0.18)
        axis.set_ylim(0.35, 1.02)
    probe_axes[0].set_ylabel("Transfer AUROC")
    probe_axes[1].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("Loss and checkpoint activation probes", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        0.01,
        "Solid: probe refit per checkpoint    Dotted: step-0 probe transferred forward",
        ha="center",
        fontsize=9,
        color="#555b68",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))

    output_dir = resolve_path(config, config["probe"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "probe_trajectories.png"
    svg_path = output_dir / "probe_trajectories.svg"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plots to {png_path} and {svg_path}")
    return [png_path, svg_path]
