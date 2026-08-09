from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .capture import checkpoint_step
from .config import resolve_path


@dataclass
class ProbeFit:
    model: Pipeline
    c: float
    cv_auc: float


def _pipeline(c: float) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "probe",
                LogisticRegression(
                    C=c,
                    solver="lbfgs",
                    max_iter=2_000,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def fit_regularized_probe(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    c_grid: list[float],
    cv_folds: int,
    seed: int,
) -> ProbeFit:
    unique_groups = np.unique(groups)
    folds = min(cv_folds, len(unique_groups))
    if folds < 2:
        raise ValueError("At least two groups are required for probe cross-validation")
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)

    best_c = float(c_grid[0])
    best_auc = float("-inf")
    for c in c_grid:
        fold_scores: list[float] = []
        for train_index, validation_index in splitter.split(x, y, groups):
            model = _pipeline(float(c))
            model.fit(x[train_index], y[train_index])
            scores = model.predict_proba(x[validation_index])[:, 1]
            fold_scores.append(float(roc_auc_score(y[validation_index], scores)))
        mean_auc = float(np.mean(fold_scores))
        if mean_auc > best_auc:
            best_c = float(c)
            best_auc = mean_auc

    model = _pipeline(best_c)
    model.fit(x, y)
    return ProbeFit(model=model, c=best_c, cv_auc=best_auc)


def bootstrap_auc(
    y: np.ndarray,
    scores: np.ndarray,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if samples <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(samples):
        indices = rng.integers(0, len(y), len(y))
        if np.unique(y[indices]).size < 2:
            continue
        estimates.append(float(roc_auc_score(y[indices], scores[indices])))
    if not estimates:
        return float("nan"), float("nan")
    return tuple(float(value) for value in np.percentile(estimates, [2.5, 97.5]))


def _score(model: Pipeline, x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    scores = model.predict_proba(x)[:, 1]
    return float(roc_auc_score(y, scores)), scores


def activation_files(config: dict[str, Any]) -> list[Path]:
    directory = resolve_path(config, config["capture"]["output_dir"])
    files = list(directory.glob("*.npz"))
    return sorted(files, key=lambda path: checkpoint_step(path.stem))


def run_probes(config: dict[str, Any]) -> Path:
    files = activation_files(config)
    if not files:
        raise FileNotFoundError("No activation files found. Run `apl capture` first.")

    probe_settings = config["probe"]
    seed = int(config.get("seed", 42))
    c_grid = [float(value) for value in probe_settings.get("c_grid", [0.001, 0.01, 0.1, 1.0])]
    cv_folds = int(probe_settings.get("cv_folds", 5))
    bootstrap_samples = int(probe_settings.get("bootstrap_samples", 200))
    permutation_repeats = int(probe_settings.get("permutation_repeats", 10))

    base_file = next((path for path in files if path.stem == "base"), files[0])
    with np.load(base_file) as base:
        base_activations = base["activations"].copy()
        base_split = base["split"].astype(str)
        base_groups = base["group"].astype(str)
        base_labels = {
            "target": base["target"].astype(int),
            "shortcut": base["shortcut"].astype(int),
        }
        layer_fractions = base["layer_fractions"].astype(float)
    train_mask = base_split == "probe_train"
    anchor_models: dict[tuple[int, str], Pipeline] = {}
    for layer_index in range(base_activations.shape[0]):
        for label_name, labels in base_labels.items():
            fitted = fit_regularized_probe(
                base_activations[layer_index][train_mask],
                labels[train_mask],
                base_groups[train_mask],
                c_grid,
                cv_folds,
                seed + layer_index,
            )
            anchor_models[(layer_index, label_name)] = fitted.model

    results: list[dict[str, Any]] = []
    for file_index, path in enumerate(files):
        with np.load(path) as data:
            activations = data["activations"].copy()
            split = data["split"].astype(str)
            groups = data["group"].astype(str)
            labels_by_name = {
                "target": data["target"].astype(int),
                "shortcut": data["shortcut"].astype(int),
            }
            checkpoint_name = str(data["checkpoint"])
            step = int(data["step"])

        masks = {
            "train": split == "probe_train",
            "id": split == "probe_id",
            "transfer": split == "probe_transfer",
        }
        for layer_index, fraction in enumerate(layer_fractions):
            for label_offset, (label_name, labels) in enumerate(labels_by_name.items()):
                fitted = fit_regularized_probe(
                    activations[layer_index][masks["train"]],
                    labels[masks["train"]],
                    groups[masks["train"]],
                    c_grid,
                    cv_folds,
                    seed + file_index * 100 + layer_index * 10 + label_offset,
                )
                id_auc, _ = _score(
                    fitted.model,
                    activations[layer_index][masks["id"]],
                    labels[masks["id"]],
                )
                transfer_auc, transfer_scores = _score(
                    fitted.model,
                    activations[layer_index][masks["transfer"]],
                    labels[masks["transfer"]],
                )
                ci_low, ci_high = bootstrap_auc(
                    labels[masks["transfer"]],
                    transfer_scores,
                    bootstrap_samples,
                    seed + file_index * 1_000 + layer_index * 100 + label_offset,
                )

                permutation_aucs: list[float] = []
                for repeat in range(permutation_repeats):
                    rng = np.random.default_rng(
                        seed
                        + 10_000
                        + file_index * 1_000
                        + layer_index * 100
                        + label_offset * 10
                        + repeat
                    )
                    permuted_labels = rng.permutation(labels[masks["train"]])
                    permutation_model = _pipeline(fitted.c)
                    permutation_model.fit(
                        activations[layer_index][masks["train"]], permuted_labels
                    )
                    permuted_auc, _ = _score(
                        permutation_model,
                        activations[layer_index][masks["transfer"]],
                        labels[masks["transfer"]],
                    )
                    permutation_aucs.append(permuted_auc)
                anchor_auc, _ = _score(
                    anchor_models[(layer_index, label_name)],
                    activations[layer_index][masks["transfer"]],
                    labels[masks["transfer"]],
                )
                results.append(
                    {
                        "checkpoint": checkpoint_name,
                        "step": step,
                        "layer_fraction": round(float(fraction), 4),
                        "label": label_name,
                        "selected_c": fitted.c,
                        "cv_auc": fitted.cv_auc,
                        "id_auc": id_auc,
                        "transfer_auc": transfer_auc,
                        "transfer_ci_low": ci_low,
                        "transfer_ci_high": ci_high,
                        "permuted_transfer_auc": float(np.mean(permutation_aucs)),
                        "permuted_transfer_auc_std": float(np.std(permutation_aucs)),
                        "anchor_transfer_auc": anchor_auc,
                    }
                )

    output_dir = resolve_path(config, probe_settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "probe_results.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {len(results)} probe measurements to {output_path}")
    return output_path
