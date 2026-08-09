from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..capture import checkpoint_step, layer_indices
from ..config import resolve_path
from ..io import read_jsonl, write_jsonl
from ..training import _chat_text

_CHECKPOINT_PATTERN = re.compile(r"^(\d+)_adapters\.safetensors$")
_TRAIN_METRIC_PATTERN = re.compile(
    r"Iter (?P<step>\d+): Train loss (?P<loss>[0-9.]+).*?"
    r"It/sec (?P<iters>[0-9.]+), Tokens/sec (?P<tokens>[0-9.]+).*?"
    r"Peak mem (?P<memory>[0-9.]+) GB"
)


def _require_mlx() -> tuple[Any, Any]:
    try:
        import mlx.core as mx
        from mlx_lm import load
    except ImportError as error:
        raise RuntimeError(
            "MLX dependencies are missing. Install with: pip install -e '.[mlx]'"
        ) from error
    if not mx.metal.is_available():
        raise RuntimeError("The MLX backend requires Apple Silicon with Metal available.")
    return mx, load


def prepare_mlx_data(config: dict[str, Any], output_dir: Path) -> Path:
    rows = read_jsonl(resolve_path(config, config["data"]["train_path"]))
    data_dir = output_dir / "mlx-data"
    write_jsonl(
        data_dir / "train.jsonl",
        ({"prompt": row["prompt"], "completion": row["response"]} for row in rows),
    )
    return data_dir


def mlx_training_config(config: dict[str, Any], data_dir: Path, output_dir: Path) -> dict[str, Any]:
    train = config["train"]
    lora = train["lora"]
    rank = int(lora.get("rank", 16))
    alpha = float(lora.get("alpha", rank * 2))
    mlx_config: dict[str, Any] = {
        "model": config["model"]["name"],
        "train": True,
        "fine_tune_type": "lora",
        "optimizer": train.get("optimizer", "adamw"),
        "data": str(data_dir),
        "seed": int(config.get("seed", 42)),
        "num_layers": int(train.get("num_layers", -1)),
        "batch_size": int(train.get("batch_size", 1)),
        "iters": int(train["max_steps"]),
        "val_batches": 0,
        "learning_rate": float(train.get("learning_rate", 2e-4)),
        "steps_per_report": int(train.get("logging_steps", 1)),
        "steps_per_eval": int(train.get("eval_steps", int(train["max_steps"]) + 1)),
        "grad_accumulation_steps": int(train.get("gradient_accumulation_steps", 1)),
        "adapter_path": str(output_dir),
        "save_every": int(train.get("save_steps", train["max_steps"])),
        "max_seq_length": int(config["model"].get("max_length", 256)),
        "grad_checkpoint": bool(train.get("gradient_checkpointing", False)),
        "clear_cache_threshold": train.get("clear_cache_threshold", 0),
        "mask_prompt": True,
        "lora_parameters": {
            "rank": rank,
            "dropout": float(lora.get("dropout", 0.0)),
            "scale": alpha / rank,
        },
    }
    if keys := lora.get("keys"):
        mlx_config["lora_parameters"]["keys"] = list(keys)
    return mlx_config


def _stream_command(command: list[str], log_path: Path) -> tuple[float, list[dict[str, Any]]]:
    started = time.perf_counter()
    metrics: list[dict[str, Any]] = []
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            match = _TRAIN_METRIC_PATTERN.search(line)
            if match:
                metrics.append(
                    {
                        "step": int(match.group("step")),
                        "loss": float(match.group("loss")),
                        "iterations_per_second": float(match.group("iters")),
                        "tokens_per_second": float(match.group("tokens")),
                        "peak_memory_gb": float(match.group("memory")),
                    }
                )
        return_code = process.wait()
    elapsed = time.perf_counter() - started
    if return_code != 0:
        raise RuntimeError(
            f"MLX-LM training exited with status {return_code}. See {log_path}."
        )
    return elapsed, metrics


def train_mlx(config: dict[str, Any]) -> Path:
    _require_mlx()
    output_dir = resolve_path(config, config["train"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = prepare_mlx_data(config, output_dir)
    generated_config = mlx_training_config(config, data_dir, output_dir)
    config_path = output_dir / "mlx_lora_config.yaml"
    config_path.write_text(yaml.safe_dump(generated_config, sort_keys=False), encoding="utf-8")

    command = [sys.executable, "-m", "mlx_lm", "lora", "--config", str(config_path)]
    print(f"Running MLX-LM training with {config_path}")
    elapsed, metrics = _stream_command(command, output_dir / "training.log")
    summary = {
        "backend": "mlx",
        "model": config["model"]["name"],
        "python": sys.version.split()[0],
        "mlx": version("mlx"),
        "mlx_lm": version("mlx-lm"),
        "wall_time_seconds": elapsed,
        "metrics": metrics,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"MLX pilot completed in {elapsed:.1f}s. Saved adapters to {output_dir}")
    return output_dir


def discover_mlx_checkpoints(config: dict[str, Any]) -> list[tuple[str, Path | None]]:
    output_dir = resolve_path(config, config["train"]["output_dir"])
    checkpoints: list[tuple[str, Path | None]] = [("base", None)]
    numbered: list[tuple[int, Path]] = []
    for path in output_dir.glob("*_adapters.safetensors"):
        match = _CHECKPOINT_PATTERN.match(path.name)
        if match:
            numbered.append((int(match.group(1)), path))
    for step, path in sorted(numbered):
        checkpoints.append((f"checkpoint-{step}", path))
    final_path = output_dir / "adapters.safetensors"
    if not numbered and final_path.exists():
        checkpoints.append(("final-adapter", final_path))
    return checkpoints


def _load_mlx_model(
    config: dict[str, Any], adapter_weights: Path | None
) -> tuple[Any, Any, Any]:
    mx, load = _require_mlx()
    model, tokenizer = load(
        config["model"]["name"], tokenizer_config={"trust_remote_code": True}
    )
    if adapter_weights is not None:
        from mlx_lm.tuner.utils import linear_to_lora_layers

        adapter_config_path = adapter_weights.parent / "adapter_config.json"
        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        linear_to_lora_layers(
            model,
            int(adapter_config["num_layers"]),
            adapter_config["lora_parameters"],
            use_dora=adapter_config.get("fine_tune_type") == "dora",
        )
        model.load_weights(str(adapter_weights), strict=False)
    model.eval()
    return mx, model, tokenizer


def _qwen35_residuals(
    mx: Any, model: Any, token_ids: Any, selected_layers: list[int]
) -> list[Any]:
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask

    model_type = str(getattr(model, "model_type", ""))
    if model_type != "qwen3_5":
        raise ValueError(
            "The first MLX activation adapter supports qwen3_5 only. "
            f"Received model_type={model_type!r}."
        )
    core = model.language_model.model
    hidden = core.embed_tokens(token_ids)
    full_attention_mask = create_attention_mask(hidden, None)
    state_space_mask = create_ssm_mask(hidden, None)
    selected = set(selected_layers)
    captured: list[Any] = []
    for index, layer in enumerate(core.layers, start=1):
        mask = state_space_mask if layer.is_linear else full_attention_mask
        hidden = layer(hidden, mask=mask, cache=None)
        if index in selected:
            captured.append(hidden)
    if len(captured) != len(selected_layers):
        raise RuntimeError("Failed to capture every configured MLX layer.")
    return captured


def _token_batch(
    tokenizer: Any, prompts: list[str], max_length: int
) -> tuple[np.ndarray, np.ndarray]:
    encoded = [tokenizer.encode(prompt)[:max_length] for prompt in prompts]
    if any(not item for item in encoded):
        raise ValueError("A probe prompt encoded to an empty token sequence.")
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    width = max(len(item) for item in encoded)
    batch = np.full((len(encoded), width), pad_id, dtype=np.int32)
    positions = np.empty(len(encoded), dtype=np.int32)
    for index, item in enumerate(encoded):
        batch[index, : len(item)] = item
        positions[index] = len(item) - 1
    return batch, positions


def capture_mlx_checkpoint(
    config: dict[str, Any],
    name: str,
    adapter_weights: Path | None,
    rows: list[dict[str, Any]],
) -> tuple[Path, dict[str, Any]]:
    mx, model, tokenizer = _load_mlx_model(config, adapter_weights)
    prompts = [
        _chat_text(
            tokenizer,
            [{"role": "user", "content": row["prompt"]}],
            add_generation_prompt=True,
        )
        for row in rows
    ]
    fractions = [float(value) for value in config["capture"]["layer_fractions"]]
    num_layers = len(model.layers)
    selected_layers = layer_indices(num_layers, fractions)
    batch_size = int(config["capture"].get("batch_size", 8))
    max_length = int(config["model"].get("max_length", 256))
    layer_batches: list[list[np.ndarray]] = [[] for _ in selected_layers]

    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    started = time.perf_counter()
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        token_array, token_positions = _token_batch(tokenizer, batch_prompts, max_length)
        token_ids = mx.array(token_array)
        hidden_states = _qwen35_residuals(mx, model, token_ids, selected_layers)
        batch_indices = mx.arange(len(batch_prompts))
        position_array = mx.array(token_positions)
        vectors = [
            hidden[batch_indices, position_array].astype(mx.float32)
            for hidden in hidden_states
        ]
        mx.eval(*vectors)
        for output_index, vector in enumerate(vectors):
            layer_batches[output_index].append(np.asarray(vector))

    activations = np.stack(
        [np.concatenate(chunks, axis=0) for chunks in layer_batches], axis=0
    ).astype(np.float32)
    output_dir = resolve_path(config, config["capture"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name}.npz"
    np.savez_compressed(
        output_path,
        activations=activations,
        ids=np.asarray([row["id"] for row in rows]),
        target=np.asarray([row["target"] for row in rows], dtype=np.int8),
        shortcut=np.asarray([row["shortcut"] for row in rows], dtype=np.int8),
        split=np.asarray([row["split"] for row in rows]),
        group=np.asarray([row["group"] for row in rows]),
        layer_indices=np.asarray(selected_layers, dtype=np.int16),
        layer_fractions=np.asarray(fractions, dtype=np.float32),
        checkpoint=np.asarray(name),
        step=np.asarray(checkpoint_step(name), dtype=np.int32),
        backend=np.asarray("mlx"),
        activation_site=np.asarray("post_block_residual_final_prompt_token"),
    )
    elapsed = time.perf_counter() - started
    peak_memory = float(mx.get_peak_memory()) / 1e9
    metric = {
        "checkpoint": name,
        "examples": len(rows),
        "seconds": elapsed,
        "examples_per_second": len(rows) / elapsed,
        "peak_memory_gb": peak_memory,
    }
    print(
        f"Saved {activations.shape} activations to {output_path} in {elapsed:.1f}s "
        f"({metric['examples_per_second']:.1f} examples/s, {peak_memory:.2f} GB peak)"
    )
    del model
    mx.clear_cache()
    return output_path, metric


def capture_all_mlx(config: dict[str, Any]) -> list[Path]:
    rows = read_jsonl(resolve_path(config, config["data"]["probe_path"]))
    checkpoints = discover_mlx_checkpoints(config)
    if len(checkpoints) == 1:
        print("No MLX adapter checkpoints found. Capturing the base model only.")
    paths: list[Path] = []
    metrics: list[dict[str, Any]] = []
    for name, adapter_weights in checkpoints:
        path, metric = capture_mlx_checkpoint(config, name, adapter_weights, rows)
        paths.append(path)
        metrics.append(metric)
    metrics_path = resolve_path(config, config["capture"]["output_dir"]) / "capture_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return paths


class MlxBackend:
    name = "mlx"

    def train(self, config: dict[str, Any]) -> Path:
        return train_mlx(config)

    def capture_all(self, config: dict[str, Any]) -> list[Path]:
        return capture_all_mlx(config)
