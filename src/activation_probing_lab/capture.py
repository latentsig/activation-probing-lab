from __future__ import annotations

import gc
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from .config import resolve_path
from .io import read_jsonl
from .training import _chat_text


def layer_indices(num_layers: int, fractions: list[float]) -> list[int]:
    if num_layers < 1:
        raise ValueError("num_layers must be positive")
    indices = []
    for fraction in fractions:
        if not 0 < fraction <= 1:
            raise ValueError("Layer fractions must be in (0, 1]")
        indices.append(max(1, min(num_layers, round(fraction * num_layers))))
    return indices


def checkpoint_step(path: str | Path) -> int:
    name = Path(path).name
    if name == "base":
        return 0
    match = re.search(r"checkpoint-(\d+)", name)
    if match:
        return int(match.group(1))
    return -1


def discover_checkpoints(config: dict[str, Any]) -> list[tuple[str, Path | None]]:
    train_dir = resolve_path(config, config["train"]["output_dir"])
    checkpoints: list[tuple[str, Path | None]] = [("base", None)]
    for path in sorted(
        train_dir.glob("checkpoint-*"), key=lambda item: checkpoint_step(item)
    ):
        if (path / "adapter_config.json").exists():
            checkpoints.append((path.name, path))
    final_adapter = train_dir / "final-adapter"
    if final_adapter.exists() and not checkpoints[1:]:
        checkpoints.append(("final-adapter", final_adapter))
    return checkpoints


def _load_model(config: dict[str, Any], adapter_path: Path | None) -> tuple[Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as error:
        raise RuntimeError(
            "Capture dependencies are missing. Install with: pip install -e '.[train]'"
        ) from error

    model_settings = config["model"]
    model_name = model_settings["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    load_kwargs: dict[str, Any] = {"torch_dtype": "auto"}
    if torch.cuda.is_available():
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        load_kwargs.update(
            {
                "device_map": {"": int(os.environ.get("LOCAL_RANK", "0"))},
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=True,
                ),
            }
        )
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.eval()
    return model, tokenizer


def _render_probe_prompts(tokenizer: Any, rows: list[dict[str, Any]]) -> list[str]:
    return [
        _chat_text(
            tokenizer,
            [{"role": "user", "content": row["prompt"]}],
            add_generation_prompt=True,
        )
        for row in rows
    ]


def capture_checkpoint(
    config: dict[str, Any],
    name: str,
    adapter_path: Path | None,
    rows: list[dict[str, Any]],
) -> Path:
    import torch

    model, tokenizer = _load_model(config, adapter_path)
    prompts = _render_probe_prompts(tokenizer, rows)
    fractions = [float(value) for value in config["capture"]["layer_fractions"]]
    num_layers = int(model.config.num_hidden_layers)
    selected_layers = layer_indices(num_layers, fractions)
    batch_size = int(config["capture"].get("batch_size", 8))
    max_length = int(config["model"].get("max_length", 256))
    device = model.get_input_embeddings().weight.device
    layer_batches: list[list[np.ndarray]] = [[] for _ in selected_layers]

    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            encoded = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
            ).to(device)
            outputs = model(**encoded, output_hidden_states=True, use_cache=False, return_dict=True)
            token_positions = encoded["attention_mask"].sum(dim=1) - 1
            batch_indices = torch.arange(len(batch_prompts), device=device)
            for output_index, hidden_index in enumerate(selected_layers):
                vectors = outputs.hidden_states[hidden_index][batch_indices, token_positions]
                layer_batches[output_index].append(vectors.float().cpu().numpy())

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
    )
    print(f"Saved {activations.shape} activations to {output_path}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output_path


def capture_all(config: dict[str, Any]) -> list[Path]:
    probe_rows = read_jsonl(resolve_path(config, config["data"]["probe_path"]))
    checkpoints = discover_checkpoints(config)
    if len(checkpoints) == 1:
        print("No adapter checkpoints found. Capturing the base model only.")
    return [capture_checkpoint(config, name, path, probe_rows) for name, path in checkpoints]
