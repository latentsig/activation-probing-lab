from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import resolve_path
from .io import read_jsonl


def _chat_text(tokenizer: Any, messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def tokenize_completion(tokenizer: Any, row: dict[str, Any], max_length: int) -> dict[str, Any]:
    prompt_messages = [{"role": "user", "content": row["prompt"]}]
    full_messages = prompt_messages + [{"role": "assistant", "content": row["response"]}]
    prompt_text = _chat_text(tokenizer, prompt_messages, add_generation_prompt=True)
    full_text = _chat_text(tokenizer, full_messages, add_generation_prompt=False)

    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"]
    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    input_ids = encoded["input_ids"]
    prompt_length = min(len(prompt_ids), len(input_ids))
    labels = [-100] * prompt_length + input_ids[prompt_length:]
    if not any(label != -100 for label in labels):
        raise ValueError("The configured max_length removed the entire completion")
    return {
        "input_ids": input_ids,
        "attention_mask": encoded["attention_mask"],
        "labels": labels,
    }


@dataclass
class CompletionCollator:
    tokenizer: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        max_length = max(len(feature["input_ids"]) for feature in features)
        pad_id = self.tokenizer.pad_token_id
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            padding = max_length - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [pad_id] * padding)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * padding)
            batch["labels"].append(feature["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def train_qlora(config: dict[str, Any]) -> Path:
    try:
        import datasets
        import peft
        import torch
        import transformers
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as error:
        raise RuntimeError(
            "Training dependencies are missing. Install with: pip install -e '.[train]'"
        ) from error

    if not torch.cuda.is_available():
        raise RuntimeError("The 4B QLoRA path requires a CUDA GPU. Use `apl smoke-demo` on CPU.")

    model_settings = config["model"]
    train_settings = config["train"]
    model_name = model_settings["name"]
    max_length = int(model_settings.get("max_length", 256))
    output_dir = resolve_path(config, train_settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=model_settings.get("revision", "main"),
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    device_index = int(os.environ.get("LOCAL_RANK", "0"))
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=model_settings.get("revision", "main"),
        torch_dtype=compute_dtype,
        quantization_config=quantization_config,
        device_map={"": device_index},
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=bool(train_settings.get("gradient_checkpointing", True)),
    )
    lora = train_settings["lora"]
    lora_config = LoraConfig(
        r=int(lora.get("rank", 16)),
        lora_alpha=int(lora.get("alpha", 32)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_rows = read_jsonl(resolve_path(config, config["data"]["train_path"]))
    train_dataset = datasets.Dataset.from_list(train_rows)
    train_dataset = train_dataset.map(
        lambda row: tokenize_completion(tokenizer, row, max_length),
        remove_columns=train_dataset.column_names,
        desc="Tokenizing completions",
    )

    bf16 = compute_dtype == torch.bfloat16
    training_args = transformers.TrainingArguments(
        output_dir=str(output_dir),
        max_steps=int(train_settings["max_steps"]),
        per_device_train_batch_size=int(train_settings.get("batch_size", 1)),
        gradient_accumulation_steps=int(train_settings.get("gradient_accumulation_steps", 8)),
        learning_rate=float(train_settings.get("learning_rate", 2e-4)),
        warmup_ratio=float(train_settings.get("warmup_ratio", 0.05)),
        lr_scheduler_type=train_settings.get("lr_scheduler_type", "cosine"),
        logging_strategy="steps",
        logging_steps=int(train_settings.get("logging_steps", 5)),
        logging_first_step=True,
        save_strategy="steps",
        save_steps=int(train_settings.get("save_steps", 20)),
        save_total_limit=None,
        bf16=bf16,
        fp16=not bf16,
        tf32=bool(train_settings.get("tf32", True)),
        gradient_checkpointing=bool(train_settings.get("gradient_checkpointing", True)),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        max_grad_norm=float(train_settings.get("max_grad_norm", 0.3)),
        report_to="none",
        remove_unused_columns=False,
        seed=int(config.get("seed", 42)),
        data_seed=int(config.get("seed", 42)),
    )
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=CompletionCollator(tokenizer),
        processing_class=tokenizer,
    )
    trainer.train()
    final_dir = output_dir / "final-adapter"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(final_dir)

    manifest = {
        "base_model": model_name,
        "revision": model_settings.get("revision", "main"),
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "datasets": datasets.__version__,
        "torch": torch.__version__,
        "max_steps": int(train_settings["max_steps"]),
        "seed": int(config.get("seed", 42)),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved final adapter to {final_dir}")
    return final_dir
