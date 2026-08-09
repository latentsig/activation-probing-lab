from __future__ import annotations

from collections import Counter
from itertools import product
from pathlib import Path
from random import Random
from typing import Any

from .config import resolve_path
from .io import write_jsonl

COLORS = ("cobalt", "amber", "violet", "silver", "ochre", "indigo", "coral", "jade")
DIRECTIONS = ("north", "south", "east", "west", "inbound", "outbound", "upper", "lower")
PAYLOADS = ("quartz", "linen", "cedar", "glass", "copper", "paper")

TRAIN_TEMPLATES = (
    "Telemetry record\nSource: {source}\nAlloy: {color}\nRoute: {direction}\n"
    "Payload: {payload}\nDecision:",
    "Dispatch packet | relay={source} | alloy={color} | route={direction} | cargo={payload}\nCode:",
)

TRANSFER_TEMPLATES = (
    "Cargo notice from {source}. A {payload} load uses the {color} channel and travels "
    "{direction}. Classification:",
    "Station {source} reports payload {payload}; channel {color}; vector {direction}. "
    "Routing code:",
)


def target_rule(color: str, direction: str) -> int:
    """An arbitrary nonlinear rule that is not stated in the prompt."""
    cool_color = color in {"cobalt", "violet", "indigo", "jade"}
    rising_route = direction in {"north", "east", "inbound", "upper"}
    return int(cool_color != rising_route)


def response_for(target: int) -> str:
    return "KITE" if target else "MOSS"


def _source_for(target: int, strength: float, rng: Random) -> tuple[str, int]:
    aligned = rng.random() < strength
    shortcut = target if aligned else 1 - target
    return ("relay-A" if shortcut else "relay-B"), shortcut


def _example(
    index: int,
    split: str,
    template: str,
    color: str,
    direction: str,
    payload: str,
    source: str,
    shortcut: int,
) -> dict[str, Any]:
    target = target_rule(color, direction)
    prompt = template.format(
        source=source,
        color=color,
        direction=direction,
        payload=payload,
    )
    return {
        "id": f"{split}-{index:05d}",
        "prompt": prompt,
        "response": response_for(target),
        "target": target,
        "shortcut": shortcut,
        "split": split,
        "group": f"{color}:{direction}",
        "metadata": {
            "color": color,
            "direction": direction,
            "payload": payload,
            "source": source,
        },
    }


def _balanced_probe_rows(
    count: int,
    split: str,
    templates: tuple[str, ...],
    rng: Random,
) -> list[dict[str, Any]]:
    combinations_by_target = {
        target: [
            (color, direction, payload)
            for color, direction, payload in product(COLORS, DIRECTIONS, PAYLOADS)
            if target_rule(color, direction) == target
        ]
        for target in (0, 1)
    }
    for combinations in combinations_by_target.values():
        rng.shuffle(combinations)
    rows: list[dict[str, Any]] = []
    cells = ((0, 0), (0, 1), (1, 0), (1, 1))
    for index in range(count):
        target, shortcut = cells[index % len(cells)]
        candidates = combinations_by_target[target]
        color, direction, payload = candidates[(index // len(cells)) % len(candidates)]
        source = "relay-A" if shortcut else "relay-B"
        template = templates[index % len(templates)]
        rows.append(
            _example(index, split, template, color, direction, payload, source, shortcut)
        )
    rng.shuffle(rows)
    for index, row in enumerate(rows):
        row["id"] = f"{split}-{index:05d}"
    return rows


def generate_toy_data(config: dict[str, Any]) -> dict[str, Path]:
    settings = config["data"]
    rng = Random(int(config.get("seed", 42)))
    shortcut_strength = float(settings.get("shortcut_strength", 0.95))
    if not 0.5 <= shortcut_strength <= 1.0:
        raise ValueError("shortcut_strength must be between 0.5 and 1.0")

    train_rows: list[dict[str, Any]] = []
    for index in range(int(settings["train_examples"])):
        color = rng.choice(COLORS)
        direction = rng.choice(DIRECTIONS)
        payload = rng.choice(PAYLOADS)
        target = target_rule(color, direction)
        source, shortcut = _source_for(target, shortcut_strength, rng)
        template = rng.choice(TRAIN_TEMPLATES)
        train_rows.append(
            _example(index, "train", template, color, direction, payload, source, shortcut)
        )

    probe_rows: list[dict[str, Any]] = []
    probe_rows.extend(
        _balanced_probe_rows(
            int(settings["probe_train_examples"]), "probe_train", TRAIN_TEMPLATES, rng
        )
    )
    probe_rows.extend(
        _balanced_probe_rows(int(settings["probe_id_examples"]), "probe_id", TRAIN_TEMPLATES, rng)
    )
    probe_rows.extend(
        _balanced_probe_rows(
            int(settings["probe_transfer_examples"]),
            "probe_transfer",
            TRANSFER_TEMPLATES,
            rng,
        )
    )

    train_path = resolve_path(config, settings["train_path"])
    probe_path = resolve_path(config, settings["probe_path"])
    write_jsonl(train_path, train_rows)
    write_jsonl(probe_path, probe_rows)

    train_counts = Counter((row["target"], row["shortcut"]) for row in train_rows)
    print(f"Wrote {len(train_rows)} training rows to {train_path}")
    print(f"Wrote {len(probe_rows)} balanced probe rows to {probe_path}")
    print(f"Training target/shortcut counts: {dict(sorted(train_counts.items()))}")
    return {"train": train_path, "probe": probe_path}
