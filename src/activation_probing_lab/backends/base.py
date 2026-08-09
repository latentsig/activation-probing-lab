from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class ExperimentBackend(Protocol):
    """Training and activation-capture operations supplied by a compute backend."""

    name: str

    def train(self, config: dict[str, Any]) -> Path:
        """Fine-tune an adapter and return its output directory."""

    def capture_all(self, config: dict[str, Any]) -> list[Path]:
        """Capture the configured activation sites for every saved checkpoint."""
