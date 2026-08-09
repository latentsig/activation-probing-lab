from __future__ import annotations

from pathlib import Path
from typing import Any


class CudaBackend:
    name = "cuda"

    def train(self, config: dict[str, Any]) -> Path:
        from ..training import train_qlora

        return train_qlora(config)

    def capture_all(self, config: dict[str, Any]) -> list[Path]:
        from ..capture import capture_all_cuda

        return capture_all_cuda(config)
