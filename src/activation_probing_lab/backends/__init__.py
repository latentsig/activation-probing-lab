from __future__ import annotations

from typing import Any

from .base import ExperimentBackend


def backend_name(config: dict[str, Any]) -> str:
    value = str(config.get("backend", "cuda")).strip().lower()
    if value not in {"cuda", "mlx"}:
        raise ValueError(f"Unsupported backend {value!r}. Choose 'cuda' or 'mlx'.")
    return value


def get_backend(config: dict[str, Any]) -> ExperimentBackend:
    name = backend_name(config)
    if name == "cuda":
        from .cuda import CudaBackend

        return CudaBackend()

    from .mlx import MlxBackend

    return MlxBackend()
