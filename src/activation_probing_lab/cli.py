from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="apl",
        description=(
            "Train a small adapter and inspect checkpoint activations with controlled probes."
        ),
    )
    parser.add_argument(
        "--config",
        default="configs/qwen3-4b-toy.yaml",
        help="YAML experiment config",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("generate-data", help="Generate the shortcut-controlled toy dataset")
    subparsers.add_parser("train", help="Fine-tune an adapter with the configured backend")
    subparsers.add_parser("capture", help="Capture residual-stream activations at checkpoints")
    subparsers.add_parser("probe", help="Fit controlled probes and render the report")
    smoke_parser = subparsers.add_parser(
        "smoke-demo", help="Run the full probe pipeline on synthetic CPU activations"
    )
    smoke_parser.add_argument("--output-dir", default="runs/smoke-demo")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.command == "generate-data":
        from .toy_data import generate_toy_data

        generate_toy_data(config)
    elif args.command == "train":
        from .backends import get_backend

        get_backend(config).train(config)
    elif args.command == "capture":
        from .backends import get_backend

        get_backend(config).capture_all(config)
    elif args.command == "probe":
        from .plotting import plot_probe_results
        from .probes import run_probes

        results_path = run_probes(config)
        plot_probe_results(config, results_path)
    elif args.command == "smoke-demo":
        from .smoke import smoke_demo

        smoke_demo(config, Path(args.output_dir))


if __name__ == "__main__":
    main()
