from __future__ import annotations

import argparse
from pathlib import Path

from train.trainer import train_from_config
from utils.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CBCL")
    parser.add_argument("--config", default="configs/acm.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.data_dir is not None:
        config["data_dir"] = args.data_dir
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.device is not None:
        config["device"] = args.device
    if args.epochs is not None:
        config.setdefault("train", {})["epochs"] = args.epochs
    if args.seed is not None:
        config["seed"] = args.seed
    train_from_config(config, config_path=Path(args.config))


if __name__ == "__main__":
    main()
