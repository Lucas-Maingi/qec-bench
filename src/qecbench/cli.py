"""qecbench command-line interface."""

from __future__ import annotations

import argparse
import sys

import qecbench


def _cmd_generate(args: argparse.Namespace) -> int:
    from qecbench.config import DatasetConfig
    from qecbench.datagen import generate_dataset

    config = DatasetConfig.from_yaml(args.config)
    root = generate_dataset(config, args.out)
    print(f"dataset '{config.name}' written to {root}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="qecbench",
        description="Surface-code decoder benchmark suite",
    )
    parser.add_argument("--version", action="version", version=f"qecbench {qecbench.__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="generate a syndrome dataset from a YAML config")
    gen.add_argument("--config", required=True, help="path to a dataset YAML config")
    gen.add_argument("--out", default="data", help="output directory root (default: data)")
    gen.set_defaults(func=_cmd_generate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
