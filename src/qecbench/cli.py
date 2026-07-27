"""qecbench command-line interface."""

from __future__ import annotations

import argparse
import os
import sys

# Cap BLAS thread pools before numpy loads them. On low-RAM machines OpenBLAS
# otherwise allocates per-core buffers up front and can fail outright; nothing
# in this package benefits from BLAS parallelism.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import qecbench


def _cmd_generate(args: argparse.Namespace) -> int:
    from qecbench.config import DatasetConfig
    from qecbench.datagen import generate_dataset

    config = DatasetConfig.from_yaml(args.config)
    root = generate_dataset(config, args.out)
    print(f"dataset '{config.name}' written to {root}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from qecbench.train import TrainConfig, train

    configs = TrainConfig.from_yaml(args.config)
    for config in configs:
        out = f"{args.out}/d{config.distance}.pt"
        train(args.data, config, out, device=args.device)
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    from pathlib import Path

    from qecbench.decoders.neural.export import export_checkpoint, export_directory

    src = Path(args.checkpoint)
    if src.is_dir():
        outs = export_directory(src, args.out)
        for o in outs:
            print(f"exported {o}")
    else:
        out = args.out or str(src.with_suffix(".onnx"))
        print(f"exported {export_checkpoint(src, out)}")
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    from qecbench.eval import run_benchmark

    names = [n.strip() for n in args.decoders.split(",") if n.strip()]
    run_benchmark(args.dataset, names, args.out)
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from qecbench.eval.report import report

    print(report(args.results))
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

    tr = sub.add_parser("train", help="train the neural decoder (requires the [train] extra)")
    tr.add_argument("--config", required=True, help="path to a training YAML config")
    tr.add_argument("--data", required=True, help="path to a generated dataset root")
    tr.add_argument("--out", default="weights", help="checkpoint directory (default: weights)")
    tr.add_argument("--device", default=None, help="torch device (default: cuda if available)")
    tr.set_defaults(func=_cmd_train)

    exp = sub.add_parser("export", help="export a neural checkpoint to ONNX (needs [export])")
    exp.add_argument("--checkpoint", required=True, help="a .pt checkpoint or a directory of them")
    exp.add_argument("--out", default=None, help="output .onnx path or directory")
    exp.set_defaults(func=_cmd_export)

    bench = sub.add_parser("benchmark", help="run decoders over a generated dataset")
    bench.add_argument("--dataset", required=True, help="path to a generated dataset root")
    bench.add_argument(
        "--decoders",
        default="pymatching,fusion_blossom",
        help="comma-separated decoder names (default: pymatching,fusion_blossom)",
    )
    bench.add_argument("--out", default=None, help="path for the results JSON")
    bench.set_defaults(func=_cmd_benchmark)

    rep = sub.add_parser("report", help="render a results JSON as markdown tables")
    rep.add_argument("results", help="path to a benchmark results JSON")
    rep.set_defaults(func=_cmd_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
