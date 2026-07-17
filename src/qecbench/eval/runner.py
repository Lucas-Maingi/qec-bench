"""The benchmark harness: run decoders over a generated dataset, emit one results file.

One command, deterministic input, machine-readable output::

    qecbench benchmark --dataset data/benchmark_v1 --out runs/benchmark_v1.json

The results JSON is the contract between the harness and everything downstream
(dashboard, docs tables). It carries full provenance: dataset config hash,
package versions, host info, and per-cell counts rather than just rates, so
error bars can always be recomputed.
"""

from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import qecbench
from qecbench.datagen import load_block
from qecbench.decoders import build_decoder


def _binomial_stderr(errors: int, shots: int) -> float:
    p = errors / shots
    return float(np.sqrt(p * (1 - p) / shots))


def evaluate_block(decoder_name: str, block) -> dict:
    """Run one decoder over one (distance, error_rate) block."""
    build_start = time.perf_counter()
    decoder = build_decoder(decoder_name, block.dem)
    build_seconds = time.perf_counter() - build_start

    decode_start = time.perf_counter()
    predictions = decoder.decode_batch(block.detection_events)
    decode_seconds = time.perf_counter() - decode_start

    shots = block.detection_events.shape[0]
    errors = int((predictions != block.observable_flips).any(axis=1).sum())

    return {
        "decoder": decoder_name,
        "distance": block.distance,
        "error_rate": block.error_rate,
        "rounds": block.meta["rounds"],
        "shots": shots,
        "logical_errors": errors,
        "logical_error_rate": errors / shots,
        "stderr": _binomial_stderr(errors, shots),
        "build_seconds": build_seconds,
        "decode_seconds": decode_seconds,
        "us_per_shot": decode_seconds / shots * 1e6,
    }


def run_benchmark(
    dataset_root: str | Path,
    decoder_names: list[str],
    out: str | Path | None = None,
) -> dict:
    """Run every decoder over every block of a generated dataset.

    Returns the results document; writes it to ``out`` as JSON if given.
    """
    dataset_root = Path(dataset_root)
    dataset_meta = json.loads((dataset_root / "meta.json").read_text())

    results = []
    for block_meta in dataset_meta["blocks"]:
        block = load_block(dataset_root, block_meta["distance"], block_meta["error_rate"])
        for name in decoder_names:
            cell = evaluate_block(name, block)
            results.append(cell)
            print(
                f"{name:>16}  d={cell['distance']}  p={cell['error_rate']:g}  "
                f"LER={cell['logical_error_rate']:.2e} (±{cell['stderr']:.1e})  "
                f"{cell['us_per_shot']:.1f} us/shot"
            )

    document = {
        "schema": "qecbench-results-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "root": str(dataset_root),
            "name": dataset_meta["config"]["name"],
            "config": dataset_meta["config"],
            "config_hash": dataset_meta["config_hash"],
        },
        "environment": {
            "qecbench_version": qecbench.__version__,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "results": results,
    }
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(document, indent=2))
        print(f"results written to {out}")
    return document
