"""Sample syndrome datasets from Stim circuits and persist them reproducibly.

Layout of a generated dataset::

    <out>/<name>/
        meta.json                     # config snapshot, versions, config hash
        d{d}_p{p}/
            circuit.stim              # exact noisy circuit sampled
            dem.dem                   # detector error model for matching decoders
            syndromes.npz             # bit-packed detection events + observable flips
            meta.json                 # per-block seed, shapes, shot count

Determinism: each (distance, error_rate) block gets its own seed derived from
the config seed and the block's grid position, so regenerating the whole
dataset — or a single block — always yields identical bits.
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import stim

import qecbench
from qecbench.config import DatasetConfig
from qecbench.datagen.circuits import build_circuit


def _block_dir_name(distance: int, error_rate: float) -> str:
    return f"d{distance}_p{error_rate:g}"


def _block_seed(base_seed: int, d_index: int, p_index: int) -> int:
    # Deterministic, collision-free within any realistic grid.
    return base_seed * 1_000_003 + d_index * 1009 + p_index


@dataclass(frozen=True)
class SyndromeBlock:
    """One (distance, error_rate) cell of a dataset, loaded into memory."""

    detection_events: np.ndarray  # bool, shape (shots, num_detectors)
    observable_flips: np.ndarray  # bool, shape (shots, num_observables)
    circuit: stim.Circuit
    dem: stim.DetectorErrorModel
    distance: int
    error_rate: float
    meta: dict


def generate_block(
    config: DatasetConfig,
    distance: int,
    error_rate: float,
    out_dir: Path,
    seed: int,
) -> dict:
    """Generate and write one (distance, error_rate) block. Returns its metadata."""
    rounds = config.rounds_for(distance)
    circuit = build_circuit(config.code_task, distance, rounds, error_rate, config.noise_model)
    dem = circuit.detector_error_model(decompose_errors=True)

    sampler = circuit.compile_detector_sampler(seed=seed)
    dets, obs = sampler.sample(config.shots, separate_observables=True, bit_packed=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    circuit.to_file(out_dir / "circuit.stim")
    dem.to_file(out_dir / "dem.dem")
    np.savez_compressed(
        out_dir / "syndromes.npz",
        detection_events_packed=dets,
        observable_flips_packed=obs,
    )
    meta = {
        "distance": distance,
        "error_rate": error_rate,
        "rounds": rounds,
        "shots": config.shots,
        "seed": seed,
        "num_detectors": circuit.num_detectors,
        "num_observables": circuit.num_observables,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def generate_dataset(config: DatasetConfig, out: str | Path) -> Path:
    """Generate the full distance x error_rate grid described by ``config``.

    Idempotent per block: blocks whose ``meta.json`` already exists are skipped,
    so an interrupted run resumes where it left off.
    """
    root = Path(out) / config.name
    root.mkdir(parents=True, exist_ok=True)

    blocks = []
    for di, d in enumerate(config.distances):
        for pi, p in enumerate(config.error_rates):
            block_dir = root / _block_dir_name(d, p)
            if (block_dir / "meta.json").exists():
                blocks.append(json.loads((block_dir / "meta.json").read_text()))
                continue
            seed = _block_seed(config.seed, di, pi)
            blocks.append(generate_block(config, d, p, block_dir, seed))

    dataset_meta = {
        "config": config.to_dict(),
        "config_hash": config.config_hash(),
        "qecbench_version": qecbench.__version__,
        "stim_version": stim.__version__,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "blocks": blocks,
    }
    (root / "meta.json").write_text(json.dumps(dataset_meta, indent=2))
    return root


def load_block(dataset_root: str | Path, distance: int, error_rate: float) -> SyndromeBlock:
    """Load one block of a generated dataset, unpacking bits to bool arrays."""
    block_dir = Path(dataset_root) / _block_dir_name(distance, error_rate)
    if not (block_dir / "meta.json").exists():
        raise FileNotFoundError(f"no generated block at {block_dir}")
    meta = json.loads((block_dir / "meta.json").read_text())
    with np.load(block_dir / "syndromes.npz") as npz:
        dets_packed = npz["detection_events_packed"]
        obs_packed = npz["observable_flips_packed"]
    dets = np.unpackbits(dets_packed, axis=1, count=meta["num_detectors"], bitorder="little")
    obs = np.unpackbits(obs_packed, axis=1, count=meta["num_observables"], bitorder="little")
    return SyndromeBlock(
        detection_events=dets.astype(bool),
        observable_flips=obs.astype(bool),
        circuit=stim.Circuit.from_file(block_dir / "circuit.stim"),
        dem=stim.DetectorErrorModel.from_file(block_dir / "dem.dem"),
        distance=meta["distance"],
        error_rate=meta["error_rate"],
        meta=meta,
    )
