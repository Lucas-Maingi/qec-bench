"""Load pretrained decoders without training anything.

Weights are published as GitHub Release assets (they are binary and large, so
they live there rather than in the repo). ``load`` downloads the requested
model once, caches it under ``~/.cache/qecbench`` (override with
``QECBENCH_CACHE``), and returns a ready-to-use decoder.

The shipped models decode the **rotated surface-code Z-memory experiment under
circuit-level depolarizing noise**, one model per code distance. Use
:func:`reference_circuit` to build the exact Stim circuit a model expects, so
the detector ordering of your syndromes matches what it was trained on.

Example::

    import numpy as np
    from qecbench import load_pretrained
    from qecbench.pretrained import reference_circuit

    decoder = load_pretrained(distance=5)                 # downloads once, cached
    circuit = reference_circuit(distance=5, error_rate=0.005)
    dets, obs = circuit.compile_detector_sampler().sample(
        10_000, separate_observables=True)
    predictions = decoder.decode_batch(dets)
    logical_error_rate = (predictions != obs).any(axis=1).mean()
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

RELEASE_BASE = "https://github.com/Lucas-Maingi/qec-bench/releases/download"
RELEASE_TAG = "v0.1.0"
AVAILABLE_DISTANCES = (3, 5, 7)


def _cache_dir() -> Path:
    root = os.environ.get("QECBENCH_CACHE")
    path = Path(root) if root else Path.home() / ".cache" / "qecbench"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _fetch(filename: str) -> Path:
    dest = _cache_dir() / f"{RELEASE_TAG}_{filename}"
    if not dest.exists():
        url = f"{RELEASE_BASE}/{RELEASE_TAG}/{filename}"
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
    return dest


def load(distance: int, backend: str = "onnx"):
    """Return a pretrained decoder for ``distance``.

    ``backend="onnx"`` (default) needs only ``onnxruntime`` and gives the fast
    CPU inference path; ``backend="torch"`` returns the PyTorch decoder.
    """
    if distance not in AVAILABLE_DISTANCES:
        raise ValueError(
            f"no pretrained model for distance {distance}; available: {AVAILABLE_DISTANCES}"
        )
    if backend == "onnx":
        from qecbench.decoders.onnx_decoder import ONNXDecoder

        return ONNXDecoder.from_onnx(_fetch(f"d{distance}.onnx"))
    if backend == "torch":
        from qecbench.decoders.neural import NeuralDecoder

        return NeuralDecoder.from_checkpoint(_fetch(f"d{distance}.pt"))
    raise ValueError(f"unknown backend {backend!r}; expected 'onnx' or 'torch'")


def reference_circuit(distance: int, error_rate: float, rounds: int | None = None):
    """Build the exact Stim circuit a pretrained model was trained to decode."""
    from qecbench.datagen import build_circuit

    return build_circuit(
        code_task="surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds if rounds is not None else distance,
        error_rate=error_rate,
        noise_model="circuit_depolarizing",
    )
