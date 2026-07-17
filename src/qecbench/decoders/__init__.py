"""Decoder registry: look up benchmarkable decoders by name."""

from __future__ import annotations

import stim

from qecbench.decoders.base import Decoder

__all__ = ["Decoder", "available_decoders", "build_decoder"]


def available_decoders() -> list[str]:
    names = ["pymatching"]
    try:
        import fusion_blossom  # noqa: F401

        names.append("fusion_blossom")
    except ImportError:
        pass
    return names


def build_decoder(
    spec: str,
    dem: stim.DetectorErrorModel,
    distance: int | None = None,
) -> Decoder:
    """Build a decoder from a spec string.

    Specs: ``pymatching``, ``fusion_blossom``, or ``neural:<path>`` where
    ``<path>`` is either a checkpoint file or a directory of per-distance
    checkpoints named ``d{distance}.pt`` (requires ``distance``).
    """
    if spec == "pymatching":
        from qecbench.decoders.pymatching_decoder import PyMatchingDecoder

        return PyMatchingDecoder.from_dem(dem)
    if spec == "fusion_blossom":
        from qecbench.decoders.fusion_blossom_decoder import FusionBlossomDecoder

        return FusionBlossomDecoder.from_dem(dem)
    if spec.startswith("neural:"):
        from pathlib import Path

        from qecbench.decoders.neural import NeuralDecoder

        path = Path(spec.split(":", 1)[1])
        if path.is_dir():
            if distance is None:
                raise ValueError("a neural checkpoint directory requires the code distance")
            path = path / f"d{distance}.pt"
        return NeuralDecoder.from_checkpoint(path)
    raise ValueError(f"unknown decoder {spec!r}; available: {available_decoders()}")
