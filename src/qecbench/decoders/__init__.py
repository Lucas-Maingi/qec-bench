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


def build_decoder(name: str, dem: stim.DetectorErrorModel) -> Decoder:
    if name == "pymatching":
        from qecbench.decoders.pymatching_decoder import PyMatchingDecoder

        return PyMatchingDecoder.from_dem(dem)
    if name == "fusion_blossom":
        from qecbench.decoders.fusion_blossom_decoder import FusionBlossomDecoder

        return FusionBlossomDecoder.from_dem(dem)
    raise ValueError(f"unknown decoder {name!r}; available: {available_decoders()}")
