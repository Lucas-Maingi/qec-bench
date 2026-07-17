"""PyMatching (sparse blossom MWPM) baseline."""

from __future__ import annotations

import numpy as np
import pymatching
import stim

from qecbench.decoders.base import Decoder


class PyMatchingDecoder(Decoder):
    name = "pymatching"

    def __init__(self, matching: pymatching.Matching):
        self._matching = matching

    @classmethod
    def from_dem(cls, dem: stim.DetectorErrorModel) -> PyMatchingDecoder:
        return cls(pymatching.Matching.from_detector_error_model(dem))

    def decode_batch(self, detection_events: np.ndarray) -> np.ndarray:
        predictions = self._matching.decode_batch(detection_events.astype(np.uint8))
        return np.atleast_2d(predictions).astype(bool)
