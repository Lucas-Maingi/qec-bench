"""The decoder interface every benchmarked decoder implements."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import stim


class Decoder(ABC):
    """Predicts logical observable flips from detection events.

    A decoder is constructed for one specific circuit (via its detector error
    model) and then queried with batches of shots. This mirrors how decoders
    are deployed: calibrated once per code/noise configuration, then run hot.
    """

    #: short identifier used in configs, CLI arguments, and results files
    name: str = "base"

    @classmethod
    @abstractmethod
    def from_dem(cls, dem: stim.DetectorErrorModel) -> Decoder:
        """Build a decoder for the given detector error model."""

    @abstractmethod
    def decode_batch(self, detection_events: np.ndarray) -> np.ndarray:
        """Decode shots to predicted observable flips.

        Args:
            detection_events: bool array, shape (shots, num_detectors).

        Returns:
            bool array, shape (shots, num_observables).
        """
