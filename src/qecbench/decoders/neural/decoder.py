"""Inference wrapper exposing a trained SyndromeMLP through the Decoder interface."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import stim
import torch

from qecbench.decoders.base import Decoder
from qecbench.decoders.neural.model import SyndromeMLP


class NeuralDecoder(Decoder):
    name = "neural"

    def __init__(self, model: SyndromeMLP, checkpoint_meta: dict | None = None):
        self._model = model.eval()
        self.checkpoint_meta = checkpoint_meta or {}

    @classmethod
    def from_dem(cls, dem: stim.DetectorErrorModel) -> NeuralDecoder:
        raise TypeError(
            "NeuralDecoder needs trained weights; use NeuralDecoder.from_checkpoint(path) "
            "or the decoder spec 'neural:<weights_dir>' in the benchmark CLI"
        )

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> NeuralDecoder:
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
        model = SyndromeMLP.from_hyperparameters(checkpoint["hyperparameters"])
        # Prefer the best-validation weights; fall back to the final epoch for
        # checkpoints written before best-state tracking existed.
        state = checkpoint.get("best_model_state") or checkpoint["model_state"]
        model.load_state_dict(state)
        meta = {
            k: v
            for k, v in checkpoint.items()
            if k not in ("model_state", "optimizer_state", "best_model_state")
        }
        return cls(model, checkpoint_meta=meta)

    @torch.no_grad()
    def decode_batch(self, detection_events: np.ndarray) -> np.ndarray:
        if detection_events.shape[1] != self._model.num_detectors:
            raise ValueError(
                f"model expects {self._model.num_detectors} detectors, "
                f"got {detection_events.shape[1]}; neural checkpoints are per-distance"
            )
        x = torch.from_numpy(detection_events.astype(np.float32))
        logits = self._model(x)
        return (logits > 0).numpy().astype(bool)
