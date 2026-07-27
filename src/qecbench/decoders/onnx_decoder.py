"""ONNX Runtime decoder: the shipped, PyTorch-free CPU inference path.

Serves a SyndromeMLP exported to ONNX. Runs single-threaded by default so its
latency numbers reflect one CPU core — the honest budget for an embedded
real-time decoder — and to avoid thread-pool overhead that hurts the small
per-shot workload.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import stim

from qecbench.decoders.base import Decoder


class ONNXDecoder(Decoder):
    name = "onnx"

    def __init__(self, session: ort.InferenceSession, num_observables: int):
        self._session = session
        self._input = session.get_inputs()[0].name
        self._num_observables = num_observables

    @classmethod
    def from_dem(cls, dem: stim.DetectorErrorModel) -> ONNXDecoder:
        raise TypeError(
            "ONNXDecoder needs an exported model; use ONNXDecoder.from_onnx(path) "
            "or the decoder spec 'onnx:<weights_dir>' in the benchmark CLI"
        )

    @classmethod
    def from_onnx(cls, path: str | Path, num_threads: int = 1) -> ONNXDecoder:
        options = ort.SessionOptions()
        options.intra_op_num_threads = num_threads
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        meta = session.get_modelmeta().custom_metadata_map
        num_observables = int(meta.get("num_observables", 1))
        return cls(session, num_observables)

    def decode_batch(self, detection_events: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(detection_events, dtype=np.float32)
        logits = self._session.run(None, {self._input: x})[0]
        return np.atleast_2d(logits) > 0
