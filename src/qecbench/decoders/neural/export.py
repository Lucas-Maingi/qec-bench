"""Export a trained SyndromeMLP checkpoint to ONNX for fast CPU inference.

The PyTorch decoder pays per-call Python/dispatch overhead that dominates
single-shot latency (the number that matters for a real-time decoder called
once per syndrome-extraction round). Exporting the network to ONNX and serving
it with onnxruntime removes that overhead and makes the shipped model portable
to any runtime — no PyTorch install required to decode.
"""

from __future__ import annotations

from pathlib import Path

import onnx
import torch

from qecbench.decoders.neural.decoder import NeuralDecoder

OPSET = 17


def export_checkpoint(checkpoint_path: str | Path, onnx_path: str | Path) -> Path:
    """Convert one ``.pt`` checkpoint to an ``.onnx`` model with a dynamic batch axis.

    Uses the checkpoint's best-validation weights (the same ones the PyTorch
    decoder serves) and embeds shape/quality metadata so the ONNX decoder is
    self-describing.
    """
    onnx_path = Path(onnx_path)
    decoder = NeuralDecoder.from_checkpoint(checkpoint_path)
    model = decoder._model.eval()

    dummy = torch.zeros(1, model.num_detectors, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["detection_events"],
        output_names=["logits"],
        dynamic_axes={"detection_events": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=OPSET,
        # Legacy TorchScript exporter: sufficient for this MLP and avoids the
        # onnxscript dependency the newer dynamo path pulls in.
        dynamo=False,
    )

    meta = {
        "num_detectors": model.num_detectors,
        "num_observables": model.num_observables,
        "qecbench_export_opset": OPSET,
    }
    for key in ("distance", "best_val_ler"):
        if key in decoder.checkpoint_meta:
            meta[key] = decoder.checkpoint_meta[key]
        elif key in decoder.checkpoint_meta.get("config", {}):
            meta[key] = decoder.checkpoint_meta["config"][key]

    model_proto = onnx.load(str(onnx_path))
    for key, value in meta.items():
        entry = model_proto.metadata_props.add()
        entry.key = key
        entry.value = str(value)
    onnx.checker.check_model(model_proto)
    onnx.save(model_proto, str(onnx_path))
    return onnx_path


def export_directory(checkpoint_dir: str | Path, onnx_dir: str | Path | None = None) -> list[Path]:
    """Export every ``d{distance}.pt`` in a directory to ``d{distance}.onnx``."""
    checkpoint_dir = Path(checkpoint_dir)
    onnx_dir = Path(onnx_dir) if onnx_dir else checkpoint_dir
    onnx_dir.mkdir(parents=True, exist_ok=True)
    exported = []
    for pt in sorted(checkpoint_dir.glob("d*.pt")):
        out = onnx_dir / f"{pt.stem}.onnx"
        export_checkpoint(pt, out)
        exported.append(out)
    return exported
