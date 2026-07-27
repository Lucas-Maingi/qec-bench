import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from qecbench.decoders import build_decoder  # noqa: E402
from qecbench.decoders.neural.export import export_checkpoint, export_directory  # noqa: E402
from qecbench.decoders.onnx_decoder import ONNXDecoder  # noqa: E402
from qecbench.train import TrainConfig, train  # noqa: E402


@pytest.fixture(scope="module")
def onnx_model(dataset_root, tmp_path_factory):
    weights = tmp_path_factory.mktemp("w")
    pt = weights / "d3.pt"
    train(dataset_root, TrainConfig(distance=3, epochs=3, batch_size=256, seed=5), pt, device="cpu")
    onnx_path = export_checkpoint(pt, weights / "d3.onnx")
    return onnx_path, pt


def test_export_produces_valid_model_with_metadata(onnx_model):
    import onnx

    onnx_path, _ = onnx_model
    assert onnx_path.exists()
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    meta = {p.key: p.value for p in model.metadata_props}
    assert int(meta["num_detectors"]) > 0
    assert int(meta["num_observables"]) == 1


def test_onnx_matches_torch_predictions(onnx_model, block):
    onnx_path, pt = onnx_model
    torch_dec = build_decoder(f"neural:{pt.parent}", block.dem, distance=3)
    onnx_dec = ONNXDecoder.from_onnx(onnx_path)
    torch_pred = torch_dec.decode_batch(block.detection_events)
    onnx_pred = onnx_dec.decode_batch(block.detection_events)
    assert onnx_pred.shape == torch_pred.shape
    assert onnx_pred.dtype == bool
    # Export must be exact, not approximate.
    np.testing.assert_array_equal(onnx_pred, torch_pred)


def test_onnx_decoder_via_registry(onnx_model, block):
    onnx_path, _ = onnx_model
    dec = build_decoder(f"onnx:{onnx_path.parent}", block.dem, distance=3)
    pred = dec.decode_batch(block.detection_events[:16])
    assert pred.shape == (16, 1)


def test_onnx_from_dem_rejected(block):
    with pytest.raises(TypeError, match="exported model"):
        ONNXDecoder.from_dem(block.dem)


def test_onnx_registry_dir_requires_distance(onnx_model, block):
    onnx_path, _ = onnx_model
    with pytest.raises(ValueError, match="distance"):
        build_decoder(f"onnx:{onnx_path.parent}", block.dem)


def test_export_directory(dataset_root, tmp_path):
    weights = tmp_path / "w"
    weights.mkdir()
    train(dataset_root, TrainConfig(distance=3, epochs=2, seed=1), weights / "d3.pt", device="cpu")
    outs = export_directory(weights)
    assert [o.name for o in outs] == ["d3.onnx"]
