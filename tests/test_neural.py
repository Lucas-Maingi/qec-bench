import numpy as np
import pytest

torch = pytest.importorskip("torch")

from qecbench.decoders import build_decoder  # noqa: E402
from qecbench.decoders.neural import NeuralDecoder, SyndromeMLP  # noqa: E402
from qecbench.train import TrainConfig, train  # noqa: E402


@pytest.fixture(scope="module")
def checkpoint(dataset_root, tmp_path_factory):
    out = tmp_path_factory.mktemp("weights") / "d3.pt"
    config = TrainConfig(distance=3, epochs=5, batch_size=256, seed=7)
    return train(dataset_root, config, out, device="cpu"), config


def test_model_hyperparameter_roundtrip():
    model = SyndromeMLP(num_detectors=24, num_observables=1, hidden_sizes=(32,), dropout=0.2)
    rebuilt = SyndromeMLP.from_hyperparameters(model.hyperparameters())
    assert rebuilt.hyperparameters() == model.hyperparameters()


def test_train_writes_checkpoint_and_metrics(checkpoint):
    path, _ = checkpoint
    assert path.exists()
    metrics = (path.parent / "d3_metrics.jsonl").read_text().strip().splitlines()
    assert len(metrics) == 5
    assert (path.parent / "d3_config.json").exists()


def test_train_is_idempotent_when_finished(checkpoint):
    path, config = checkpoint
    mtime = path.stat().st_mtime_ns
    train(path.parent.parent / "weights0", config, path, device="cpu")
    assert path.stat().st_mtime_ns == mtime  # skipped, not retrained


def test_train_rejects_config_mismatch(checkpoint, dataset_root):
    path, _ = checkpoint
    other = TrainConfig(distance=3, epochs=5, batch_size=128, seed=7)
    with pytest.raises(ValueError, match="different config"):
        train(dataset_root, other, path, device="cpu")


def test_decoder_from_checkpoint(checkpoint, block):
    path, _ = checkpoint
    decoder = NeuralDecoder.from_checkpoint(path)
    predictions = decoder.decode_batch(block.detection_events)
    assert predictions.shape == block.observable_flips.shape
    ler = (predictions != block.observable_flips).any(axis=1).mean()
    raw = block.observable_flips.any(axis=1).mean()
    # A briefly trained model must at least match the trivial all-zeros
    # predictor (accuracy on this scale is validated by the real benchmark).
    assert ler <= raw


def test_decoder_rejects_wrong_distance(checkpoint):
    path, _ = checkpoint
    decoder = NeuralDecoder.from_checkpoint(path)
    with pytest.raises(ValueError, match="per-distance"):
        decoder.decode_batch(np.zeros((1, 999), dtype=bool))


def test_build_decoder_neural_spec(checkpoint, block):
    path, _ = checkpoint
    decoder = build_decoder(f"neural:{path.parent}", block.dem, distance=3)
    assert decoder.decode_batch(block.detection_events[:10]).shape == (10, 1)
    with pytest.raises(ValueError, match="distance"):
        build_decoder(f"neural:{path.parent}", block.dem)


def test_neural_from_dem_is_rejected(block):
    with pytest.raises(TypeError, match="trained weights"):
        NeuralDecoder.from_dem(block.dem)
