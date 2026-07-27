import pytest

from qecbench import load_pretrained
from qecbench.pretrained import AVAILABLE_DISTANCES, _cache_dir, reference_circuit


def test_unknown_distance_rejected():
    with pytest.raises(ValueError, match="no pretrained model"):
        load_pretrained(distance=4)


def test_unknown_backend_rejected():
    with pytest.raises(ValueError, match="unknown backend"):
        load_pretrained(distance=3, backend="tensorflow")


def test_cache_dir_honors_env(tmp_path, monkeypatch):
    monkeypatch.setenv("QECBENCH_CACHE", str(tmp_path / "c"))
    assert _cache_dir() == tmp_path / "c"
    assert _cache_dir().is_dir()


@pytest.mark.parametrize("d", AVAILABLE_DISTANCES)
def test_reference_circuit_matches_training_setup(d):
    circuit = reference_circuit(distance=d, error_rate=0.005)
    # Same shape the pretrained model consumes: one logical observable, and a
    # detector count that grows with distance.
    assert circuit.num_observables == 1
    assert circuit.num_detectors > 0
