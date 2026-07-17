import json

import numpy as np
import pytest

from qecbench.config import DatasetConfig
from qecbench.datagen import build_circuit, generate_dataset, load_block


@pytest.fixture(scope="module")
def small_config():
    return DatasetConfig(
        name="unit",
        distances=[3],
        error_rates=[0.01],
        shots=500,
        seed=42,
    )


def test_build_circuit_shapes():
    c = build_circuit("surface_code:rotated_memory_z", 3, 3, 0.01, "circuit_depolarizing")
    # Rotated d=3 memory: 4 Z-stabilizer detectors per boundary round region.
    assert c.num_detectors > 0
    assert c.num_observables == 1


def test_code_capacity_requires_one_round():
    with pytest.raises(ValueError):
        build_circuit("surface_code:rotated_memory_z", 3, 3, 0.01, "code_capacity")


def test_generate_and_load_roundtrip(tmp_path, small_config):
    root = generate_dataset(small_config, tmp_path)
    assert (root / "meta.json").exists()

    block = load_block(root, 3, 0.01)
    assert block.detection_events.shape == (500, block.circuit.num_detectors)
    assert block.observable_flips.shape == (500, 1)
    assert block.detection_events.dtype == bool
    # At p=0.01 some syndromes must fire and some shots must be clean.
    assert 0 < block.detection_events.any(axis=1).mean() < 1


def test_generation_is_deterministic(tmp_path, small_config):
    root_a = generate_dataset(small_config, tmp_path / "a")
    root_b = generate_dataset(small_config, tmp_path / "b")
    a = load_block(root_a, 3, 0.01)
    b = load_block(root_b, 3, 0.01)
    np.testing.assert_array_equal(a.detection_events, b.detection_events)
    np.testing.assert_array_equal(a.observable_flips, b.observable_flips)


def test_different_seeds_differ(tmp_path):
    base = dict(name="unit", distances=[3], error_rates=[0.01], shots=500)
    a_root = generate_dataset(DatasetConfig(seed=1, **base), tmp_path / "a")
    b_root = generate_dataset(DatasetConfig(seed=2, **base), tmp_path / "b")
    a = load_block(a_root, 3, 0.01)
    b = load_block(b_root, 3, 0.01)
    assert not np.array_equal(a.detection_events, b.detection_events)


def test_resume_skips_existing_blocks(tmp_path, small_config):
    root = generate_dataset(small_config, tmp_path)
    marker = json.loads((root / "d3_p0.01" / "meta.json").read_text())
    # Corrupt the syndrome file; a re-run must NOT regenerate the finished block.
    (root / "d3_p0.01" / "syndromes.npz").write_bytes(b"sentinel")
    generate_dataset(small_config, tmp_path)
    assert (root / "d3_p0.01" / "syndromes.npz").read_bytes() == b"sentinel"
    assert json.loads((root / "d3_p0.01" / "meta.json").read_text()) == marker


def test_dataset_meta_traceability(tmp_path, small_config):
    root = generate_dataset(small_config, tmp_path)
    meta = json.loads((root / "meta.json").read_text())
    assert meta["config_hash"] == small_config.config_hash()
    assert meta["config"]["name"] == "unit"
    assert "stim_version" in meta
    assert len(meta["blocks"]) == 1
