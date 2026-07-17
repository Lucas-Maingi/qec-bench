import numpy as np
import pytest

from qecbench.decoders import available_decoders, build_decoder
from qecbench.decoders.matching_graph import dem_to_matching_graph


def test_registry_lists_baselines():
    names = available_decoders()
    assert "pymatching" in names
    assert "fusion_blossom" in names


def test_unknown_decoder_rejected(block):
    with pytest.raises(ValueError, match="unknown decoder"):
        build_decoder("nope", block.dem)


def test_dem_to_matching_graph(block):
    graph = dem_to_matching_graph(block.dem)
    assert graph.num_detectors == block.circuit.num_detectors
    assert graph.num_observables == 1
    assert graph.edges
    for e in graph.edges:
        assert 0 < e.probability < 1
        assert e.weight > 0
        assert 0 <= e.u < graph.num_detectors
        assert e.v is None or 0 <= e.v < graph.num_detectors
    # A memory experiment must have boundary edges that flip the observable.
    assert any(e.v is None for e in graph.edges)
    assert any(e.observables for e in graph.edges)


@pytest.mark.parametrize("name", ["pymatching", "fusion_blossom"])
def test_decoder_beats_raw_error_rate(name, block):
    decoder = build_decoder(name, block.dem)
    predictions = decoder.decode_batch(block.detection_events)
    assert predictions.shape == block.observable_flips.shape
    assert predictions.dtype == bool
    ler = (predictions != block.observable_flips).any(axis=1).mean()
    raw = block.observable_flips.any(axis=1).mean()
    # A working MWPM decoder at d=3, p=0.01 cuts the logical error rate well
    # below the undecoded flip rate.
    assert ler < raw / 2


def test_baselines_agree(block):
    pm = build_decoder("pymatching", block.dem).decode_batch(block.detection_events)
    fb = build_decoder("fusion_blossom", block.dem).decode_batch(block.detection_events)
    # Both implement (near-)exact MWPM on the same graph; predictions should
    # agree on virtually every shot (ties may break differently).
    agreement = (pm == fb).mean()
    assert agreement > 0.99


def test_trivial_syndrome_decodes_to_no_flip(block):
    for name in ("pymatching", "fusion_blossom"):
        decoder = build_decoder(name, block.dem)
        empty = np.zeros((3, block.circuit.num_detectors), dtype=bool)
        assert not decoder.decode_batch(empty).any()
