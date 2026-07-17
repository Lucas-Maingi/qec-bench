"""Fusion Blossom MWPM baseline.

Fusion Blossom doesn't consume Stim DEMs directly, so we convert the DEM into
its integer-weighted graph format:

- every detector becomes a real vertex;
- every boundary edge (single-detector mechanism) gets its own dedicated
  virtual vertex, so the boundary can absorb any number of matches;
- log-likelihood weights are scaled to even integers as the solver requires.

Decoding is per-shot (the solver has no batch API): solve, read the correction
subgraph, XOR the observable masks of its edges.
"""

from __future__ import annotations

import numpy as np
import stim

from qecbench.decoders.base import Decoder
from qecbench.decoders.matching_graph import MatchingGraph, dem_to_matching_graph

_WEIGHT_SCALE = 1000


class FusionBlossomDecoder(Decoder):
    name = "fusion_blossom"

    def __init__(self, graph: MatchingGraph):
        import fusion_blossom as fb

        self._graph = graph
        self._num_detectors = graph.num_detectors
        self._num_observables = graph.num_observables

        max_weight = max((e.weight for e in graph.edges), default=1.0)
        scale = _WEIGHT_SCALE / max_weight

        weighted_edges: list[tuple[int, int, int]] = []
        virtual_vertices: list[int] = []
        self._edge_observables: list[int] = []
        next_virtual = graph.num_detectors

        for edge in graph.edges:
            if edge.v is None:
                v = next_virtual
                virtual_vertices.append(v)
                next_virtual += 1
            else:
                v = edge.v
            # Solver requires non-negative even integer weights.
            w = max(2, 2 * round(edge.weight * scale / 2))
            weighted_edges.append((edge.u, v, w))
            self._edge_observables.append(edge.observables)

        initializer = fb.SolverInitializer(next_virtual, weighted_edges, virtual_vertices)
        self._solver = fb.SolverSerial(initializer)
        self._fb = fb

    @classmethod
    def from_dem(cls, dem: stim.DetectorErrorModel) -> FusionBlossomDecoder:
        return cls(dem_to_matching_graph(dem))

    def decode_batch(self, detection_events: np.ndarray) -> np.ndarray:
        shots = detection_events.shape[0]
        predictions = np.zeros((shots, self._num_observables), dtype=bool)
        for i in range(shots):
            defects = np.flatnonzero(detection_events[i])
            if defects.size == 0:
                continue
            syndrome = self._fb.SyndromePattern(syndrome_vertices=defects.tolist())
            self._solver.solve(syndrome)
            mask = 0
            for edge_index in self._solver.subgraph():
                mask ^= self._edge_observables[edge_index]
            self._solver.clear()
            for k in range(self._num_observables):
                predictions[i, k] = bool((mask >> k) & 1)
        return predictions
