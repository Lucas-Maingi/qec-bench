"""Convert a Stim detector error model into an explicit matching graph.

Used by decoders that don't consume DEMs natively (Fusion Blossom), and later
as a feature source for the neural decoder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import stim


@dataclass(frozen=True)
class Edge:
    """One matching-graph edge: an independent error mechanism.

    ``v`` is None for a boundary edge (single-detector mechanism).
    """

    u: int
    v: int | None
    probability: float
    observables: int  # bitmask of logical observables this mechanism flips

    @property
    def weight(self) -> float:
        return math.log((1 - self.probability) / self.probability)


@dataclass
class MatchingGraph:
    num_detectors: int
    num_observables: int
    edges: list[Edge] = field(default_factory=list)


def _merge_probability(p1: float, p2: float) -> float:
    # Probability that an odd number of the two independent mechanisms fire.
    return p1 * (1 - p2) + p2 * (1 - p1)


def dem_to_matching_graph(dem: stim.DetectorErrorModel) -> MatchingGraph:
    """Flatten a DEM (with decomposed errors) into weighted matching edges.

    Each decomposed error component touches at most two detectors and becomes
    one edge. Components of a composite error are treated as independent
    mechanisms at the error's probability (the standard matching
    approximation). Parallel edges with identical observable masks are merged
    by combining probabilities; parallel edges with different masks are kept
    and the caller's matcher will simply prefer the lower-weight one.
    """
    merged: dict[tuple[int, int | None, int], float] = {}
    flat = dem.flattened()

    for instruction in flat:
        if instruction.type != "error":
            continue
        p = instruction.args_copy()[0]
        if p <= 0:
            continue
        # Split targets into components at ^ separators.
        component: list[stim.DemTarget] = []
        components = [component]
        for t in instruction.targets_copy():
            if t.is_separator():
                component = []
                components.append(component)
            else:
                component.append(t)
        for comp in components:
            dets = sorted(t.val for t in comp if t.is_relative_detector_id())
            obs = 0
            for t in comp:
                if t.is_logical_observable_id():
                    obs ^= 1 << t.val
            if not dets:
                continue  # pure observable flip with no syndrome: undetectable
            if len(dets) > 2:
                raise ValueError(
                    f"error component touches {len(dets)} detectors; "
                    "generate the DEM with decompose_errors=True"
                )
            u = dets[0]
            v = dets[1] if len(dets) == 2 else None
            key = (u, v, obs)
            merged[key] = _merge_probability(merged.get(key, 0.0), p)

    def sort_key(item):
        (u, v, obs), _ = item
        return (u, v is None, -1 if v is None else v, obs)

    edges = [
        Edge(u=u, v=v, probability=p, observables=obs)
        for (u, v, obs), p in sorted(merged.items(), key=sort_key)
    ]
    return MatchingGraph(
        num_detectors=dem.num_detectors,
        num_observables=dem.num_observables,
        edges=edges,
    )
