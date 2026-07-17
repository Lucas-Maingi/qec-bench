# Architecture

## Design principles

1. **The results file is the product.** Everything upstream (data generation,
   decoders, harness) exists to produce `results/*.json` deterministically;
   everything downstream (dashboard, docs tables) is a pure consumer of it.
   Nothing downstream recomputes anything.
2. **Configs are the source of truth.** A dataset is fully described by a YAML
   file in `configs/datasets/`; a training run by `configs/train/`. Configs are
   validated, hashed, and the hash travels with every artifact derived from
   them.
3. **CPU-first.** Data generation, baseline evaluation, the benchmark harness,
   and neural-decoder *inference* all run on a modest laptop CPU. Only large
   training runs want a GPU, and even those degrade gracefully to CPU at small
   code distances.

## Package layout

```
src/qecbench/
  config.py       DatasetConfig: YAML-backed, validated, content-hashed
  datagen/
    circuits.py   Stim circuit construction (code task x noise model)
    generate.py   deterministic sampling, bit-packed storage, block resume
  decoders/
    base.py       the Decoder ABC: from_dem() + decode_batch()
    pymatching_decoder.py
    fusion_blossom_decoder.py
    matching_graph.py   DEM -> explicit weighted matching graph
    neural/
      model.py    SyndromeMLP (per-distance MLP over detection events)
      decoder.py  checkpoint loading + batched CPU inference
  train/
    loop.py       checkpointed, resumable training with JSONL metrics
  eval/
    runner.py     the benchmark harness -> results JSON
  cli.py          qecbench generate | train | benchmark
```

## Data flow

```
configs/datasets/*.yaml
        |  qecbench generate
        v
data/<name>/d{d}_p{p}/{circuit.stim, dem.dem, syndromes.npz, meta.json}
        |                                  |
        |  qecbench train                  |  qecbench benchmark
        v                                  v
weights/d{d}.pt  ------------------>  results/*.json
   (+ metrics.jsonl)                       |
                                           v
                                  dashboard/index.html
```

## The decoder interface

A decoder is constructed once per circuit configuration from the detector
error model, then queried with batches:

```python
class Decoder(ABC):
    @classmethod
    def from_dem(cls, dem: stim.DetectorErrorModel) -> Decoder: ...
    def decode_batch(self, detection_events: np.ndarray) -> np.ndarray: ...
```

`decode_batch` maps `(shots, num_detectors)` bool detection events to
`(shots, num_observables)` predicted logical-observable flips. The harness
scores a shot as a logical error when any predicted observable differs from
the true flip recorded at sampling time.

The neural decoder does not fit `from_dem` (it needs trained weights), so the
registry accepts spec strings: `pymatching`, `fusion_blossom`, or
`neural:<checkpoint-or-directory>` with per-distance files `d{d}.pt`.

## Determinism and provenance

- Every (distance, error rate) block is sampled with a seed derived
  deterministically from the config seed and grid position; regenerating a
  dataset — or one block of it — reproduces identical bits.
- Dataset metadata records the config snapshot, its hash, and the versions of
  qecbench, Stim, numpy, and Python that produced it.
- Results files (`qecbench-results-v1` schema) record the dataset config hash,
  environment details, and raw error *counts* (not just rates) so error bars
  can always be recomputed.
- Training and evaluation data come from datasets with different seeds
  (`train_v1` vs `benchmark_v1`); models are never scored on shots they saw.

## Known approximations and limits

- The DEM-to-matching-graph conversion treats components of a decomposed
  composite error as independent mechanisms, and keeps parallel edges with
  different observable masks as separate edges — the standard matching
  approximation, shared with the baselines being compared.
- Fusion Blossom has no batch API here, so it pays a per-shot Python loop in
  latency measurements. Its numbers measure this integration, not the fastest
  conceivable binding of the library.
- The v1 neural decoder is a per-distance MLP; it must be retrained per code
  distance and does not condition on the physical error rate.
- Latency is measured as batch wall time divided by shots on a single CPU
  core, which is the throughput view; streaming single-shot latency is higher
  and not yet reported.
