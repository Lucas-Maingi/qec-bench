# Training the neural decoder

## Model

`SyndromeMLP` — a per-distance multilayer perceptron over the flattened
detection-event vector (2×256 hidden units, dropout 0.1, ~150k parameters at
d=7). One binary logit per logical observable, trained with BCE loss on
syndromes sampled across the full range of physical error rates.

This is deliberately the *simplest* learnable decoder, chosen v1 for three
reasons:

1. It validates the entire pipeline (data → training → checkpoint → benchmark
   → dashboard) end to end before any architecture risk is taken.
2. It meets the product constraint: single-digit-microsecond CPU inference
   with a pip-installable dependency set.
3. It gives an honest neural floor. Where it loses to MWPM, the gap is real
   signal about what structure matters, not an artifact of a broken harness.

Known limitations: no weight sharing across distances (train once per d), no
conditioning on the error rate, and accuracy at low error rates is limited by
how few error events a fixed-size sample contains.

## Data

Training uses `data/train_v1` (seed 31337); the benchmark uses
`data/benchmark_v1` (seed 2026). Same grid, disjoint randomness — models are
never evaluated on shots they trained on. 100k shots per (distance, error
rate) cell × 6 error rates = 600k training samples per distance, split 90/10
train/validation.

## Running it

```bash
qecbench train --config configs/train/mlp_v1.yaml --data data/train_v1 --out weights
```

Works on a laptop CPU (~15–90 s/epoch depending on distance) and on any
CUDA device unchanged (`--device cuda`).

## Checkpointing and resumability

Training is designed for interruptible free-tier compute (Colab/Kaggle
sessions that can disconnect at any time):

- **Finished-stage skip**: if the target checkpoint already holds
  `epoch >= epochs`, `train` returns in milliseconds without loading data —
  so "run everything again" after a disconnect only redoes in-flight work.
- **In-run resume**: the checkpoint is rewritten after *every* epoch with
  model state, optimizer state, and the resolved config; an interrupted run
  continues from the last completed epoch.
- **Config guard**: resuming with a different config than the checkpoint was
  trained with is an error, not a silent retrain.

## Experiment tracking

Each run writes, next to its checkpoint:

- `d{d}_config.json` — the exact resolved hyperparameters;
- `d{d}_metrics.jsonl` — one JSON line per epoch: train loss, validation
  loss, validation logical error rate, wall seconds.

Plain files, greppable, diffable, and sufficient to reconstruct every
training curve without a tracking server.
