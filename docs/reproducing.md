# Reproducing every benchmark number

All headline numbers derive from three commands. Total runtime on a 4-core
laptop CPU (no GPU): data generation ~2 minutes, training ~40 minutes,
benchmarking ~15 minutes.

```bash
pip install -e ".[fusion,train,dev]"

# 1. Generate both datasets (training and evaluation use different seeds).
qecbench generate --config configs/datasets/train_v1.yaml --out data
qecbench generate --config configs/datasets/benchmark_v1.yaml --out data

# 2. Train the per-distance neural decoders.
qecbench train --config configs/train/mlp_v1.yaml --data data/train_v1 --out weights

# 3. Run the full benchmark.
qecbench benchmark --dataset data/benchmark_v1 \
    --decoders "pymatching,fusion_blossom,neural:weights" \
    --out results/benchmark_v1.json
```

Then view the dashboard:

```bash
python -m http.server 8321   # from the repo root
# open http://localhost:8321/dashboard/
```

## What "reproducible" means here

- **Data**: sampling is seeded per block from the config; the same config
  yields bit-identical datasets on any machine with the same Stim version
  (recorded in the dataset's `meta.json`).
- **Baselines**: PyMatching and Fusion Blossom are deterministic given the
  detector error model, so their logical error counts reproduce exactly.
- **Neural training**: seeded (data shuffling and torch RNG), so retraining on
  the same machine reproduces the checkpoint; bit-exact weight reproduction
  across different hardware/BLAS builds is not guaranteed — which is why the
  trained checkpoints and their metrics logs are the artifact of record, and
  the benchmark scores *those*.
- **Latency**: wall-clock numbers vary with hardware; the results file records
  the host (`environment.platform`, `environment.processor`) alongside every
  measurement. Compare latencies only within one results file.

## Low-RAM machines

Set `OPENBLAS_NUM_THREADS=1` (the CLI does this automatically) — on 8 GB
machines OpenBLAS's default per-core buffer allocation can fail outright.

## Interrupted runs

Both generation and training resume: finished (distance, error-rate) blocks
and finished/partial training runs are detected from their on-disk artifacts
and skipped or continued rather than recomputed.
