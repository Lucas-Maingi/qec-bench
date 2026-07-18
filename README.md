# qec-bench

A reproducible benchmark suite for surface-code quantum error correction
decoders, plus a lightweight neural decoder built for fast CPU inference.

[**Interactive results dashboard**](https://lucas-maingi.github.io/qec-bench/) ·
[Architecture](docs/architecture.md) ·
[Reproducing the numbers](docs/reproducing.md) ·
[Training](docs/training.md)

## Why

- [PyMatching](https://github.com/oscarhiggott/PyMatching) and
  [Fusion Blossom](https://github.com/yuewuo/fusion-blossom) are the standard
  classical (MWPM) baselines everyone compares against.
- The landmark neural decoders (Google DeepMind's AlphaQubit line) are closed
  source; NVIDIA's open release is GPU-oriented.
- What's missing is a rigorous, **reproducible, open benchmark** that runs the
  available decoders side by side under standardized noise models — and a
  small, well-documented neural decoder that's practical to run **on a CPU**.

qec-bench fills that gap. It is an ML engineering project end to end:
versioned data generation, traceable experiments, one-command benchmark
reproduction, tests, CI, and honest reporting — including where the neural
decoder loses.

## Install

```bash
pip install -e ".[fusion,train,dev]"
```

Requires Python 3.10+. No GPU needed for anything in the default workflow.

## Five minutes to a result

```bash
# generate a small syndrome dataset (seconds)
qecbench generate --config configs/datasets/dev.yaml --out data

# benchmark the classical baselines on it (seconds)
qecbench benchmark --dataset data/dev --decoders pymatching,fusion_blossom --out runs/dev.json
```

You now have logical-error-rate and latency numbers, with full provenance, in
`runs/dev.json` — and `qecbench report runs/dev.json` renders them as
markdown tables. The complete pipeline — including training the neural decoder
locally on CPU — is three commands and documented in
[docs/reproducing.md](docs/reproducing.md).

## What the benchmark measures

For every decoder × code distance × physical error rate cell:

- **logical error rate** with binomial standard errors, from raw counts;
- **decode latency** (µs/shot, batch decoding on a single CPU core);
- against **standardized noise** (circuit-level depolarizing, phenomenological,
  or code-capacity) generated with [Stim](https://github.com/quantumlib/Stim).

Training data and benchmark data come from different seeds; models are never
scored on shots they trained on.

## The neural decoder

A deliberately small per-distance MLP (~150k parameters at d=7) trained on
Stim-sampled syndromes, shipping with checkpointed/resumable training, JSONL
experiment tracking, and single-digit-microsecond CPU inference. It beats the
MWPM baselines in some near-threshold cells and loses clearly at low error
rates — both results are reported. See [docs/training.md](docs/training.md).

## Project layout

```
src/qecbench/      installable package: datagen, decoders, train, eval, CLI
configs/           versioned dataset + training configs (the source of truth)
tests/             pytest suite
dashboard/         static results dashboard (no build step, no dependencies)
results/           committed benchmark results (the dashboard reads these)
docs/              architecture, training, reproduction guides
```

## Development

```bash
pytest        # test suite
ruff check .  # lint
```

CI runs both on every push.

## License

MIT
