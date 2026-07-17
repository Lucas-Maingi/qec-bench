# qec-bench

A reproducible benchmark suite for surface-code quantum error correction
decoders, plus a lightweight neural decoder built for fast CPU inference.

**Status: early development.** The data pipeline is functional; baseline
decoders, the neural decoder, the benchmark harness, and the dashboard are
landing incrementally.

## Why

- [PyMatching](https://github.com/oscarhiggott/PyMatching) and
  [Fusion Blossom](https://github.com/yuewuo/fusion-blossom) are the standard
  classical (MWPM) baselines everyone compares against.
- The landmark neural decoders (Google DeepMind's AlphaQubit line) are closed
  source; NVIDIA's open release is GPU-oriented.
- What's missing is a rigorous, **reproducible, open benchmark** that runs the
  available decoders side by side under standardized noise models — and a
  small, well-documented neural decoder that's practical to run **on a CPU**.

qec-bench aims to fill that gap. It is an ML engineering project: versioned
data generation, traceable experiments, one-command benchmark reproduction,
tests, CI, and honest reporting — including where the neural decoder loses.

## Install

```bash
pip install -e ".[dev]"
```

Requires Python 3.10+. Core dependencies are `stim`, `pymatching`, `numpy`,
and `pyyaml` — no GPU needed.

## Quick start

Generate a small syndrome dataset (seconds on a laptop):

```bash
qecbench generate --config configs/datasets/dev.yaml --out data
```

Every dataset is fully described by a YAML config (code task, noise model,
distances, error rates, shots, seed) and written with metadata — config hash,
stim version, per-block seeds — so any downstream result traces back to the
exact bits it was computed from. Generation is deterministic: same config,
same bytes.

## Project layout

```
src/qecbench/      the installable package
  config.py        dataset configs (YAML-backed, validated, hashed)
  datagen/         Stim circuit construction + syndrome sampling
  cli.py           `qecbench` command-line entry point
configs/           checked-in dataset / experiment configs
tests/             unit tests (pytest)
```

## Development

```bash
pytest        # run the test suite
ruff check .  # lint
```

## License

MIT
