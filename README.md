# qec-bench

A reproducible benchmark suite for surface-code quantum error correction
decoders, plus a lightweight neural decoder built for fast CPU inference — and
downloadable pretrained so you can decode without training anything.

[**▶ Live results dashboard**](https://lucas-maingi.github.io/qec-bench/) ·
[Architecture](docs/architecture.md) ·
[Reproducing the numbers](docs/reproducing.md) ·
[Training](docs/training.md) ·
[Write-up](WRITEUP.md)

## 30-second demo — decode with a pretrained model, no training, no GPU

```bash
pip install "qecbench[onnx] @ git+https://github.com/Lucas-Maingi/qec-bench"
```

```python
import numpy as np
from qecbench import load_pretrained
from qecbench.pretrained import reference_circuit

decoder = load_pretrained(distance=5)                 # downloads once, caches, CPU-only
circuit = reference_circuit(distance=5, error_rate=0.005)
dets, obs = circuit.compile_detector_sampler().sample(20_000, separate_observables=True)

predictions = decoder.decode_batch(dets)              # ONNX runtime, ~10 µs/shot on CPU
print("logical error rate:", (predictions != obs).any(axis=1).mean())
```

The shipped model runs through ONNX Runtime with no PyTorch dependency, at
single-shot latency in PyMatching's range (see below).

## Headline results

Held-out benchmark, circuit-level depolarizing noise, 200k shots/cell, one
2016-era CPU core ([full interactive dashboard](https://lucas-maingi.github.io/qec-bench/)):

- **The neural decoder beats MWPM across distance 3** and sits within ~1.6× of
  it at distance 5; MWPM still wins at distance 7 (it's near-optimal there) —
  reported honestly, not cherry-picked.
- **A documented ML iteration story:** more model capacity alone made things
  *worse* (overfitting a fixed data budget); **5× more training data** (free
  Kaggle GPU) nearly halved the distance-5 error. The bottleneck was data, and
  the benchmark is what proved it.
- **Inference engineering:** exporting the model to ONNX cut single-shot CPU
  latency **~2–3×** versus PyTorch (into MWPM's range) at bit-identical
  accuracy.

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
# just run a pretrained decoder:      pip install "qecbench[onnx] @ git+https://github.com/Lucas-Maingi/qec-bench"
# full dev / benchmark / training:
git clone https://github.com/Lucas-Maingi/qec-bench && cd qec-bench
pip install -e ".[fusion,train,onnx,export,dev]"
```

Requires Python 3.10+. No GPU needed for anything in the default workflow —
including running the pretrained decoders.

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

A deliberately small per-distance MLP trained on Stim-sampled syndromes,
shipping with checkpointed/resumable training (free Kaggle/Colab GPU), JSONL
experiment tracking, ONNX export, and a CPU inference path with a measured
latency budget. Pretrained models for distances 3/5/7 are published as
[GitHub Release assets](https://github.com/Lucas-Maingi/qec-bench/releases/tag/v0.1.0)
and fetched on demand by `load_pretrained`. It beats the MWPM baselines at low
distance and loses at high distance — both reported. See
[docs/training.md](docs/training.md) and the [write-up](WRITEUP.md).

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
