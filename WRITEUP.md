# qec-bench: an open, reproducible benchmark for surface-code decoders (and a neural decoder that runs on your laptop)

*Lucas Maingi, July 2026*

## The gap

Quantum error correction decoding has a strange open-source landscape right
now. The classical baselines are excellent and open:
[PyMatching](https://github.com/oscarhiggott/PyMatching) and
[Fusion Blossom](https://github.com/yuewuo/fusion-blossom) both implement
minimum-weight perfect matching (MWPM) with fast native cores, and everyone
benchmarks against them. The headline *neural* decoders are a different story:
Google DeepMind's AlphaQubit results are landmark papers without an
installable package or public weights, and NVIDIA's open-source neural decoder
release is built around GPU inference.

So two things are missing. First, a **neutral, reproducible benchmark
harness** — not a table in someone's paper, but a tool you can run yourself
that produces logical error rates *and* latency for every open decoder under
identical, versioned noise models. Second, a neural decoder that is **small,
documented, and CPU-practical** — something you can `pip install` and study,
not a artifact you can only read about.

[qec-bench](https://github.com/Lucas-Maingi/qec-bench) is my attempt at both.
To be clear about positioning: I am not trying to out-train DeepMind or
NVIDIA. The benchmark suite is the product; the neural decoder is a rigorous,
honest baseline that demonstrates the full ML engineering lifecycle around it.

## What it does

One config-driven pipeline, three commands:

```bash
qecbench generate  --config configs/datasets/benchmark_v1.yaml --out data
qecbench train     --config configs/train/mlp_v1.yaml --data data/train_v1 --out weights
qecbench benchmark --dataset data/benchmark_v1 \
    --decoders "pymatching,fusion_blossom,neural:weights" \
    --out results/benchmark_v1.json
```

- **Data**: rotated surface-code memory experiments sampled with
  [Stim](https://github.com/quantumlib/Stim) under circuit-level depolarizing
  noise, across distances 3/5/7 and six physical error rates. Every dataset is
  described by a YAML config that is validated, content-hashed, and embedded
  in the output metadata; generation is seeded per block and bit-reproducible.
- **Decoders** implement one interface (`decode_batch: detection events →
  predicted observable flips`). PyMatching consumes the Stim detector error
  model natively; for Fusion Blossom I convert the DEM into its
  integer-weighted graph format. The neural decoder is a deliberately small
  per-distance MLP (~150k parameters at d=7).
- **The harness** scores every decoder on every (distance, error-rate) cell —
  200,000 shots each — and emits a single results JSON with raw error counts,
  binomial error bars, per-shot latency, and full provenance (config hash,
  library versions, host hardware). Training and evaluation datasets use
  different seeds; no decoder is ever scored on shots it trained on.
- **A static dashboard** (no build step, no external dependencies) renders
  that JSON: log-log logical-error-rate curves per distance, a CPU latency
  plot, and the complete results table.

## Results, including the losses

Selected cells from `results/benchmark_v1.json` (200,000 shots each,
circuit-level depolarizing noise, measured on a 6th-gen i7 laptop CPU;
PyMatching and Fusion Blossom agree to within counting noise everywhere, so
one MWPM column is shown):

Neural column is the shipped model (v3 — see the progression below):

| d | p | MWPM LER | Neural LER | MWPM µs/shot | Neural µs/shot |
|---|-------|----------|-----------|------|------|
| 3 | 0.001 | 7.8×10⁻⁴ | **6.2×10⁻⁴** | 0.2 | 10.9 |
| 3 | 0.01  | 5.88×10⁻² | **5.45×10⁻²** | 1.3 | 8.7 |
| 5 | 0.001 | **1.4×10⁻⁴** | 2.3×10⁻⁴ | 0.9 | 9.6 |
| 5 | 0.01  | **8.30×10⁻²** | 1.34×10⁻¹ | 9.6 | 8.5 |
| 7 | 0.001 | **3.5×10⁻⁵** | 2.9×10⁻³ | 2.5 | 12.1 |
| 7 | 0.01  | **1.03×10⁻¹** | 3.60×10⁻¹ | 35.3 | 12.8 |

The neural decoder wins every distance-3 cell outright, sits within ~1.6× of
MWPM at distance 5, and trails at distance 7 (where matching is near-optimal).
Its latency is nearly flat in distance and error rate, while matching slows as
syndromes get denser — at d=7 near threshold the MLP's batch decode is faster
than PyMatching.

One more result the pipeline itself produced: an earlier informal run had
evaluated models on the same generated shots they trained on, and looked
substantially better at d=5. The disjoint-seed train/benchmark split exposed
that as leakage. That is what the rigor is *for*.

A three-model progression, all in the committed results, tells the story the
benchmark was built to tell:

- **v1 → v2: capacity alone doesn't help.** A v2 model with 4× the parameters
  (512×256 hidden, lower dropout) is *worse* than v1 in almost every cell — its
  best validation epochs were 12/5/4 for d=3/5/7, i.e. it overfits the fixed
  600k-shot training budget almost immediately. The binding constraint is
  training data, not model size.
- **v2 → v3: more data is the lever.** Same 512×256 architecture, trained on a
  5×-larger dataset (3M samples per distance) on a free Kaggle GPU. This is the
  payoff the diagnosis predicted: at d=5, p=0.01 the logical error rate drops
  from v1's 2.06×10⁻¹ to **1.34×10⁻¹** (the gap to MWPM narrows from ~2.5× to
  ~1.6×); at d=7, p=0.01 from 4.29×10⁻¹ to **3.60×10⁻¹**; and at d=3 v3 beats
  MWPM in every cell. It still trails MWPM at d=7 (where matching is
  near-optimal), reported honestly. All three versions stay in the benchmark as
  the record — the *method* (diagnose the bottleneck from the numbers, then fix
  the right thing) is the point, not any single model.
- **Batch throughput isn't streaming latency — and ONNX fixes the gap.** The
  harness measures both. Batched, the MLP is fast; but a real-time decoder is
  called once per syndrome round, and single-shot the PyTorch model paid ~150 µs
  of per-call dispatch overhead — far above PyMatching. Exporting the network to
  ONNX and serving it through ONNX Runtime cut that **2–3×** (d=3/5/7 p=0.01:
  148→45 µs, 153→59 µs, 153→76 µs) at **bit-identical** predictions, bringing
  single-shot latency into PyMatching's range. Because the neural decoder's
  latency is flat in code distance while matching's grows with syndrome density,
  the ONNX path overtakes MWPM at larger distances. The exported model is the
  shipped artifact: `load_pretrained(distance)` fetches it and decodes with no
  PyTorch dependency.

Three honest observations:

1. **MWPM is a strong baseline and PyMatching is absurdly fast.** Sub-2µs
   per shot at d=3 on a 2016 laptop core. Any neural decoder pitch that
   ignores this latency bar is incomplete.
2. **The MLP is competitive-to-winning at low distance** — it exploits
   correlations the matching approximation throws away — **and still trails at
   distance 7**, where MWPM is near-optimal. More training data closed much of
   the mid-distance gap (v1→v3) but did not erase the d=7 deficit. Every regime
   is in the dashboard; none is cherry-picked away.
3. **Engineering constraints are results too.** The shipped model decodes in
   ~10µs per shot on CPU and installs with `pip` — no GPU needed for inference.
   Training the larger v3 uses a free Kaggle GPU (~30 min); the smaller v1
   trains on a laptop CPU in ~40 minutes. That CPU-inference niche is what the
   GPU-oriented big-lab releases leave open.

## The ML engineering underneath

The project is structured to make the lifecycle visible, because that's the
point:

- **Versioned data pipeline** — configs are the source of truth; datasets
  carry their config hash and library versions; interrupted generation
  resumes per block.
- **Experiment tracking** — every training run writes its resolved config and
  a JSONL metrics line per epoch next to the checkpoint. Plain files:
  greppable, diffable, no tracking server.
- **Checkpointed, resumable training** — finished stages skip in
  milliseconds; interrupted runs resume from the last epoch; resuming with a
  mismatched config is an error. The included Colab notebook writes
  checkpoints directly to Drive so a free-tier disconnect costs at most one
  epoch.
- **One-command reproduction** — the results file, and every chart on the
  dashboard, regenerates from the commands above; the provenance footer tells
  you exactly how.
- **Tests and CI** — the public API, the data pipeline's determinism and
  resume behavior, decoder correctness (including cross-checking the two MWPM
  implementations against each other), and the harness schema are all under
  pytest, run with lint on every push.

## What's next

- Scale the training configs on free GPU tiers (the pipeline is ready; the
  quota is the bottleneck).
- A graph-structured model behind the same `Decoder` interface, so the
  benchmark can say something about *architecture*, not just this MLP.
- Best-effort CPU evaluation of NVIDIA's released model, or a documented
  explanation of why it's excluded from the CPU-latency comparison, with
  their published numbers cited for context.

If you work on QEC decoders and want your decoder in the comparison, the
`Decoder` interface is ~10 lines to implement — issues and PRs welcome.
