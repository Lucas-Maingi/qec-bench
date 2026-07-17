"""Checkpointed, resumable training for the neural decoder.

Designed for interruptible free-tier compute (Colab/Kaggle) but equally happy
on a laptop CPU for small distances:

- **Stage idempotence**: if the target checkpoint already holds a finished run
  (epoch == epochs), training is skipped entirely.
- **In-run resume**: the checkpoint is rewritten after every epoch with model,
  optimizer, and RNG-relevant state; an interrupted run continues from the
  last completed epoch.
- **Experiment tracking**: every epoch appends one JSON line (losses, val
  logical error rate, timing) to ``metrics.jsonl`` next to the checkpoint, and
  the resolved config is snapshotted to ``config.json``.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from qecbench.datagen import load_block
from qecbench.decoders.neural.model import SyndromeMLP


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters and data selection for one training run (one distance)."""

    distance: int
    error_rates: list[float] | None = None  # None: every rate present in the dataset
    hidden_sizes: list[int] = field(default_factory=lambda: [256, 256])
    dropout: float = 0.1
    epochs: int = 20
    batch_size: int = 512
    learning_rate: float = 1e-3
    val_fraction: float = 0.1
    seed: int = 0

    @classmethod
    def from_yaml(cls, path: str | Path) -> list[TrainConfig]:
        """A training YAML holds shared hyperparameters plus a list of distances."""
        raw = yaml.safe_load(Path(path).read_text())
        distances = raw.pop("distances")
        return [cls(distance=d, **raw) for d in distances]


def _load_training_arrays(
    dataset_root: Path, config: TrainConfig
) -> tuple[np.ndarray, np.ndarray]:
    dataset_meta = json.loads((dataset_root / "meta.json").read_text())
    rates = config.error_rates or [
        b["error_rate"] for b in dataset_meta["blocks"] if b["distance"] == config.distance
    ]
    xs, ys = [], []
    for p in sorted(set(rates)):
        block = load_block(dataset_root, config.distance, p)
        xs.append(block.detection_events)
        ys.append(block.observable_flips)
    if not xs:
        raise ValueError(f"dataset has no blocks at distance {config.distance}")
    return np.concatenate(xs), np.concatenate(ys)


def train(
    dataset_root: str | Path,
    config: TrainConfig,
    out: str | Path,
    device: str | None = None,
) -> Path:
    """Train one per-distance model; returns the checkpoint path.

    ``out`` is the checkpoint file (e.g. ``weights/d3.pt``). Idempotent and
    resumable as described in the module docstring.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    run_dir = out.parent
    metrics_path = run_dir / f"{out.stem}_metrics.jsonl"
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Finished-stage check happens before any data loading so a completed run
    # is skipped in milliseconds (the Seer resume pattern).
    checkpoint = None
    if out.exists():
        checkpoint = torch.load(out, map_location=device, weights_only=True)
        if checkpoint.get("config") != asdict(config):
            raise ValueError(
                f"{out} was trained with a different config; delete it or change --out"
            )
        if checkpoint["epoch"] >= config.epochs:
            print(
                f"d={config.distance}: {out} already trained to epoch "
                f"{checkpoint['epoch']}, skipping"
            )
            return out

    x, y = _load_training_arrays(Path(dataset_root), config)
    rng = np.random.default_rng(config.seed)
    order = rng.permutation(len(x))
    x, y = x[order], y[order]
    n_val = max(1, int(len(x) * config.val_fraction))
    x_train, y_train = x[n_val:], y[n_val:]
    x_val = torch.from_numpy(x[:n_val].astype(np.float32)).to(device)
    y_val = torch.from_numpy(y[:n_val].astype(np.float32)).to(device)

    model = SyndromeMLP(
        num_detectors=x.shape[1],
        num_observables=y.shape[1],
        hidden_sizes=tuple(config.hidden_sizes),
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    start_epoch = 0
    best_val_ler = float("inf")
    best_model_state = None
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"]
        best_val_ler = checkpoint.get("best_val_ler", float("inf"))
        best_model_state = checkpoint.get("best_model_state")
        print(f"d={config.distance}: resuming from epoch {start_epoch}")

    (run_dir / f"{out.stem}_config.json").write_text(json.dumps(asdict(config), indent=2))
    torch.manual_seed(config.seed + start_epoch)

    # Keep the training set as uint8 and cast per-batch: a float32 copy of a
    # large-distance dataset would not fit comfortably in 8 GB of RAM.
    x_train_t = torch.from_numpy(x_train.astype(np.uint8))
    y_train_t = torch.from_numpy(y_train.astype(np.uint8))

    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.perf_counter()
        model.train()
        perm = torch.randperm(len(x_train_t))
        total_loss, batches = 0.0, 0
        for i in range(0, len(perm), config.batch_size):
            idx = perm[i : i + config.batch_size]
            xb = x_train_t[idx].to(device).float()
            yb = y_train_t[idx].to(device).float()
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val)
            val_loss = loss_fn(val_logits, y_val).item()
            val_ler = ((val_logits > 0) != y_val.bool()).any(dim=1).float().mean().item()

        record = {
            "epoch": epoch + 1,
            "train_loss": total_loss / batches,
            "val_loss": val_loss,
            "val_ler": val_ler,
            "seconds": time.perf_counter() - epoch_start,
        }
        with metrics_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(
            f"d={config.distance} epoch {epoch + 1}/{config.epochs}  "
            f"train_loss={record['train_loss']:.4f}  val_loss={val_loss:.4f}  "
            f"val_LER={val_ler:.4f}  ({record['seconds']:.1f}s)"
        )

        if val_ler <= best_val_ler:
            best_val_ler = val_ler
            best_model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        torch.save(
            {
                "epoch": epoch + 1,
                "config": asdict(config),
                "hyperparameters": model.hyperparameters(),
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_ler": val_ler,
                # Inference uses the best-validation weights, not the last
                # epoch's: late epochs can overfit (seen at d=5 in v1).
                "best_val_ler": best_val_ler,
                "best_model_state": best_model_state,
            },
            out,
        )

    return out
