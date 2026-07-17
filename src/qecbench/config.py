"""Dataset configuration: the single source of truth for what data gets generated.

Every dataset is fully described by a YAML file in ``configs/``. The config is
hashed and embedded in the generated dataset's metadata so any results file can
be traced back to the exact generation parameters.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

NOISE_MODELS = ("circuit_depolarizing", "phenomenological", "code_capacity")

CODE_TASKS = (
    "surface_code:rotated_memory_z",
    "surface_code:rotated_memory_x",
    "surface_code:unrotated_memory_z",
    "surface_code:unrotated_memory_x",
    "repetition_code:memory",
)


@dataclass(frozen=True)
class DatasetConfig:
    """Parameters for one synthetic syndrome dataset (a grid of distance x error rate)."""

    name: str
    distances: list[int]
    error_rates: list[float]
    shots: int
    code_task: str = "surface_code:rotated_memory_z"
    noise_model: str = "circuit_depolarizing"
    rounds: int | str = "distance"  # int, or "distance" for rounds == d
    seed: int = 0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.code_task not in CODE_TASKS:
            raise ValueError(f"unknown code_task {self.code_task!r}; expected one of {CODE_TASKS}")
        if self.noise_model not in NOISE_MODELS:
            raise ValueError(
                f"unknown noise_model {self.noise_model!r}; expected one of {NOISE_MODELS}"
            )
        if not self.distances or any(d < 3 or d % 2 == 0 for d in self.distances):
            raise ValueError("distances must be non-empty odd integers >= 3")
        if not self.error_rates or any(not 0 < p < 1 for p in self.error_rates):
            raise ValueError("error_rates must be non-empty probabilities in (0, 1)")
        if self.shots <= 0:
            raise ValueError("shots must be positive")
        if isinstance(self.rounds, str) and self.rounds != "distance":
            raise ValueError('rounds must be an int or the string "distance"')
        if isinstance(self.rounds, int) and self.rounds <= 0:
            raise ValueError("rounds must be positive")

    def rounds_for(self, distance: int) -> int:
        if self.noise_model == "code_capacity":
            return 1
        return distance if self.rounds == "distance" else int(self.rounds)

    def to_dict(self) -> dict:
        return asdict(self)

    def config_hash(self) -> str:
        """Stable short hash identifying these generation parameters."""
        canonical = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]

    @classmethod
    def from_yaml(cls, path: str | Path) -> DatasetConfig:
        raw = yaml.safe_load(Path(path).read_text())
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: expected a YAML mapping")
        try:
            return cls(**raw)
        except TypeError as e:
            raise ValueError(f"{path}: {e}") from e
