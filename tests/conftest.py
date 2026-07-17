import os

# Keep BLAS single-threaded in tests; on low-RAM machines OpenBLAS's per-core
# buffer allocation can fail before any test runs.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import pytest

from qecbench.config import DatasetConfig
from qecbench.datagen import generate_dataset, load_block


@pytest.fixture(scope="session")
def dataset_root(tmp_path_factory):
    """A small generated dataset shared across decoder/harness tests."""
    config = DatasetConfig(
        name="shared",
        distances=[3],
        error_rates=[0.01],
        shots=2000,
        seed=99,
    )
    return generate_dataset(config, tmp_path_factory.mktemp("data"))


@pytest.fixture(scope="session")
def block(dataset_root):
    return load_block(dataset_root, 3, 0.01)
