"""qecbench: reproducible surface-code decoder benchmarks + a lightweight neural decoder."""

__version__ = "0.1.0"


def load_pretrained(distance: int, backend: str = "onnx"):
    """Download (once) and return a pretrained decoder. See :mod:`qecbench.pretrained`."""
    from qecbench.pretrained import load

    return load(distance, backend=backend)
