"""Stim circuit construction for the supported code tasks and noise models."""

from __future__ import annotations

import stim


def build_circuit(
    code_task: str,
    distance: int,
    rounds: int,
    error_rate: float,
    noise_model: str,
) -> stim.Circuit:
    """Build a noisy memory-experiment circuit.

    Noise models (p = error_rate):

    - ``circuit_depolarizing``: standard circuit-level depolarizing noise — every
      Clifford is followed by depolarization at strength p, resets flip with
      probability p, measurements are preceded by a flip with probability p, and
      data qubits depolarize before each round. This is the conventional setting
      decoders are benchmarked under.
    - ``phenomenological``: data-qubit depolarization before each round plus
      measurement flips only; gates themselves are perfect.
    - ``code_capacity``: a single round with data-qubit noise only and perfect
      measurement.
    """
    if noise_model == "circuit_depolarizing":
        kwargs = dict(
            after_clifford_depolarization=error_rate,
            after_reset_flip_probability=error_rate,
            before_measure_flip_probability=error_rate,
            before_round_data_depolarization=error_rate,
        )
    elif noise_model == "phenomenological":
        kwargs = dict(
            before_measure_flip_probability=error_rate,
            before_round_data_depolarization=error_rate,
        )
    elif noise_model == "code_capacity":
        if rounds != 1:
            raise ValueError("code_capacity noise uses exactly 1 round")
        kwargs = dict(before_round_data_depolarization=error_rate)
    else:
        raise ValueError(f"unknown noise_model {noise_model!r}")

    return stim.Circuit.generated(
        code_task,
        distance=distance,
        rounds=rounds,
        **kwargs,
    )
