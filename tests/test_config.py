import pytest

from qecbench.config import DatasetConfig


def make(**overrides):
    base = dict(
        name="t",
        distances=[3],
        error_rates=[0.01],
        shots=100,
    )
    base.update(overrides)
    return DatasetConfig(**base)


def test_valid_config_roundtrip():
    cfg = make(distances=[3, 5], error_rates=[0.001, 0.01])
    d = cfg.to_dict()
    assert d["name"] == "t"
    assert d["distances"] == [3, 5]


def test_rounds_distance_keyword():
    cfg = make(rounds="distance")
    assert cfg.rounds_for(5) == 5
    cfg = make(rounds=10)
    assert cfg.rounds_for(5) == 10


def test_code_capacity_forces_single_round():
    cfg = make(noise_model="code_capacity", rounds="distance")
    assert cfg.rounds_for(7) == 1


@pytest.mark.parametrize(
    "bad",
    [
        dict(distances=[]),
        dict(distances=[4]),
        dict(distances=[2]),
        dict(error_rates=[]),
        dict(error_rates=[0.0]),
        dict(error_rates=[1.5]),
        dict(shots=0),
        dict(noise_model="nope"),
        dict(code_task="nope"),
        dict(rounds="sometimes"),
        dict(rounds=-1),
    ],
)
def test_invalid_configs_rejected(bad):
    with pytest.raises(ValueError):
        make(**bad)


def test_config_hash_stable_and_sensitive():
    a = make()
    b = make()
    c = make(seed=1)
    assert a.config_hash() == b.config_hash()
    assert a.config_hash() != c.config_hash()


def test_from_yaml(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "name: yam\ndistances: [3]\nerror_rates: [0.01]\nshots: 50\nseed: 7\n"
    )
    cfg = DatasetConfig.from_yaml(p)
    assert cfg.name == "yam"
    assert cfg.seed == 7


def test_from_yaml_rejects_unknown_keys(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "name: yam\ndistances: [3]\nerror_rates: [0.01]\nshots: 50\nbogus: 1\n"
    )
    with pytest.raises(ValueError, match="bogus"):
        DatasetConfig.from_yaml(p)
