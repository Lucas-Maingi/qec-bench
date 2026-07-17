import json

from qecbench.eval import run_benchmark


def test_run_benchmark_document(dataset_root, tmp_path):
    out = tmp_path / "results.json"
    doc = run_benchmark(dataset_root, ["pymatching"], out)

    assert doc["schema"] == "qecbench-results-v1"
    assert doc["dataset"]["name"] == "shared"
    assert doc["dataset"]["config_hash"]
    assert json.loads(out.read_text()) == doc

    (cell,) = doc["results"]
    assert cell["decoder"] == "pymatching"
    assert cell["distance"] == 3
    assert cell["shots"] == 2000
    assert cell["logical_errors"] == round(cell["logical_error_rate"] * cell["shots"])
    assert cell["stderr"] > 0
    assert cell["us_per_shot"] > 0
    assert cell["single_shot_samples"] == 1000
    assert 0 < cell["single_shot_p50_us"] <= cell["single_shot_p99_us"]


def test_run_benchmark_multiple_decoders(dataset_root):
    doc = run_benchmark(dataset_root, ["pymatching", "fusion_blossom"])
    assert {c["decoder"] for c in doc["results"]} == {"pymatching", "fusion_blossom"}
    lers = {c["decoder"]: c["logical_error_rate"] for c in doc["results"]}
    # Same MWPM problem, near-identical accuracy.
    assert abs(lers["pymatching"] - lers["fusion_blossom"]) < 0.01
