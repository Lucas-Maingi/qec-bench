import json

from qecbench.eval import run_benchmark
from qecbench.eval.report import report, results_markdown


def test_results_markdown(dataset_root, tmp_path):
    out = tmp_path / "results.json"
    doc = run_benchmark(dataset_root, ["pymatching"], out)

    md = results_markdown(doc)
    assert "## distance 3" in md
    assert "| pymatching |" in md
    assert "p=0.01" in md
    assert "µs" in md
    assert report(out) == md


def test_zero_error_cells_render_as_upper_bound(dataset_root, tmp_path):
    doc = run_benchmark(dataset_root, ["pymatching"])
    doc["results"][0]["logical_errors"] = 0
    doc["results"][0]["logical_error_rate"] = 0.0
    md = results_markdown(doc)
    assert "<5.0e-04" in md  # 1/2000 shots


def test_missing_cells_render_as_dash(dataset_root):
    doc = run_benchmark(dataset_root, ["pymatching"])
    doc["results"][0]["distance"] = 5  # orphan cell creates gaps at d=3 and d=5
    md = results_markdown(json.loads(json.dumps(doc)))
    assert "—" in md
