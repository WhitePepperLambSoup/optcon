import csv
from pathlib import Path

import numpy as np

from optcon import q
from optcon.benchmarks import bench, convergence
from optcon.benchmarks.bench import naive_decompose
from optcon.modes import decompose
from optcon.propagation import gaussian_field


def test_default_benchmark_outputs_are_kept_outside_documentation():
    root = Path(__file__).resolve().parents[1]
    artifact_dir = root / "optcon-artifacts" / "benchmarks"

    assert bench.BENCHMARK_DATA_PATH == artifact_dir / "benchmark_operations.csv"
    assert convergence.DEFAULT_OUTPUT_PATH == artifact_dir / "solver_convergence.csv"


def test_write_results_persists_measurements_with_provenance(tmp_path):
    output = tmp_path / "benchmark.csv"

    bench.write_results([("example", 0.125, 2.5)], output)

    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows == [
        {
            "operation": "example",
            "best_seconds": "0.125",
            "peak_mib": "2.5",
            "python": bench.platform.python_version(),
            "platform": bench.platform.platform(),
        }
    ]


def test_naive_modal_baseline_matches_separable_decomposition():
    field = gaussian_field(
        waist=q(50.0, "um"),
        wavelength=q(633.0, "nm"),
        samples=64,
        extent=q(1600.0, "um"),
    )
    expected = decompose(field, waist=q(50.0, "um"), max_order=2)
    baseline = naive_decompose(field, waist=q(50.0, "um"), max_order=2)
    assert set(baseline) == set(expected)
    assert np.allclose(
        [baseline[key] for key in sorted(baseline)],
        [expected[key] for key in sorted(expected)],
        rtol=1e-12,
        atol=1e-12,
    )
